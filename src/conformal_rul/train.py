"""Training CLI.

    python -m conformal_rul.train --subset FD001 --model lstm
    python -m conformal_rul.train --all

Artifacts land in models/<subset>/<model>/: weights, the fitted preprocessor,
calibration/test predictions for the conformal step, and a manifest with every
knob and metric of the run. MLflow tracks the same numbers locally (./mlruns).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import COVERAGES, QUANTILES, SUBSETS, DataConfig, TrainConfig, train_config
from .data import Bundle, prepare, window_stats
from .evaluate import point_metrics

try:
    import mlflow
except ImportError:  # tracking is optional; training must not depend on it
    mlflow = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pinball_loss(q_pred, y, quantiles):
    e = y.unsqueeze(1) - q_pred
    return torch.maximum(quantiles * e, (quantiles - 1.0) * e).mean()


def predict_net(net, X: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    """[N, 1+Q] predictions with quantile columns monotonized by sorting
    (Chernozhukov et al. rearrangement) so intervals cannot cross."""
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            outs.append(net(xb).cpu().numpy())
    pred = np.concatenate(outs)
    pred[:, 1:] = np.sort(pred[:, 1:], axis=1)
    return pred


def train_net(bundles: dict[str, Bundle], cfg: TrainConfig, device) -> tuple:
    from torch.utils.data import DataLoader, TensorDataset

    from .models.nets import build_net

    set_seed(cfg.seed)
    tr, va = bundles["train"], bundles["val"]
    net = build_net(cfg.model, tr.X.shape[-1], cfg).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.y)),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    quantiles = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_rmse, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(cfg.epochs):
        net.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            out = net(xb)
            loss = torch.nn.functional.mse_loss(out[:, 0], yb) + cfg.pinball_weight * pinball_loss(
                out[:, 1:], yb, quantiles
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)

        val_pred = predict_net(net, va.X, device)
        val_rmse = float(np.sqrt(np.mean((val_pred[:, 0] - va.y) ** 2)))
        sched.step(val_rmse)
        history.append({"epoch": epoch, "train_loss": total / len(tr.X), "val_rmse": val_rmse})
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break

    net.load_state_dict(best_state)
    return net, history, best_rmse


def run(subset: str, model: str, out_root: str = "models", **overrides) -> dict:
    t0 = time.time()
    data_cfg = DataConfig(subset=subset)
    cfg = train_config(model, **overrides)
    bundles, pre = prepare(data_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "gbm":
        from .models.gbm import fit_gbm, predict_gbm

        stats = {k: window_stats(b.X) for k, b in bundles.items()}
        boosters = fit_gbm(stats["train"], bundles["train"].y, stats["val"], bundles["val"].y,
                           seed=cfg.seed)
        preds = {k: predict_gbm(boosters, s) for k, s in stats.items()}
        val_rmse = float(np.sqrt(np.mean((preds["val"][:, 0] - bundles["val"].y) ** 2)))
        history, net = [], None
    else:
        net, history, val_rmse = train_net(bundles, cfg, device)
        preds = {k: predict_net(net, b.X, device) for k, b in bundles.items()}

    metrics = point_metrics(preds["test"][:, 0], bundles["test"].y, data_cfg.rul_cap)
    manifest = {
        "subset": subset,
        "model": model,
        "data": asdict(data_cfg),
        "train": asdict(cfg),
        "quantiles": QUANTILES,
        "coverages": COVERAGES,
        "n_features": pre.n_features,
        "n_regimes": pre.n_regimes,
        "feature_names": pre.feature_names,
        "sizes": {k: len(b.y) for k, b in bundles.items()},
        "val_rmse": round(val_rmse, 3),
        "test": metrics,
        "epochs_ran": len(history),
        "runtime_s": round(time.time() - t0, 1),
        "device": str(device),
        "platform": platform.platform(),
    }

    out = Path(out_root) / subset / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocessor.json").write_text(pre.to_json())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(
        out / "preds.npz",
        cal_pred=preds["cal"],
        cal_y=bundles["cal"].y,
        cal_regime=bundles["cal"].regime,
        test_pred=preds["test"],
        test_y=bundles["test"].y,
        test_regime=bundles["test"].regime,
    )
    if net is not None:
        torch.save(net.state_dict(), out / "model.pt")

    if mlflow is not None:
        mlflow.set_experiment("conformal-rul")
        with mlflow.start_run(run_name=f"{subset}-{model}"):
            mlflow.log_params({"subset": subset, "model": model, **asdict(cfg)})
            test_metrics = {f"test_{k}": v for k, v in metrics.items()}
            mlflow.log_metrics({"val_rmse": val_rmse, **test_metrics})
            mlflow.log_artifact(str(out / "manifest.json"))

    print(
        f"{subset}/{model}: val RMSE {val_rmse:.2f} | test RMSE {metrics['rmse']:.2f} "
        f"| NASA {metrics['nasa_score']:.0f} | {manifest['epochs_ran']} epochs "
        f"| {manifest['runtime_s']}s"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=SUBSETS, default="FD001")
    ap.add_argument("--model", choices=("lstm", "transformer", "gbm"), default="lstm")
    ap.add_argument("--all", action="store_true", help="every subset x model combination")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    overrides = {"epochs": args.epochs} if args.epochs else {}
    if args.all:
        for subset in SUBSETS:
            for model in ("gbm", "lstm", "transformer"):
                run(subset, model, args.out, **overrides)
    else:
        run(args.subset, args.model, args.out, **overrides)


if __name__ == "__main__":
    main()
