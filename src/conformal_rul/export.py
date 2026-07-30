"""Export trained nets to ONNX so serving carries onnxruntime, not PyTorch.

Writes models/<subset>/<model>/model.onnx (dynamic batch axis, opset 17) and
models/<subset>/best.json pointing at the sequence model with the lowest test
RMSE — the one the API serves. Every export is parity-checked against the
torch forward pass before being accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import SUBSETS, train_config
from .models.nets import build_net


def export_model(model_dir: str | Path, atol: float = 1e-4) -> Path:
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    cfg = train_config(manifest["model"], **{
        k: manifest["train"][k]
        for k in ("hidden", "layers", "dropout", "heads", "ff_mult")
    })
    net = build_net(manifest["model"], manifest["n_features"], cfg)
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    net.eval()

    window = manifest["data"]["window"]
    dummy = torch.zeros(1, window, manifest["n_features"])
    out_path = model_dir / "model.onnx"
    torch.onnx.export(
        net,
        dummy,
        str(out_path),
        input_names=["windows"],
        output_names=["pred"],
        dynamic_axes={"windows": {0: "batch"}, "pred": {0: "batch"}},
        opset_version=17,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(4, window, manifest["n_features"])).astype(
        np.float32
    )
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"windows": x})[0]
    diff = float(np.abs(ref - got).max())
    if diff > atol:
        raise RuntimeError(f"{model_dir}: ONNX/torch mismatch {diff:.2e} > {atol}")
    print(f"{model_dir}: exported, torch parity {diff:.2e}")
    return out_path


def select_best(subset_dir: str | Path) -> str:
    """Lowest test RMSE among exported sequence models wins serving duty."""
    subset_dir = Path(subset_dir)
    candidates = {}
    for model in ("lstm", "transformer"):
        mf = subset_dir / model / "manifest.json"
        if mf.exists() and (subset_dir / model / "model.onnx").exists():
            candidates[model] = json.loads(mf.read_text())["test"]["rmse"]
    if not candidates:
        raise FileNotFoundError(f"no exported models under {subset_dir}")
    best = min(candidates, key=candidates.get)
    (subset_dir / "best.json").write_text(
        json.dumps({"model": best, "test_rmse": candidates[best]}, indent=2)
    )
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-root", default="models")
    args = ap.parse_args()
    root = Path(args.models_root)
    for subset in SUBSETS:
        for model in ("lstm", "transformer"):
            d = root / subset / model
            if (d / "model.pt").exists():
                export_model(d)
        best = select_best(root / subset)
        print(f"{subset}: serving {best}")


if __name__ == "__main__":
    main()
