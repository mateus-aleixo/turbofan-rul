"""The training and export path, on random weights and synthetic data.

These modules had no tests: the API, the conformal layer and the data pipeline
were covered, while the networks, the loss, the quantile rearrangement and the
ONNX export were not. Everything here runs on CPU in seconds with no dataset and
no downloads, matching the CI contract of the sibling repos.
"""

import numpy as np
import pytest

# The fast CI job installs .[dev,serve] on purpose: serving is torch-free and its
# tests run against the committed models/ registry on onnxruntime alone. These
# tests exercise the training path, so they need the train extra and skip without
# it. The `train-tests` CI job installs it so they are actually run somewhere.
pytest.importorskip("torch", reason="training path requires the train extra")

import torch  # noqa: E402

from conformal_rul.config import QUANTILES, TrainConfig  # noqa: E402
from conformal_rul.models.nets import build_net  # noqa: E402
from conformal_rul.train import pinball_loss, predict_net, set_seed  # noqa: E402

N_FEATURES = 11
WINDOW = 30


def _windows(n: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, WINDOW, N_FEATURES)).astype(np.float32)


@pytest.mark.parametrize("model", ["lstm", "transformer"])
def test_net_shape_is_mean_plus_one_column_per_quantile(model):
    net = build_net(model, N_FEATURES, TrainConfig())
    out = net(torch.from_numpy(_windows(8)))
    assert out.shape == (8, 1 + len(QUANTILES))


@pytest.mark.parametrize("model", ["lstm", "transformer"])
def test_predict_net_rearranges_quantiles(model):
    """Untrained heads emit unordered quantiles, which would produce intervals
    whose lower bound sits above the upper. predict_net sorts them, so a crossing
    is impossible before it ever reaches the conformal layer."""
    net = build_net(model, N_FEATURES, TrainConfig())
    pred = predict_net(net, _windows(16), device="cpu")

    assert pred.shape == (16, 1 + len(QUANTILES))
    quantile_cols = pred[:, 1:]
    assert np.all(np.diff(quantile_cols, axis=1) >= 0), "quantiles must be non-decreasing"


def test_build_net_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown net"):
        build_net("gru", N_FEATURES, TrainConfig())


def test_pinball_loss_is_asymmetric_in_the_right_direction():
    """The whole point of quantile regression: the 0.9 quantile must be punished
    harder for under-predicting than for over-predicting, and the 0.1 quantile the
    other way round. A symmetric loss here would silently give median behaviour."""
    y = torch.zeros(1)
    q_hi = torch.tensor([[0.9]])
    q_lo = torch.tensor([[0.1]])

    under_hi = pinball_loss(torch.tensor([[-1.0]]), y, q_hi)  # predicted below truth
    over_hi = pinball_loss(torch.tensor([[1.0]]), y, q_hi)   # predicted above truth
    assert under_hi > over_hi

    under_lo = pinball_loss(torch.tensor([[-1.0]]), y, q_lo)
    over_lo = pinball_loss(torch.tensor([[1.0]]), y, q_lo)
    assert over_lo > under_lo

    assert pinball_loss(torch.zeros(1, 1), y, q_hi).item() == pytest.approx(0.0)


def test_set_seed_makes_initialisation_reproducible():
    set_seed(42)
    a = build_net("lstm", N_FEATURES, TrainConfig())
    set_seed(42)
    b = build_net("lstm", N_FEATURES, TrainConfig())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("model", ["lstm", "transformer"])
def test_one_optimisation_step_reduces_the_loss(model):
    """Not a claim about accuracy: a claim that gradients reach the heads at all.
    A net wired so the quantile heads are detached would train silently forever."""
    set_seed(0)
    net = build_net(model, N_FEATURES, TrainConfig())
    x = torch.from_numpy(_windows(32, seed=1))
    y = torch.from_numpy(np.linspace(0, 125, 32).astype(np.float32))
    quantiles = torch.tensor(QUANTILES, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)

    def loss_now():
        out = net(x)
        return torch.nn.functional.mse_loss(out[:, 0], y) + pinball_loss(
            out[:, 1:], y, quantiles
        )

    before = loss_now().item()
    for _ in range(5):
        opt.zero_grad()
        loss_now().backward()
        opt.step()
    assert loss_now().item() < before


@pytest.mark.parametrize("model", ["lstm", "transformer"])
def test_onnx_export_matches_torch(tmp_path, model):
    """The serving image runs onnxruntime, never torch, so an export that drifts
    from the network is served silently. conformal-seg gates its export on parity;
    this repo checked it only inside the export script."""
    ort = pytest.importorskip("onnxruntime")

    set_seed(3)
    net = build_net(model, N_FEATURES, TrainConfig()).eval()
    path = tmp_path / "model.onnx"
    dummy = torch.from_numpy(_windows(2, seed=5))
    torch.onnx.export(
        net, dummy, str(path),
        input_names=["windows"], output_names=["prediction"],
        dynamic_axes={"windows": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=18, verbose=False,
    )

    x = _windows(6, seed=7)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {"windows": x})[0]

    assert got.shape == ref.shape
    assert float(np.abs(ref - got).max()) < 1e-4
