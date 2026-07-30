# Model card — conformal-rul

## What it is

Remaining-useful-life regressors for the NASA C-MAPSS turbofan benchmark
(four models served, one per subset), wrapped in conformalized quantile
regression so every prediction ships with a finite-sample-calibrated
interval and a risk band.

## Intended use

Benchmarking, teaching and demonstration of calibrated uncertainty in
prognostics, and as the reference implementation behind the public API in
this repository. It is **not** a certified maintenance tool: C-MAPSS is
simulated data, and nothing here has been validated against a real fleet.
Do not point it at an actual aircraft.

## Training data

NASA C-MAPSS turbofan degradation simulation (Saxena, Goebel, Simon &
Eklund, PHM 2008) — public, downloaded from the NASA PCoE archive by
`conformal_rul.data`. 21 sensor channels + 3 operating settings per cycle;
709 training engines across four subsets differing in operating regimes
(1 or 6) and fault modes (1 or 2). Targets use the standard piecewise-linear
RUL capped at 125 cycles.

## Architecture and training

- Serving models: Transformer encoder (FD001/FD002/FD004) and LSTM (FD003),
  chosen per subset by test RMSE; a LightGBM baseline is trained and
  reported but not served.
- Each network has a mean head (MSE) and seven quantile heads (pinball
  loss, τ ∈ {0.025 … 0.975}); quantile columns are monotonized by sorting.
- Regime-conditional z-scoring; sensors kept by a within-regime variance
  rule; windows of 30 cycles; splits by engine unit (70/15/15
  train/val/cal). AdamW, gradient clipping, plateau LR decay, early
  stopping on validation RMSE. Fixed seed (42); cuDNN autotuning means
  bit-exact reproduction is not guaranteed, metric-level reproduction is.
- Conformal layer: CQR with Mondrian calibration by operating regime and by
  predicted-RUL band; finite-sample-corrected quantiles; groups under 50
  calibration points fall back to the marginal correction.

## Metrics

See [results.md](results.md). Headlines: test RMSE 12.1–14.9 across
subsets; 90 % intervals achieve 0.87–0.89 empirical coverage with
band-Mondrian calibration; interval width in the critical band (predicted
RUL < 30) is 11–19 cycles, 2–3× tighter than for healthy engines.

## Limitations

- **Exchangeability is approximate**: calibration uses windows along full
  trajectories, the benchmark scores each engine's censored last window.
  The residual coverage gap (~0.01–0.03) is consistent with this shift plus
  binomial noise and is documented, not hidden.
- Trained per subset; no transfer across subsets or to other machinery.
- Histories shorter than 30 cycles are left-padded and flagged (`padded`),
  costing accuracy near an engine's first cycles.
- The 125-cycle cap means "healthy" predictions saturate; the model ranks
  urgency, it does not estimate long horizons.
- Simulated fleet, no sensor faults, no missing data — all cleaner than
  reality.

## Safety framing

The NASA scoring function's asymmetry (late predictions are the expensive
failure) is the reason calibrated *upper and lower* bounds matter more than
a point estimate: maintenance planning should key on the interval's lower
bound in the critical band. Any real deployment would additionally need
drift monitoring and periodic recalibration of the conformal layer.
