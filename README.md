# conformal-rul

Probabilistic remaining-useful-life (RUL) prediction for turbofan engines, with
**calibrated uncertainty** — because in predictive maintenance a point estimate
without a trustworthy error bar is an invitation to either scrap a healthy machine
or run a failing one.

> **Status: under active development.** This README states the intent and design
> up front; results land as they are produced.

## What this is

- **Data**: NASA C-MAPSS turbofan degradation benchmark (all four subsets,
  FD001–FD004: 1–6 operating regimes, 1–2 fault modes).
- **Models**: LSTM and Transformer encoders trained from scratch in PyTorch, against
  a LightGBM baseline on windowed summary statistics — the baseline is there to be
  beaten, and to show when it isn't.
- **Uncertainty**: split conformal prediction and conformalized quantile regression
  (CQR), including **Mondrian (per-group) calibration** by operating regime and by
  predicted-RUL band — distribution-free finite-sample coverage, not Gaussian hand-waving.
- **Serving**: models exported to ONNX, served by a FastAPI service with no PyTorch
  dependency at inference time; Docker image deployed to **AWS Lambda behind API
  Gateway via Terraform**, CI/CD through GitHub Actions.

## Why these choices

- *Conformal rather than MC-dropout/ensembles*: exact finite-sample marginal coverage
  under exchangeability, model-agnostic, and cheap at inference.
- *Mondrian buckets*: marginal coverage hides regime-level under-coverage; calibrating
  per operating condition surfaces it and fixes it.
- *ONNX at the boundary*: training stacks should not dictate serving images. The
  Lambda container stays small and cold starts stay tolerable.
- *Serverless*: a demo endpoint that costs ~€0 at rest and needs no pet server.

## Layout

    src/conformal_rul/    package: data, models, conformal, training, serving
    tests/                pytest suite
    infra/                Terraform (ECR, Lambda, API Gateway, IAM/OIDC)
    docker/               serving image
    docs/                 architecture, results, deployment runbook, model card

## Results

Coming as runs complete — RMSE and NASA PHM08 score per subset, plus interval
coverage/width at 80/90/95 %.

## License

MIT
