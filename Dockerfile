# One image, two habitats: runs as a normal web server locally
# (docker compose up) and unchanged on AWS Lambda — the Lambda Web Adapter
# extension translates function invocations into plain HTTP, so there is no
# Lambda-specific code path to drift.
# The adapter is its own stage so CI can build against a stand-in: anonymous pulls
# from ECR Public fail on shared runners ("toomanyrequests: Data limit exceeded").
ARG LAMBDA_ADAPTER_IMAGE=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1
FROM ${LAMBDA_ADAPTER_IMAGE} AS lambda-adapter

FROM python:3.12-slim

COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

COPY requirements/serve.txt requirements/serve.txt
RUN pip install --no-cache-dir -r requirements/serve.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Serving registry only — .dockerignore strips training artifacts (*.pt, preds.npz)
COPY models ./models

ENV MODEL_ROOT=/app/models \
    PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

EXPOSE 8000
CMD ["uvicorn", "conformal_rul.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
