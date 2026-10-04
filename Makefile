.PHONY: data train calibrate export figures test lint serve image all \
        k8s-up k8s-verify k8s-down

KIND_CLUSTER ?= conformal-rul
K8S_RELEASE  ?= rul
K8S_IMAGE    ?= conformal-rul:ci

data:
	python -m conformal_rul.data

train:
	python -m conformal_rul.train --all

calibrate:
	python -m conformal_rul.conformal

export:
	python -m conformal_rul.export

figures:
	python scripts/make_figures.py

test:
	pytest

lint:
	ruff check src tests scripts

serve:
	uvicorn conformal_rul.serve.app:app --reload

image:
	docker build -t conformal-rul:dev .

# ---- Kubernetes -------------------------------------------------------------
# Mirrors .github/workflows/k8s.yml step for step, so a green local run and a
# green CI run mean the same thing.

k8s-up:
	docker build -t $(K8S_IMAGE) .
	kind create cluster --name $(KIND_CLUSTER) --config deploy/kind/cluster.yaml
	kind load docker-image $(K8S_IMAGE) --name $(KIND_CLUSTER)
	helm install $(K8S_RELEASE) deploy/helm/conformal-rul \
		-f deploy/kind/values.yaml --wait --timeout 5m
	kubectl get pods -o wide

k8s-verify:
	bash deploy/scripts/no-downtime.sh rollout
	bash deploy/scripts/no-downtime.sh kill-pod
	helm test $(K8S_RELEASE) --logs --timeout 3m

k8s-down:
	kind delete cluster --name $(KIND_CLUSTER)

all: data train calibrate export figures test
