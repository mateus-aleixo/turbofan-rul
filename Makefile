.PHONY: data train calibrate export figures test lint serve image all

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

all: data train calibrate export figures test
