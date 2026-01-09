.PHONY: help install install-dev data train test test-cov lint format type serve app docker-build docker-up clean

help:
	@echo "fraud-shield — common tasks"
	@echo "  make install        editable install (runtime only)"
	@echo "  make install-dev    editable install + dev tools + pre-commit hooks"
	@echo "  make data           download the ULB credit card fraud dataset (requires Kaggle API token)"
	@echo "  make train          train the LightGBM model with the default config"
	@echo "  make test           run pytest"
	@echo "  make test-cov       run pytest with coverage report"
	@echo "  make lint           run ruff check"
	@echo "  make format         run ruff format"
	@echo "  make type           run mypy strict"
	@echo "  make serve          run the FastAPI inference service"
	@echo "  make app            run the Streamlit demo"
	@echo "  make docker-build   build the docker image"
	@echo "  make docker-up      docker compose up the full stack"
	@echo "  make clean          remove caches and build artifacts"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"
	pre-commit install

data:
	python -m scripts.download_data

train:
	python -m scripts.train --config configs/lgbm.yaml

test:
	pytest

test-cov:
	pytest --cov=src/fraud_shield --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

format:
	ruff format src tests

type:
	mypy src

serve:
	uvicorn fraud_shield.api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t fraud-shield:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
