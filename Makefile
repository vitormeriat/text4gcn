.PHONY: help install test test-cov lint format format-check typecheck all-checks clean build docs docs-serve pre-commit pre-commit-run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with development and docs dependencies
	poetry install --with dev,docs

test: ## Run tests
	poetry run pytest tests/ -q -o addopts=

test-cov: ## Run tests with coverage
	poetry run pytest tests/ --cov=text4gcn --cov-report=html --cov-report=term

lint: ## Lint (the CI gate)
	poetry run ruff check .

format: ## Format code
	poetry run ruff format text4gcn tests scripts

format-check: ## Check formatting without changing files
	poetry run ruff format --check text4gcn tests scripts

typecheck: ## Type check (not yet a CI gate)
	poetry run mypy text4gcn

all-checks: lint test ## Run the CI checks

clean: ## Clean build artifacts
	rm -rf build/ dist/ site/ htmlcov/ .pytest_cache/ .coverage coverage.xml *.egg-info/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

build: ## Build sdist and wheel
	poetry build

docs: ## Build the documentation (strict, as in CI)
	poetry run mkdocs build --strict

docs-serve: ## Serve the documentation locally
	poetry run mkdocs serve

pre-commit: ## Install pre-commit hooks
	poetry run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	poetry run pre-commit run --all-files
