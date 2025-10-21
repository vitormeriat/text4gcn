.PHONY: help install install-dev test lint format clean build publish docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	poetry install --no-dev

install-dev: ## Install the package with development dependencies
	poetry install

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=text4gcn --cov-report=html --cov-report=term

lint: ## Run linting
	poetry run flake8 text4gcn tests
	poetry run mypy text4gcn

format: ## Format code
	poetry run black text4gcn tests
	poetry run isort text4gcn tests

format-check: ## Check code formatting
	poetry run black --check text4gcn tests
	poetry run isort --check-only text4gcn tests

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	poetry build

publish: ## Publish to PyPI
	poetry publish

docs: ## Build documentation
	cd docs && poetry run sphinx-build -b html . _build/html

docs-serve: ## Serve documentation locally
	cd docs && poetry run sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

pre-commit: ## Install pre-commit hooks
	poetry run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	poetry run pre-commit run --all-files

all-checks: format-check lint test ## Run all checks (format, lint, test)
