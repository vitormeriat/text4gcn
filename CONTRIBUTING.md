# Contributing to Text4GCN

Thank you for your interest in contributing to Text4GCN! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Git

### Installation

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/text4gcn.git
   cd text4gcn
   ```

3. Install dependencies:
   ```bash
   poetry install --with dev,docs
   ```

4. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

### Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   make test
   ```

3. Run the linter (the same check as CI):
   ```bash
   make lint
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request on GitHub

## Code Style

- **[Ruff](https://docs.astral.sh/ruff/)** for linting; its rules live in `pyproject.toml` and CI fails on any violation
- **mypy** for type checking (runs in CI but does not block yet: annotations are incomplete)
- Line length 100; NumPy-style docstrings in new code

### Running Quality Checks

```bash
make lint          # ruff check, as in CI
make typecheck     # mypy
make all-checks    # lint + tests, as in CI
```

`make format` applies `ruff format`. The code base has not been formatted in one
pass yet, so format only the files you change.

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=text4gcn

# Run specific test file
poetry run pytest tests/test_builder.py

# Run with markers
poetry run pytest -m "not slow"
```

### Writing Tests

- Write tests for new functionality
- Use descriptive test names
- Include both unit and integration tests
- Use fixtures for test data
- Mark slow tests with `@pytest.mark.slow`

### Test Structure

```
tests/
├── test_builder.py           # Builder classes and the Builder enum
├── test_rwf_builders.py      # Word2Vec RWF, cosine similarity and LIWC edges
├── test_graph_refinement.py  # RefinementConfig / apply_refinement
├── test_graph_stats.py       # GraphBuildStats and the DCAI 2022 reproduction script
├── test_training_protocol.py # Seeds, dropout, early stopping, RWF weights
├── test_models.py            # GNN and data loading
├── test_dataset.py           # Input files and datasets helpers
├── test_device.py            # Device selection
├── test_unicode_corpus.py    # Non-ASCII corpora end to end
├── test_integration.py       # Public API and pipeline integration
└── data/                     # Test data files
```

## Documentation

### Building Documentation

The documentation is built with MkDocs (Material) and `mkdocs-jupyter`:

```bash
make docs        # mkdocs build --strict, as in CI
make docs-serve  # live preview at http://127.0.0.1:8000
```

### Documentation Standards

- Use NumPy-style docstrings (`Parameters` / `Returns` sections)
- Include type hints
- Provide examples in docstrings
- Update README.md for user-facing changes

## Pull Request Guidelines

### Before Submitting

1. Ensure all tests pass
2. Run quality checks (formatting, linting, type checking)
3. Update documentation if needed
4. Add tests for new functionality
5. Update `docs/changelog.md` if applicable

### PR Description

Include:
- Description of changes
- Motivation for changes
- Any breaking changes
- Screenshots (if UI changes)
- Test results

### Review Process

- All PRs require review
- Address reviewer feedback
- Keep PRs focused and reasonably sized
- Use draft PRs for work in progress

## Release Process

### Version Bumping

- Update the version in `pyproject.toml` and `text4gcn/__init__.py` (a test checks that they match)
- Update `docs/changelog.md`
- Create a GitHub release with the tag `v<version>`, e.g. `v2.0.0`

### Publishing

Publishing a GitHub release whose tag starts with `v` runs `publish.yml`: it checks
that the tag matches the version in `pyproject.toml`, runs the tests, builds the
package and uploads it to PyPI with the `PYPI_API_TOKEN` secret.

## Issue Guidelines

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Minimal code example

### Feature Requests

Include:
- Description of the feature
- Use case and motivation
- Proposed implementation (if applicable)
- Alternatives considered

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the golden rule

## Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for sensitive issues

Thank you for contributing to Text4GCN! 🚀
