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
   poetry install
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
   poetry run pytest
   ```

3. Run linting and formatting:
   ```bash
   poetry run black text4gcn tests
   poetry run isort text4gcn tests
   poetry run flake8 text4gcn tests
   poetry run mypy text4gcn
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

We follow these coding standards:

- **Black** for code formatting (line length: 88)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **Google-style** docstrings

### Running Quality Checks

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Run all checks
make all-checks
```

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
├── test_builder.py      # Builder module tests
├── test_dataset.py      # Dataset module tests
├── test_models.py       # Model tests
├── test_integration.py  # Integration tests
└── data/               # Test data files
```

## Documentation

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

### Documentation Standards

- Use Google-style docstrings
- Include type hints
- Provide examples in docstrings
- Update README.md for user-facing changes

## Pull Request Guidelines

### Before Submitting

1. Ensure all tests pass
2. Run quality checks (formatting, linting, type checking)
3. Update documentation if needed
4. Add tests for new functionality
5. Update CHANGELOG.md if applicable

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

- Update version in `pyproject.toml` and `text4gcn/__init__.py`
- Update CHANGELOG.md
- Create a release tag

### Publishing

Releases are automatically published to PyPI when a GitHub release is created.

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
