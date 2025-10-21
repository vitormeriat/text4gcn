# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with unit and integration tests
- GitHub Actions workflows for CI/CD
- Pre-commit hooks for code quality
- Sphinx documentation setup
- Makefile for development tasks
- Contributing guidelines
- Type hints throughout the codebase
- Improved error handling

### Changed
- Updated project structure and organization
- Enhanced README.md with better documentation
- Improved code formatting and linting
- Updated dependencies and requirements

### Fixed
- Version consistency between pyproject.toml and __init__.py
- Import organization and structure
- Documentation formatting

## [2.0.0] - 2024-01-XX

### Added
- Initial release of Text4GCN framework
- Support for multiple text classification datasets (R8, R52, 20NG)
- Graph building methods (semantic, syntactic, frequency-based)
- GNN model implementations (GCN, GAT)
- Memory-optimized text graph construction
- Support for various adjacency matrix builders
- Rich console output for better user experience

### Features
- **Datasets**: Module for downloading and managing text classification datasets
- **Preprocess**: Text processing pipeline with NLP techniques
- **Builder**: Adjacency matrix construction with multiple methods
- **Models**: Graph Neural Network implementations for document classification

### Supported Datasets
- R8 (Reuters News Dataset with 8 labels)
- R52 (Reuters News Dataset with 52 labels)
- 20NG (Newsgroup Dataset)
- AG_NEWS (AG News Dataset)

### Supported Builders
- **Semantic**: LLM, BERT, Word2Vec, Cosine Similarity
- **Syntactic**: Constituency Parsing, Dependency Parsing, LIWC
- **Frequency**: TF-IDF

### Supported Models
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Custom GNN architectures

## [1.0.0] - 2023-XX-XX

### Added
- Initial development version
- Basic framework structure
- Core functionality implementation
