# Contributing

The canonical contributing guide lives at the repository root:

**[CONTRIBUTING.md](https://github.com/vitormeriat/text4gcn/blob/main/CONTRIBUTING.md)**

Quick commands:

```bash
poetry install
poetry run pytest tests/ -q -o addopts=
poetry run flake8 text4gcn --max-line-length=100 --count
poetry run mypy text4gcn --ignore-missing-imports
```

For academic regression checks (DCAI 2022):

```bash
python scripts/reproduce_dcai2022.py --reference-only
poetry run pytest tests/test_training_protocol.py tests/test_graph_refinement.py -q -o addopts=
```
