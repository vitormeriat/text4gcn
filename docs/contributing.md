# Contributing

The canonical contributing guide lives at the repository root:

**[CONTRIBUTING.md](https://github.com/vitormeriat/text4gcn/blob/main/CONTRIBUTING.md)**

Quick commands:

```bash
poetry install --with dev,docs
make test        # pytest
make lint        # ruff, as in CI
make docs        # mkdocs build --strict
```

For academic regression checks (DCAI 2022):

```bash
python scripts/reproduce_dcai2022.py --reference-only
poetry run pytest tests/test_training_protocol.py tests/test_graph_refinement.py -q -o addopts=
```
