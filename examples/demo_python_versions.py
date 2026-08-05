#!/usr/bin/env python3
"""
Smoke-check Text4GCN imports across supported Python versions.

Validates the public API surface (not a full training run).
"""

from __future__ import annotations

import sys


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_python_version() -> bool:
    print_header("Python version")
    version = sys.version_info
    print(f"  {sys.executable}")
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    ok = version.major == 3 and version.minor >= 10
    print("  OK" if ok else "  Need Python 3.10+")
    return ok


def demonstrate_text4gcn_import() -> bool:
    print_header("Import text4gcn")
    try:
        import text4gcn
        from text4gcn import GNN, TextPipeline, Builder, Layer, GraphBuildStats

        print(f"  version: {text4gcn.__version__}")
        print(f"  exports: GNN={GNN.__name__}, Layer={list(Layer)}")
        _ = TextPipeline, Builder, GraphBuildStats
        return True
    except ImportError as exc:
        print(f"  Import error: {exc}")
        return False


def demonstrate_datasets() -> bool:
    print_header("Datasets")
    try:
        from text4gcn.datasets import data

        data.list()
        return True
    except Exception as exc:
        print(f"  Error: {exc}")
        return False


def demonstrate_builders() -> bool:
    print_header("Builders")
    try:
        from text4gcn.builder import Frequency, Semantic, Syntactic

        print(f"  {Frequency.__name__}, {Semantic.__name__}, {Syntactic.__name__}")
        return True
    except Exception as exc:
        print(f"  Error: {exc}")
        return False


def demonstrate_models() -> bool:
    print_header("Models")
    try:
        from text4gcn.models import GNN, Layer
        from text4gcn.models.model import GCN

        print(f"  GNN={GNN.__name__}, GCN={GCN.__name__}, Layer={list(Layer)}")
        return True
    except Exception as exc:
        print(f"  Error: {exc}")
        return False


def main() -> int:
    print_header("Text4GCN multi-version smoke check")
    if not check_python_version():
        return 1
    if not demonstrate_text4gcn_import():
        print("Install with: pip install -e '.[cpu]'")
        return 1

    ok = all(
        [
            demonstrate_datasets(),
            demonstrate_builders(),
            demonstrate_models(),
        ]
    )
    print_header("Summary")
    print("  All checks passed." if ok else "  Some checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
