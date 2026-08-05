#!/usr/bin/env python3
"""
End-to-end synthetic example for Text4GCN.

Generates a small corpus, runs TextPipeline → Frequency.tfidf → GNN.fit().
Use this as the canonical offline smoke / tutorial script.

Usage (from repo root or examples/)::

    python examples/end_to_end_synthetic.py
    python examples/end_to_end_synthetic.py --docs 300 --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=400, help="Number of synthetic documents")
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--data-dir",
        default=str(EXAMPLES / "synthetic_data"),
        help="Directory for .txt/.meta and pipeline outputs",
    )
    args = parser.parse_args()

    os.chdir(EXAMPLES)

    from synthetic_data_generator import SyntheticDataGenerator
    from text4gcn import GNN, TextPipeline, Builder, Layer
    from text4gcn.builder import Frequency

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = "SyntheticR8"

    print("=" * 60)
    print("1/4 Generate synthetic data")
    print("=" * 60)
    SyntheticDataGenerator(str(data_dir)).generate_dataset(
        dataset, num_documents=args.docs, train_ratio=0.8
    )

    print("=" * 60)
    print("2/4 TextPipeline")
    print("=" * 60)
    TextPipeline(
        dataset_name=dataset,
        dataset_path=str(data_dir),
        rare_count=3,
        language="english",
        seed=args.seed,
    ).execute(validation_ratio=0.10)

    print("=" * 60)
    print("3/4 Frequency.tfidf")
    print("=" * 60)
    stats = Frequency(dataset_name=dataset, dataset_path=str(data_dir)).tfidf(window_size=20)
    print(stats)

    print("=" * 60)
    print("4/4 GNN.fit")
    print("=" * 60)
    result = GNN(
        dataset_name=dataset,
        dataset_path=str(data_dir),
        log_dir=str(EXAMPLES / "logs"),
        builder=Builder.Tfidf,
        layer=Layer.GCN,
        seed=args.seed,
        epochs=args.epochs,
        early_stopping=10,
        dropout=0.5,
        lr=0.02,
        nhid=200,
        device=None,
    ).fit()

    print(result)
    print(f"test_accuracy={result.test_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
