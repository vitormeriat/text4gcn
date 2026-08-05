#!/usr/bin/env python3
"""Reproduce (or compare against) DCAI 2022 Tables 2 and 3.

Pereira & de Castro, *Natural Language Processing Based on a Text Graph
Convolutional Network*, DCAI 2022.

Usage
-----
Reference tables only (no data required)::

    python scripts/reproduce_dcai2022.py --reference-only

Compare local adjacency files to Table 2 order-of-magnitude::

    python scripts/reproduce_dcai2022.py --data-dir data/ --dataset R8 \\
        --builders frequency,semantic

Full training accuracy (Table 3) requires preprocessed data + built graphs::

    python scripts/reproduce_dcai2022.py --data-dir data/ --dataset R8 \\
        --builders frequency --train --epochs 200 --seed 1234

Notes
-----
- Table 2 counts are from the paper; local nnz may differ slightly due to
  cleaning / NLTK versions. The script flags order-of-magnitude agreement.
- ``semantic`` maps to the Word2Vec RWF builder (``embedding.word2vec``).
- Training is **transductive** (test docs in the graph), as in Yao and DCAI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from repo root without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Paper reference values (DCAI 2022, Tables 2 and 3)
TABLE2_EDGES: Dict[str, Dict[str, int]] = {
    # dataset -> builder_family -> edge count
    "20NG": {
        "tgcn": 24_689_966,
        "semantic": 2_286_228,
        "syntactic": 3_130_672,
        "linguistic": 2_529_518,
    },
    "R8": {
        "tgcn": 3_165_430,
        "semantic": 324_464,
        "syntactic": 548_166,
        "linguistic": 393_562,
    },
    "R52": {
        "tgcn": 3_981_246,
        "semantic": 407_694,
        "syntactic": 682_675,
        "linguistic": 492_946,
    },
    "MR": {
        "tgcn": 1_701_424,
        "semantic": 226_468,
        "syntactic": 407_508,
        "linguistic": 219_116,
    },
}

TABLE3_ACCURACY: Dict[str, Dict[str, float]] = {
    "20NG": {
        "tgcn": 0.86126,
        "semantic": 0.85068,
        "syntactic": 0.84665,
        "linguistic": 0.84241,
    },
    "R8": {
        "tgcn": 0.96848,
        "semantic": 0.96895,
        "syntactic": 0.95021,
        "linguistic": 0.95432,
    },
    "R52": {
        "tgcn": 0.93497,
        "semantic": 0.92445,
        "syntactic": 0.91978,
        "linguistic": 0.91783,
    },
}

# Map CLI builder names -> (paper key, adjacency filename pattern)
BUILDER_MAP = {
    "frequency": ("tgcn", "ind.frequency.tfidf.{dataset}.adj"),
    "tgcn": ("tgcn", "ind.frequency.tfidf.{dataset}.adj"),
    "semantic": ("semantic", "ind.embedding.word2vec.{dataset}.adj"),
    "word2vec": ("semantic", "ind.embedding.word2vec.{dataset}.adj"),
}


def print_reference_tables(datasets: List[str]) -> None:
    print("=" * 72)
    print("DCAI 2022 — Table 2 (edge counts) [reference]")
    print("=" * 72)
    for ds in datasets:
        if ds not in TABLE2_EDGES:
            continue
        print(f"\n{ds}:")
        for name, n in TABLE2_EDGES[ds].items():
            print(f"  {name:12s}  {n:>12,}")

    print("\n" + "=" * 72)
    print("DCAI 2022 — Table 3 (test accuracy) [reference]")
    print("=" * 72)
    for ds in datasets:
        if ds not in TABLE3_ACCURACY:
            continue
        print(f"\n{ds}:")
        for name, acc in TABLE3_ACCURACY[ds].items():
            print(f"  {name:12s}  {acc:.5f}")


def order_of_magnitude_ok(observed: int, reference: int, factor: float = 3.0) -> bool:
    """True if observed is within ``factor`` of reference (same ballpark)."""
    if reference <= 0 or observed <= 0:
        return False
    ratio = max(observed, reference) / min(observed, reference)
    return ratio <= factor


def compare_edge_files(
    data_dir: Path,
    dataset: str,
    builders: List[str],
) -> List[Tuple[str, Optional[int], Optional[int], str]]:
    from text4gcn.graph import load_adjacency_stats

    rows = []
    adj_dir = data_dir / f"{dataset}.adjacency"
    for b in builders:
        if b not in BUILDER_MAP:
            rows.append((b, None, None, "unknown builder"))
            continue
        paper_key, pattern = BUILDER_MAP[b]
        path = adj_dir / pattern.format(dataset=dataset)
        ref = TABLE2_EDGES.get(dataset, {}).get(paper_key)
        if not path.is_file():
            rows.append((b, None, ref, f"missing: {path}"))
            continue
        stats = load_adjacency_stats(
            path, builder=paper_key, dataset_name=dataset
        )
        status = "OK" if ref and order_of_magnitude_ok(stats.n_edges, ref) else "CHECK"
        if ref is None:
            status = "no paper ref"
        rows.append((b, stats.n_edges, ref, status))
        print(stats)
    return rows


def maybe_train(
    data_dir: Path,
    dataset: str,
    builder_cli: str,
    epochs: int,
    seed: int,
) -> Optional[float]:
    from text4gcn import GNN
    from text4gcn.models.builder import Builder

    builder_enum = {
        "frequency": Builder.Tfidf,
        "tgcn": Builder.Tfidf,
        "semantic": Builder.Word2Vec,
        "word2vec": Builder.Word2Vec,
    }[builder_cli]

    log_dir = data_dir / "logs_reproduce"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = GNN(
        dataset_name=dataset,
        dataset_path=str(data_dir),
        log_dir=str(log_dir),
        builder=builder_enum,
        epochs=epochs,
        seed=seed,
        device="cpu",
    ).fit()
    paper_key = BUILDER_MAP[builder_cli][0]
    ref = TABLE3_ACCURACY.get(dataset, {}).get(paper_key)
    print(
        f"\nTest accuracy={result.test_accuracy:.5f} "
        f"(paper {paper_key}={ref if ref is not None else 'n/a'})"
    )
    return result.test_accuracy


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Print DCAI 2022 Tables 2/3 and exit",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="R8", choices=sorted(TABLE2_EDGES.keys()))
    parser.add_argument(
        "--builders",
        default="frequency,semantic",
        help="Comma-separated: frequency,semantic (aliases: tgcn,word2vec)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also run GNN.fit for the first builder (Table 3 check)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    builders = [b.strip().lower() for b in args.builders.split(",") if b.strip()]
    datasets = [args.dataset] if not args.reference_only else list(TABLE2_EDGES.keys())

    print_reference_tables(datasets if args.reference_only else [args.dataset])

    if args.reference_only:
        print(
            "\nNicho: RWF reduces edges (Tab. 2) while keeping accuracy ≈ TGCN "
            "(Tab. 3). Model is transductive — test documents enter the graph."
        )
        return 0

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Local edge counts vs Table 2 — {args.dataset} @ {args.data_dir}")
    print("=" * 72)
    rows = compare_edge_files(args.data_dir, args.dataset, builders)
    print(f"\n{'builder':12s}  {'observed':>12s}  {'paper':>12s}  status")
    for name, obs, ref, status in rows:
        obs_s = f"{obs:,}" if obs is not None else "-"
        ref_s = f"{ref:,}" if ref is not None else "-"
        print(f"{name:12s}  {obs_s:>12s}  {ref_s:>12s}  {status}")

    if args.train:
        first = builders[0]
        print("\n" + "=" * 72)
        print(f"Training ({first}) — Table 3 check")
        print("=" * 72)
        maybe_train(args.data_dir, args.dataset, first, args.epochs, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
