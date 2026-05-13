#!/usr/bin/env python3
"""
Graph refinement example for Text4GCN.

Shows how to use RefinementConfig to filter low-quality nodes and prune weak
edges in the doc-word graph before training.  Run this after the preprocessing
and adjacency-building steps have already been completed for your dataset.

Usage::

    # Minimal — auto-detects device, uses TF-IDF adjacency
    python graph_refinement_example.py

    # Custom dataset via environment variables
    TEXT4GCN_DATASET=20ng TEXT4GCN_PATH=data/ python graph_refinement_example.py
"""

import os

from text4gcn import GNN, Builder, RefinementConfig


def main() -> None:
    dataset_name = os.environ.get("TEXT4GCN_DATASET", "R8")
    dataset_path = os.environ.get("TEXT4GCN_PATH", "data")
    log_dir = os.environ.get("TEXT4GCN_LOG_DIR", "logs")

    # -------------------------------------------------------------------------
    # Option A — pass refinement_config directly to GNN.fit()
    #
    # The most common workflow: GNN handles everything internally, loads the
    # corpus, refines the graph, trains, and returns structured results.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Option A: GNN with inline refinement")
    print("=" * 60)

    result = GNN(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        log_dir=log_dir,
        builder=Builder.Tfidf,
        epochs=200,
        refinement_config=RefinementConfig(
            # Keep only words that appear in at least 2 documents
            min_doc_freq=2,
            # Drop words with very low connectivity in the graph
            min_word_degree=1.0,
            # Remove noisy / very weak edges
            edge_weight_threshold=0.1,
            # Always preserve all document nodes
            keep_docs=True,
            # Attach node-type metadata for inspection
            build_metadata=True,
        ),
    ).fit()

    print(f"\nTraining complete: {result}")

    # -------------------------------------------------------------------------
    # Option B — apply refinement manually to inspect stats before training
    #
    # Use apply_refinement() when you want to examine what gets removed before
    # committing to a full training run.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Option B: Manual refinement to inspect stats first")
    print("=" * 60)

    from text4gcn.graph import apply_refinement
    from text4gcn.models.utils import LoadData  # internal — only for inspection

    config = RefinementConfig(
        min_doc_freq=2,
        min_word_degree=1.0,
        edge_weight_threshold=0.1,
        build_metadata=True,
    )

    predata_raw = LoadData(
        path=dataset_path,
        dataset=dataset_name,
        builder=Builder.Tfidf,
    ).load_corpus()

    predata_refined, info = apply_refinement(
        predata_raw,
        config,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        return_metadata=True,
    )

    if info:
        print("\nRefinement stats:", info["stats"])
        if info.get("metadata"):
            print("\nSample node metadata (first 5 nodes):")
            for m in info["metadata"][:5]:
                print(" ", m.to_dict())

    print(f"\nNodes before: {predata_raw[0].shape[0]}")
    print(f"Nodes after:  {predata_refined[0].shape[0]}")


if __name__ == "__main__":
    main()
