from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from text4gcn.graph import RefinementConfig, apply_refinement
from text4gcn.models.builder import Builder
from text4gcn.models.layers import Layer
from text4gcn.models.model import GCN
from text4gcn.models.train_eval import TextGCNTrainer
from text4gcn.models.utils import LoadData
from text4gcn.utils.device import get_device_manager


@dataclass(frozen=True)
class TrainingResult:
    """Results returned by :meth:`GNN.fit`.

    Attributes
    ----------
    test_accuracy : float
        Accuracy on the held-out test set.
    test_loss : float
        Cross-entropy loss on the held-out test set.
    model_path : str
        Path of the saved checkpoint (``.pth``) inside ``log_dir``: the weights with
        the best validation loss.
    epochs_trained : int
        Number of epochs completed before stopping.
    """

    test_accuracy: float
    test_loss: float
    model_path: str
    epochs_trained: int

    def __str__(self) -> str:
        return (
            f"TrainingResult(accuracy={self.test_accuracy:.4f}, "
            f"loss={self.test_loss:.5f}, "
            f"epochs={self.epochs_trained}, "
            f"model_path={self.model_path!r})"
        )


class GNN:
    """Text classification using Graph Convolutional Networks.

    Builds a heterogeneous word-document graph and trains a GCN to classify
    documents using the graph structure.

    Parameters
    ----------
    dataset_name : str
        Dataset name (must match preprocessed files on disk).
    dataset_path : str
        Path to the dataset directory.
    log_dir : str
        Directory for model checkpoints and training metric plots.
        Defaults to the current working directory (``"."``).
    builder : Builder
        Adjacency builder type. Defaults to ``Builder.Tfidf``.
    lr : float
        Learning rate. Default ``0.02``.
    nhid : int
        Hidden layer dimension. Default ``200``.
    seed : int
        Random seed. Default ``1234``.
    epochs : int
        Maximum training epochs. Default ``200``.
    dropout : float
        Dropout rate. Default ``0.5``.
    layer : Layer
        GNN layer type. Default ``Layer.GCN``.
    early_stopping : int
        Stop training if validation loss does not improve for this many
        consecutive epochs. Default ``10``.
    device : str or None
        PyTorch device string (``'cpu'``, ``'cuda'``). Auto-detected if
        ``None``.
    refinement_config : RefinementConfig or None
        Optional graph refinement settings applied after loading the corpus.

    Examples
    --------
    Minimal usage (auto-detects device, uses TF-IDF adjacency):

    >>> from text4gcn import GNN
    >>> result = GNN(dataset_name="R8", dataset_path="data/").fit()
    >>> print(result.test_accuracy)

    Custom configuration:

    >>> from text4gcn import GNN, Builder, Layer
    >>> result = GNN(
    ...     dataset_name="R8",
    ...     dataset_path="data/",
    ...     log_dir="logs/",
    ...     builder=Builder.Tfidf,
    ...     epochs=200,
    ...     lr=0.02,
    ...     nhid=200,
    ... ).fit()
    >>> print(result)

    With graph refinement:

    >>> from text4gcn import GNN, Builder, RefinementConfig
    >>> result = GNN(
    ...     dataset_name="R8",
    ...     dataset_path="data/",
    ...     builder=Builder.Tfidf,
    ...     refinement_config=RefinementConfig(
    ...         min_doc_freq=2,
    ...         edge_weight_threshold=0.1,
    ...     ),
    ... ).fit()
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | os.PathLike[str],
        log_dir: str = ".",
        builder: Builder = Builder.Tfidf,
        lr: float = 0.02,
        nhid: int = 200,
        seed: int = 1234,
        epochs: int = 200,
        dropout: float = 0.5,
        layer: Layer = Layer.GCN,
        early_stopping: int = 10,
        device: Optional[str] = None,
        refinement_config: Optional[RefinementConfig] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.builder = builder
        self.lr = lr
        self.nhid = nhid
        self.seed = seed
        self.epochs = epochs
        self.dropout = dropout
        self.layer = layer
        self.early_stopping = early_stopping
        self.refinement_config = refinement_config

        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            raise FileNotFoundError(
                f"dataset_path does not exist: {dataset_path}\n\n"
                "Create the directory and place corpus files there, then run "
                "TextPipeline / builders before GNN.fit()."
            )
        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")

        if layer != Layer.GCN:
            raise TypeError(
                f"Unsupported layer type '{layer}'. "
                "Only Layer.GCN is supported in this version."
            )

        self.device_manager = get_device_manager(device)
        self.device = self.device_manager.device
        self.device_manager.set_seed(seed)

        device_info = self.device_manager.get_device_info()
        print(f"Using device: {self.device}")
        if self.device_manager.is_cuda:
            print(f"GPU: {device_info['gpu_name']}")
            print(f"GPU Memory: {device_info['gpu_memory_gb']:.1f} GB")

    def _validate_data_paths(self) -> None:
        """Raise a descriptive error if required pipeline outputs are missing."""
        base = Path(self.dataset_path)
        node_feat_dir = base / f"{self.dataset_name}.node_features"
        adj_dir = base / f"{self.dataset_name}.adjacency"

        if not node_feat_dir.exists():
            raise FileNotFoundError(
                f"Node feature files not found at '{node_feat_dir}'.\n\n"
                f"Run the preprocessing step first:\n\n"
                f"  from text4gcn import TextPipeline\n"
                f"  TextPipeline(\n"
                f"      dataset_name='{self.dataset_name}',\n"
                f"      dataset_path='{self.dataset_path}',\n"
                f"  ).execute()\n"
            )

        if not adj_dir.exists():
            raise FileNotFoundError(
                f"Adjacency files not found at '{adj_dir}'.\n\n"
                f"Run the graph builder step after preprocessing:\n\n"
                f"  from text4gcn.builder import Frequency\n"
                f"  Frequency(\n"
                f"      dataset_name='{self.dataset_name}',\n"
                f"      dataset_path='{self.dataset_path}',\n"
                f"  ).tfidf()  # or .word2vec(), .cosine_similarity(), ...\n"
            )

    def _build_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            early_stopping=self.early_stopping,
            seed=self.seed,
            max_epoch=self.epochs,
            dataset=self.dataset_name,
            dropout=self.dropout,
            log_dir=self.log_dir,
            layer=self.layer,
            nhid=self.nhid,
            lr=self.lr,
            device=self.device,
        )

    def fit(self) -> TrainingResult:
        """Load data, optionally refine the graph, and train the GCN.

        Returns
        -------
        TrainingResult
            Test accuracy, loss, model checkpoint path, and epoch count.

        Raises
        ------
        FileNotFoundError
            If preprocessing or graph-building steps have not been run.
        TypeError
            If an unsupported layer type is requested.
        """
        self._validate_data_paths()

        args = self._build_args()
        print(f"\n{'='*30} MODEL CONFIGURATION\n{args}\n")

        predata = LoadData(
            path=self.dataset_path,
            dataset=self.dataset_name,
            builder=self.builder,
        ).load_corpus()

        if self.refinement_config is not None:
            predata, info = apply_refinement(
                predata,
                self.refinement_config,
                dataset_path=self.dataset_path,
                dataset_name=self.dataset_name,
                return_metadata=True,
            )
            if info and "stats" in info:
                print("\nGraph refinement stats:", info["stats"])

        result = TextGCNTrainer(model=GCN, args=args, pre_data=predata).fit()
        return TrainingResult(
            test_accuracy=result["test_accuracy"],
            test_loss=result["test_loss"],
            model_path=result["model_path"],
            epochs_trained=result["epochs_trained"],
        )
