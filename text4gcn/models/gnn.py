import argparse

import numpy as np

from text4gcn.graph import RefinementConfig, apply_refinement
from text4gcn.models.layers import Layer
from text4gcn.models.model import GCN
from text4gcn.models.train_eval import TextGCNTrainer
from text4gcn.models.utils import LoadData
from text4gcn.utils.device import check_pytorch_installation, get_device_manager


class GNN:
    """Text classification using Graph Convolutional Networks.

    Builds a heterogeneous word-document graph and trains a GCN to classify
    documents using the graph structure.

    Parameters
    ----------
    dataset : str
        Dataset name (must match preprocessed files on disk).
    path : str
        Path to the dataset directory.
    log_dir : str
        Directory for model checkpoints and training metric plots.
    builder : Builder
        Adjacency builder type (e.g. ``Builder.Tfidf``).
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
    val_ratio : float
        Fraction of training data reserved for validation. Default ``0.1``.
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
    >>> from text4gcn.models import GNN, Builder, Layer
    >>> model = GNN(
    ...     dataset="20ng",
    ...     path="data/",
    ...     log_dir="logs/",
    ...     builder=Builder.Tfidf,
    ... )
    >>> model.fit()
    """

    def __init__(
        self,
        dataset: str,
        path: str,
        log_dir: str,
        builder,
        lr: float = 0.02,
        nhid: int = 200,
        seed: int = 1234,
        epochs: int = 200,
        dropout: float = 0.5,
        val_ratio: float = 0.1,
        layer: Layer = Layer.GCN,
        early_stopping: int = 10,
        device=None,
        refinement_config: RefinementConfig = None,
    ) -> None:
        pytorch_info = check_pytorch_installation()
        if not pytorch_info["torch_available"]:
            raise ImportError(
                "PyTorch is not installed. Install it with:\n"
                "  pip install text4gcn[cpu]   # CPU\n"
                "  pip install text4gcn[gpu]   # GPU"
            )

        self.dataset = dataset
        self.path = path
        self.log_dir = log_dir
        self.builder = builder
        self.lr = lr
        self.nhid = nhid
        self.seed = seed
        self.epochs = epochs
        self.dropout = dropout
        self.val_ratio = val_ratio
        self.layer = layer
        self.early_stopping = early_stopping
        self.refinement_config = refinement_config

        self.device_manager = get_device_manager(device)
        self.device = self.device_manager.device

        device_info = self.device_manager.get_device_info()
        print(f"Using device: {self.device}")
        if self.device_manager.is_cuda:
            print(f"GPU: {device_info['gpu_name']}")
            print(f"GPU Memory: {device_info['gpu_memory_gb']:.1f} GB")

    def _build_args(self) -> argparse.Namespace:
        """Assemble training configuration as a Namespace."""
        return argparse.Namespace(
            early_stopping=self.early_stopping,
            seed=np.random.randint(1, 100_000),
            val_ratio=self.val_ratio,
            max_epoch=self.epochs,
            dataset=self.dataset,
            dropout=self.dropout,
            log_dir=self.log_dir,
            layer=self.layer,
            nhid=self.nhid,
            lr=self.lr,
            device=self.device,
        )

    def fit(self) -> None:
        """Load data, optionally refine the graph, and train the GCN."""
        args = self._build_args()
        print(f"\n{'='*30} MODEL CONFIGURATION\n{args}\n")

        predata = LoadData(
            path=self.path,
            dataset=self.dataset,
            builder=self.builder,
        ).load_corpus()

        if self.refinement_config is not None:
            predata, info = apply_refinement(
                predata,
                self.refinement_config,
                dataset_path=self.path,
                dataset_name=self.dataset,
                return_metadata=True,
            )
            if info and "stats" in info:
                print("\nGraph refinement stats:", info["stats"])

        if self.layer != Layer.GCN:
            raise TypeError(
                f"Unsupported layer type '{self.layer}'. "
                "Only Layer.GCN is supported in this version."
            )

        TextGCNTrainer(model=GCN, args=args, pre_data=predata).fit()
