"""Unit tests for Fase A0 — academic rigor (seed, dropout, early-stop, RWF weights)."""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pytest
import torch as th

from text4gcn.models.gnn import GNN
from text4gcn.models.model import GCN
from text4gcn.models.train_eval import TextGCNTrainer
from text4gcn.modules import adjacency as adjcy


class TestSeedWiring:
    def test_build_args_uses_user_seed(self, tmp_path):
        gnn = GNN(
            dataset_name="toy",
            dataset_path=str(tmp_path),
            seed=4242,
            device="cpu",
        )
        args = gnn._build_args()
        assert args.seed == 4242
        assert args.dropout == 0.5

    def test_val_ratio_mismatch_emits_deprecation(self, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GNN(
                dataset_name="toy",
                dataset_path=str(tmp_path),
                val_ratio=0.25,
                device="cpu",
            )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


class TestRwfWeights:
    def test_max_min_not_zscore(self):
        docs = [["a", "b"], ["a", "b"], ["a", "c"]]
        rela = {"a,b": 10, "b,a": 10, "a,c": 2, "c,a": 2}
        word_window_freq = {"a": 3, "b": 2, "c": 1}
        word_id_map = {"a": 0, "b": 1, "c": 2}
        min_c, max_c, mean_c, std_c = adjcy.relation_pair_statitcs(rela)
        row, col = [], []
        weights = adjcy.compute_weights_with_PMI(
            docs,
            rela,
            word_window_freq,
            train_size=0,
            word_id_map=word_id_map,
            min_count1=min_c,
            max_count1=max_c,
            count_mean1=mean_c,
            count_std1=std_c,
            row=row,
            col=col,
        )
        assert weights, "expected at least one positive-PMI edge"
        assert all(0.0 <= w <= 1.0 for w in weights)
        assert 1.0 in weights

    def test_relation_pair_stats_min_not_stuck_at_zero(self):
        rela = {"a,b": 5, "b,a": 5, "a,c": 3, "c,a": 3}
        min_c, max_c, _, _ = adjcy.relation_pair_statitcs(rela)
        assert min_c == 3.0
        assert max_c == 5.0


class TestEarlyStopBestCheckpoint:
    def test_restores_best_val_weights(self, tmp_path):
        n_nodes = 6
        n_classes = 2
        features = np.eye(n_nodes, dtype=np.float32)
        y = np.zeros((n_nodes, n_classes), dtype=np.float32)
        y[0, 0] = y[2, 0] = y[4, 0] = 1.0
        y[1, 1] = y[3, 1] = y[5, 1] = 1.0
        train_mask = np.array([True, True, False, False, False, False])
        val_mask = np.array([False, False, True, True, False, False])
        test_mask = np.array([False, False, False, False, True, True])
        adj = np.eye(n_nodes, dtype=np.float32)

        predata = (
            adj,
            features,
            y.copy(),
            y.copy(),
            y.copy(),
            train_mask,
            val_mask,
            test_mask,
            2,
            2,
        )
        args = argparse.Namespace(
            early_stopping=50,
            seed=7,
            val_ratio=0.1,
            max_epoch=3,
            dataset="toy",
            dropout=0.0,
            log_dir=str(tmp_path),
            layer="GCN",
            nhid=4,
            lr=0.02,
            device=th.device("cpu"),
        )
        trainer = TextGCNTrainer(model=GCN, args=args, pre_data=predata)
        trainer.prepare_data()
        trainer.convert_tensor()
        trainer.model = GCN(
            input_dim=n_nodes,
            output_dim=4,
            support=trainer.support,
            num_classes=n_classes,
            dropout_rate=0.0,
        )
        trainer.criterion = th.nn.CrossEntropyLoss()
        trainer.optimizer = th.optim.Adam(trainer.model.parameters(), lr=0.02)

        val_losses_seq = [1.0, 0.5, 0.9]
        call_idx = {"i": 0}
        snapshots = {}

        def fake_evaluate(model, criterion, features, labels, mask):
            i = call_idx["i"]
            loss = val_losses_seq[min(i, len(val_losses_seq) - 1)]
            snapshots[i] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            call_idx["i"] += 1
            return loss, 0.5, np.zeros(n_nodes), labels.detach().cpu().numpy(), 0.0

        trainer._evaluate = fake_evaluate  # type: ignore[method-assign]
        trainer._test = lambda t1: (0.0, 0.0)  # type: ignore[method-assign]
        trainer._train()

        # Best val is epoch index 1 (loss 0.5); model must match that snapshot
        best = snapshots[1]
        for key, tensor in best.items():
            assert th.allclose(trainer.model.state_dict()[key], tensor), key


class TestDropoutWiring:
    def test_gcn_receives_dropout_from_args(self):
        support = [th.eye(4)]
        model = GCN(
            input_dim=4,
            output_dim=2,
            support=support,
            num_classes=2,
            dropout_rate=0.5,
        )
        assert model.layer1.dropout.p == 0.5
        assert model.layer2.dropout.p == 0.5

    def test_trainer_fit_passes_dropout(self, tmp_path, monkeypatch):
        captured = {}

        class FakeGCN(th.nn.Module):
            def __init__(self, input_dim, output_dim, support, num_classes=10, dropout_rate=0.0):
                super().__init__()
                captured["dropout_rate"] = dropout_rate
                self.layer1 = th.nn.Linear(1, 1)
                self.layer1.embedding = th.zeros(4, 1)
                self.W = th.nn.Parameter(th.zeros(1))

            def forward(self, x):
                return th.zeros(x.shape[0], 2)

        n = 4
        y = np.eye(2, dtype=np.float32).repeat(2, axis=0)
        predata = (
            np.eye(n, dtype=np.float32),
            np.eye(n, dtype=np.float32),
            y,
            y,
            y,
            np.array([True, True, False, False]),
            np.array([False, False, True, False]),
            np.array([False, False, False, True]),
            2,
            1,
        )
        args = argparse.Namespace(
            early_stopping=10,
            seed=1,
            val_ratio=0.1,
            max_epoch=1,
            dataset="toy",
            dropout=0.5,
            log_dir=str(tmp_path),
            layer="GCN",
            nhid=2,
            lr=0.02,
            device=th.device("cpu"),
        )
        trainer = TextGCNTrainer(model=FakeGCN, args=args, pre_data=predata)
        monkeypatch.setattr(trainer, "_train", lambda: {
            "test_accuracy": 0.0,
            "test_loss": 0.0,
            "model_path": None,
            "epochs_trained": 0,
        })
        trainer.fit()
        assert captured["dropout_rate"] == 0.5


class TestLayerApi:
    def test_gat_not_in_public_layer(self):
        from text4gcn.models.layers import Layer

        assert hasattr(Layer, "GCN")
        assert not hasattr(Layer, "GAT")

    def test_unsupported_layer_rejected_at_init(self, tmp_path):
        with pytest.raises(TypeError, match="Only Layer.GCN"):
            GNN(
                dataset_name="toy",
                dataset_path=str(tmp_path),
                layer="GAT",
                device="cpu",
            )


class TestDevicePlacement:
    def test_all_training_tensors_follow_trainer_device(self, tmp_path):
        """Regression: `support` stayed on CPU while features went to CUDA."""
        n_nodes, n_classes = 4, 2
        y = np.zeros((n_nodes, n_classes), dtype=np.float32)
        y[::2, 0] = y[1::2, 1] = 1.0
        mask = np.array([True, True, False, False])
        predata = (np.eye(n_nodes, dtype=np.float32), np.eye(n_nodes, dtype=np.float32),
                   y, y, y, mask, mask, mask, 2, 2)
        # "meta" stands in for CUDA: any tensor not explicitly moved stays on "cpu".
        args = argparse.Namespace(
            early_stopping=10, seed=7, val_ratio=0.1, max_epoch=1, dataset="toy",
            dropout=0.0, log_dir=str(tmp_path), layer="GCN", nhid=4, lr=0.02,
            device=th.device("meta"),
        )
        trainer = TextGCNTrainer(model=GCN, args=args, pre_data=predata)
        trainer.prepare_data()
        trainer.convert_tensor()

        tensors = [trainer.features, trainer.y_train, trainer.y_val, trainer.y_test,
                   trainer.train_mask, trainer.val_mask, trainer.test_mask,
                   trainer.tm_train_mask, *trainer.support]
        assert {t.device.type for t in tensors} == {"meta"}

