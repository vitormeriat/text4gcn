from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn import metrics

from text4gcn.models.utils import preprocess_adj, preprocess_features, unique_name
from text4gcn.utils.logging import get_logger
from text4gcn.utils.reporting import Reporter, format_count, format_duration

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (no class state needed)
# ---------------------------------------------------------------------------

def evaluation_tables(test_labels: Sequence[int], test_pred: Sequence[int],
                      class_names: Sequence[str]) -> Tuple[Dict[str, float], "object", "object"]:
    """Test metrics, per-class report and confusion matrix (pandas), with class names."""
    import pandas as pd

    labels = list(range(len(class_names)))
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        test_labels, test_pred, labels=labels, zero_division=0)
    report = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support},
                          index=pd.Index(class_names, name="class"))
    confusion = pd.DataFrame(metrics.confusion_matrix(test_labels, test_pred, labels=labels),
                             index=pd.Index(class_names, name="true"),
                             columns=pd.Index(class_names, name="predicted"))
    summary = {
        "accuracy": float(metrics.accuracy_score(test_labels, test_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(metrics.f1_score(test_labels, test_pred, labels=labels,
                                              average="weighted", zero_division=0)),
    }
    return summary, report, confusion


def plot_training_history(log_dir, train_acc, val_acc, train_loss, val_loss) -> None:
    """Save accuracy and loss plots to *log_dir*."""
    from pathlib import Path

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_acc, "b", label="Training acc")
    plt.plot(epochs, val_acc, "r", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig(f"{log_dir}/metrics-acc.png", dpi=250)
    plt.close()

    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(f"{log_dir}/metrics-loss.png", dpi=250)
    plt.close()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TextGCNTrainer:
    def __init__(self, model, args, pre_data, report: Optional[Reporter] = None,
                 class_names: Optional[Sequence[str]] = None):
        self.args = args
        self.model = model
        self.device = args.device
        self.max_epoch = args.max_epoch
        self.nhid = args.nhid
        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = args.early_stopping
        self.log_dir = args.log_dir
        self.dropout = getattr(args, "dropout", 0.5)
        self.report = report if report is not None else Reporter(getattr(args, "verbose", 1))
        self.class_names = list(class_names) if class_names is not None else None

    def fit(self) -> Dict:
        """Prepare data, train the model, evaluate on the test set.

        Returns
        -------
        dict
            Keys: ``test_accuracy``, ``test_loss``, ``model_path``, ``epochs_trained``,
            ``best_epoch``, ``history``, ``metrics``, ``report``, ``confusion_matrix``.
        """
        self.prepare_data()
        self.convert_tensor()

        self.model = self.model(
            input_dim=self.features.shape[0],
            output_dim=self.nhid,
            support=self.support,
            num_classes=self.y_train.shape[1],
            dropout_rate=self.dropout,
        ).to(self.device)

        self.criterion = th.nn.CrossEntropyLoss()
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_classes = self.y_train.shape[1]
        self.report.write(f"  model     GCN {self.features.shape[0]:,} → {self.nhid} → {n_classes} · "
                          f"{format_count(n_params)} parameters · dropout {self.dropout} · lr {self.args.lr}")
        logger.debug("model structure:\n%s", self.model)

        return self._train()

    def prepare_data(self) -> None:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = self.predata

        self.features = preprocess_features(sp.identity(features.shape[0]))
        self.adj = adj
        self.support = [preprocess_adj(adj)]
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_size = train_size
        self.test_size = test_size

    def convert_tensor(self) -> None:
        n_nodes = self.features.shape[0]
        n_words = n_nodes - self.train_size - self.test_size
        self.report.write(
            f"  graph     {n_nodes:,} nodes: {int(self.train_mask.sum()):,} train · "
            f"{int(self.val_mask.sum()):,} validation · {int(self.test_mask.sum()):,} test documents "
            f"+ {n_words:,} words")

        self.features = th.from_numpy(self.features).float().to(self.device)
        self.y_train = th.from_numpy(self.y_train).float().to(self.device)
        self.y_val = th.from_numpy(self.y_val).float().to(self.device)
        self.y_test = th.from_numpy(self.y_test).float().to(self.device)
        self.train_mask = th.from_numpy(self.train_mask).float().to(self.device)
        self.val_mask = th.from_numpy(self.val_mask).float().to(self.device)
        self.test_mask = th.from_numpy(self.test_mask).float().to(self.device)
        self.tm_train_mask = th.transpose(
            th.unsqueeze(self.train_mask, 0), 1, 0
        ).repeat(1, self.y_train.shape[1])
        self.support = [th.Tensor(s).to(self.device) for s in self.support]

    def _evaluate(self, model, criterion, features, labels, mask) -> Tuple:
        t = perf_counter()
        model.eval()
        with th.no_grad():
            logits = model(features)
            t_mask = th.as_tensor(mask, dtype=th.float32, device=logits.device)
            tm_mask = th.transpose(th.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = criterion(logits * tm_mask, th.max(labels, 1)[1])
            pred = th.max(logits, 1)[1]
            try:
                acc = (
                    (pred == th.max(labels, 1)[1]).float() * t_mask
                ).sum().item() / t_mask.sum().item()
            except ZeroDivisionError:
                acc = 0.0
        return float(loss), acc, pred.cpu().numpy(), labels.cpu().numpy(), (perf_counter() - t)

    def _test(self) -> Dict:
        """Evaluate on the test set and report the metrics."""
        start = perf_counter()
        test_loss, test_acc, pred, labels, _ = self._evaluate(
            self.model, self.criterion, self.features, self.y_test, self.test_mask
        )
        test_mask = self.test_mask.cpu().numpy().astype(bool)
        test_pred = pred[test_mask]
        test_labels = np.argmax(labels[test_mask], axis=1)
        class_names = self.class_names or [str(i) for i in range(labels.shape[1])]
        summary, report, confusion = evaluation_tables(test_labels, test_pred, class_names)

        self.report.write(
            f"  ✓ test      accuracy {test_acc:.1%} · macro-F1 {summary['macro_f1']:.1%} · "
            f"loss {test_loss:.4f} · {len(test_labels):,} documents  ({format_duration(perf_counter() - start)})")
        self.report.write()
        table = report.reset_index().to_string(index=False, float_format=lambda v: f"{v:.3f}")
        for line in table.splitlines():
            self.report.write(f"    {line}")
        self.report.write()
        return {"test_accuracy": float(test_acc), "test_loss": float(test_loss), "metrics": summary,
                "report": report, "confusion_matrix": confusion}

    def _save_model(self) -> str:
        """Save model state dict; return the file path."""
        from pathlib import Path

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        path = str(Path(self.log_dir) / f"{unique_name()}.pth")
        th.save(self.model.state_dict(), path)
        return path

    def _log_epoch(self, bar, epoch: int, row: Dict) -> None:
        text = (f"loss {row['loss']:.4f} · acc {row['accuracy']:.1%} · "
                f"val_loss {row['val_loss']:.4f} · val_acc {row['val_accuracy']:.1%}")
        if bar is not None:
            bar.set_postfix_str(text, refresh=False)
            bar.update()
        elif self.report.verbose == 2:
            self.report.write(f"    epoch {epoch:>3}/{self.max_epoch} · {text} · {row['seconds']:.1f} s")

    def _train(self) -> Dict:
        import pandas as pd

        start = perf_counter()
        seed = int(self.args.seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

        history: List[Dict] = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        early_stop = False
        bar = self.report.bar(total=self.max_epoch, desc="    epochs", unit="epoch")

        for epoch in range(self.max_epoch):
            epoch_start = perf_counter()
            self.model.train()

            logits = self.model(self.features)
            loss = self.criterion(logits * self.tm_train_mask, th.max(self.y_train, 1)[1])

            try:
                acc = (
                    (th.max(logits, 1)[1] == th.max(self.y_train, 1)[1]).float()
                    * self.train_mask
                ).sum().item() / self.train_mask.sum().item()
            except ZeroDivisionError:
                acc = 0.0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            val_loss, val_acc, _, _, _ = self._evaluate(
                self.model, self.criterion, self.features, self.y_val, self.val_mask
            )
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = deepcopy(self.model.state_dict())

            row = {"epoch": epoch + 1, "loss": float(loss.detach()), "accuracy": float(acc),
                   "val_loss": float(val_loss), "val_accuracy": float(val_acc),
                   "seconds": perf_counter() - epoch_start}
            history.append(row)
            self._log_epoch(bar, epoch + 1, row)

            if epoch > self.earlystopping and val_losses[-1] > np.mean(
                val_losses[-(self.earlystopping + 1): -1]
            ):
                early_stop = True
                break

        if bar is not None:
            bar.close()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        model_path = self._save_model()

        stop = "early stop" if early_stop else "maximum reached"
        self.report.write(
            f"  ✓ train     {len(history)} epochs ({stop}) · best epoch {best_epoch} "
            f"(val_loss {best_val_loss:.4f}, val_acc {history[best_epoch - 1]['val_accuracy']:.1%})  "
            f"({format_duration(perf_counter() - start)})")

        plot_training_history(
            self.log_dir,
            [r["accuracy"] for r in history], [r["val_accuracy"] for r in history],
            [r["loss"] for r in history], [r["val_loss"] for r in history],
        )

        result = self._test()
        self.report.write(f"  model     {model_path}")
        result.update({
            "model_path": model_path,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "history": pd.DataFrame(history).set_index("epoch"),
        })
        return result
