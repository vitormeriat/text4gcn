from __future__ import annotations

from copy import deepcopy
from time import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn import metrics
from tabulate import tabulate

from text4gcn.models.utils import preprocess_adj, preprocess_features, unique_name


# ---------------------------------------------------------------------------
# Module-level helpers (no class state needed)
# ---------------------------------------------------------------------------

def show_metrics(test_labels, test_pred) -> None:
    """Print classification report and all metric aggregations."""
    print("\nTest Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_labels, test_pred, digits=4))
    print("\nMacro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average="macro"))
    print("\nMicro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average="micro"))
    print("\nWeighted average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average="weighted"))
    print("\nConfusion Matrix...")
    print(metrics.confusion_matrix(test_labels, test_pred))
    print("F1-Score (macro)...  ", metrics.f1_score(test_labels, test_pred, average="macro"))
    print("Precision (macro)... ", metrics.precision_score(test_labels, test_pred, average="macro"))
    print("Recall (macro)...    ", metrics.recall_score(test_labels, test_pred, average="macro"))
    print("Accuracy...          ", metrics.accuracy_score(test_labels, test_pred))


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
    def __init__(self, model, args, pre_data):
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

    def fit(self) -> Dict:
        """Prepare data, train the model, evaluate on test set.

        Returns
        -------
        dict
            Keys: ``test_accuracy``, ``test_loss``, ``model_path``,
            ``epochs_trained``.
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

        print("\nModel structure:\n\n", self.model, "\n")
        print(f"Dropout rate: {self.dropout}")
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}")

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
        print("\nData statistics\n")
        print(tabulate(
            [
                ["Edges", "Classes", "Train samples", "Val samples", "Test samples"],
                [
                    self.features.shape[0],
                    self.y_train.shape[1],
                    self.train_mask.sum(),
                    self.val_mask.sum(),
                    self.test_mask.sum(),
                ],
            ],
            headers="firstrow",
        ))

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
        t = time()
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
        return float(loss), acc, pred.cpu().numpy(), labels.cpu().numpy(), (time() - t)

    def _test(self, t1: float) -> Tuple[float, float]:
        """Evaluate on the test set, print metrics, return (accuracy, loss)."""
        test_loss, test_acc, pred, labels, test_duration = self._evaluate(
            self.model, self.criterion, self.features, self.y_test, self.test_mask
        )
        print(
            f"\nTest set results: loss={test_loss:.5f}, accuracy={test_acc:.5f}, "
            f"time={test_duration:.5f}"
        )

        test_mask = self.test_mask.cpu().numpy()
        test_pred, test_labels = [], []
        for i in range(len(test_mask)):
            if test_mask[i]:
                test_pred.append(pred[i])
                test_labels.append(np.argmax(labels[i]))

        show_metrics(test_labels, test_pred)

        tmp = self.model.layer1.embedding.detach().cpu().numpy()
        word_embeddings = tmp[self.train_size: self.adj.shape[0] - self.test_size]
        train_doc_embeddings = tmp[: self.train_size]
        test_doc_embeddings = tmp[self.adj.shape[0] - self.test_size:]

        print(
            f"\nEmbeddings:"
            f"\n  Word:      {len(word_embeddings)}"
            f"\n  Train doc: {len(train_doc_embeddings)}"
            f"\n  Test doc:  {len(test_doc_embeddings)}"
        )
        print(f"Elapsed: {time() - t1:.2f}s")

        return float(test_acc), float(test_loss)

    def _save_model(self) -> str:
        """Save model state dict; return the file path."""
        from pathlib import Path

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        path = f"{self.log_dir}/{unique_name()}.pth"
        th.save(self.model.state_dict(), path)
        return path

    def _train(self) -> Dict:
        t1 = time()
        seed = int(self.args.seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
        print(f"\n{'∎'*20} Torch Seed: {seed}")

        train_loss_hist, train_acc_hist = [], []
        eval_loss_hist, eval_acc_hist = [], []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        model_path: Optional[str] = None
        epoch = 0

        for epoch in range(self.max_epoch):
            epoch_start = time()
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
                best_state = deepcopy(self.model.state_dict())

            print(
                f"Epoch:{epoch+1:04d}, train_loss={float(loss.detach()):.5f}, train_acc={acc:.5f}, "
                f"val_loss={val_loss:.5f}, val_acc={val_acc:.5f}, time={time()-epoch_start:.5f}"
            )

            train_loss_hist.append(float(loss.detach()))
            train_acc_hist.append(float(acc))
            eval_loss_hist.append(float(val_loss))
            eval_acc_hist.append(float(val_acc))

            if epoch > self.earlystopping and val_losses[-1] > np.mean(
                val_losses[-(self.earlystopping + 1): -1]
            ):
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Restoring best validation checkpoint (val_loss={best_val_loss:.5f})."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        model_path = self._save_model()

        print(
            f"\nCompleted {epoch+1} epochs — "
            f"train_loss={float(loss.detach()):.4f}, best_val_loss={best_val_loss:.4f}, "
            f"last_train_acc={acc:.4f}"
        )

        plot_training_history(
            self.log_dir, train_acc_hist, eval_acc_hist, train_loss_hist, eval_loss_hist
        )
        print("\nOptimization Finished!")

        test_acc, test_loss = self._test(t1)
        return {
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "model_path": model_path,
            "epochs_trained": epoch + 1,
        }
