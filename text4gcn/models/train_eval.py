import matplotlib.pyplot as plt
from tabulate import tabulate
# from ..models.utils import *
from text4gcn.models.utils import *
from sklearn import metrics
import scipy.sparse as sp
from time import time
import numpy as np
import torch as th


class TextGCNTrainer:
    def __init__(self, model, args, pre_data):
        self.args = args
        self.model = model
        self.device = args.device
        self.max_epoch = self.args.max_epoch
        self.nhid = self.args.nhid
        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = args.early_stopping
        self.log_dir = args.log_dir

    def fit(self):
        # Extract data
        self.prepare_data()
        # Load corpus, unpack values & convert to tensor
        self.convert_tensor()

        self.model = self.model(
            input_dim=self.features.shape[0], 
            output_dim=self.nhid, 
            support=self.support, 
            num_classes=self.y_train.shape[1]
        ).to(self.device)

        # Loss and optimizer
        self.criterion = th.nn.CrossEntropyLoss()
        self.optimizer = th.optim.Adam(
            self.model.parameters(), lr=self.args.lr)

        print("\nModel structure:\n\n", self.model, "\n")
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}")

        self.train_model()

    def prepare_data(self):
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = self.predata

        # featureless
        self.features = sp.identity(features.shape[0])
        # feature of nodes
        self.features = preprocess_features(self.features)
        # adjacency matrix
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

    def convert_tensor(self):

        print("\nData statistics\n")
        print(tabulate([
            ["Edges", "Classes", "Train samples", "Val samples", "Test samples"],
            [self.features.shape[0], self.y_train.shape[1], self.train_mask.sum(), self.val_mask.sum(), self.test_mask.sum()]],
            headers="firstrow"))

        self.features = th.from_numpy(self.features).float().to(self.device)
        self.y_train = th.from_numpy(self.y_train).float().to(self.device)
        self.y_val = th.from_numpy(self.y_val).float().to(self.device)
        self.y_test = th.from_numpy(self.y_test).float().to(self.device)
        self.train_mask = th.from_numpy(
            self.train_mask).float().to(self.device)
        self.val_mask = th.from_numpy(self.val_mask).float().to(self.device)
        self.test_mask = th.from_numpy(self.test_mask).float().to(self.device)
        self.tm_train_mask = th.transpose(th.unsqueeze(
            self.train_mask, 0), 1, 0).repeat(1, self.y_train.shape[1])
        self.support = [th.Tensor(self.support[i])
                        for i in range(len(self.support))]

    def evaluate_model(self, model, criterion, features, labels, mask):
        t_test = time()

        model.eval()
        with th.no_grad():
            logits = model(features)
            t_mask = th.from_numpy(np.array(mask * 1., dtype=np.float32))
            tm_mask = th.transpose(th.unsqueeze(
                t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = criterion(logits * tm_mask, th.max(labels, 1)[1])
            pred = th.max(logits, 1)[1]

            try:
                acc = ((pred == th.max(labels, 1)[1]).float(
                ) * t_mask).sum().item() / t_mask.sum().item()
            except ZeroDivisionError:
                acc = 0

        return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time() - t_test)

    def show_metrics(self, test_labels, test_pred):
        """ Calucate evaluation metrics for precision, recall, and f1.

        Arguments
        ---------
            y_pred: ndarry, the predicted result list
            y_true: ndarray, the ground truth label list

        Returns
        -------
            precision: float, precision value
            recall: float, recall value
            f1: float, f1 measure value
        """
        print("\nTest Precision, Recall and F1-Score...")
        print(metrics.classification_report(test_labels, test_pred, digits=4))
        print("\nMacro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(
            test_labels, test_pred, average='macro'))
        print("\nMicro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(
            test_labels, test_pred, average='micro'))
        print("\nWeighted average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(
            test_labels, test_pred, average='weighted'))
        print("\nConfusion Matrix...")
        print(metrics.confusion_matrix(test_labels, test_pred))
        print("\nF1-Score...")
        print(metrics.f1_score(test_labels, test_pred, average="macro"))
        print("\nPrecision-Score...")
        print(metrics.precision_score(test_labels, test_pred, average="macro"))
        print("\nRecall-Score...")
        print(metrics.recall_score(test_labels, test_pred, average="macro"))
        print("\nAccuracy-Score...")
        print(metrics.accuracy_score(test_labels, test_pred))

    def test(self, t1):
        # Testing
        test_loss, test_acc, pred, labels, test_duration = self.evaluate_model(
            self.model, self.criterion, self.features, self.y_test, self.test_mask)
        print("\nTest set results: loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(
            test_loss, test_acc, test_duration))

        test_pred = []
        test_labels = []
        for i in range(len(self.test_mask)):
            if self.test_mask[i]:
                test_pred.append(pred[i])
                test_labels.append(np.argmax(labels[i]))

        elapsed = time() - t1

        # print("\nTest Precision, Recall and F1-Score...")
        # print(metrics.classification_report(test_labels, test_pred, digits=4))
        # print("\nMacro average Test Precision, Recall and F1-Score...")
        # print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
        # print("\nMicro average Test Precision, Recall and F1-Score...")
        # print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))
        # print("\nWeighted average Test Precision, Recall and F1-Score...")
        # print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='weighted'))
        # print("\nConfusion Matrix...")
        # print(metrics.confusion_matrix(test_labels, test_pred))
        # print("\nF1-Score...")
        # print(metrics.f1_score(test_labels, test_pred, average="macro"))
        # print("\nPrecision-Score...")
        # print(metrics.precision_score(test_labels, test_pred, average="macro"))
        # print("\nRecall-Score...")
        # print(metrics.recall_score(test_labels, test_pred, average="macro"))
        # print("\nAccuracy-Score...")
        # print(metrics.accuracy_score(test_labels, test_pred))
        self.show_metrics(test_labels, test_pred)

        # doc and word embeddings
        tmp = self.model.layer1.embedding.numpy()
        word_embeddings = tmp[self.train_size: self.adj.shape[0] - self.test_size]
        train_doc_embeddings = tmp[:self.train_size]
        test_doc_embeddings = tmp[self.adj.shape[0] - self.test_size:]

        print('\nEmbeddings:')
        print(f'Word_embeddings:{len(word_embeddings)}')
        print(f'Train_doc_embeddings:{len(train_doc_embeddings)}')
        print(f'Test_doc_embeddings:{len(test_doc_embeddings)}')
        print("\nElapsed time is %f seconds." % elapsed)

    def saveModel(self):
        name = unique_name()
        #path = f"{self.log_dir}/NetModel.pth"
        path = f"{self.log_dir}/{unique_name()}.pth"
        th.save(self.model.state_dict(), path)

    def plot_accuracy(self, train_acc_hist, eval_acc_hist, train_loss_hist, eval_loss_hist):
        epochs = range(1, len(train_acc_hist) + 1)
        plt.figure(figsize=(16, 10))
        plt.plot(epochs, train_acc_hist, 'b', label='Training acc')
        plt.plot(epochs, eval_acc_hist, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(f'{self.log_dir}/metrics-acc.png', dpi=250)

        plt.figure()
        plt.figure(figsize=(16, 10))
        plt.plot(epochs, train_loss_hist, 'b', label='Training loss')
        plt.plot(epochs, eval_loss_hist, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(f'{self.log_dir}/metrics-loss.png', dpi=250)

    def train_model(self):

        t1 = time()

        th.manual_seed(self.args.seed)
        print(f"\n{'∎'*20} Torch Seed: {th.seed()}")  # "∎"

        train_loss_hist = []
        eval_loss_hist = []

        train_acc_hist = []
        eval_acc_hist = []

        val_losses = []

        # Train model
        for epoch in range(self.max_epoch):
            epoch_start_time = time()

            # Forward pass: Compute predicted y by passing x to the model
            logits = self.model(self.features)
            # Compute loss
            loss = self.criterion(
                logits * self.tm_train_mask, th.max(self.y_train, 1)[1])

            try:
                acc = ((th.max(logits, 1)[1] == th.max(
                    self.y_train, 1)[1]).float(
                ) * self.train_mask).sum().item() / self.train_mask.sum().item()
            except ZeroDivisionError:
                acc = 0

            # Backward and optimize
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            # for each parameter, calculate d(loss)/d(weight)
            loss.backward()
            # update weights, causes the optimizer to take a step based on the gradients of the parameters
            self.optimizer.step()

            # Validation
            val_loss, val_acc, pred, labels, duration = self.evaluate_model(
                self.model, self.criterion, self.features, self.y_val, self.val_mask)
            val_losses.append(val_loss)

            print("Epoch:{:04d}, train_loss={:.5f}, train_acc={:.5f}, val_loss={:.5f}, val_acc={:.5f}, time={:.5f}".format(
                epoch + 1, loss, acc, val_loss, val_acc, time() - epoch_start_time))

            if epoch > self.earlystopping and val_losses[-1] > np.mean(val_losses[-(self.earlystopping + 1):-1]):
                print(
                    f'Early stopping... Validation loss decreased ({val_loss:.6f}).  Saving model ...')
                self.saveModel()
                break

            train_loss_hist.append(float(loss))
            train_acc_hist.append(float(acc))
            eval_loss_hist.append(float(val_loss))
            eval_acc_hist.append(float(val_acc))

        print('\nCompleted training batch', epoch+1, 'Training Loss is: %.4f' %
              loss, 'Validation Loss is: %.4f' % val_loss, 'Accuracy is %.4f' % (acc))

        # epochs = range(1, len(train_acc_hist) + 1)
        # plt.figure(figsize=(16, 10))
        # plt.plot(epochs, train_acc_hist, 'b', label='Training acc')
        # plt.plot(epochs, eval_acc_hist, 'r', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.legend()
        # plt.savefig(f'{self.log_dir}/metrics-acc.png', dpi=250)

        # plt.figure()
        # plt.figure(figsize=(16, 10))
        # plt.plot(epochs, train_loss_hist, 'b', label='Training loss')
        # plt.plot(epochs, eval_loss_hist, 'r', label='Validation loss')
        # plt.title('Training and validation loss')
        # plt.legend()
        # plt.savefig(f'{self.log_dir}/metrics-loss.png', dpi=250)
        self.plot_accuracy(train_acc_hist, eval_acc_hist, train_loss_hist, eval_loss_hist)

        print("\nOptimization Finished!")

        self.test(t1)
