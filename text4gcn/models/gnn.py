from text4gcn.models.train_eval import TextGCNTrainer
from text4gcn.models.utils import LoadData
from text4gcn.models.layers import Layer
from text4gcn.models.model import GCN
import numpy as np
import torch as th
import argparse

# # parser = argparse.ArgumentParser(description='TextGCN')
# # #parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
# # parser.add_argument('--dataset', type=str, help='Dataset to train')
# # parser.add_argument('--lr', type=float, default=0.01, help='Initial learing rate')
# # parser.add_argument('--epoches', type=int, default=200, help='Number of traing epoches')
# # parser.add_argument('--hidden_dim', type=list, default=200, help='Dimensions of hidden layers')
# # parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
# # parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
# # parser.add_argument('--log_interval', type=int, default=10, help='Print iterval')
# # parser.add_argument('--log_dir', type=str, default='experiments', help='Train/val loss and accuracy logs and plots')
# # parser.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint saved interval')
# # parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
# # args = parser.parse_args()


class GNN():
    """
        The graph2seq model consists the following components: 1) node embedding 2) graph embedding 3) decoding. # noqa
        Since the full pipeline will consist all parameters, so we will add prefix to the original parameters
         in each component as follows (except the listed four parameters):
            1) emb_ + parameter_name (eg: ``emb_input_size``)
            2) gnn_ + parameter_name (eg: ``gnn_direction_option``)
            3) dec_ + parameter_name (eg: ``dec_max_decoder_step``)
        Considering neatness, we will only present the four hyper-parameters which don't meet regulations.

    Examples
    -------
        # Build a vocab model from scratch
        >>> "It is just a how-to-use example."
        >>> from graph4nlp.pytorch.modules.config import get_basic_args
        >>> opt = get_basic_args(graph_construction_name="node_emb", graph_embedding_name="gat", decoder_name="stdrnn")
        >>> graph2seq = Graph2Seq.from_args(opt=opt, vocab_model=vocab_model, device=torch.device("cuda:0"))
        >>> batch_graph = [GraphData() for _ in range(2)]
        >>> tgt_seq = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> seq_out, _, _ = graph2seq(batch_graph=batch_graph, tgt_seq=tgt_seq)
        >>> print(seq_out.shape) # [2, 6, 5] (assume the vocabulary size is 5 and max_decoder_step is 6)

    Attributes
    ----------
    dataset : str
        first name of the person
    path : str
        family name of the person
    log_dir : int
        age of the person
    seed: int, default=1234
        The seed for random function.

    Methods
    -------
    config():
        Prints the person's name and age.
    fit():
        Prints the person's name and age.
    """

    def __init__(
        self,
        dataset,
        path,
        log_dir,
        builder,
        lr=0.02,
        nhid=200,
        seed=1234,
        epoches=200,
        dropout=0.5,
        val_ratio=0.1,
        layer=Layer.GCN,
        early_stopping=10
    ) -> None:

        self.early_stopping = early_stopping
        self.val_ratio = val_ratio
        self.dataset = dataset
        self.epoches = epoches
        self.dropout = dropout
        self.log_dir = log_dir
        self.builder = builder
        self.layer = layer
        self.seed = seed
        self.nhid = nhid
        self.path = path
        self.lr = lr

    def config(self):
        print(f"\n{'='*30} MODEL CONFIGURATION\n")

        parser = argparse.ArgumentParser(description='TextGCN')
        self.args = parser.parse_args()

        self.args.early_stopping = self.early_stopping
        self.args.seed = np.random.randint(1, 100000)
        self.args.val_ratio = self.val_ratio
        self.args.max_epoch = self.epoches
        self.args.dataset = self.dataset
        self.args.dropout = self.dropout
        self.args.log_dir = self.log_dir
        self.args.layer = self.layer
        self.args.nhid = self.nhid
        self.args.lr = self.lr
        self.args.device = th.device(
            'cuda') if th.cuda.is_available() else th.device('cpu')

        print(f'{self.args}\n')

    def fit(self):
        self.config()

        predata = LoadData(
            path=self.path,
            dataset=self.dataset,
            builder=self.builder).load_corpus()

        if self.layer == Layer.GCN:
            model = GCN
        else:
            raise TypeError(
                "Invalide Layer. In this version only the GCN layer is valid!")

        framework = TextGCNTrainer(
            model=model,
            args=self.args,
            pre_data=predata
        )

        framework.fit()
