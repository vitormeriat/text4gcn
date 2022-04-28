#import torch
#from torch_geometric.nn import GCNConv, ChebConv, GATConv
#import torch.nn.functional as F
#import torch.nn as nn
#import torch.nn.functional as Func
#from torch.nn.parameter import Parameter
#import math

# class GCN(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(nfeat, nhid, cached=True, normalize=True)
#         self.conv2 = GCNConv(nhid, nclass, cached=True, normalize=True)
#         # self.conv1 = ChebConv(nfeat, nhid, K=2)
#         # self.conv2 = ChebConv(nhid, nclass, K=2)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = torch.dropout(x, self.dropout, train=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return F.log_softmax(x, dim=1)


# class GAT(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(nfeat, nhid, heads=8, dropout=dropout)
#         self.conv2 = GATConv(nhid * 8, nclass, heads=1, concat=False,
#                              dropout=dropout)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=-1)


# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, acti=True):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
#         self.acti = nn.ReLU(inplace=True) if acti else None
#         self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))

#     # def forward(self, F):
#     #     output = self.linear(F)
#     #     if not self.acti:
#     #         return output
#     #     return self.acti(output)
#     def forward(self, infeatn, adj):
#         support = torch.spmm(infeatn, self.weight)
#         output = torch.spmm(adj, support)
#         return output + self.acti if self.acti is not None else output


# class GCN(nn.Module):
#     #def __init__(self, input_dim, hidden_dim, num_classes, p):
#     def __init__(self, nfeat, nhid, nclass, dropout):

#         super(GCN, self).__init__()
#         self.gcn_layer1 = GCNLayer(nfeat, nhid)
#         self.gcn_layer2 = GCNLayer(nhid, nclass, acti=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, X, A):
#         #A = torch.from_numpy(preprocess_adj(A)).float()
#         #X = self.dropout(X.float())
#         #F = torch.mm(A, X)
#         # F = self.gcn_layer1(F)
#         # F = self.dropout(F)
#         # F = torch.mm(A, F)
#         # return self.gcn_layer2(F)
#         X = self.gcn_layer1(X, A)
#         X = self.dropout(X)
#         #F = torch.mm(A, F)
#         return self.gcn_layer2(X, A)


# class GraphConvolution(nn.Module):
#     """
#     Simple pygGCN layer
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, infeatn, adj):
#         support = torch.spmm(infeatn, self.weight)
#         output = torch.spmm(adj, support)
#         return output + self.bias if self.bias is not None else output

#     # def __repr__(self):
#     #     return ((f'{self.__class__.__name__} (' + +str(self.in_features)) + ' -> ') + str(self.out_features) + ')'

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout) -> None:
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = self.gc1(x, adj)
#         x = torch.relu(x)
#         x = torch.dropout(x, self.dropout, train=self.training)
#         x = self.gc2(x, adj)
#         return x


import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support, act_func=None, featureless=False, dropout_rate=0., bias=False):

        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(self, f'W{i}', nn.Parameter(
                torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, f'W{i}')
            else:
                pre_sup = x.mm(getattr(self, f'W{i}'))

            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__(self, input_dim, support, dropout_rate=0., num_classes=10):
        super(GCN, self).__init__()

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(
        ), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(
            200, num_classes, support, dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
