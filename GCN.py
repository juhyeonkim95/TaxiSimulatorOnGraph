'''
Codes are from
https://github.com/dmlc/dgl/tree/master/examples/pytorch
'''

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv


def positive_safe_sigmoid(x):
    return torch.sigmoid(x) + 1e-8


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=positive_safe_sigmoid))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h


class GAT(nn.Module):
    def __init__(self,
                 g,
                 activation,
                 in_dim=1,
                 num_classes=1,
                 num_layers=1,
                 num_hidden=8,
                 num_heads=8,
                 num_out_heads=1,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False):

        heads = ([num_heads] * num_layers) + [num_out_heads]

        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, positive_safe_sigmoid))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
