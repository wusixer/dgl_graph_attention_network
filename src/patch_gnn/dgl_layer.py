#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Graph attention layer implemented in DGL package"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling
from dgl.nn.pytorch.conv import GATConv


class GATN(torch.nn.Module):
    """Modified from from https://github.com/dmlc/dgl/issues/1887"""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        gat_n_heads: int,
        gat_out_dim: int,
        dense_out1: int,
        dense_out2: int,
        dense_out3: int,
    ):
        super(GATN, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim)
        self.gatlayer = GATConv(embed_dim, gat_out_dim, gat_n_heads)
        self.poolinglayer = (
            AvgPooling()
        )  # use avgpooling to adjust for different size of graph
        self.dense1 = nn.Linear(gat_out_dim * gat_n_heads, dense_out1)
        self.dense2 = nn.Linear(dense_out1, dense_out2)
        self.dense3 = nn.Linear(dense_out2, dense_out3)

    # there are multiple disconnected graph, will be easier to think of them as ind graph (one graph)
    def forward(self, graph, feature):
        h = self.dense(
            feature
        )  # (graph_sample_size, in_dim) --> (graph_sample_size, embed_dim)
        # print(f"h after embedding is {h.size()}")
        h, atten = self.gatlayer(
            graph, h, get_attention=True
        )  # --> h: (n_nodes, num_heads, gat_out_dim)
        # print(f"h size after gat is {h.size()}")
        h = h.flatten(start_dim=1)  # (n_nodes, gat_out_dim*num_heads )
        # add graph summation across 1st axis - axis 0
        h = self.poolinglayer(graph, h)  # (batch_size, gat_out_dim*num_heads)
        # use sumlayer insteadh = torch.sum(h, dim=0)
        # print(f"h size after summation is {h.size()}")
        h = F.elu(h)
        h = self.dense1(h)
        h = F.elu(h)
        h = self.dense2(h)
        h = F.elu(h)
        h = self.dense3(h)

        return h, atten
