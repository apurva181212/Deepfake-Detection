import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class STGCN(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index)
        return x.mean(dim=0)  # temporal pooling
