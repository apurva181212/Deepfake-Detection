from torch_geometric.nn import GATConv
import torch
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=4)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)
