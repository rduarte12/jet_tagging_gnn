from torch import nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv, global_mean_pool
from torch.nn import functional as F


def classifier_head():
    return nn.Sequential(
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
    )


class JetGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 64)
        self.classifier = classifier_head()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

class JetGAT(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.conv1 = GATConv(4,  16, heads=heads, concat=True)   # -> 64
        self.conv2 = GATConv(64, 64, heads=heads, concat=False)   # -> 64
        self.classifier = classifier_head()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class JetEdgeConv(nn.Module):
    def __init__(self):
        super().__init__()
        def edge_mlp(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(in_ch * 2, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Linear(out_ch,    out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(),
            )
        self.conv1 = EdgeConv(edge_mlp(4,  64), aggr='max')
        self.conv2 = EdgeConv(edge_mlp(64, 64), aggr='max')
        self.classifier = classifier_head()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)
