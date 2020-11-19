import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_res, dropout, device):
        super(GCN,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(len(num_layers)):
            num_layer = num_layers[i]
            convs = torch.nn.ModuleList()
            convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            for _ in range(num_layer - 2):
                convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            convs.append(GCNConv(hidden_channels, out_channels, cached=True))
            convs = convs.to(device)
            self.convs.append(convs)
        self.dropout = dropout
        self.use_res = use_res
        self.number = len(self.num_layers)

    def reset_parameters(self):
        for convlist in self.convs:
            for conv in convlist:
                conv.reset_parameters()

    def forward(self, x, adj_t):
        x_all = []
        for convlist in self.convs:
            x1 = x
            x1 = convlist[0](x1, adj_t)
            for conv in convlist[1:]:
                x2 = x1
                x1 = F.relu(x1)
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
                if self.use_res:
                    x1 = conv(x1, adj_t) + x2
                else:
                    x1 = conv(x1, adj_t)
            x_all.append(x1)
        x_final = x_all[0]
        for i in range(1, len(x_all)):
            x_final += x_all[i]
        x_final /= self.number
        return x_final
