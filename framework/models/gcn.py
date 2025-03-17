import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# class GCN(nn.Module):
#     def __init__(self, args, **kwargs):
#         super().__init__()
        
#         self.conv1 = GCNConv(args.in_dim, args.hidden_dim)
#         self.conv2 = GCNConv(args.hidden_dim, args.out_dim)
#         # self.dropout = nn.Dropout(args.dropout)

#     def forward(self, x, edge_index, return_all_emb=False):
#         x1 = self.conv1(x, edge_index)
#         x = F.relu(x1)
#         # x = self.dropout(x)
#         x2 = self.conv2(x, edge_index)

#         if return_all_emb:
#             return x1, x2
        
#         return x2

#     def decode(self, z, pos_edge_index, neg_edge_index=None):
#         if neg_edge_index is not None:
#             edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
#             logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

#         else:
#             edge_index = pos_edge_index
#             logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

#         return logits


class GCN(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        self.conv1 = GCN_encoder_scatter(args.in_dim, args.hidden_dim)
        self.conv2 = GCN_encoder_scatter(args.hidden_dim, args.out_dim)
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, edge_index, return_all_emb=False, df_mask=None, edge_influence=None):
        x1 = self.conv1(x, edge_index, df_mask, edge_influence)
        x = F.relu(x1)
        # x = self.dropout(x)
        x2 = self.conv2(x, edge_index, df_mask, edge_influence)

        if return_all_emb:
            return x1, x2
        
        return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits


from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter

def propagate(x, edge_index, df_mask=None, edge_influence=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    # update message function
    if df_mask != None:
        adjust_row, adjust_col = df_mask
        adjust_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        adjust_mask |= torch.isin(edge_index[0], adjust_row) & torch.isin(edge_index[1], adjust_col)
        score = (edge_influence[row] + edge_influence[col]).to('cuda')
        
        edge_weight[adjust_mask] *= torch.exp(- score[adjust_mask])  # NPO+MPNN+GD(DBLP)
          
        # x = x * torch.exp(-edge_influence.view(-1, 1))  # NPO+MPNN+GD(PubMed)
        


    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN_encoder_scatter, self).__init__()
        self.lin = Linear(in_dim, out_dim, bias=False)
        self.bias = Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, df_mask=None, edge_influence=None):
        h = self.lin(x)
        # h = propagate(h, edge_index)
        h = propagate(h, edge_index, df_mask, edge_influence) + self.bias
        return h

class GCN_encoder_spmm(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN_encoder_spmm, self).__init__()
        self.lin = Linear(in_dim, out_dim, bias=False)
        # self.bias = Parameter(torch.Tensor(args.hidden))

    def reset_parameters(self):
        self.lin.reset_parameters()
        # self.bias.data.fill_(0.0)

    def forward(self, x, adj_norm_sp):
        h = self.lin(x)
        h = torch.spmm(adj_norm_sp, h)
        # h = torch.spmm(adj_norm_sp, h) + self.bias
        return h

