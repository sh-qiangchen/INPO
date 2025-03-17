import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from . import GCN, GAT, GIN


class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        # self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(x[mask], self.deletion_weight)
            return new_rep

        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class GCNDelete(GCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.conv1(x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GATDelete(GAT):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        with torch.no_grad():
            x1 = self.conv1(x, edge_index)
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GINDelete(GIN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        with torch.no_grad():
            x1 = self.conv1(x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

#=======================LoRA=============================

from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter

def propagate2(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)        # deg_inv_sqrt = torch.clamp(deg.pow(-0.5), min=1e-5, max=1e5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LoRALayer, self).__init__()
        self.r = 3
        # self.lin = Linear(in_dim, out_dim, bias=False)
        self.B = Linear(in_dim, self.r, bias=False)
        self.A = Linear(self.r, out_dim, bias=False)
        self.bias = Parameter(torch.Tensor(out_dim)) 
        self.reset_parameters()

    def reset_parameters(self):
        # self.lin.reset_parameters()
        nn.init.constant_(self.B.weight, 0)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01) 
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.A(self.B(x)) 
        h = propagate2(h, edge_index) + self.bias
        return h

class LoRAGCNDelete(GCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = LoRALayer(args.in_dim, args.hidden_dim)
        self.deletion2 = LoRALayer(args.hidden_dim, args.out_dim)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # print("======LoRA layer")
        x1 = self.conv1(x, edge_index) + self.deletion1(x, edge_index)
        x = F.leaky_relu(x1)  # x = F.relu(x1)
        x2 = self.conv2(x, edge_index) + self.deletion2(x, edge_index)
        
        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)
