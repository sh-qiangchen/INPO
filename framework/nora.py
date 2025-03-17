import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import argparse
from tqdm import tqdm
import copy
import time
import dgl
import dgl.function as fn
import os

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# graph_ori: original graph Data instance
def nora(args, model, graph_ori, pyg_data, device):
    graph_ori = graph_ori.to(device)
    graph = graph_ori.clone()
    # graph = clone_dgl_graph(graph_ori)
    
    deg = graph.remove_self_loop().in_degrees().float()
    num_node = graph.num_nodes()
    mean_deg = deg.float().mean()

    model.eval()  
    
    
    # x = copy.deepcopy(graph.ndata['feat'])
    x = pyg_data.x
    x.requires_grad = True
    
    # out, hidden_list = model(graph, x, return_hidden=True)
    h1, out = model(x, pyg_data.train_pos_edge_index, return_all_emb=True)  # pyg
    
    hidden_list = [x, h1, out]
    out = F.softmax(out, dim=1)

    for hs in hidden_list:
        hs.retain_grad()    # 保留中间变量的梯度
    out.backward(gradient=out, retain_graph=True)   # out is not a scalar, add gradient=out.
    hidden_grad_list = []
    for i in range(len(hidden_list)):
        hidden_grad_list.append(hidden_list[i].grad.detach())

    gradient = torch.zeros(num_node, device=device)
    rate = 1.0
    assert len(hidden_list) == args.num_layers + 1
    for i in range(len(hidden_list) - 2, -1, -1):
        new_grad = hidden_grad_list[i] * hidden_list[i]
        new_grad = torch.norm(new_grad, p=1, dim=1)  # L1
        new_grad = new_grad * deg / (deg + args.self_buff)
        gradient = gradient + new_grad * rate
        rate = rate * (1 - deg / (num_node - 1) / (mean_deg + args.self_buff))

    assert (gradient < 0).sum() == 0
    deg_delta1 = 1 / torch.sqrt(deg - 1) - 1 / torch.sqrt(deg)
    deg_delta2 = 1 / (deg-1) - 1 / deg
    deg_delta1[deg_delta1 == np.nan] = 1.0
    deg_delta2[deg_delta2 == np.nan] = 1.0
    deg_delta1[deg_delta1.abs() == np.inf] = 1.0
    deg_delta2[deg_delta2.abs() == np.inf] = 1.0
    deg_delta = args.k1 * deg_delta1 + (1 - args.k1) * deg_delta2
    deg_inv = args.k2[0] / torch.sqrt(deg) + args.k2[1] / deg + (1 - args.k2[0] - args.k2[1])
    
    # i != r
    graph = graph.remove_self_loop()

    graph.ndata.update({'deg_inv': deg_inv})
    graph.update_all(fn.copy_u("deg_inv", "m1"), fn.sum("m1", "deg_inv_sum"))
    deg_gather = graph.ndata['deg_inv_sum']
    graph.ndata.update({'deg_delta': deg_gather * deg_delta})
    graph.update_all(fn.copy_u("deg_delta", "m2"), fn.sum("m2", "deg_gather"))
    deg_gather = graph.ndata['deg_gather']
    deg_gather = deg_gather / deg_gather.mean() * gradient.mean()  # Normalize
    influence = gradient + args.k3 * deg_gather
    influence = influence.abs().detach().cpu().numpy()
    
    def min_max_normalize(vector):
        return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    
    influence = min_max_normalize(influence)
    
    return influence