from ogb.linkproppred import PygLinkPropPredDataset
import os
import math
import torch
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import k_hop_subgraph, is_undirected, to_undirected, negative_sampling, subgraph
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Planetoid, Reddit2, Flickr
from dataclasses import dataclass
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


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

@dataclass
class Args:
    num_layers: int = 2
    self_buff: int = 8
    k1: float = 1.0
    k2: list = (0.5, 0.0) 
    k3: int = 1

def get_original_data(d):
    data_dir = './data'    

    if d in ['Cora', 'PubMed', 'DBLP']:
        dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Cora_p', 'PubMed_p', 'Citeseer_p']:
        dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures())
    elif d in ['CS', 'Physics']:
        dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Amazon']:
        dataset = Amazon(os.path.join(data_dir, d), 'Photo', transform=T.NormalizeFeatures())
    elif d in ['Reddit']:   # On 4090: need minibatch
        dataset = Reddit2(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Flickr']:
        dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif 'ogbl' in d:
        dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)
    else:
        raise NotImplementedError(f"{d} not supported.")
    data = dataset[0]
    return data


def gen_inout_mask(data):
    _, local_edges, _, mask = k_hop_subgraph(
        data.val_pos_edge_index.flatten().unique(), 
        2, 
        data.train_pos_edge_index, 
        num_nodes=data.num_nodes)
    distant_edges = data.train_pos_edge_index[:, ~mask]
    print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

    in_mask = mask
    out_mask = ~mask

    return {'in': in_mask, 'out': out_mask}


def sort_by_influence(data, model, ):
    return 

def split_forget_retain(data, df_size, subset='in', model1=None):
    if df_size >= 100:     # df_size is number of nodes/edges to be deleted
        df_size = int(df_size)
    else:                       # df_size is the ratio
        df_size = int(df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')
    df_mask_all = gen_inout_mask(data)[subset]
    df_nonzero = df_mask_all.nonzero().squeeze()        # subgraph子图内/外的edge idx序号
    
    if model1 is None:
        idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    else:    
        # 根据边的影响性进行排序，挑选影响性最大或最小的比例
        infer_hp = Args()     
        import dgl
        num_nodes = data.x.size(0)
        src, dst = data.edge_index
        graph_dgl = dgl.graph((src, dst), num_nodes=num_nodes)
        graph_dgl.ndata['feat'] = data.x.clone()  # deepcopy
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        influence_score = torch.from_numpy(nora(infer_hp, model1.to(device), graph_dgl.to(device), data.to(device), device)).to(device) 
        selected_edges = data.train_pos_edge_index[:, df_nonzero]
        source_nodes_influence = influence_score[selected_edges[0]]  
        target_nodes_influence = influence_score[selected_edges[1]]  
        edge_influences = source_nodes_influence + target_nodes_influence
        idx = torch.argsort(edge_influences, descending=True)[:df_size] 
        # sorted_edge_influences = edge_influences[sorted_indices]
        # sorted_edges = selected_edges[:, sorted_indices] 
        
        # idx = torch.argsort(edge_influences, descending=True)[:3 * df_size] 
        # idx = torch.randperm(idx.shape[0])[:df_size]

    # idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # Collect enclosing subgraph of Df for loss computation
    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        2, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        1, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    assert not is_undirected(data.train_pos_edge_index)

    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
    # to_undirected return full undirected edges and corresponding mask for given edge_attrs
    two_hop_mask = two_hop_mask.bool()  
    df_mask = df_mask.bool()
    dr_mask = ~df_mask

    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask
    return data

def train_test_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.05, test_ratio: float = 0.05, two_hop_degree=None):
    '''Avoid adding neg_adj_mask'''

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    if two_hop_degree is not None:          # Use low degree edges for test sets
        low_degree_mask = two_hop_degree < 50

        low = low_degree_mask.nonzero().squeeze()
        high = (~low_degree_mask).nonzero().squeeze()

        low = low[torch.randperm(low.size(0))]
        high = high[torch.randperm(high.size(0))]

        perm = torch.cat([low, high])

    else:
        perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]

    # Train
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.train_pos_edge_index, data.train_pos_edge_attr = None
    else:
        data.train_pos_edge_index = data.train_pos_edge_index
    
    assert not is_undirected(data.train_pos_edge_index)

    
    # Test
    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data


def load_dict(filename):
    '''Load entity and relation to id mapping'''

    mapping = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            mapping[l[0]] = l[1]

    return mapping

def load_edges(filename):
    with open(filename, 'r') as f:
        r = f.readlines()
    r = [i.strip().split('\t') for i in r]

    return r

def generate_true_dict(all_triples):
    heads = {(r, t) : [] for _, r, t in all_triples}
    tails = {(h, r) : [] for h, r, _ in all_triples}

    for h, r, t in all_triples:
        heads[r, t].append(h)
        tails[h, r].append(t)

    return heads, tails

def get_loader(args, delete=[]):
    prefix = os.path.join('./data', args.dataset)

    # Edges
    train = load_edges(os.path.join(prefix, 'train.txt'))
    valid = load_edges(os.path.join(prefix, 'valid.txt'))
    test = load_edges(os.path.join(prefix, 'test.txt'))
    train = [(int(i[0]), int(i[1]), int(i[2])) for i in train]
    valid = [(int(i[0]), int(i[1]), int(i[2])) for i in valid]
    test = [(int(i[0]), int(i[1]), int(i[2])) for i in test]
    train_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in train]
    valid_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in valid]
    test_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in test]
    train = train + train_rev
    valid = valid + valid_rev
    test = test + test_rev
    all_edge = train + valid + test

    true_triples = generate_true_dict(all_edge)

    edge = torch.tensor([(int(i[0]), int(i[2])) for i in all_edge], dtype=torch.long).t()
    edge_type = torch.tensor([int(i[1]) for i in all_edge], dtype=torch.long)#.view(-1, 1)

    # Masks
    train_size = len(train)
    valid_size = len(valid)
    test_size = len(test)
    total_size = train_size + valid_size + test_size

    train_mask = torch.zeros((total_size,)).bool()
    train_mask[:train_size] = True

    valid_mask = torch.zeros((total_size,)).bool()
    valid_mask[train_size:train_size + valid_size] = True
    
    test_mask = torch.zeros((total_size,)).bool()
    test_mask[-test_size:] = True

    # Graph size
    num_nodes = edge.flatten().unique().shape[0]
    num_edges = edge.shape[1]
    num_edge_type = edge_type.unique().shape[0]

    # Node feature
    x = torch.rand((num_nodes, args.in_dim))

    # Delete edges
    if len(delete) > 0:
        delete_idx = torch.tensor(delete, dtype=torch.long)
        num_train_edges = train_size // 2
        train_mask[delete_idx] = False
        train_mask[delete_idx + num_train_edges] = False
        train_size -= 2 * len(delete)
    
    node_id = torch.arange(num_nodes)
    dataset = Data(
        edge_index=edge, edge_type=edge_type, x=x, node_id=node_id, 
        train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    dataloader = GraphSAINTRandomWalkSampler(
        dataset, batch_size=args.batch_size, walk_length=args.walk_length, num_steps=args.num_steps)

    print(f'Dataset: {args.dataset}, Num nodes: {num_nodes}, Num edges: {num_edges//2}, Num relation types: {num_edge_type}')
    print(f'Train edges: {train_size//2}, Valid edges: {valid_size//2}, Test edges: {test_size//2}')
    
    return dataloader, valid, test, true_triples, num_nodes, num_edges, num_edge_type


