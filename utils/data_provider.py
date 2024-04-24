import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import sys
sys.path.append("./")
from torch_geometric.data import Data
import os
import json
import torch.nn.functional as F


def compressed_data(data, adj_t, colors, split_idx, num_classes):
    y_onehot = F.one_hot(data.y, num_classes=num_classes)
    # y_onehot[split_idx['valid']] = torch.zeros(1, num_classes).type(torch.LongTensor)
    y_onehot[split_idx['test']] = torch.zeros(1, num_classes).type(torch.LongTensor)
    new_y = torch.zeros(len(colors), num_classes)
    for i, x in enumerate(colors):
        new_y[i] = torch.sum(y_onehot[x], dim=0)
        new_y[i] = new_y[i] / torch.sum(new_y[i])
    new_y = torch.argmax(new_y, dim=1, keepdim=True)

    new_x = torch.zeros(len(colors), data.x.shape[1])
    for i, X in enumerate(colors):
        new_x[i] = torch.mean(data.x[X], dim=0)

    adj_t = torch.tensor(adj_t)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t.to_sparse_coo()

    
    data = Data(x=new_x, edge_index=adj_t.indices(), y=new_y)
    data.adj_t = adj_t.to_sparse_csr()

    return data


def data_loader(dataset_name, num_colors, directed=True, logger=None):
    dataset = PygNodePropPredDataset(name=f'ogbn-{dataset_name}', transform=T.ToSparseTensor())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    num_classes = dataset.num_classes
    if num_colors == 0:
        if dataset_name == 'mag':
            rel_data = dataset[0]
            data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
            data = T.ToSparseTensor()(data)
        data.train_idx = split_idx['train']
        data.adj_t = data.adj_t.to_symmetric()
    else:
        store_root = os.path.join("data_compressed", dataset_name)
        if directed:
            store_root = os.path.join(store_root, "directed")
        else:
            store_root = os.path.join(store_root, "undirected")
        with open(os.path.join(store_root, f"{num_colors:04d}.json")) as f:
            y = json.load(f)
        colors = y[0]
        adj_t = y[1]
        time_cost = y[2]
        logger.info(f"compressed_time_cost: {time_cost}")
        data = compressed_data(data, adj_t, colors, split_idx, num_classes)
        data.train_idx = torch.arange(0, len(colors))

    data.num_classes = num_classes
    return data, split_idx