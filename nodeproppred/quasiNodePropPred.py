import sys
sys.path.append("./")
from utils.quasi_stable_coloring import QuasiStableColoring
from torch_geometric import transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data


def quasi(dataset_name, directed):
    dataset = PygNodePropPredDataset(name='ogbn-' + dataset_name, transform=T.ToSparseTensor())
    data = dataset[0]
    if dataset_name == 'mag':
        rel_data = data
        data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
        data = T.ToSparseTensor()(data)

    if not directed:
        data.adj_t = data.adj_t.to_symmetric()
    data.adj_t = data.adj_t.to_torch_sparse_coo_tensor()
    store_root = f"data_compressed/{dataset_name}/"
    store_root += "directed" if directed else "undirected"
    qsc = QuasiStableColoring(data, store_root, directed)
    qsc.q_color(n_colors=3000)


if __name__ == '__main__':
    for dataset_name in ['arxiv', 'products', 'mag', 'proteins']:
        if dataset_name in ['arxiv', 'mag']:
            for directed in [True, False]:
                quasi(dataset_name, directed)
            continue
        
        if dataset_name in ['products', 'proteins']:
            quasi(dataset_name, False)

    