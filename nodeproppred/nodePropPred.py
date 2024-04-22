import argparse
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import sys
sys.path.append("./")
from model.GCN import GCN
from utils.logger import Logger
from utils.utils import train, test
import time
from torch_geometric.data import Data
import logging
import os
import json
import torch.nn.functional as F


def compressed_data(data, adj_t, colors, split_idx, num_classes):
    new_x = torch.zeros(len(colors), data.x.shape[1])
    for i in range(new_x.shape[0]):
        new_x[i] = torch.mean(data.x[torch.tensor(colors[i])], dim=0)
    adj_t = torch.tensor(adj_t).to_sparse_coo()
    y_onehot = F.one_hot(data.y, num_classes=num_classes)
    y_onehot[split_idx['valid']] = torch.zeros(1, 40).type(torch.LongTensor)
    y_onehot[split_idx['test']] = torch.zeros(1, 40).type(torch.LongTensor)
    new_y = torch.zeros(len(colors), num_classes)
    for i, x in enumerate(colors):
        new_y[i] = torch.sum(y_onehot[x], dim=0)
        new_y[i] = new_y[i] / torch.sum(new_y[i])
    data = Data(x=new_x, edge_index=adj_t.indices, y=new_y.unsqueeze(1))
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
        train_idx = torch.sum(data.y.squeeze(1), dim=1)
        train_idx = train_idx != 0.0
        train_idx = torch.nonzero(train_idx, as_tuple=False)
        train_idx = train_idx.squeeze(1)
        data.train_idx = train_idx

    data.num_classes = num_classes
    return data, split_idx


def main():
    # args parser
    parser = argparse.ArgumentParser(description='OGBN-NodePropPred')
    parser.add_argument('--dataset', type=str, default='arxiv', help="products, proteins, arxiv, papers100M, mag")
    parser.add_argument('--compress_directed', type=bool, default=False)
    parser.add_argument('--num_colors', type=int, default=3000, help="please make sure it mod 100 = 0")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()

    # device setup
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # check if dataset exists
    datasetList = ["products", "proteins", "arxiv", "papers100M", "mag"]
    if args.dataset not in datasetList:
        print("Dataset not exist.")
        return

    # check log exists and set log file
    store_root = os.path.join("logs", "NodePropPred", args.dataset)
    if not os.path.exists(store_root):
        os.makedirs(store_root)
    
    if args.num_colors != 0:
        file_name = f"{args.dataset}_with_{args.num_colors:04d}_colors"
    else:
        file_name = f"{args.dataset}"

    # set logger
    logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(store_root, f'{file_name}.log'), mode='w'), 
                                      logging.StreamHandler()])
    logger = logging.getLogger('my_logger')
    logger.info(args)

    data_original, split_idx = data_loader(args.dataset, 0, logger=logger)
    if args.num_colors == 0:
        data = data_original
    else:
        data, split_idx = data_loader(args.dataset, args.num_colors, args.compress_directed, logger)
    data = data.to(device)
    data_original = data_original.to(device)

    # set model, evaluator and metric_logger
    model = GCN(data.num_features, args.hidden_channels, data.num_classes, args.num_layers, args.dropout).to(device)
    evaluator = Evaluator(name=f'ogbn-{args.dataset}')
    metric_Logger = Logger(runs=args.runs)

    # train and evaluate model
    time_over_all_list = []
    
    for run in range(args.runs):
        time_list = []
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            time1 = time.time()
            loss = train(model, data, data.train_idx, optimizer)
            time2 = time.time()
            time_list.append(time2 - time1)
            result = test(model, data_original, split_idx, evaluator)
            metric_Logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                logger.info(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_acc:.2f}%, '
                            f'Valid: {100 * valid_acc:.2f}%, '
                            f'Test: {100 * test_acc:.2f}%')
                
        logger.info(metric_Logger.print_statistics(run))
        time_cost = torch.sum(torch.tensor(time_list))
        logger.info(f"Run:{run+1} time cost {time_cost:.2f} seconds")
        time_over_all_list.append(time_cost)
    logger.info(metric_Logger.print_statistics())
    time_over_all = torch.mean(torch.tensor(time_over_all_list))
    logger.info(f"Time cost ten runs average: {time_over_all:.2f}, epoch average: {time_over_all / args.epochs:.2f} seconds")

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':
    main()
    