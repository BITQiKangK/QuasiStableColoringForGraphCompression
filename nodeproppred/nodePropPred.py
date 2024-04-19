import argparse
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import sys
sys.path.append("./")
from model.GCN import GCN
from utils.logger import Logger
from utils.utils import train, test
import logging
import os


def main():
    parser = argparse.ArgumentParser(description='OGBN-NodePropPred')
    parser.add_argument('--dataset', type=str, default='arxiv', help="products, proteins, arxiv, papers100M, mag")
    parser.add_argument('--num_colors', type=int, default=0, help="please make sure it mod 100 = 0")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    datasetList = ["products", "proteins", "arxiv", "papers100M", "mag"]

    if args.dataset not in datasetList:
        print("Dataset not exist.")
        return

    store_root = os.path.join("logs", "NodePropPred", args.dataset)
    if not os.path.exists(store_root):
        os.makedirs(store_root)
    
    if args.num_colors != 0:
        file_name = f"{args.dataset}_with_{args.num_colors:4d}_colors"
    else:
        file_name = f"{args.dataset}"

    logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(store_root, f'{file_name}.log'), mode='w'), 
                                      logging.StreamHandler()])
    logger = logging.getLogger('my_logger')
    logger.info(args)


    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)



if __name__ == '__main__':
    main()
    