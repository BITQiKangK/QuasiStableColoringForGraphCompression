import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import sys
sys.path.append("./")
from utils.logger import Logger
from model.GCN import GCN
import logging
import os
from utils.data_provider import data_loader


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--num_colors', type=int, default=10000)
    args = parser.parse_args()
    print(args)

    # set logger
    store_root = os.path.join("logs", "products")
    if not os.path.exists(store_root):
        os.makedirs(store_root)
    
    if args.num_colors != 0:
        file_name = f'products_with_{args.num_colors:04d}_colors'
    else:
        file_name = "products"
    
    logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(store_root, f'{file_name}.log'), mode='w'), 
                                      logging.StreamHandler()])
    logger = logging.getLogger()
    logger.info(args)

    # device setup
    device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # data load
    data_original, split_idx = data_loader("products", num_colors=0, directed=False, logger=logger)
    if args.num_colors == 0:
        data = data_original
    else:
        data, split_idx = data_loader("products", args.num_colors, directed=False, logger=logger)
    data = data.to(device)
    data_test_x = data_original.x[split_idx['test']].to(device)



    model = GCN(data.num_features, args.hidden_channels,
                data.num_classes, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')
    metric_logger = Logger(args.runs)


    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, data.train_idx, optimizer)
            result = test(model, data_original, split_idx, evaluator)
            metric_logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                logger.info(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_acc:.2f}%, '
                            f'Valid: {100 * valid_acc:.2f}% '
                            f'Test: {100 * test_acc:.2f}% ')

        logger.info(metric_logger.print_statistics(run))
    logger.info(metric_logger.print_statistics())


if __name__ == "__main__":
    main()
