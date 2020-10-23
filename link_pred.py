import torch
import argparse
import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.transforms as T
from transforms import Normalize
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from graphsage import GraphSAGE
from gcn import GCN
from gat import GAT

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device
    
class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, extra_num, device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout
        self.num_layers=num_layers
        self.extra_num = extra_num
        if extra_num >= 1:
            self.edge_layers = torch.nn.ModuleList()
            self.edge_layers.append(torch.nn.Linear(extra_num, 1))
            self.concat_layers = torch.nn.ModuleList()
            self.concat_layers.append(torch.nn.Linear(2, 1))
    def forward(self, x_i, x_j, edge_info):
        y = x_i * x_j
        for layer in self.layers[:-1]:
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.layers[-1](y)
        if self.extra_num == 0:
            return torch.sigmoid(y)
        for layer in self.edge_layers[:-1]:
            edge_info = layer(edge_info)
            edge_info = F.relu(edge_info)
            edge_info = F.dropout(edge_info, p=self.dropout, training=self.training)
        edge_info = self.edge_layers[-1](edge_info)
        y = torch.cat([y, edge_info], dim = 1)
        for layer in self.concat_layers[:-1]:
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.concat_layers[-1](y)
        return torch.sigmoid(y)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.extra_num >= 1:
            for layer in self.edge_layers:
                layer.reset_parameters()
            for layer in self.concat_layers:
                layer.reset_parameters()
                
def train(model, predictor, emb, data, split_edge, edge_info, optimizer, batch_size, device):
    if 1.0 * train_pos_edge.size(0) / data.num_nodes / data.num_nodes > 0.05:
        flag = True
    else:
        flag = False
    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim = 0)
    
    model.train()
    predictor.train()
    x = torch.cat([emb, data.x.to(device)], dim = 1).to(device)
    train_pos_edge = split_edge['train']['edge'].to(device)
    train_pos_edge_info = edge_info['train']['edge'].to(device)
    train_neg_edge_info = edge_info['train']['edge_neg'].to(device)
    total_loss = 0.0
    total_examples = 0
    perms = []
    cnt = 0
    for perm in DataLoader(range(train_pos_edge.size(0)), batch_size, shuffle=True):
        perms.append(perm)
    for perm in DataLoader(range(train_pos_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(x, data.adj_t)
        edge = train_pos_edge[perm].t()
        edge_info = train_pos_edge_info[perm].to(device)
        pos_out = predictor(h[edge[0]], h[edge[1]], edge_info)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        if flag == True:
            edge = negative_sampling(edge_index, num_nodes=x.size(0), num_neg_samples = perm.size(0), method='dense')
        else:
            edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=device)
        edge_info = train_neg_edge_info[perms[cnt]].to(device)
        cnt = cnt + 1
        neg_out = predictor(h[edge[0]], h[edge[1]], edge_info)
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(emb, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def test(eval_metric, model, predictor, emb, data, split_edge, edge_info, evaluator, batch_size, device)
    model.eval()
    predictor.eval()
    x = torch.cat([emb, data.x.to(device)], dim = 1)
    h = model(x, data.adj_t)
    valid_pos_edge = split_edge['valid']['edge']
    valid_neg_edge = split_edge['valid']['edge_neg']
    test_pos_edge = split_edge['test']['edge']
    test_neg_edge = split_edge['test']['edge_neg']
    valid_pos_edge_info = edge_info['valid']['edge']
    valid_neg_edge_info = edge_info['valid']['edge_neg']
    test_pos_edge_info = edge_info['test']['edge']
    test_neg_edge_info = edge_info['test']['edge_neg']
    if eval_metric == 'hits':
        pos_valid_preds = []
        for perm in DataLoader(range(valid_pos_edge.size(0)), batch_size):
            edge = valid_pos_edge[perm].t()
            edge_info = valid_pos_edge_info[perm].to(device)
            pos_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
            pos_valid_preds += [pos_out.squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim = 0)
        neg_valid_preds = []
        for perm in DataLoader(range(valid_neg_edge.size(0)), batch_size):
            edge = valid_neg_edge[perm].t()
            edge_info = valid_neg_edge_info[perm].to(device)
            neg_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
            neg_valid_preds += [neg_out.squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim = 0)
        pos_test_preds = []
        for perm in DataLoader(range(test_pos_edge.size(0)), batch_size):
            edge = test_pos_edge[perm].t()
            edge_info = test_pos_edge_info[perm].to(device)
            pos_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
            pos_test_preds += [pos_out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim = 0)
        neg_test_preds = []
        for perm in DataLoader(range(test_neg_edge.size(0)), batch_size):
            edge = test_neg_edge[perm].t()
            edge_info = test_neg_edge_info[perm].to(device)
            neg_out = predictor(h[edge[0]].to(device), h[edge[1]].to(device), edge_info)
            neg_test_preds += [neg_out.squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
            results[f'Hits@{K}'] = (valid_hits, test_hits)
        return results

def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = (valid_hits, test_hits)
    return results

def main():
    parser = argparse.ArgumentParser(description='Link-Pred')
    parser.add_argument('--dataset', type = str, default = 'ogbl-ddi')
    parser.add_argument('--model', type = str, default = 'GraphSAGE')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--num_layers', type = list, default = [2])
    parser.add_argument('--node_emb', type = int, default = 500)
    parser.add_argument('--hidden_channels', type = int, default = 500)
    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--batch_size', type = int, default = 70000)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--runs', type = int, default = 10)
    parser.add_argument('--use_save', type = bool, default = False)
    parser.add_argument('--eval_epoch', type = int, default = 100)
    parser.add_argument('--use_res', type = bool, default = False)
    parser.add_argument('--eval_metric', type = str, default = 'auc')
    parser.add_argument('--extra_data_dir', type = str, default = 'data/')
    parser.add_argument('--extra_data_list', type = list, default = ['random_tree'])
    args = parser.parse_args()
    device = gpu_setup(True, args.device)
    if args.dataset.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args.dataset, transform=T.ToSparseTensor())
        split_edge = datset.get_edge_split()
    if args.dataset == 'ogbl-citation':
        args.eval_metric = 'mrr'
    elif args.dataset.startswith('ogbl'):
        args.eval_metric = 'hits'
    if args.dataset.startwith('ogbl'):
        evaluator = Evaluator(name = args.dataset)
    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }
    elif args.eval_metric == 'auc':
        loggers = {
            'AUC': Logger(args.runs, args),
        }
    num = len(args.extra_data_list)
    edge_info = {'train': {'edge', 'edge_neg'}, 'valid': {'edge', 'edge_neg'}, 'test': {'edge', 'edge_neg'}}
    edge_info['train']['edge'] = torch.FloatTensor(split_edge['train']['edge'].size(0), num).to(device)
    edge_info['train']['edge_neg'] = torch.FloatTensor(split_edge['train']['edge_neg'].size(0), num).to(device)
    edge_info['valid']['edge'] = torch.FloatTensor(split_edge['valid']['edge'].size(0), num).to(device)
    edge_info['valid']['edge_neg'] = torch.FloatTensor(split_edge['valid']['edge_neg'].size(0), num).to(device)
    edge_info['test']['edge'] = torch.FloatTensor(split_edge['test']['edge'].size(0), num).to(device)
    edge_info['test']['edge'] = torch.FloatTensor(split_edge['test']['edge_neg'].size(0), num).to(device)
    '''
    for extra_data in extra_data_list:
        f = open(args.extra_data_dir + extra_data + "_train_pos.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        info = transforms.Normalize(torch.FloatTensor(np.array(ret).reshape(-1, 1)))
        train_pos_edge_info = torch.cat(train_pos_edge_info
        f.close()
    '''
    if args.model == 'GCN':
        model = GCN(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim = 1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
    elif args.model == 'GraphSAGE':
        model = GraphSAGE(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    elif args.model == 'GAT':
        model = GAT(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout, num, device).to(device)
    emb = torch.nn.Embedding(data.num_nodes, args.node_emb).to(device)
    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()) + list(emb.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, data, split_edge, edge_info, optimizer, args.batch_size, device)
            if epoch > args.eval_epoch:
                results = test(args.eval_metric, model, predictor, emb.weight, data, split_edge, edge_info, evaluator, args.batch_size, device)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Valid: {100 * valid_hits:.2f}%, '
                          f'Test: {100 * test_hits:.2f}%, ')
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

if __name__ == "__main__":
    main()
