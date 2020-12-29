import torch
import argparse
import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor
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

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.2f}')
            print(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, extra_data_layer, device):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        self.extra_num = len(extra_data_layer)
        self.edge_layers = torch.nn.ModuleList()
        for layer in extra_data_layer:
            new_layer = torch.nn.ModuleList()
            for _ in range(layer):
                new_layer.append(torch.nn.Linear(1, 1))
            self.edge_layers.append(new_layer)
        self.concat_layers = torch.nn.ModuleList()
        self.concat_layers.append(torch.nn.Linear(self.extra_num + out_dim, 1))
    
    def forward(self, x_i, x_j, edge_info):
        y = x_i * x_j
        for layer in self.layers[:-1]:
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.layers[-1](y)
        if self.extra_num == 0:
            return torch.sigmoid(y)
        i = 0
        new_edge_info = torch.unsqueeze(edge_info, 2)
        for layer_list in self.edge_layers:
            tmp_edge_info = new_edge_info[:, i]
            for layer in layer_list[:-1]:
                tmp_edge_info = layer(tmp_edge_info)
                tmp_edge_info = F.relu(tmp_edge_info)
            tmp_edge_info = layer_list[-1](tmp_edge_info)
            i = i + 1
            y = torch.cat([y, tmp_edge_info], dim = 1)
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
            for layer_list in self.edge_layers:
                for layer in layer_list:
                    layer.reset_parameters()
            for layer in self.concat_layers:
                layer.reset_parameters()
                
def train(model, predictor, emb, data, split_edge, train_pos_edge_info, train_neg_edge_info, optimizer, batch_size, ratio, device):
    train_pos_edge = split_edge['train']['edge'].to(device)
    if 1.0 * train_pos_edge.size(0) / data.num_nodes / data.num_nodes > ratio:
        flag = True
    else:
        flag = False
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim = 0)
    model.train()
    predictor.train()
    if data.x == None:
        x = emb
    else:
        x = torch.cat([emb, data.x.to(device)], dim = 1).to(device)
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
def test(eval_metric, model, predictor, emb, data, split_edge, valid_pos_edge_info, valid_neg_edge_info, test_pos_edge_info, test_neg_edge_info, evaluator, batch_size, device):
    model.eval()
    predictor.eval()
    if data.x == None:
        x = emb
    else:
        x = torch.cat([emb, data.x.to(device)], dim = 1)
    valid_pos_edge = split_edge['valid']['edge']
    valid_neg_edge = split_edge['valid']['edge_neg']
    test_pos_edge = split_edge['test']['edge']
    test_neg_edge = split_edge['test']['edge_neg']
    if eval_metric == 'hits':
        h = model(x, data.adj_t)
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
        h = model(x, data.full_adj_t)
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
            
def main():
    parser = argparse.ArgumentParser(description='Link-Pred')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, nargs='+', default=[3])
    parser.add_argument('--node_emb', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=70000)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--use_res', type=bool, default=False)
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--negative_sample_ratio', type=float, default=0.05)
    parser.add_argument('--use_valedges_as_input', type=bool, default=False)
    parser.add_argument('--eval_metric', type=str, default='auc')
    parser.add_argument('--extra_data_dir', type=str, default='../../data/collab_')
    parser.add_argument('--extra_data_list', type=str, nargs='+', default=['adamic_adar'])
    parser.add_argument('--extra_data_weight', type=float, nargs='+', default=[1.0])
    parser.add_argument('--extra_data_layer', type=int, nargs='+', default=[2])
    args = parser.parse_args()
    device = gpu_setup(True, args.device)
    if args.dataset.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args.dataset)
        data = dataset[0]
        edge_index = data.edge_index
        if data.x != None:
            data.x = data.x.to(torch.float)
        if args.dataset == 'ogbl-collab':
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
        split_edge = dataset.get_edge_split()
        if args.use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t
        data = data.to(device)
    if args.dataset == 'ogbl-citation':
        args.eval_metric = 'mrr'
    elif args.dataset.startswith('ogbl'):
        args.eval_metric = 'hits'
    if args.dataset.startswith('ogbl'):
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
    train_pos_list = []
    train_neg_list = []
    valid_pos_list = []
    valid_neg_list = []
    test_pos_list = []
    test_neg_list = []
    cnt = 0
    for extra_data in args.extra_data_list:
        weight = args.extra_data_weight[cnt]
        cnt = cnt + 1
        f = open(args.extra_data_dir + extra_data + "_train_pos.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        train_pos_info = torch.FloatTensor(np.array(ret).reshape(-1, 1))
        f.close()
        f = open(args.extra_data_dir + extra_data + "_train_neg.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        train_neg_info = torch.FloatTensor(np.array(ret).reshape(-1, 1))
        f.close()
        f = open(args.extra_data_dir + extra_data + "_valid_pos.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        valid_pos_info = torch.FloatTensor(np.array(ret).reshape(-1, 1))
        f.close()
        f = open(args.extra_data_dir + extra_data + "_valid_neg.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        valid_neg_info = torch.FloatTensor(np.array(ret).reshape(-1, 1))
        f.close()
        f = open(args.extra_data_dir + extra_data + "_test_pos.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        test_pos_info = torch.FloatTensor(np.array(ret).reshape(-1, 1))
        f.close()
        f = open(args.extra_data_dir + extra_data + "_test_neg.txt", "r")
        lines = f.readlines()
        ret = [float(x) for x in lines]
        test_neg_info = torch.FloatTensor(np.array(ret).reshape(-1, 1))
        f.close()
        max_info = torch.max(torch.max(train_pos_info), torch.max(train_neg_info))
        train_pos_info /= max_info
        train_neg_info /= max_info
        valid_pos_info /= max_info
        valid_neg_info /= max_info
        test_pos_info /= max_info
        test_neg_info /= max_info
        train_pos_list.append(train_pos_info * weight)
        train_neg_list.append(train_neg_info * weight)
        valid_pos_list.append(valid_pos_info * weight)
        valid_neg_list.append(valid_neg_info * weight)
        test_pos_list.append(test_pos_info * weight)
        test_neg_list.append(test_neg_info * weight)
    if len(args.extra_data_list) > 0:
        train_pos_edge_info = torch.cat(train_pos_list, dim=1)
        train_neg_edge_info = torch.cat(train_neg_list, dim=1)
        valid_pos_edge_info = torch.cat(valid_pos_list, dim=1)
        valid_neg_edge_info = torch.cat(valid_neg_list, dim=1)
        test_pos_edge_info = torch.cat(test_pos_list, dim=1)
        test_neg_edge_info = torch.cat(test_neg_list, dim=1)
    else:
        train_pos_edge_info = torch.FloatTensor(split_edge['train']['edge'].size(0), 1)
        train_neg_edge_info = torch.FloatTensor(split_edge['train']['edge'].size(0), 1)
        valid_pos_edge_info = torch.FloatTensor(split_edge['valid']['edge'].size(0), 1)
        valid_neg_edge_info = torch.FloatTensor(split_edge['valid']['edge_neg'].size(0), 1)
        test_pos_edge_info = torch.FloatTensor(split_edge['test']['edge'].size(0), 1)
        test_neg_edge_info = torch.FloatTensor(split_edge['test']['edge_neg'].size(0), 1)
    print(train_pos_edge_info.mean())
    print(train_neg_edge_info.mean())
    print(valid_pos_edge_info.mean())
    print(valid_neg_edge_info.mean())
    print(test_pos_edge_info.mean())
    print(test_neg_edge_info.mean())
    if args.model == 'GCN':
        model = GCN(data.num_features + args.node_emb, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    elif args.model == 'GraphSAGE':
        model = GraphSAGE(data.num_features + args.node_emb, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    elif args.model == 'GAT':
        model = GAT(data.num_features + args.node_emb, args.hidden_channels, args.hidden_channels, args.num_layers, args.use_res, args.dropout, device).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, args.out_dim, 3, args.dropout, args.extra_data_layer, device).to(device)
    emb = torch.nn.Embedding(data.num_nodes, args.node_emb).to(device)
    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()) + list(emb.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, data, split_edge, train_pos_edge_info, train_neg_edge_info, optimizer, args.batch_size, args.negative_sample_ratio, device)
            if epoch > args.eval_epoch:
                results = test(args.eval_metric, model, predictor, emb.weight, data, split_edge, valid_pos_edge_info, valid_neg_edge_info, test_pos_edge_info, test_neg_edge_info, evaluator, args.batch_size, device)
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
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

if __name__ == "__main__":
    main()
