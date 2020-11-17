import torch
import os
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset
import argparse
import random
import torch_geometric.transforms as T
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device
def main():
    parser = argparse.ArgumentParser(description='Process-Data')
    parser.add_argument('--name', type=str, default='ogbl-ppa')
    parser.add_argument('--dir', type=str, default='/blob2/v-bonli/data/ppa_edges.txt')
    parser.add_argument('--use_valedges_as_input', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.name.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args.name, transform=T.ToSparseTensor())
        device = gpu_setup(True, args.gpu)
        data = dataset[0].to(device)
        split_edge = dataset.get_edge_split()
        train_pos_edge = split_edge['train']['edge']
        if args.name == 'ogbl-citation':
            f = open("citation/train_pos_edge.txt","w")
            ret = split_edge['train']['source_node']
            for x in ret:
                f.write(str(x.item())+"\n")
            ret = split_edge['train']['target_node']
            for x in ret:
                f.write(str(x.item())+"\n")
            f.close()
            train_neg_edge = torch.randint(0, data.num_nodes, split_edge['train']['source_node'].size(), dtype=torch.long)
            f = open("citation/train_neg_edge.txt","w")
            for x in train_neg_edge:
                f.write(str(x.item())+"\n")
            train_neg_edge = torch.randint(0, data.num_nodes, split_edge['train']['source_node'].size(), dtype=torch.long)
            for x in train_neg_edge:
                f.write(str(x.item())+"\n")
            f.close()
            f = open("citation/valid_edge.txt","w")
            ret = split_edge['valid']['source_node']
            for x in ret:
                f.write(str(x.item())+"\n")
            ret = split_edge['valid']['target_node']
            for x in ret:
                f.write(str(x.item())+"\n")
            f.close()
            ret = split_edge['valid']['target_node_neg']
            for i in range(1000):
                tmp = ret[i]
                f = open("citation/valid_neg"+str(i)+".txt","w")
                for x in tmp:
                    f.write(str(x.item())+"\n")
                f.close()
            f = open("citation/test_edge.txt","w")
            ret = split_edge['test']['source_node']
            for x in ret:
                f.write(str(x.item())+"\n")
            ret = split_edge['test']['target_node']
            for x in ret:
                f.write(str(x.item())+"\n")
            f.close()
            ret = split_edge['test']['target_node_neg']
            for i in range(1000):
                tmp = ret[i]
                f = open("citation/test_neg"+str(i)+".txt","w")
                for x in tmp:
                    f.write(str(x.item())+"\n")
                f.close()
            return
        edge_index = data.edge_index
        valid_pos_edge = split_edge['valid']['edge']
        valid_neg_edge = split_edge['valid']['edge_neg']
        test_pos_edge = split_edge['test']['edge']
        test_neg_edge = split_edge['test']['edge_neg']
        if args.use_valedges_as_input:
            val_edge_index = valid_pos_edge.t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            data.adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.adj_t = data.adj_t.to_symmetric()
            train_pos_edge = torch.cat([train_pos_edge, valid_pos_edge], dim = 1)
        if args.name == 'ogbl-ddi':
            adj_t = data.adj_t.to(device)
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([col, row], dim=0)
            train_neg_edge = negative_sampling(edge_index, num_nodes = data.num_nodes, num_neg_samples = train_pos_edge.size(0), method = 'dense').t()
        else:
            train_neg_edge = torch.randint(0, data.num_nodes, train_pos_edge.size(), dtype=torch.long)
        f = open(args.dir, "w")
        f.write(str(data.num_nodes)+"\n")
        f.write(str(train_pos_edge.size(0))+"\n")
        ret = train_pos_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
        f.write(str(train_neg_edge.size(0))+"\n")
        ww1 = 0
        ww2 = 1
        for x in train_neg_edge:
            v1, v2 = x
            w1 = v1.item()
            w2 = v2.item()
            while w1 == w2:
                if args.name == 'ogbl-ddi':
                    w1, w2 = ww1, ww2
                else:
                    w1 = random.randint(0, data.num_nodes - 1)
                    w2 = random.randint(0, data.num_nodes - 1)
            f.write(str(w1)+"\n")
            f.write(str(w2)+"\n")
            ww1 = w1
            ww2 = w2
        f.write(str(valid_pos_edge.size(0))+"\n")
        ret = valid_pos_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
        f.write(str(valid_neg_edge.size(0))+"\n")
        ret = valid_neg_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
        f.write(str(test_pos_edge.size(0))+"\n")
        ret = test_pos_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
        f.write(str(test_neg_edge.size(0))+"\n")
        ret = test_neg_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
        f.close()
if __name__ == "__main__":
    main()
