def main():
    parser = argparse.ArgumentParser(description='Process-Data')
    parser.add_argument('--name', type=string, default='ogbl-ddi')
    parser.add_argument('--dir', type=string, default='all.txt')
    if args.name.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args.name, transform=T.ToSparseTensor())
        data = dataset[0]
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
        if args.name == 'ogbl-ddi':
            adj_t = data.adj_t.to(device)
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([col, row], dim=0)
            train_neg_edge = negative_sampling(edge_index, num_nodes = data.num_nodes, num_neg_samples = train_pos_edge.size(0), method = 'dense').t()
        else:
            train_neg_edge = torch.randint(0, data.num_nodes, train_pos_edge.size(), dtype=torch.long)
        valid_pos_edge = split_edge['valid']['edge']
        valid_neg_edge = split_edge['valid']['edge_neg']
        test_pos_edge = split_edge['test']['edge']
        test_neg_edge = split_edge['test']['edge_neg']
        f = open(args.dir, "w")
        f.write(str(data.num_nodes)+"\n")
        f.write(str(train_pos_edge.size(0))+"\n")
        ret = train_pos_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
        f.write(str(train_neg_edge.size(0))+"\n")
        ret = train_neg_edge.reshape(-1)
        for x in ret:
            f.write(str(x.item())+"\n")
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