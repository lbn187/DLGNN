# DLGNN

## DDI

### Process Data

python process_data.py --name 'ogbl-ddi' --dir 'ddi_edges.txt'

### Generate Feature

g++ -std=c++11 -o generate_feature generate_feature.cpp random_tree.cpp shortest_path.cpp katz_distance.cpp anchor_distance.cpp

./generate_feature random_tree ddi_edges.txt /data/ddi_ 10000

./generate_feature anchor_distance ddi_edges.txt /data/ddi_ 10000

./generate_feature shortest_path ddi_edges.txt /data/ddi_ 10

./generate_feature katz_distance ddi_edges.txt /data/ddi_ 0.03 10

### Experiments

#### BaseLine

python link_pred.py --dataset ogbl-ddi --model GraphSAGE --num_layers [2] --node_emb 500 --hidden_channels 500 --dropout 0.3 --batch_size 70000 --lr 0.003 --epochs 500 --runs 50 --eval_epoch 50 --extra_data_dir /data/ddi_ --extra_data_list [] --extra_data_weight []

#### One Feature

python link_pred.py --dataset ogbl-ddi --model GraphSAGE --num_layers [2] --node_emb 500 --hidden_channels 500 --dropout 0.3 --batch_size 70000 --lr 0.003 --epochs 500 --runs 50 --eval_epoch 50 --extra_data_dir /data/ddi_ --extra_data_list ['random_tree'] --extra_data_weight [0.1]


#### All Features

python link_pred.py --dataset ogbl-ddi --model GraphSAGE --num_layers [2] --node_emb 500 --hidden_channels 500 --dropout 0.3 --batch_size 70000 --lr 0.003 --epochs 500 --runs 50 --eval_epoch 50 --extra_data_dir /data/ddi_ --extra_data_list ['random_tree', 'anchor_distance', 'shortest_path'] --extra_data_weight [0.05, 0.1, 0.05]

## COLLAB

### Process Data

python process_data.py --name 'ogbl-collab' --dir 'collab_edges.txt'

### Generate Feature

g++ -std=c++11 -o generate_feature generate_feature.cpp random_tree.cpp shortest_path.cpp katz_distance.cpp anchor_distance.cpp

./generate_feature random_tree collab_edges.txt /data/collab_ 5000

./generate_feature anchor_distance collab_edges.txt /data_collab_ 5000 200000

./generate_feature shortest_path collab_edges.txt /data/collab_ 20

### Experiments

#### BaseLine

python link_pred.py --dataset ogbl-collab --model GCN --num_layers [3] --node_emb 200 --hidden_channels 500 --dropout 0.1 --batch_size 70000 --lr 0.003 --epochs 1000 --runs 10 --eval_epoch 50 --extra_data_dir /data/collab_ --extra_data_list [] --extra_data_weight []

#### One Feature

python link_pred.py --dataset ogbl-collab --model GCN --num_layers [3] --node_emb 200 --hidden_channels 500 --dropout 0.1 --batch_size 70000 --lr 0.003 --epochs 1000 --runs 10 --eval_epoch 50 --extra_data_dir /data/collab_ --extra_data_list ['random_tree'] --extra_data_weight [0.1]

#### All Features

python link_pred.py --dataset ogbl-collab --model GCN --num_layers [3] --node_emb 200 --hidden_channels 500 --dropout 0.1 --batch_size 70000 --lr 0.003 --epochs 1000 --runs 10 --eval_epoch 50 --extra_data_dir /data/collab_ --extra_data_list ['random_tree', 'anchor_distance', 'shortest_path'] --extra_data_weight [0.1, 0.01, 0.1]
