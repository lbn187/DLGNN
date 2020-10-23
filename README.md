# DLGNN

python process_data.py --name 'ogbl-ddi' --dir 'all.txt'
g++ -std=c++11 -o generate_feature generate_feature.cpp random_tree.cpp shortest_path.cpp katz_distance.cpp anchor_distance.cpp
./generate_feature random_tree 'all.txt' /data 10000
./generate_feature anchor_distance 'all.txt' /data 10000
python link_pred.py --dataset 'ogbl-ddi' --model 'GraphSAGE' --num_layers [2] --node_emb 500 --hidden_channels 500 --dropout 0.3 --lr 0.001 --epochs 1000 --runs 10 --eval_epoch 100 --extra_data_dir 'data/' --extra_data_list ['random_tree', 'anchor_distance']
