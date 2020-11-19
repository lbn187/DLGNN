# DLGNN

## Process Data

python process_data.py --name $DATASET --dir $EDGE_DIR

## Generate Feature

g++ -std=c++11 -o generate_feature generate_feature.cpp random_tree.cpp shortest_path.cpp katz_distance.cpp anchor_distance.cpp

### Random Tree

./generate_feature random_tree $EDGE_DIR $INFO_DIR $NUM_TREES $USE_VAL

### Anchor Distance

./generate_feature anchor_distance $EDGE_DIR $INFO_DIR $NUM_ANCHOR_NODES $MAX_DISTANCE $EXTRA_RANDOM_EDGES $USE_VAL

### Shortest Path

./generate_feature shortest_path $EDGE_DIR $INFO_DIR $MAX_DISTANCE $USE_VAL

### Katz Distance

./generate_feature katz_distance $EDGE_DIR $INFO_DIR $BETA $MAX_LENGTH

## Run Jobs

### BaseLine

python link_pred.py

### One Feature

python link_pred.py --extra_data_dir $INFO_DIR --extra_data_list $FEATURE_NAME --extra_data_weight $FEATURE_WEIGHT --extra_data_layer $FEATURE_LAYER

### Multiple Features

python link_pred.py --extra_data_dir $INFO_DIR --extra_data_list $FEATURE_NAME_1 $FEATURE_NAME_2 ... $FEATURE_NAME_N --extra_data_weight $FEATURE_WEIGHT_1 $FEATURE_WEIGHT_2 ... $FEATURE_WEIGHT_N --extra_data_layer $FEATURE_LAYER_1 $FEATURE_LAYER_2 ... $FEATURE_LAYER_N

## Our Settings

### DDI

$DATASET = 'ogbl-ddi'

$NUM_TREES = 10000

$NUM_ANCHOR_NODES = 50000

$MAX_DISTANCE = 10

$BETA = 0.03

$MAX_LENGTH = 10

#### BaseLine

python link_pred.py --dataset ogbl-ddi --model GraphSAGE --num_layers 2 --node_emb 500 --hidden_channels 500 --dropout 0.3 --batch_size 70000 --lr 0.003 --epochs 500 --runs 100 --eval_epoch 50

#### One Feature

python link_pred.py --extra_data_list random_tree --extra_data_weight 0.1 --extra_data_layer 1

python link_pred.py --extra_data_list anchor_distance --extra_data_weight 0.1 --extra_data_layer 1

python link_pred.py --extra_data_list shortest_path --extra_data_weight 0.1 --extra_data_layer 1

python link_pred.py --extra_data_list katz_distance --extra_data_weight 0.1 --extra_data_layer 2

#### Multiple Features

python link_pred.py --extra_data_list random_tree anchor_distance shortest_path katz_distance --extra_data_weight 0.05 0.1 0.03 0.03 --extra_data_layer 1 1 1 2

### COLLAB

$DATASET = 'ogbl-collab'

$NUM_TREES = 2500

$NUM_ANCHOR_NODES = 2500

$EXTRA_RANDOM_EDGES = 200000

$MAX_DISTANCE = 20

#### BaseLine

python link_pred.py --dataset ogbl-collab --model GCN --num_layers 3 --node_emb 200 --hidden_channels 500 --dropout 0.1 --batch_size 70000 --lr 0.003 --epochs 1000 --runs 10 --eval_epoch 50

#### One Feature

python link_pred.py --extra_data_list random_tree --extra_data_weight 1.0 --extra_data_layer 1

python link_pred.py --extra_data_list anchor_distance --extra_data_weight 1.0 --extra_data_layer 1

python link_pred.py --extra_data_list shortest_path --extra_data_weight 1.0 --extra_data_layer 1

#### All Features

python link_pred.py --extra_data_list random_tree anchor_distance shortest_path --extra_data_weight 1.0 0.05 1.0 --extra_data_layer 1 1 1
