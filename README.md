#  Distance-Enhanced Graph Neural Network for Link Prediction

We propose a simple yet effective method for link predciton, where the GNN is incorported with a distance, and we can observer performance improvement.

## Requirements

- Python>=3.6
- PyTorch>=1.2
- torch-geometric>=1.6.0
- ogb>=1.2.3

## Step-1: Data Porcessing

    python process_data.py --name $DATASET --dir $EDGE_DIR
    
In this step, the edges of the datasets are stored in `$EDGE_DIR` to prepare for the next step of extracting information. 
`$DATASET` is the data we process, it can be "ogbl-ddi", "ogbl-collab" or "ogbl-ppa". 
`$EDGE_DIR` is the place we store the edges.

## Step-2: Generate Feature

Compile:

    g++ -std=c++11 -o generate_feature generate_feature.cpp random_tree.cpp shortest_path.cpp katz_distance.cpp anchor_distance.cpp common_neighbors.cpp jaccard_coefficient.cpp adamic_adar.cpp resource_allocation.cpp
    
Generate Feature:

    ./generate_feature $Distance_NAME $EDGE_DIR $INFO_DIR <$NUM_TREES> <$NUM_ANCHOR_NODES> <$MAX_DISTANCE> <$USE_MIN_ANCHOR> <$EXTRA_RANDOM_EDGES> <$BETA> <$MAX_LENGTH> <$USE_VAL>

In this work, we implement variants of distances

- `$Distance_NAME` is the name of distance. 
- `$EDGE_DIR` is the place we store the edges. 
- `$INFO_DIR` is the place we store edges' infomation. 
- `$NUM_ANCHOR_NODES` is the anchor nodes we selected. Used for anchor-based distance (recommended)
- `$USE_MIN_ANCHOR` is whether use the minimum distance among anchor points.
- `$EXTRA_RANDOM_EDGES` is the number of random edges we added to the graph each time we select a anchor node.
- `$NUM_TREES` is the random tree we generated. Used for tree-based distance.
- `$MAX_DISTANCE` is the default maximum distance. 
- `$BETA` is the parameter of Katz Index. 
- `$MAX_LENGTH` is the maximum number of steps. 
- `$USE_VAL` is whether we use the validation data as the input of test data.

We can generate distances using the following commands:
- `Random Tree`: `./generate_feature random_tree $EDGE_DIR $INFO_DIR $NUM_TREES $USE_VAL`
- `Anchor Distance`: `./generate_feature anchor_distance $EDGE_DIR $INFO_DIR $NUM_ANCHOR_NODES $MAX_DISTANCE $USE_MIN_ANCHOR $EXTRA_RANDOM_EDGES $USE_VAL`
- `Shortest Path`: `./generate_feature shortest_path $EDGE_DIR $INFO_DIR $MAX_DISTANCE $USE_VAL`
- `Katz Distance`: `./generate_feature katz_distance $EDGE_DIR $INFO_DIR $BETA $MAX_LENGTH`
- `Common Neighbors & Jaccard Coefficient & Adamic Adar & Resource Allocation`: `./generate_feature common_neighbors/jaccard_coefficient/adamic_adar/resource_allocation $USE_VAL`

## Step-3: Run Jobs

- Baseline: `python link_pred.py`
- Using one kind of distance: `python link_pred.py --extra_data_dir $INFO_DIR --extra_data_list $FEATURE_NAME --extra_data_weight $FEATURE_WEIGHT --extra_data_layer $FEATURE_LAYER`
- Using multiple kinds of distances: `python link_pred.py --extra_data_dir $INFO_DIR --extra_data_list $FEATURE_NAME_1 $FEATURE_NAME_2 ... $FEATURE_NAME_N --extra_data_weight $FEATURE_WEIGHT_1 $FEATURE_WEIGHT_2 ... $FEATURE_WEIGHT_N --extra_data_layer $FEATURE_LAYER_1 $FEATURE_LAYER_2 ... $FEATURE_LAYER_N`

    
## Our Setting for DDI

### Data processing
```
DATASET='ogbl-ddi'
python process_data.py --name $DATASET --dir ./DDI_edge

EDGE_DIR=DDI_edge
INFO_DIR=DDI_info
NUM_ANCHOR_NODES=1000
MAX_DISTANCE=10
USE_MIN_ANCHOR=
EXTRA_RANDOM_EDGES=
USE_VAL=

./generate_feature anchor_distance $EDGE_DIR $INFO_DIR $NUM_ANCHOR_NODES $MAX_DISTANCE $USE_MIN_ANCHOR $EXTRA_RANDOM_EDGES $USE_VAL
```

### Running our algorithm
```
L=2
hid=512

python link_pred.py --extra_data_dir './DDI_info'  \
--extra_data_weight 0.1 \
--extra_data_layer 1 \
--dataset ogbl-ddi \
--extra_data_list anchor_distance \
--model GraphSAGE \
--num_layers $L --node_emb $hid --hidden_channels $hid --dropout 0.3 --batch_size 70000 --lr 0.003 --epochs 500 --runs 100 --eval_epoch 50
```

### Results
We repeat the experiments for 30 times. The validation Hits@20 is `82.0566 $$\pm$$ 2.9849`. The test Hits@20 is `82.3877 $$\pm$$ 4.3709`.
