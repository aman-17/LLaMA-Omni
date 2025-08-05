#!/bin/bash
set -e

CONFIG_PATH=${1:-"/data/input/amanr/LLaMA-Omni/training/configs/stage1_config.json"}
DATA_PATH=${2:-"/data/input/amanr/LLaMA-Omni/InstructS2S-200K/instruct_en_train.json"}
OUTPUT_DIR=${3:-"./outputs/olmo7b_tiny_stage1"}
NNODES=${4:-1}
NPROC_PER_NODE=${5:-$(nvidia-smi -L | wc -l)}
NODE_RANK=${6:-0}
MASTER_ADDR=${7:-"localhost"}
MASTER_PORT=${8:-"12355"}

mkdir -p $OUTPUT_DIR

python -c "
import json
import sys

config_path = '$CONFIG_PATH'
data_path = '$DATA_PATH'
output_dir = '$OUTPUT_DIR'

with open(config_path, 'r') as f:
    config = json.load(f)

config['data_path'] = data_path
config['output_dir'] = output_dir

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f'Updated config: {config_path}')
"

# Check if distributed training is needed
if [ "$NNODES" -gt 1 ] || [ "$NPROC_PER_NODE" -gt 1 ]; then
    echo "Starting distributed training with $NNODES nodes and $NPROC_PER_NODE processes per node"
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        training/train.py $CONFIG_PATH
else
    echo "Starting single-GPU training"
    python training/train.py $CONFIG_PATH
fi

echo "Stage 1 training completed!"
echo "Model saved to: $OUTPUT_DIR"