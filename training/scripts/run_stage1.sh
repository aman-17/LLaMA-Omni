#!/bin/bash
set -e

OUTPUT_DIR=${1:-"./outputs/olmo1b_whisper_large_stage1"}
NNODES=${4:-1}
NPROC_PER_NODE=${5:-$(nvidia-smi -L | wc -l)}
NODE_RANK=${6:-0}
MASTER_ADDR=${7:-"localhost"}
MASTER_PORT=${8:-"12355"}

mkdir -p $OUTPUT_DIR

if [ "$NNODES" -gt 1 ] || [ "$NPROC_PER_NODE" -gt 1 ]; then
    echo "Starting distributed training with $NNODES nodes and $NPROC_PER_NODE processes per node"
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        training/train.py /myfiles/amanr/LLaMA-Omni/training/configs/stage1_config.json
else
    echo "Starting single-GPU training"
    python training/train.py /myfiles/amanr/LLaMA-Omni/training/configs/stage1_config.json
fi

echo "Stage 1 training completed!"