#!/bin/bash

# Stage 2 Training Script for LLaMA-Omni
# Usage: bash run_stage2.sh [config_path] [data_path] [stage1_model_path] [kmeans_model_path] [output_dir]

set -e

# Default configurations
CONFIG_PATH=${1:-"configs/stage2_config.json"}
DATA_PATH=${2:-"/path/to/InstructS2S-200K"}
STAGE1_MODEL_PATH=${3:-"./outputs/stage1/final_model"}
KMEANS_MODEL_PATH=${4:-"/path/to/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"}
OUTPUT_DIR=${5:-"./outputs/stage2"}

echo "Starting LLaMA-Omni Stage 2 Training"
echo "=================================="
echo "Config: $CONFIG_PATH"
echo "Data: $DATA_PATH"
echo "Stage 1 Model: $STAGE1_MODEL_PATH"
echo "K-means Model: $KMEANS_MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=================================="

# Validate Stage 1 model exists
if [ ! -d "$STAGE1_MODEL_PATH" ]; then
    echo "Error: Stage 1 model not found at $STAGE1_MODEL_PATH"
    echo "Please run Stage 1 training first or provide correct path"
    exit 1
fi

# Validate K-means model exists
if [ ! -f "$KMEANS_MODEL_PATH" ]; then
    echo "Error: K-means model not found at $KMEANS_MODEL_PATH"
    echo "Please download the K-means model from:"
    echo "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Update config with provided paths
python -c "
import json
import sys

config_path = '$CONFIG_PATH'
data_path = '$DATA_PATH'
stage1_model_path = '$STAGE1_MODEL_PATH'
kmeans_model_path = '$KMEANS_MODEL_PATH'
output_dir = '$OUTPUT_DIR'

with open(config_path, 'r') as f:
    config = json.load(f)

config['data_path'] = data_path
config['stage1_model_path'] = stage1_model_path
config['kmeans_model_path'] = kmeans_model_path
config['output_dir'] = output_dir

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f'Updated config: {config_path}')
"

# Run training
python train.py $CONFIG_PATH

echo "Stage 2 training completed!"
echo "Final model saved to: $OUTPUT_DIR"