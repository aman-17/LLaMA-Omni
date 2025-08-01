#!/bin/bash
set -e

CONFIG_PATH=${1:-"/data/input/amanr/LLaMA-Omni/training/configs/stage1_config.json"}
DATA_PATH=${2:-"/data/input/amanr/LLaMA-Omni/InstructS2S-200K/instruct_en_train.json"}
OUTPUT_DIR=${3:-"./outputs/stage1"}


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
python training/train.py $CONFIG_PATH

echo "Stage 1 training completed!"
echo "Model saved to: $OUTPUT_DIR"