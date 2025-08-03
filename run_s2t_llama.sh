#!/bin/bash

# Speech-to-Text inference for LLaMA models (no S2S)
ROOT=${1:-./omni_speech/infer/examples}

# Check if we're already in a virtual environment (by checking for specific packages)
if ! python -c "import torch" 2>/dev/null; then
    if [[ -f "./venv/bin/activate" ]]; then
        echo "Activating virtual environment..."
        source ./venv/bin/activate
    else
        echo "Warning: No virtual environment found. Make sure you have the required packages installed."
    fi
fi

# Add current directory to Python path so omni_speech module can be found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set environment variable to handle meta tensors
export PYTORCH_DISABLE_META_TENSORS=1

python ./omni_speech/infer/infer.py \
    --model-path ./outputs/llama_stage1_freezed_backbone/checkpoint-epoch-3 \
    --model-base meta-llama/Llama-3.1-8B-Instruct \
    --question-file $ROOT/question.json \
    --answer-file $ROOT/answer.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 80

echo "S2T inference completed. Results saved to $ROOT/answer.json"