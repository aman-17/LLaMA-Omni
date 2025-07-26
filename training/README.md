# LLaMA-Omni Training

This directory contains the complete training pipeline for LLaMA-Omni, implementing the two-stage training approach described in the paper.

## Overview

LLaMA-Omni training consists of two stages:

1. **Stage 1**: Speech-to-Text instruction following - trains the speech adaptor and LLM while keeping the speech encoder frozen
2. **Stage 2**: Speech generation with CTC - trains the speech decoder while keeping all other components frozen

## Directory Structure

```
training/
├── README.md                 # This file
├── train.py                 # Main training script
├── data_utils.py            # Data loading utilities
├── stage1_trainer.py        # Stage 1 trainer implementation
├── stage2_trainer.py        # Stage 2 trainer implementation
├── configs/                 # Configuration files
│   ├── stage1_config.json  # Stage 1 training config
│   └── stage2_config.json  # Stage 2 training config
└── scripts/                 # Training scripts
    ├── run_stage1.sh       # Stage 1 training script
    └── run_stage2.sh       # Stage 2 training script
```

## Requirements

### Dependencies

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install datasets
pip install librosa
pip install fairseq
pip install wandb  # Optional, for logging
pip install tqdm
```

### Data Preparation

1. **InstructS2S-200K Dataset**: Prepare your speech instruction dataset following the format:
   ```json
   {
     "speech_instruction_path": "/path/to/instruction.wav",
     "text_instruction": "Hey, can you help me with...",
     "text_response": "Sure! I'd be happy to help...",
     "speech_response_path": "/path/to/response.wav"  // Required for Stage 2
   }
   ```

2. **Models Required**:
   - Whisper-large-v3: `openai/whisper-large-v3`
   - Llama-3.1-8B-Instruct: `meta-llama/Llama-3.1-8B-Instruct`
   - K-means model (for Stage 2): Download from [here](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin)

## Usage

### Quick Start

1. **Stage 1 Training**:
   ```bash
   cd training
   bash scripts/run_stage1.sh configs/stage1_config.json /path/to/InstructS2S-200K ./outputs/stage1
   ```

2. **Stage 2 Training**:
   ```bash
   bash scripts/run_stage2.sh configs/stage2_config.json /path/to/InstructS2S-200K ./outputs/stage1/final_model /path/to/kmeans_model.bin ./outputs/stage2
   ```

### Manual Training

You can also run training manually with custom configurations:

```bash
# Stage 1
python train.py configs/stage1_config.json

# Stage 2
python train.py configs/stage2_config.json
```

## Configuration

### Stage 1 Configuration

Key parameters in `configs/stage1_config.json`:

- `data_path`: Path to InstructS2S-200K dataset
- `model_name_or_path`: Base LLM model (Llama-3.1-8B-Instruct)
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per device (default: 32)
- `learning_rate`: Learning rate (default: 2e-5)
- `speech_projector_lr`: Separate learning rate for speech projector
- `freeze_backbone`: Whether to freeze the LLM backbone

### Stage 2 Configuration

Key parameters in `configs/stage2_config.json`:

- `stage1_model_path`: Path to trained Stage 1 model
- `kmeans_model_path`: Path to K-means model for unit extraction
- `ctc_loss_weight`: Weight for CTC loss (default: 1.0)
- `unit_vocab_size`: Size of discrete unit vocabulary (default: 1000)
- `ctc_upsample_factor`: Upsampling factor for CTC (default: 25)

## Training Details

### Stage 1: Speech-to-Text

- **Objective**: Cross-entropy loss for text generation
- **Frozen components**: Speech encoder (Whisper-large-v3)
- **Trainable components**: Speech adaptor, LLM, LM head
- **Duration**: ~3 epochs, ~32 hours on 4x L40 GPUs
- **Batch size**: 32
- **Learning rate**: 2e-5 with cosine schedule and 3% warmup

### Stage 2: Speech Generation

- **Objective**: CTC loss for discrete unit prediction
- **Frozen components**: Speech encoder, speech adaptor, LLM
- **Trainable components**: Speech decoder only
- **Duration**: ~3 epochs, ~33 hours on 4x L40 GPUs
- **Batch size**: 32
- **Learning rate**: 2e-4 with cosine schedule and 3% warmup

## Monitoring

The training scripts support Weights & Biases logging. Set `report_to: "wandb"` in your config and ensure you have wandb installed and logged in:

```bash
pip install wandb
wandb login
```

## Hardware Requirements

- **Minimum**: 1x GPU with 24GB VRAM (e.g., RTX 3090, A100)
- **Recommended**: 4x GPU with 40GB+ VRAM (e.g., A100, L40)
- **Memory**: 64GB+ RAM recommended
- **Storage**: 500GB+ for dataset and checkpoints

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Data Loading Errors**: Check dataset format and file paths
3. **Model Loading Errors**: Verify model paths and ensure Stage 1 is completed before Stage 2

### Performance Tips

- Use mixed precision training with `fp16: true` in config
- Enable gradient checkpointing for large models
- Use multiple GPUs with data parallel training
- Optimize data loading with more workers

## Results

After successful training, you should achieve:

- **Stage 1**: Good speech-to-text instruction following
- **Stage 2**: Low-latency speech generation (236ms as reported in paper)
- **Overall**: End-to-end speech interaction with <250ms latency

## Citation

If you use this training code, please cite the original paper:

```bibtex
@article{fang-etal-2024-llama-omni,
  title={LLaMA-Omni: Seamless Speech Interaction with Large Language Models},
  author={Fang, Qingkai and Guo, Shoutao and Zhou, Yan and Ma, Zhengrui and Zhang, Shaolei and Feng, Yang},
  journal={arXiv preprint arXiv:2409.06666},
  year={2024}
}
```