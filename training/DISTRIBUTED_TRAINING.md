# Distributed Training Setup for LLaMA-Omni Stage 1

## Overview
The Stage 1 trainer now supports distributed training using PyTorch's DistributedDataParallel (DDP) for multi-GPU and multi-node training.

## Usage

### Single Node, Multi-GPU Training
```bash
# Run with 4 GPUs on a single node
./training/scripts/run_stage1.sh \
    "/path/to/config.json" \
    "/path/to/data.json" \
    "./outputs/stage1" \
    1 \
    4
```

### Multi-Node Training
```bash
# Node 0 (master):
./training/scripts/run_stage1.sh \
    "/path/to/config.json" \
    "/path/to/data.json" \
    "./outputs/stage1" \
    2 \
    4 \
    0 \
    "master_node_ip" \
    "12355"

# Node 1:
./training/scripts/run_stage1.sh \
    "/path/to/config.json" \
    "/path/to/data.json" \
    "./outputs/stage1" \
    2 \
    4 \
    1 \
    "master_node_ip" \
    "12355"
```

### Script Parameters
1. `CONFIG_PATH`: Path to training configuration JSON
2. `DATA_PATH`: Path to training data JSON
3. `OUTPUT_DIR`: Output directory for checkpoints
4. `NNODES`: Number of nodes (default: 1)
5. `NPROC_PER_NODE`: Number of processes per node (default: auto-detect GPUs)
6. `NODE_RANK`: Rank of current node (default: 0)
7. `MASTER_ADDR`: Master node address (default: "localhost")
8. `MASTER_PORT`: Master node port (default: "12355")

## Features Added

### Distributed Training Support
- ✅ Multi-GPU training with DistributedDataParallel (DDP)
- ✅ Multi-node training support
- ✅ Proper gradient synchronization
- ✅ DistributedSampler for data loading
- ✅ Main process logging and checkpointing
- ✅ Process synchronization with barriers

### Automatic GPU Detection
The script automatically detects available GPUs using `nvidia-smi -L | wc -l` if `NPROC_PER_NODE` is not specified.

### Process Management
- Only the main process (rank 0) handles:
  - Logging output
  - WandB logging
  - Checkpoint saving
  - Progress bars
- All processes participate in:
  - Model training
  - Gradient computation
  - Data loading (with distributed sampling)

## Example Commands

### Quick Single-GPU Training
```bash
./training/scripts/run_stage1.sh
```

### Multi-GPU Training with Custom Config
```bash
./training/scripts/run_stage1.sh \
    "./training/configs/stage1_config.json" \
    "./InstructS2S-200K/instruct_en_train_small.json" \
    "./outputs/stage1_distributed"
```

### 8-GPU Training on Single Node
```bash
./training/scripts/run_stage1.sh \
    "./training/configs/stage1_config.json" \
    "./InstructS2S-200K/instruct_en_train_small.json" \
    "./outputs/stage1_8gpu" \
    1 \
    8
```

## Requirements
- PyTorch with distributed support
- NCCL backend for multi-GPU communication
- Proper network configuration for multi-node setups