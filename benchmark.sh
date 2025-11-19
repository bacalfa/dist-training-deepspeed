#!/bin/bash
# Example run command (needs deepspeed installed and GPUs)
# run with: docker run --gpus all -v $(pwd):/workspace -w /workspace <image> bash -c "./benchmark.sh"

NUM_GPUS=${1:-1}
DEEPSPEED_CONFIG=deepspeed_config.json

# Use deepspeed launcher
deepspeed --num_gpus $NUM_GPUS train.py --model_name gpt2 --batch_size 32 --epochs 10 --deepspeed_config $DEEPSPEED_CONFIG
