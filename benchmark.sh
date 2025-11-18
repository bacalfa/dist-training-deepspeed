#!/bin/bash
# Example run command (needs deepspeed installed and GPUs)
# run with: docker run --gpus all -v $(pwd):/workspace -w /workspace <image> bash -c "./benchmark.sh"

NUM_GPUS=${1:-1}
DEEPSPEED_CONFIG=deepspeed_config.json

# Use deepspeed launcher
deepspeed --num_gpus $NUM_GPUS train.py --model gpt2 --batch-size 4 --steps 100 --deepspeed_config $DEEPSPEED_CONFIG
