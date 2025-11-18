# Project dist-training-deepspeed

## Objectives
- Scale Transformer training using DeepSpeed's Zero Redundancy Optimizer (ZeRO) and model/data-parallel techniques
- Measure throughput
- Develop reproducing benchmarking logs

## Running Locally with Docker

Build image:

```bash
docker build -t dist-training-deepspeed:latest .
```

Run benchmark (example with 1 GPU):

```bash
docker run --gpus '"device=0"' --rm -it -v $(pwd):/workspace -w /workspace dist-training-deepspeed:latest bash -c "./benchmark.sh 1"
```

For multi-GPU, pass more devices and adjust DeepSpeed `--num_gpus`.