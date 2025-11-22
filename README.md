# Project dist-training-deepspeed

## Objectives
- Scale Transformer training using DeepSpeed's Zero Redundancy Optimizer (ZeRO) and model/data-parallel techniques
- Measure throughput
- Develop reproducing benchmarking logs

## Results

A total of 8 experiments were run (see DeepSpeed configuration files in [experiments](experiments)) using an NVIDIA H100 GPU as follows:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 NVL                Off |   00000000:4E:00.0 Off |                    0 |
| N/A   46C    P0             68W /  400W |       4MiB /  95830MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

The LLM used was GPT2 provided via Hugging Face. The aggregated results by experiment are given below:

```
                          median_tokens_per_sec             avg_step_time_s           peak_gpu_mem_mb avg_gpu_util_pct
                                         median         std          median       std          median           median
exp                                                                                                                   
zs0_fp16_gpus1_bs32_sl128             23173.730   84.275070        0.185223  0.000648          7596.0        89.507036
zs0_fp32_gpus1_bs32_sl128              7445.155   59.125772        0.575916  0.005856          8956.0        90.685191
zs1_fp16_gpus1_bs32_sl128             21506.465   20.007474        0.199738  0.000164          7582.0        86.309175
zs1_fp32_gpus1_bs32_sl128              7194.530   15.987557        0.595777  0.001469          9736.0        88.619398
zs2_fp16_gpus1_bs32_sl128             21566.020   57.459516        0.199376  0.000622          7582.0        78.075536
zs2_fp32_gpus1_bs32_sl128              7186.690   13.301731        0.596442  0.001181          9736.0        83.974565
zs3_fp16_gpus1_bs32_sl128             11922.985  282.623740        0.361432  0.008480          9758.0        46.061215
zs3_fp32_gpus1_bs32_sl128              6968.430    2.496669        0.616783  0.000467         12570.0        71.084372
```

Key observations are as follows:
- `fp16` leads to higher throughput measured by median tokens per second compared to `fp32`
- Average step time is also lower for `fp16`
- Enabling ZeRO stage 3 led to the most significant decrease in GPU utilization

Additional analysis could include plotting latency as a function of different values for batch size and sequence length.

## Running Locally

Either create and activate a `conda` environment or virtual environment (`venv`), then install the dependencies:

```bash
pip install -r requirements.txt
```

Run all experiments

```bash
bash benchmark_runner.sh
```

Summarize results

```bash
python summarize_results.py
```

## Running with Docker

Build image:

```bash
docker build -t dist-training-deepspeed:latest .
```

Run benchmark (example with 1 GPU):

```bash
docker run \
  --gpus '"device=0"' \
  --rm \
  -it \
  -v $(pwd):/workspace \
  -w /workspace dist-training-deepspeed:latest \
  /bin/bash -c "bash benchmark_runner.sh && python summarize_results.py"
```

For multi-GPU, pass more devices and adjust DeepSpeed `--num_gpus`.
