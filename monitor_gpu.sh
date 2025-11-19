#!/usr/bin/env bash
# Script to monitor and log GPU(s) with `nvidia-smi`
# Usage: ./monitor_gpu.sh results/gpu_log.csv &

OUT=${1:-results/gpu_log.csv}
mkdir -p $(dirname "$OUT")
echo "timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]" > "$OUT"

# loop-ms is higher resolution (ms). If your nvidia-smi supports --loop-ms:
if nvidia-smi --help | grep -q -- '--loop-ms'; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv --loop-ms=200 >> "$OUT"
else
    # fallback to 1-second loop
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 1 >> "$OUT"
fi