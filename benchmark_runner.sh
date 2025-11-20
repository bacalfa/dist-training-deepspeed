#!/usr/bin/env bash
set -e

# Define general configs
NUM_GPUS=${1:-1}
MODEL_NAME=${2:-gpt2}
N_SAMPLES=${3:-2000}
BATCH_SIZE=${4:-32}
EPOCHS=${5:-50}
SEQ_LEN=${6:-128}

# Define arrays of configs
declare -a ZERO_STAGES=("0" "1" "2" "3")
declare -a PRECISIONS=("fp32" "fp16")

mkdir -p results

for zs in "${ZERO_STAGES[@]}"; do
    for prec in "${PRECISIONS[@]}"; do
        EXP="zs${zs}_${prec}_gpus${NUM_GPUS}_bs${BATCH_SIZE}_sl${SEQ_LEN}"
        OUTDIR="results/${EXP}"
        mkdir -p "${OUTDIR}"

        # Choose config file
        DS_CONFIG="experiments/ds_stage${zs}_${prec}.json"

        # Run 3 repeats
        for run in 1 2 3; do
            RUN_DIR="${OUTDIR}/run${run}"
            mkdir -p "${RUN_DIR}"

            # Start nvidia-smi monitor
            ./monitor_gpu.sh "${RUN_DIR}/gpu_log.csv" &
            MON_PID=$!
            echo "Started GPU monitor PID ${MON_PID} -> ${RUN_DIR}/gpu_log.csv"

            # Set STEP_LOG env so train.py writes there
            export STEP_LOG="${RUN_DIR}/step_log.csv"

            echo "Running experiment ${EXP} run${run} config ${DS_CONFIG}"
            
            # Run DeepSpeed
            deepspeed --num_gpus $NUM_GPUS train.py \
            --model_name $MODEL_NAME \
            --n_samples $N_SAMPLES \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --seq_length $SEQ_LEN \
            --deepspeed_config $DS_CONFIG \
            || echo "deepspeed failed for ${EXP} run${run}"

            # Terminate monitor
            kill $MON_PID || true
            sleep 2
            echo "Saved logs to ${RUN_DIR}"
        done
    done
done
