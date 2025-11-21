import csv
import glob
import os
import statistics
import pandas as pd
import argparse


def summarize_run(run_dir: str, warmup_steps: int = 20):
    step_file = os.path.join(run_dir, "step_log.csv")
    gpu_file = os.path.join(run_dir, "gpu_log.csv")

    if not os.path.exists(step_file):
        return None

    steps = []
    with open(step_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(
                {
                    "step": int(row["global_step"]),
                    "step_time": float(row["step_time_s"]),
                    "tokens_per_sec": float(row["tokens_per_sec"]),
                    "gpu_max_mem_mb": float(row["gpu_max_mem_mb"]),
                }
            )

    # Ignore warmup
    steps_post = [s for s in steps if s["step"] > warmup_steps]
    if not steps_post:
        return None
    avg_step = statistics.mean([s["step_time"] for s in steps_post])
    median_tokens = statistics.median([s["tokens_per_sec"] for s in steps_post])
    peak_mem = max([s["gpu_max_mem_mb"] for s in steps_post])

    # GPU monitor
    if os.path.exists(gpu_file):
        df = pd.read_csv(gpu_file, header=1)

        # df has rows with timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]
        # compute average utilization across samples, and average memory used
        util_col = [c for c in df if "utilization.gpu" in c][0]
        mem_col = [c for c in df if "memory.used" in c][0]

        util = df[util_col].str.replace("%", "").str.strip().astype(float)
        util = util[util > 0]
        mem = df[mem_col].str.replace("MiB", "").str.strip().astype(float)
        mem = mem[mem > 0]
        avg_util = util.mean()
        avg_mem = mem.mean() if mem is not None else None
    else:
        avg_util = None
        avg_mem = None

    return {
        "avg_step_time_s": avg_step,
        "median_tokens_per_sec": median_tokens,
        "peak_gpu_mem_mb": peak_mem,
        "avg_gpu_util_pct": avg_util,
        "avg_gpu_mem_reported_mib": avg_mem,
    }


def scan_results(top: str = "results", warmup: int = 20):
    records = []
    for expdir in glob.glob(os.path.join(top, "*")):
        for rundir in glob.glob(os.path.join(expdir, "run*")):
            summary = summarize_run(rundir, warmup)
            if summary:
                rec = {"exp": os.path.basename(expdir), "run": os.path.basename(rundir)}
                rec.update(summary)
                records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        print("No results found.")
        return

    # Group by exp and compute median/std across runs
    grouped = df.groupby("exp").agg(
        {
            "median_tokens_per_sec": ("median", "std"),
            "avg_step_time_s": ("median", "std"),
            "peak_gpu_mem_mb": ("median",),
            "avg_gpu_util_pct": ("median",),
        }
    )
    print(grouped)

    df.to_csv(f"{top}/summary_raw.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--warmup", type=int, default=20)

    args = parser.parse_args()
    scan_results(args.out_dir, args.warmup)


if __name__ == "__main__":
    main()
