import argparse
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import deepspeed
import os
import csv
from model import load_model


class DummyTextDataset(Dataset):
    def __init__(
        self, tokenizer: AutoTokenizer, n_samples: int = 1000, seq_length: int = 100
    ):
        """
        Creates a dummy/synthetic text dataset.

        :param tokenizer: Tokenizer instance
        :param n_samples: Number of samples
        :param seq_length: Sequence length
        """

        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Synthetic tokens
        ids = torch.randint(
            0, self.tokenizer.vocab_size - 1, (self.seq_length,), dtype=torch.long
        )
        return {"input_ids": ids, "labels": ids}


def collate_fn(batch):
    """
    Collate function for the dataset.
    :param batch: Batch of data
    :return: Dictionary of input_ids and labels
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def parse_args():
    """
    Parses command line arguments.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json")
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_args()
    print(args)

    # Set up experiment logging
    log_path = os.environ.get("STEP_LOG", "results/step_log.csv")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    csvfile = open(log_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "timestamp",
            "global_step",
            "step_time_s",
            "loss",
            "tokens_per_sec",
            "gpu_max_mem_mb",
        ]
    )

    # Compute tokens per step
    tokens_per_step = args.batch_size * args.seq_length

    # Initialize LLM and DeepSpeed engine
    model, tokenizer = load_model(args.model_name)
    ds_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # config=args.deepspeed_config,
    )

    # Initialize dataset and loader
    dataset = DummyTextDataset(tokenizer, args.n_samples, args.seq_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Run training loop
    ds_engine.train()
    total_steps = 0
    t_start = time.time()
    t0 = time.time()
    for epoch in range(args.epochs):
        steps = 0
        t0 = time.time()
        for batch in loader:
            input_ids = batch["input_ids"].to(ds_engine.device)
            labels = batch["labels"].to(ds_engine.device)
            loss = ds_engine(input_ids, labels=labels).loss
            ds_engine.backward(loss)
            ds_engine.step()
            t1 = time.time()
            step_time = t1 - t0
            steps += 1

            # Get max memory reserved on current device (MB)
            if torch.cuda.is_available():
                device = torch.device("cuda", ds_engine.local_rank)
                max_mem = torch.cuda.max_memory_reserved(device) / (1024**2)
            else:
                max_mem = 0.0

            if ds_engine.global_rank == 0:
                tokens_per_sec = tokens_per_step / step_time
                writer.writerow(
                    [
                        time.time(),
                        steps,
                        f"{step_time:.6f}",
                        f"{loss.item():.6f}",
                        f"{tokens_per_sec:.2f}",
                        f"{max_mem:.1f}",
                    ]
                )
                if steps % 10 == 0:
                    print(
                        f"Epoch {epoch + 1} - Step {steps} - Loss: {loss.item()} - Time: {step_time}"
                    )
                    t0 = t1

        total_steps += steps

    # Terminate experiment
    t_end = time.time()
    csvfile.close()
    if ds_engine.global_rank == 0:
        print(f"Trained {total_steps} steps in {t_end - t_start} seconds")


if __name__ == "__main__":
    main()
