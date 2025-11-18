import argparse
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import deepspeed
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
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--deepspeed-config", type=str, default="deepspeed_config.json")
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_args()
    print(args)

    model, tokenizer = load_model(args.model_name)
    ds_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # config=args.deepspeed_config,
    )

    dataset = DummyTextDataset(tokenizer, args.n_samples, args.seq_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Run training loop
    ds_engine.train()
    steps = 0
    total_steps = int(args.n_samples / args.batch_size)
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            steps += 1
            input_ids = batch["input_ids"].to(ds_engine.device)
            labels = batch["labels"].to(ds_engine.device)
            loss = ds_engine(input_ids, labels=labels).loss
            ds_engine.backward(loss)
            ds_engine.step()
            if steps % 10 == 0 and ds_engine.global_rank == 0:
                t1 = time.time()
                print(f"Step {steps} - Loss: {loss.item()} - Time: {t1 - t0}")
                t0 = t1
            if steps >= total_steps:
                break

    t1 = time.time()
    if ds_engine.global_rank == 0:
        print(f"Trained {steps} steps in {t1 - t0} seconds")


if __name__ == "__main__":
    main()
