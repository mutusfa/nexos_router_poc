import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from router_poc.vectors import embed
from router_poc.settings import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.input)
    df = df.drop_duplicates(subset="prompt")
    embeddings = []
    for i in range(0, len(df), args.batch_size):
        embeddings.append(embed(df["prompt"].iloc[i : i + args.batch_size].tolist()))
    embeddings = np.concatenate(embeddings)
    df = df[["prompt"]]
    df["embedding"] = embeddings.tolist()
    df.to_parquet(args.input.with_suffix(".embeddings.parquet"))


if __name__ == "__main__":
    main()
