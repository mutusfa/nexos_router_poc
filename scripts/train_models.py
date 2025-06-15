import argparse
import sys
from pathlib import Path

import pandas as pd

from router_poc import router
from router_poc import settings as S


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=S.DATA_DIR
        / "intermediate"
        / "stanford_mmlu_results.embeddings.parquet",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=S.DATA_DIR / "intermediate" / "stanford_mmlu_results.parquet",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="absolute",
        choices=["absolute", "pairwise", "relative"],
    )
    return parser.parse_args(args)


def main(args):
    labels = pd.read_parquet(args.labels_path)
    models = labels["model"].unique()
    del labels

    if args.model_type == "absolute":
        for model in models:
            print(f"Training evaluator for {model}...")
            router.AbsoluteStrength(model).train(args.embeddings_path, args.labels_path)
    elif args.model_type == "pairwise":
        for idx, model_a in enumerate(models):
            for model_b in models[idx + 1 :]:
                print(f"Training comparer for {model_a} vs {model_b}...")
                router.PairwiseStrength(model_a, model_b).train(args.embeddings_path, args.labels_path)
    elif args.model_type == "relative":
        for idx, model_a in enumerate(models):
            for model_b in models[idx + 1 :]:
                print(f"Training relative strength for {model_a} vs {model_b}...")
                router.RelativeStrength(model_a, model_b).train(args.embeddings_path, args.labels_path)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
