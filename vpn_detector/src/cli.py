import argparse
import datetime
import sys

from . import preprocess, train
from .evaluate import evaluate


def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VPN Detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("preprocess", help="Preprocess raw VNAT data into flow parquet")
    prep_parser.add_argument("--config", required=True, help="Path to config.yaml")

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--config", required=True, help="Path to config.yaml")

    eval_parser = subparsers.add_parser("eval", help="Evaluate models")
    eval_parser.add_argument("--config", required=True, help="Path to config.yaml")

    args = parser.parse_args(argv)

    if args.command == "preprocess":
        log("Starting preprocessing")
        preprocess.main(args.config)
        log("Finished preprocessing")
    elif args.command == "train":
        log("Starting training")
        train.train_models(args.config)
        log("Finished training")
    elif args.command == "eval":
        log("Starting evaluation")
        evaluate(args.config)
        log("Finished evaluation")


if __name__ == "__main__":
    main(sys.argv[1:])
