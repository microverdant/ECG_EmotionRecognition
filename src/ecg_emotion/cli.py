"""Command-line interface for local experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic import generate_demo_dataset
from .training import train_feature_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecg-emotion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo-data", help="create a synthetic smoke-test dataset")
    demo.add_argument("--output", type=Path, default=Path("data/demo_ecg.npz"))
    demo.add_argument("--seed", type=int, default=42)

    baseline = subparsers.add_parser("train-baseline", help="train a feature-based baseline")
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    baseline.add_argument("--model", default="logistic-regression")
    baseline.add_argument("--sample-rate", type=float, default=256.0)
    baseline.add_argument("--seed", type=int, default=42)

    tiny_cnn = subparsers.add_parser("train-tiny-cnn", help="train the optional compact 1D-CNN")
    tiny_cnn.add_argument("--data", type=Path, required=True)
    tiny_cnn.add_argument("--output", type=Path, default=Path("artifacts/tiny-cnn"))
    tiny_cnn.add_argument("--sample-rate", type=float, default=256.0)
    tiny_cnn.add_argument("--batch-size", type=int, default=128)
    tiny_cnn.add_argument("--epochs", type=int, default=80)
    tiny_cnn.add_argument("--patience", type=int, default=10)
    tiny_cnn.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "make-demo-data":
        dataset = generate_demo_dataset(args.output, random_seed=args.seed)
        print(
            f"Generated {dataset.num_samples} windows from {dataset.num_subjects} subjects at "
            f"{args.output}"
        )
        return

    if args.command == "train-baseline":
        result = train_feature_baseline(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.model,
            sample_rate=args.sample_rate,
            random_seed=args.seed,
        )
        test_metrics = result["metrics"]["test"]
        print(
            f"{result['model_name']} test accuracy={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        return

    if args.command == "train-tiny-cnn":
        from .torch_training import train_tiny_cnn

        result = train_tiny_cnn(
            data_path=args.data,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            early_stopping_patience=args.patience,
            random_seed=args.seed,
        )
        test_metrics = result["metrics"]["test"]
        print(
            f"{result['model_name']} test accuracy={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"epochs={result['epochs_completed']}"
        )


if __name__ == "__main__":
    main()
