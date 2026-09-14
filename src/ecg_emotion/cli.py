"""Command-line interface for local experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic import generate_demo_dataset
from .training import cross_validate_feature_baseline, train_feature_baseline


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
    tiny_cnn.add_argument("--learning-rate", type=float, default=5e-4)
    tiny_cnn.add_argument("--disable-augmentation", action="store_true")
    tiny_cnn.add_argument("--seed", type=int, default=42)

    tiny_lstm = subparsers.add_parser(
        "train-tiny-lstm", help="train the optional compact convolutional LSTM"
    )
    tiny_lstm.add_argument("--data", type=Path, required=True)
    tiny_lstm.add_argument("--output", type=Path, default=Path("artifacts/tiny-lstm"))
    tiny_lstm.add_argument("--sample-rate", type=float, default=256.0)
    tiny_lstm.add_argument("--batch-size", type=int, default=128)
    tiny_lstm.add_argument("--epochs", type=int, default=80)
    tiny_lstm.add_argument("--patience", type=int, default=10)
    tiny_lstm.add_argument("--learning-rate", type=float, default=5e-4)
    tiny_lstm.add_argument("--disable-augmentation", action="store_true")
    tiny_lstm.add_argument("--seed", type=int, default=42)

    cross_validation = subparsers.add_parser(
        "cross-validate-baseline",
        help="run stratified subject-level cross-validation for a feature baseline",
    )
    cross_validation.add_argument("--data", type=Path, required=True)
    cross_validation.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cross-validation"),
    )
    cross_validation.add_argument("--model", default="logistic-regression")
    cross_validation.add_argument("--sample-rate", type=float, default=256.0)
    cross_validation.add_argument("--folds", type=int, default=5)
    cross_validation.add_argument("--seed", type=int, default=42)

    cnn_cross_validation = subparsers.add_parser(
        "cross-validate-tiny-cnn",
        help="run subject-independent cross-validation for the optional Tiny CNN",
    )
    cnn_cross_validation.add_argument("--data", type=Path, required=True)
    cnn_cross_validation.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tiny-cnn-cross-validation"),
    )
    cnn_cross_validation.add_argument("--sample-rate", type=float, default=256.0)
    cnn_cross_validation.add_argument("--batch-size", type=int, default=128)
    cnn_cross_validation.add_argument("--epochs", type=int, default=30)
    cnn_cross_validation.add_argument("--patience", type=int, default=6)
    cnn_cross_validation.add_argument("--learning-rate", type=float, default=5e-4)
    cnn_cross_validation.add_argument("--folds", type=int, default=5)
    cnn_cross_validation.add_argument("--disable-augmentation", action="store_true")
    cnn_cross_validation.add_argument("--seed", type=int, default=42)

    lstm_cross_validation = subparsers.add_parser(
        "cross-validate-tiny-lstm",
        help="run subject-independent cross-validation for the optional Tiny LSTM",
    )
    lstm_cross_validation.add_argument("--data", type=Path, required=True)
    lstm_cross_validation.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tiny-lstm-cross-validation"),
    )
    lstm_cross_validation.add_argument("--sample-rate", type=float, default=256.0)
    lstm_cross_validation.add_argument("--batch-size", type=int, default=128)
    lstm_cross_validation.add_argument("--epochs", type=int, default=30)
    lstm_cross_validation.add_argument("--patience", type=int, default=6)
    lstm_cross_validation.add_argument("--learning-rate", type=float, default=5e-4)
    lstm_cross_validation.add_argument("--folds", type=int, default=5)
    lstm_cross_validation.add_argument("--disable-augmentation", action="store_true")
    lstm_cross_validation.add_argument("--seed", type=int, default=42)

    robustness = subparsers.add_parser(
        "audit-subject-robustness",
        help="run leave-one-subject-out and feature ablation analysis",
    )
    robustness.add_argument("--data", type=Path, required=True)
    robustness.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/subject-robustness"),
    )
    robustness.add_argument("--model", default="shrinkage-lda")
    robustness.add_argument("--sample-rate", type=float, default=256.0)
    robustness.add_argument("--bootstrap-samples", type=int, default=2000)
    robustness.add_argument("--seed", type=int, default=42)
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
            learning_rate=args.learning_rate,
            random_seed=args.seed,
            use_augmentation=not args.disable_augmentation,
        )
        test_metrics = result["metrics"]["test"]
        print(
            f"{result['model_name']} test accuracy={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"epochs={result['epochs_completed']}"
        )
        return

    if args.command == "train-tiny-lstm":
        from .torch_training import train_tiny_lstm

        result = train_tiny_lstm(
            data_path=args.data,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            early_stopping_patience=args.patience,
            learning_rate=args.learning_rate,
            random_seed=args.seed,
            use_augmentation=not args.disable_augmentation,
        )
        test_metrics = result["metrics"]["test"]
        print(
            f"{result['model_name']} test accuracy={test_metrics['accuracy']:.4f}, "
            f"macro_f1={test_metrics['macro_f1']:.4f}, "
            f"epochs={result['epochs_completed']}"
        )
        return

    if args.command == "cross-validate-baseline":
        result = cross_validate_feature_baseline(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.model,
            sample_rate=args.sample_rate,
            num_folds=args.folds,
            random_seed=args.seed,
        )
        macro_f1 = result["aggregate"]["macro_f1"]
        print(
            f"{result['model_name']} {result['num_folds']}-fold macro_f1="
            f"{macro_f1['mean']:.4f} +/- {macro_f1['std']:.4f}"
        )
        return

    if args.command == "cross-validate-tiny-cnn":
        from .torch_training import cross_validate_tiny_cnn

        result = cross_validate_tiny_cnn(
            data_path=args.data,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            early_stopping_patience=args.patience,
            learning_rate=args.learning_rate,
            num_folds=args.folds,
            random_seed=args.seed,
            use_augmentation=not args.disable_augmentation,
        )
        macro_f1 = result["aggregate"]["macro_f1"]
        print(
            f"{result['model_name']} {result['num_folds']}-fold macro_f1="
            f"{macro_f1['mean']:.4f} +/- {macro_f1['std']:.4f}"
        )
        return

    if args.command == "cross-validate-tiny-lstm":
        from .torch_training import cross_validate_tiny_lstm

        result = cross_validate_tiny_lstm(
            data_path=args.data,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            early_stopping_patience=args.patience,
            learning_rate=args.learning_rate,
            num_folds=args.folds,
            random_seed=args.seed,
            use_augmentation=not args.disable_augmentation,
        )
        macro_f1 = result["aggregate"]["macro_f1"]
        print(
            f"{result['model_name']} {result['num_folds']}-fold macro_f1="
            f"{macro_f1['mean']:.4f} +/- {macro_f1['std']:.4f}"
        )
        return

    if args.command == "audit-subject-robustness":
        from .robustness import run_loso_audit

        result = run_loso_audit(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.model,
            sample_rate=args.sample_rate,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.seed,
        )
        primary = result["feature_sets"]["all_features"]["aggregate"]["macro_f1"]
        print(
            f"{result['model_name']} LOSO macro_f1={primary['mean']:.4f} "
            f"95% CI=({primary['bootstrap_95ci'][0]:.4f}, "
            f"{primary['bootstrap_95ci'][1]:.4f})"
        )


if __name__ == "__main__":
    main()
