"""Run the reproducible subject-independent feature-model experiment suite."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ecg_emotion.evaluation import save_json
from ecg_emotion.robustness import run_loso_audit, run_loso_selective_audit
from ecg_emotion.training import cross_validate_feature_baseline, train_feature_baseline


def run_experiment_suite(
    data_path: str | Path,
    output_dir: str | Path,
    models: list[str],
    sample_rate: float = 140.0,
    folds: int = 5,
    bootstrap_samples: int = 2000,
    min_coverage: float = 0.3,
    inner_folds: int = 3,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run all formal feature-model evaluations and save a compact summary."""

    started = time.perf_counter()
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    model_summaries: dict[str, Any] = {}
    for model_name in models:
        model_dir = output_dir / model_name.replace("/", "-")
        cross_validation = cross_validate_feature_baseline(
            data_path,
            model_dir / "cross-validation",
            model_name=model_name,
            sample_rate=sample_rate,
            num_folds=folds,
            random_seed=random_seed,
        )
        robustness = run_loso_audit(
            data_path,
            model_dir / "robustness",
            model_name=model_name,
            sample_rate=sample_rate,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        selective = run_loso_selective_audit(
            data_path,
            model_dir / "selective-audit",
            model_name=model_name,
            sample_rate=sample_rate,
            min_coverage=min_coverage,
            inner_folds=inner_folds,
            random_seed=random_seed,
        )
        bundle = train_feature_baseline(
            data_path,
            model_dir / "bundle",
            model_name=model_name,
            sample_rate=sample_rate,
            random_seed=random_seed,
        )
        model_summaries[model_name] = {
            "cross_validation": {
                "artifact": str(model_dir / "cross-validation" / "cross_validation.json"),
                "aggregate": cross_validation["aggregate"],
            },
            "loso": {
                "artifact": str(model_dir / "robustness" / "loso_robustness.json"),
                "all_features": robustness["feature_sets"]["all_features"]["aggregate"],
            },
            "selective": {
                "artifact": str(model_dir / "selective-audit" / "selective_audit.json"),
                "aggregate": selective["aggregate"],
            },
            "bundle": {
                "artifact": str(model_dir / "bundle" / "model.joblib"),
                "metrics": str(model_dir / "bundle" / "metrics.json"),
                "confidence_threshold": bundle["confidence_threshold"],
            },
        }

    result = {
        "suite_name": "wesad-subject-independent-feature-model-suite",
        "data_path": str(data_path),
        "models": models,
        "sample_rate": sample_rate,
        "folds": folds,
        "bootstrap_samples": bootstrap_samples,
        "min_coverage": min_coverage,
        "inner_folds": inner_folds,
        "random_seed": random_seed,
        "model_summaries": model_summaries,
        "evaluation_seconds": round(time.perf_counter() - started, 4),
    }
    save_json(result, output_dir / "suite_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/wesad_core_140hz_30s_robust.npz"),
        help="prepared NumPy archive; ignored local data is never committed",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="optional raw WESAD root; prepare it into --data before evaluation",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/wesad-suite"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["shrinkage-lda", "balanced-shrinkage-lda"],
        help="feature models to evaluate",
    )
    parser.add_argument("--sample-rate", type=float, default=140.0)
    parser.add_argument("--source-rate", type=float, default=700.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=15.0)
    parser.add_argument("--min-label-purity", type=float, default=0.9)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input_root is not None:
        from ecg_emotion.wesad import prepare_wesad

        dataset = prepare_wesad(
            input_root=args.input_root,
            output_path=args.data,
            sample_rate=args.source_rate,
            target_sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            min_label_purity=args.min_label_purity,
            label_set="core",
        )
        print(
            f"Prepared {dataset.num_samples} windows from {dataset.num_subjects} subjects "
            f"at {args.data}"
        )

    result = run_experiment_suite(
        data_path=args.data,
        output_dir=args.output,
        models=args.models,
        sample_rate=args.sample_rate,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        min_coverage=args.min_coverage,
        inner_folds=args.inner_folds,
        random_seed=args.seed,
    )
    print(f"Completed {len(result['models'])} models in {result['evaluation_seconds']:.1f}s")
    for model_name, summary in result["model_summaries"].items():
        macro_f1 = summary["cross_validation"]["aggregate"]["macro_f1"]
        threshold = summary["bundle"]["confidence_threshold"]
        print(
            f"{model_name}: CV Macro-F1={macro_f1['mean']:.4f} +/- {macro_f1['std']:.4f}; "
            f"bundle threshold={threshold:.2f}"
        )


if __name__ == "__main__":
    main()
