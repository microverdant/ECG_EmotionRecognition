"""Prepare standard WESAD pickle files for local experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_emotion.wesad import prepare_wesad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="WESAD root directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed/wesad_ecg.npz"))
    parser.add_argument("--sample-rate", type=float, default=700.0)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--min-label-purity", type=float, default=0.8)
    parser.add_argument("--line-frequency", type=float, default=50.0)
    parser.add_argument(
        "--target-sample-rate",
        type=float,
        default=None,
        help="optional CPU-friendly output sampling rate, for example 140",
    )
    parser.add_argument(
        "--label-set",
        choices=("core", "extended"),
        default="core",
        help="core uses baseline/stress/amusement; extended also includes meditation",
    )
    args = parser.parse_args()

    dataset = prepare_wesad(
        input_root=args.input,
        output_path=args.output,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_label_purity=args.min_label_purity,
        line_frequency=args.line_frequency,
        target_sample_rate=args.target_sample_rate,
        label_set=args.label_set,
    )
    print(
        f"Prepared {dataset.num_samples} windows from {dataset.num_subjects} subjects "
        f"at {args.output}"
    )


if __name__ == "__main__":
    main()
