"""Render a compact static figure from a local LOSO robustness JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL_NAMES = ("Baseline", "Stress", "Amusement")


def plot_robustness(input_path: Path, output_path: Path) -> None:
    """Render feature ablation, pooled confusion, and subject variation panels."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    model_name = payload.get("model_name", "model")
    feature_sets = payload["feature_sets"]
    names = list(feature_sets)
    f1_means = [feature_sets[name]["aggregate"]["macro_f1"]["mean"] for name in names]
    f1_intervals = [feature_sets[name]["aggregate"]["macro_f1"]["bootstrap_95ci"] for name in names]
    lower = np.asarray(f1_means) - np.asarray([interval[0] for interval in f1_intervals])
    upper = np.asarray([interval[1] for interval in f1_intervals]) - np.asarray(f1_means)

    primary = feature_sets["all_features"]["aggregate"]
    confusion = np.asarray(primary["row_normalized_confusion_matrix"])
    subjects = feature_sets["all_features"]["subjects"]
    subjects = sorted(subjects, key=lambda item: item["metrics"]["macro_f1"])

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    axes[0].barh(names, f1_means, xerr=np.vstack((lower, upper)), color="#35618f")
    axes[0].set_title("Feature ablation")
    axes[0].set_xlabel("LOSO Macro-F1")
    axes[0].set_xlim(0.0, 0.7)
    axes[0].grid(axis="x", alpha=0.25)

    image = axes[1].imshow(confusion, vmin=0.0, vmax=1.0, cmap="Blues")
    axes[1].set_title("Pooled confusion matrix")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    axes[1].set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=30, ha="right")
    axes[1].set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            color = "white" if confusion[row, column] > 0.5 else "black"
            axes[1].text(
                column,
                row,
                f"{confusion[row, column]:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Row proportion")

    subject_names = [item["subject"] for item in subjects]
    subject_f1 = [item["metrics"]["macro_f1"] for item in subjects]
    axes[2].barh(subject_names, subject_f1, color="#5e8b69")
    axes[2].set_title("Subject-level variation")
    axes[2].set_xlabel("LOSO Macro-F1")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].grid(axis="x", alpha=0.25)

    figure.suptitle(
        f"WESAD subject-domain robustness audit — {model_name}",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_robustness(args.input, args.output)


if __name__ == "__main__":
    main()
