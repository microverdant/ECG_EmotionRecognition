# ECG Emotion Recognition

> A reproducible and lightweight pipeline for subject-independent emotion recognition from ECG signals.

## Project status

This repository is an independent, clean-room implementation focused on methodological clarity,
CPU-friendly experimentation, and reproducible evaluation. The current release contains the project
foundation and signal-processing contract. Model training and evaluation components are being added
incrementally.

## Research question

Can a compact ECG model recognize affective states while preserving subject-level separation and
remaining practical to run on a laptop CPU?

The project deliberately emphasizes reliable evaluation over a single optimistic score. Every final
experiment should report the split protocol, macro-F1, balanced accuracy, confusion matrix, and
per-subject variation.

## Planned pipeline

```text
Raw ECG
  -> quality checks and filtering
  -> subject-level split
  -> fixed-length windows
  -> feature baseline and lightweight temporal model
  -> calibrated evaluation and error analysis
  -> local inference demo
```

## Planned model tracks

1. HRV/statistical features with Logistic Regression and Random Forest baselines.
2. A compact 1D-CNN or TCN for raw ECG windows.
3. Optional fusion of interpretable ECG features and learned representations.

Large recurrent architectures are not part of the default laptop workflow.

## Data policy

Raw participant data is not included in this repository. Users must obtain the dataset through its
official access process and place prepared data under `data/`, following the contract in
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Generated data and model artifacts are ignored by
Git by default.

## Development principles

- Split by subject before creating overlapping windows.
- Fit normalization parameters on training subjects only.
- Keep raw data, generated artifacts, and credentials out of Git.
- Record seeds, configuration, metrics, and model metadata for every run.
- Treat this as an experimental research tool, not a medical diagnostic system.

## Roadmap

- [x] Establish a modern Python project and data contract.
- [ ] Implement filtering, windowing, and subject-level splits.
- [ ] Add fast feature-based baselines.
- [ ] Add the compact 1D-CNN/TCN training path.
- [ ] Add evaluation reports and reproducibility checks.
- [ ] Add a local Streamlit inference demo.

## License

Released under the [MIT License](LICENSE). Dataset access and use remain subject to the dataset's
own terms.

