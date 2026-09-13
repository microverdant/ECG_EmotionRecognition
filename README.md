# ECG Emotion Recognition

> A reproducible and lightweight pipeline for subject-independent emotion recognition from ECG signals.

## Project status

This repository is an independent, clean-room implementation focused on methodological clarity,
CPU-friendly experimentation, and reproducible evaluation. It contains signal processing, feature
baselines, an optional compact 1D-CNN, tests, and a local inference demo.

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

## Model tracks

1. HRV/statistical features with Logistic Regression and Random Forest baselines.
2. A compact 1D-CNN for raw ECG windows.
3. Future work: TCN and feature/representation fusion.

Large recurrent architectures are not part of the default laptop workflow.

## Data policy

Raw participant data is not included in this repository. Users must obtain the dataset through its
official access process and place prepared data under `data/`, following the contract in
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Generated data and model artifacts are ignored by
Git by default.

For standard WESAD pickle files, follow [`docs/WESAD_SETUP.md`](docs/WESAD_SETUP.md) and run the
included preparation script before training.

## Development principles

- Split by subject before creating overlapping windows.
- Fit normalization parameters on training subjects only.
- Keep raw data, generated artifacts, and credentials out of Git.
- Record seeds, configuration, metrics, and model metadata for every run.
- Treat this as an experimental research tool, not a medical diagnostic system.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ecg_emotion.cli make-demo-data --output data/demo_ecg.npz
python -m ecg_emotion.cli train-baseline `
  --data data/demo_ecg.npz `
  --output artifacts/demo-baseline `
  --sample-rate 128
```

The baseline command writes a serialized model and `metrics.json`. To run the optional neural model:

```powershell
python -m pip install -e ".[deep-learning]"
python -m ecg_emotion.cli train-tiny-cnn `
  --data data/demo_ecg.npz `
  --output artifacts/tiny-cnn `
  --sample-rate 128
```

To launch the local showcase after training a baseline:

```powershell
python -m pip install -e ".[demo]"
streamlit run demo/app.py
```

The bundled synthetic data is for smoke testing only. It is not a physiological dataset and its
metrics must not be presented as research results.

## Roadmap

- [x] Establish a modern Python project and data contract.
- [ ] Implement filtering, windowing, and subject-level splits.
- [x] Add fast feature-based baselines.
- [x] Add a compact 1D-CNN training path.
- [x] Add evaluation artifacts and reproducibility checks.
- [x] Add a local Streamlit inference demo.
- [ ] Evaluate the official prepared dataset with a locked subject-level protocol.

## License

Released under the [MIT License](LICENSE). Dataset access and use remain subject to the dataset's
own terms.
