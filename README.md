# ECG Emotion Recognition

[![CI](https://github.com/microverdant/ECG_EmotionRecognition/actions/workflows/ci.yml/badge.svg)](https://github.com/microverdant/ECG_EmotionRecognition/actions/workflows/ci.yml)

> A reproducible and lightweight pipeline for subject-independent emotion recognition from ECG signals.

## Project status

This independent, clean-room implementation focuses on methodological clarity, CPU-friendly
experimentation, and reproducible evaluation. It includes signal processing, interpretable feature
baselines, optional compact 1D-CNN and convolutional LSTM models, tests, and a local inference demo.

## Research question

Can a compact ECG model recognize controlled affective states while preserving participant-level
separation and remaining practical to run on a laptop CPU?

The project prioritizes reliable evaluation over a single optimistic score. Formal experiments report
the split protocol, Macro-F1, balanced accuracy, confusion matrix, and participant variation.

## WESAD benchmark

The primary subject-independent benchmark uses 15 subjects, 2,140 windows, 140 Hz ECG, 30-second
windows, and five-fold StratifiedGroupKFold evaluation. The selected feature model is a shrinkage LDA
with `0.5392 +/- 0.0227` Macro-F1. A separate `balanced-shrinkage-lda` operating mode reaches
`0.5494 +/- 0.0616` Macro-F1 and improves coverage of the Amusement class, but with lower overall
accuracy and higher subject variation. This modest result is deliberate: the protocol prioritizes
honest performance on unseen people over participant-specific accuracy.

The compact convolutional LSTM is available as a raw-signal experiment and reached `0.5300 +/- 0.1161`
Macro-F1 under the same protocol. It remains a secondary model because its cross-subject variation is
higher than shrinkage LDA.

See [`reports/BENCHMARK.md`](reports/BENCHMARK.md) for the full protocol, results, reproduction
commands, and limitations. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use and known
failure modes.

Confidence and abstention are audited separately in [`reports/SELECTIVE_AUDIT.md`](reports/SELECTIVE_AUDIT.md).
The audit uses training-only OOF threshold selection and reports when confidence is not reliable enough
to support aggressive rejection.

The participant-domain analysis is available in [`reports/ROBUSTNESS.md`](reports/ROBUSTNESS.md).

The complete reproducibility workflow is documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md),
resume-ready project bullets are available in [`docs/RESUME_BULLETS.md`](docs/RESUME_BULLETS.md), and
role-specific variants are collected in [`docs/RESUME_VARIANTS.md`](docs/RESUME_VARIANTS.md).

## Data policy

Raw participant data is not included in this repository. Users must obtain data through its official
access process and place prepared data under `data/`, following
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Generated data and model artifacts are ignored by
Git by default.

For standard WESAD pickle files, follow [`docs/WESAD_SETUP.md`](docs/WESAD_SETUP.md) and run the
included preparation script before training.

## Development principles

- Split by subject before creating overlapping windows.
- Use per-window signal normalization and fit learned feature scaling on training subjects only.
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

The baseline command writes a serialized model and `metrics.json`. To run the optional CNN or LSTM:

```powershell
python -m pip install -e ".[deep-learning]"
python -m ecg_emotion.cli train-tiny-cnn `
  --data data/demo_ecg.npz `
  --output artifacts/tiny-cnn `
  --sample-rate 128

python -m ecg_emotion.cli train-tiny-lstm `
  --data data/demo_ecg.npz `
  --output artifacts/tiny-lstm `
  --sample-rate 128
```

To launch the local showcase after training a baseline:

```powershell
python -m pip install -e ".[demo]"
streamlit run demo/app.py
```

The sidebar can switch between the conservative empirical-prior bundle and the class-balanced bundle.
To prepare the latter locally, run:

```powershell
python -m ecg_emotion.cli train-baseline `
  --data data\processed\wesad_core_140hz_30s_robust.npz `
  --output artifacts\demo-balanced-lda `
  --model balanced-shrinkage-lda `
  --sample-rate 140
```

The interface shows the bundle's training-derived threshold, class probabilities, and explicit accept /
abstain state. Model bundles and WESAD data remain local and ignored by Git.

The bundled synthetic data is for smoke testing only. It is not a physiological dataset and its
metrics must not be presented as research results.

## Formal experiment suite

Once the official WESAD data has been prepared locally, one command runs grouped cross-validation,
LOSO robustness, leakage-safe selective prediction, and local demo-bundle training for both feature
operating modes:

```powershell
python scripts/run_experiment_suite.py `
  --data data\processed\wesad_core_140hz_30s_robust.npz `
  --output artifacts\wesad-suite `
  --models shrinkage-lda balanced-shrinkage-lda `
  --sample-rate 140 `
  --seed 42
```

All generated outputs go under `artifacts/`, which is ignored by Git. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for raw-data preparation and the full experiment
contract.

## Roadmap

- [x] Establish a modern Python project and data contract.
- [x] Implement filtering, windowing, and subject-level splits.
- [x] Add fast feature-based baselines.
- [x] Add compact 1D-CNN and convolutional LSTM training paths.
- [x] Add evaluation artifacts and reproducibility checks.
- [x] Add a local Streamlit inference demo.
- [x] Evaluate the official prepared dataset with grouped subject-level baselines.
- [x] Add grouped cross-validation for Tiny CNN v2 and Tiny LSTM v1.
- [x] Add confidence diagnostics and leakage-safe calibration audit.
- [x] Add class-balanced selective prediction and abstention audit.
- [x] Add LOSO subject-domain audit and feature ablation analysis.
- [ ] Validate calibration and confidence thresholds on larger, more diverse cohorts.

## License

Released under the [MIT License](LICENSE). Dataset access and use remain subject to the dataset's own
terms.
