# WESAD Benchmark

## Scope

This benchmark evaluates subject-independent emotion recognition from the WESAD chest ECG channel.
It reports the primary three-condition task: baseline, stress, and amusement. Meditation is retained
as an optional extended task and is not mixed into the results below.

## Data protocol

| Setting | Value |
|---|---:|
| Subjects | 15 |
| Prepared windows | 2,140 |
| Source sampling rate | 700 Hz |
| Model sampling rate | 140 Hz |
| Window length | 30 seconds |
| Window stride | 15 seconds |
| Minimum label purity | 90% |
| Evaluation | 5-fold GroupKFold by subject |

Raw data was filtered before anti-aliased resampling. Subject identifiers remain attached to every
window, and no subject appears in both the train and test partitions of a fold.

## Feature baseline results

Values are the mean and population standard deviation across five subject-held-out folds.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5914 ± 0.0958 | 0.6207 ± 0.0251 | 0.6074 ± 0.0529 | 0.6207 ± 0.0251 | **0.5716 ± 0.0721** |
| Random Forest | 0.5643 ± 0.0760 | 0.4970 ± 0.0630 | 0.5311 ± 0.0724 | 0.4970 ± 0.0630 | 0.4909 ± 0.0626 |
| RBF-SVM | 0.5492 ± 0.1353 | 0.5271 ± 0.0957 | 0.5426 ± 0.1252 | 0.5271 ± 0.0957 | 0.5154 ± 0.1224 |

### Logistic Regression folds

| Fold | Held-out subjects | Accuracy | Macro-F1 |
|---:|---|---:|---:|
| 1 | S17, S3, S9 | 0.6946 | 0.6386 |
| 2 | S10, S13, S2 | 0.4789 | 0.4868 |
| 3 | S5, S6, S8 | 0.6340 | 0.6260 |
| 4 | S14, S16, S7 | 0.4743 | 0.4803 |
| 5 | S11, S15, S4 | 0.6752 | 0.6265 |

The variance between folds is material. Reporting only the strongest split would overstate expected
performance on unseen subjects.

## Tiny CNN iteration

The neural iteration below uses one locked subject-level train/validation/test split and therefore
must not be compared directly with the five-fold means above. The purpose of this table is to show
the measured impact of the architecture change on the same split.

| Model | Parameters | Epochs | Training time | Test accuracy | Test balanced accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Tiny CNN v1 | 43,939 | 13 | 55.69 s | 0.3247 | 0.4570 | 0.3003 |
| Tiny CNN v2 | 57,075 | 14 | 39.87 s | **0.5741** | **0.4781** | **0.4476** |

Tiny CNN v2 replaces BatchNorm with GroupNorm, adds temporal statistics pooling, and uses lightweight
augmentation, label smoothing, and validation-driven learning-rate reduction. The largest remaining
failure mode is poor generalization for amusement in the locked test subjects.

## Reproduction

Prepare the ignored local dataset:

```powershell
python scripts/prepare_wesad.py `
  --input D:\path\to\WESAD `
  --output data\processed\wesad_core_140hz_30s.npz `
  --sample-rate 700 `
  --target-sample-rate 140 `
  --window-seconds 30 `
  --stride-seconds 15 `
  --min-label-purity 0.9 `
  --label-set core
```

Run the primary grouped baseline:

```powershell
python -m ecg_emotion.cli cross-validate-baseline `
  --data data\processed\wesad_core_140hz_30s.npz `
  --output artifacts\cv-core-logistic `
  --model logistic-regression `
  --sample-rate 140 `
  --folds 5 `
  --seed 42
```

## Limitations and next work

- Neural results require grouped cross-validation before they can be treated as a final benchmark.
- Hyperparameter selection is not nested inside the outer subject folds.
- Short-window HRV frequency features are approximate and should not be interpreted clinically.
- WESAD labels describe controlled study conditions, not a medical diagnosis.
- Future work should assess confidence calibration and subject-domain adaptation.

No raw WESAD files, prepared windows, fitted models, or participant-level artifacts are committed to
this repository.

