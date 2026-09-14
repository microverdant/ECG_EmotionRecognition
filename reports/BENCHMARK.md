# WESAD Benchmark

## Protocol

| Setting | Value |
|---|---:|
| Subjects | 15 |
| Prepared windows | 2,140 |
| Task | Baseline, stress, amusement |
| Source / model rate | 700 Hz / 140 Hz |
| Window / stride | 30 s / 15 s |
| Minimum label purity | 90% |
| Evaluation | 5-fold StratifiedGroupKFold by subject |

Raw ECG is filtered before resampling. Each feature window is normalized independently, and all
learned scalers are fit within the training partition only. R-R intervals outside 0.30–2.00 seconds
are excluded before HRV calculations. No participant appears in both train and test partitions.

## Primary results

Values are the mean ± population standard deviation across five subject-held-out folds.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Assessment |
|---|---:|---:|---:|---|
| Shrinkage LDA | 0.6713 ± 0.0399 | 0.5496 ± 0.0283 | **0.5392 ± 0.0227** | Selected, low-variance model |
| Logistic Regression | 0.5581 ± 0.1064 | 0.5422 ± 0.1009 | 0.5192 ± 0.1011 | Linear comparison |
| Random Forest | 0.6021 ± 0.1007 | 0.5089 ± 0.0635 | 0.4758 ± 0.0767 | Reject: train Macro-F1 0.997 |
| RBF-SVM | 0.5832 ± 0.0492 | 0.5322 ± 0.0496 | 0.5156 ± 0.0601 | Reject: train Macro-F1 0.888 |

The compact Tiny CNN was evaluated under the same outer subject protocol, with an inner
participant-level validation split for early stopping:

| Model | Accuracy | Balanced accuracy | Macro-F1 | Assessment |
|---|---:|---:|---:|---|
| Tiny CNN v2 | 0.5819 ± 0.1045 | 0.5293 ± 0.0418 | 0.4913 ± 0.0370 | Robust secondary model, not selected |

The CNN is more consistent across outer folds but does not outperform the feature baseline. This
supports retaining Shrinkage LDA as the primary model and the CNN as a reproducible raw-signal
comparison.

The feature results are not strong enough to claim broad emotion recognition. They are a realistic
subject-independent baseline. Shrinkage LDA is selected because it provides the highest mean Macro-F1
and the lowest fold variation among the candidates. On the locked holdout it reached Macro-F1 0.5201,
with train/validation/test Macro-F1 of 0.6519/0.5494/0.5201.

## Robustness checks

- No zero R-peak windows, non-finite values, or mean heart rates outside 30–200 bpm occurred in the
  prepared primary data.
- R-R validity ratio was at least 0.972 for every window; fewer than 0.8 valid intervals occurred in
  zero windows.
- Participant-level Macro-F1 for Shrinkage LDA ranges from 0.297 to 0.751. This is evidence of
  unresolved domain shift, so only the fold mean and spread should be cited.

## Reproduction

```powershell
python scripts/prepare_wesad.py `
  --input D:\path\to\WESAD `
  --output data\processed\wesad_core_140hz_30s_robust.npz `
  --sample-rate 700 `
  --target-sample-rate 140 `
  --window-seconds 30 `
  --stride-seconds 15 `
  --min-label-purity 0.9 `
  --label-set core

python -m ecg_emotion.cli cross-validate-baseline `
  --data data\processed\wesad_core_140hz_30s_robust.npz `
  --output artifacts\wesad-core-robust-shrinkage-lda `
  --model shrinkage-lda `
  --sample-rate 140 `
  --folds 5 `
  --seed 42
```

To reproduce the CNN comparison:

```powershell
python -m ecg_emotion.cli cross-validate-tiny-cnn `
  --data data\processed\wesad_core_140hz_30s_robust.npz `
  --output artifacts\wesad-core-tiny-cnn-cv `
  --sample-rate 140 `
  --epochs 20 `
  --patience 5 `
  --folds 5 `
  --seed 42
```

Raw WESAD files, prepared windows, fitted models, and participant-level artifacts are not committed.
