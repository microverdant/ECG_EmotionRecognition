# Subject-Domain Robustness Audit

## Protocol

This audit uses Leave-One-Subject-Out (LOSO) evaluation on the primary WESAD task. Each participant
is held out once, the model is trained on all other participants, and metrics are first computed per
participant before averaging. This prevents participants with more windows from dominating the result.

The audit also runs four predefined feature sets. Feature-set selection is descriptive and is not used
to tune the reported LOSO test predictions.

| Setting | Value |
|---|---:|
| Model | Shrinkage LDA |
| Subjects | 15 |
| Features | 22 total; 140 Hz, 30-second windows |
| Split | Leave-One-Subject-Out |
| Bootstrap samples | 2,000 subject-level resamples |
| Confidence interval | Percentile 95% CI for the mean |

## Feature ablation

| Feature set | Features | Macro-F1 | Subject SD | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| All features | 22 | **0.5194** | 0.1286 | 0.4550-0.5810 |
| Without amplitude | 20 | 0.5185 | 0.1272 | 0.4515-0.5782 |
| Rate and HRV | 13 | 0.4744 | 0.1508 | 0.3985-0.5505 |
| Morphology and spectral | 9 | 0.4726 | 0.1258 | 0.4049-0.5323 |

The complete feature set is retained because it has the highest mean Macro-F1. Removing R-peak
amplitude features barely changes Macro-F1, suggesting that amplitude is not essential for this
cross-subject result. HRV-only and morphology/spectral-only representations lose complementary signal.

## Subject-level results

| Subject | Macro-F1 | Subject | Macro-F1 | Subject | Macro-F1 |
|---|---:|---|---:|---|---:|
| S2 | 0.311 | S7 | 0.539 | S13 | 0.484 |
| S3 | 0.588 | S8 | 0.624 | S14 | 0.621 |
| S4 | 0.350 | S9 | 0.568 | S15 | 0.391 |
| S5 | 0.381 | S10 | 0.357 | S16 | 0.609 |
| S6 | 0.603 | S11 | **0.771** | S17 | 0.594 |

The observed range is wide (`0.311-0.771`), which is evidence of participant domain shift. The mean
training Macro-F1 is 0.6171, compared with the LOSO test Macro-F1 of 0.5194. This gap is meaningful
but substantially smaller than the nonlinear overfit baselines previously rejected.

## Confidence and interpretation

The all-feature LOSO model reaches mean accuracy 0.6791 and balanced accuracy 0.5547. Its mean
expected calibration error is 0.1547. These figures support confidence-aware review, not autonomous
emotion assessment. A model trained on this cohort should not be assumed to transfer to a new device,
population, or free-living setting.

## Reproduction

```powershell
python -m ecg_emotion.cli audit-subject-robustness `
  --data data\processed\wesad_core_140hz_30s_robust.npz `
  --output artifacts\wesad-subject-robustness `
  --model shrinkage-lda `
  --sample-rate 140 `
  --bootstrap-samples 2000 `
  --seed 42
```

Raw WESAD files, prepared windows, fitted models, and participant-level artifacts are not committed.
