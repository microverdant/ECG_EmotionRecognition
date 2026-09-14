# Model Card: Subject-Independent ECG Emotion Recognition

## Selected model and operating modes

The formal feature model is a shrinkage Linear Discriminant Analysis (LDA) over 22 window-level ECG
morphology and HRV features. It uses standardized inputs and automatic covariance shrinkage, which is
appropriate when the number of participants is small relative to the feature dimension. The repository
exposes two explicit decision-prior modes:

- `shrinkage-lda` uses empirical class priors and is the conservative benchmark mode.
- `balanced-shrinkage-lda` uses equal class priors and is the recommended showcase mode when recall
  coverage across all three classes matters more than overall accuracy.

The balanced mode is a decision-rule variant, not a claim that the underlying ECG representation is
universally stronger. Both modes are retained so the accuracy/coverage trade-off remains inspectable.

## Intended use

This is a reproducible research and portfolio project for exploring controlled WESAD study conditions:
baseline, stress, and amusement. It is suitable for laptop-scale experimentation, method
demonstrations, and evaluation-pipeline discussion.

It is not a clinical, wellness, safety, hiring, or mental-health decision system.

## Data and preprocessing

- Chest ECG only; 15 WESAD participants.
- Filtered at the 700 Hz source rate, then anti-aliased resampled to 140 Hz.
- 30-second windows, 15-second stride, and 90% label purity.
- Each window is normalized independently; no full-recording participant statistics are used.
- R-R intervals outside 0.30-2.00 seconds are excluded from HRV features.

## Evaluation

The primary result is five-fold `StratifiedGroupKFold`: no participant occurs in both train and test
sets, while folds are balanced by study condition where possible. The selected model reached Macro-F1
`0.5392 +/- 0.0227` across folds. Large participant variation remains and is reported rather than
hidden; individual Macro-F1 ranges from 0.297 to 0.751.

In a separate LOSO audit, the empirical-prior mode reached mean participant-level Macro-F1 `0.5194`
with a bootstrap 95% CI of `0.4550-0.5810`; individual results ranged from `0.311` to `0.771`. The
equal-prior mode reached Macro-F1 `0.5138` with a bootstrap 95% CI of `0.4172-0.6150`, while improving
Amusement recall from `0.1326` to `0.4065` and balanced accuracy from `0.5547` to `0.5658`. This
trade-off and the wider subject variation are central deployment limitations, not noise to be omitted.

## Confidence policy

Out-of-fold probabilities are audited with Brier score, Log Loss, expected calibration error, and
coverage at confidence thresholds. The default demo exposes confidence and marks predictions below
0.80 as low-confidence; this is a review signal, not a clinical guarantee. Temperature scaling is
implemented and evaluated with cross-fitted training-only predictions, but is not declared universally
better because different calibration metrics move in different directions.

## Known limitations

- The controlled study labels are not diagnoses or a general definition of emotion.
- HRV frequency-domain estimates from 30-second windows are approximate.
- Class imbalance strongly affects the decision rule: empirical priors favor Baseline, while equal
  priors improve Amusement coverage but increase false Amusement predictions.
- Results are sensitive to participant domain shift; performance should not be assumed to transfer to
  a new device, protocol, population, or free-living setting.
- The compact CNN and convolutional LSTM have received the same grouped cross-validation protocol.
  Their Macro-F1 scores are `0.4913 +/- 0.0370` and `0.5300 +/- 0.1161`, respectively; both remain below
  the selected shrinkage LDA baseline.

## Reproducibility and privacy

Raw data, generated windows, fitted models, and participant-level artifacts are ignored by Git. Every
experiment records the split strategy, seed, model metadata, and class-aware metrics locally.
