# Model Card: Subject-Independent ECG Emotion Recognition

## Selected model

The formal feature model is a shrinkage Linear Discriminant Analysis (LDA) over 22 window-level ECG
morphology and HRV features. It uses standardized inputs and automatic covariance shrinkage, which is
appropriate when the number of participants is small relative to the feature dimension. It is selected
over Random Forest, RBF-SVM, Logistic Regression, and the neural models based on subject-independent
cross-validation.

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
- R-R intervals outside 0.30–2.00 seconds are excluded from HRV features.

## Evaluation

The primary result is five-fold `StratifiedGroupKFold`: no participant occurs in both train and test
sets, while folds are balanced by study condition where possible. The selected model reached Macro-F1
`0.5392 +/- 0.0227` across folds. Large participant variation remains and is reported rather than
hidden; individual Macro-F1 ranges from 0.297 to 0.751.

## Known limitations

- The controlled study labels are not diagnoses or a general definition of emotion.
- HRV frequency-domain estimates from 30-second windows are approximate.
- Results are sensitive to participant domain shift; performance should not be assumed to transfer to
  a new device, protocol, population, or free-living setting.
- The compact CNN and convolutional LSTM have received the same grouped cross-validation protocol.
  Their Macro-F1 scores are `0.4913 +/- 0.0370` and `0.5300 +/- 0.1161`, respectively; both remain below
  the selected shrinkage LDA baseline.

## Reproducibility and privacy

Raw data, generated windows, fitted models, and participant-level artifacts are ignored by Git. Every
experiment records the split strategy, seed, model metadata, and class-aware metrics locally.
