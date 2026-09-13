# Experiment Protocol

## Required reporting

Every tracked experiment must record:

- dataset version and subject IDs;
- sampling rate and window configuration;
- preprocessing parameters;
- random seed;
- model name and parameter count;
- training time;
- accuracy, macro-F1, balanced accuracy, precision, and recall;
- confusion matrix;
- per-subject metrics when subject-level evaluation is used.

## Recommended comparison

Start with a feature-based Logistic Regression baseline, then compare it with a compact 1D-CNN or
TCN. Use the same subject-level split and label mapping for every model.

## Interpretation

Do not compare scores produced by different data splits as if they were a controlled experiment.
Do not describe a single random split as evidence of subject-independent generalization.

