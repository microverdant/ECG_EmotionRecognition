# Data Contract

The training code consumes a prepared NumPy archive. This keeps dataset-specific parsing separate
from model code and makes experiments reproducible.

## Required archive fields

```text
signals  float32 array with shape (n_samples, signal_length)
labels   integer array with shape (n_samples,)
subjects string or integer array with shape (n_samples,)
```

Each row in `signals` must represent one fixed-length ECG window. `labels[i]` and `subjects[i]`
must describe the same window.

## Label contract

The default four-class label mapping is:

| ID | Name |
|---:|---|
| 0 | baseline |
| 1 | stress |
| 2 | amusement |
| 3 | meditation |

If another dataset or label mapping is used, update the configuration and README before reporting
results.

## Leakage prevention

Subject identifiers are mandatory. Splits must be generated from subject IDs before model fitting.
Window overlap is allowed only within a split; no subject may appear in more than one split.

## Data preparation checklist

- Verify the sampling rate and signal units.
- Remove or document invalid samples.
- Apply filtering consistently across splits.
- Fit any global normalization statistics on the training subjects only.
- Save the preprocessing configuration with the experiment output.

