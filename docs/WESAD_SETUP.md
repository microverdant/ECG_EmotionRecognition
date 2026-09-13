# WESAD Setup

## 1. Obtain the dataset

Download WESAD through the official dataset access page and follow its terms for use and
redistribution. Do not commit the raw participant files to this repository.

## 2. Place the files locally

The adapter expects the standard subject layout, for example:

```text
data/raw/WESAD/
├── S2/S2.pkl
├── S3/S3.pkl
└── ...
```

Each pickle must contain the chest ECG signal at `signal -> chest -> ECG` and synchronized labels
at `label`.

## 3. Prepare the project archive

```powershell
python scripts/prepare_wesad.py `
  --input data/raw/WESAD `
  --output data/processed/wesad_ecg.npz `
  --sample-rate 700 `
  --window-seconds 10 `
  --stride-seconds 5
```

The output follows [`DATA_CONTRACT.md`](DATA_CONTRACT.md): every row contains one ECG window, a
zero-based four-class label, and a subject identifier.

## 4. Verify the split before training

The training utilities split by subject. Never create overlapping windows across the split boundary,
and never fit normalization statistics using validation or test subjects.

