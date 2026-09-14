# Resume-Ready Project Summary

## Recommended project title

**Subject-Independent ECG Emotion Recognition | Python, SciPy, scikit-learn, PyTorch, Streamlit**

## Concise English version

- Built a reproducible ECG affect-recognition pipeline for the WESAD study, converting 700 Hz chest ECG
  into 2,140 leakage-safe 140 Hz windows and extracting 22 morphology, spectral, heart-rate, and HRV
  features.
- Designed and benchmarked shrinkage LDA, Logistic Regression, Random Forest, RBF-SVM, compact 1D-CNN,
  and convolutional LSTM models under subject-grouped cross-validation; Shrinkage LDA achieved
  `0.5392 +/- 0.0227` Macro-F1 with no participant overlap between train and test folds.
- Led a class-coverage analysis that added an equal-prior LDA operating mode, improving LOSO Amusement
  recall from `0.1326` to `0.4065` while documenting the associated accuracy and domain-variation trade-off.
- Implemented leakage-safe confidence diagnostics and abstention: thresholds are learned only from
  training-subject cross-fitted OOF predictions, with per-class acceptance metrics and subject-level
  bootstrap intervals.
- Delivered a Streamlit research showcase with model-mode selection, class probabilities, threshold
  provenance, and explicit accept/abstain states; kept raw participant data and generated artifacts out
  of Git through automated ignore and CI checks.

## Shorter two-bullet version

- Developed a subject-independent ECG emotion-recognition system with 22 engineered signal features,
  grouped cross-validation, LOSO domain-robustness analysis, and lightweight LDA/CNN/LSTM baselines;
  achieved `0.5392 +/- 0.0227` Macro-F1 on the WESAD benchmark.
- Improved minority-class coverage through an equal-prior decision mode and built leakage-safe OOF
  confidence / abstention diagnostics plus a Streamlit inference demo.

## Interview framing

The strongest story is methodological rather than claiming a high accuracy number: the project tests
whether ECG emotion models generalize to unseen participants, identifies class imbalance and domain
shift as failure modes, and exposes those limitations in the model card and demo.
