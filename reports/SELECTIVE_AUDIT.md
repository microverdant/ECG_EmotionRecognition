# Leakage-Safe Selective Prediction Audit

## Purpose

The demo exposes a confidence-aware prediction state, but a fixed confidence threshold can create a
misleading impression of safety. This audit asks whether the model can abstain from uncertain windows
without becoming overly specific to one class.

For every LOSO split, the threshold is selected from three-fold cross-fitted predictions on the
training subjects only. The held-out subject is evaluated exactly once with that threshold. The
selection objective is class-balanced selective F1 subject to a predefined minimum OOF coverage of
30%. No held-out labels are used to select a threshold.

## WESAD results

Values are means over the 15 held-out subjects. The confidence threshold is independently selected for
each held-out subject, so the spread is also reported.

| Mode | OOF threshold | Held-out coverage | Rejection rate | Selective accuracy | Selective Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Empirical-prior Shrinkage LDA | 0.5033 +/- 0.0085 | 0.9157 | 0.0843 | **0.7001** | **0.5027** |
| Equal-prior Shrinkage LDA | 0.5000 +/- 0.0000 | 0.8430 | 0.1570 | 0.6278 | 0.4876 |

The empirical-prior mode has a bootstrap 95% CI of `0.8570-0.9640` for coverage and `0.6260-0.7675`
for selective accuracy. The equal-prior mode has a bootstrap 95% CI of `0.7792-0.8956` for coverage
and `0.5244-0.7287` for selective accuracy. These intervals are subject-level bootstrap intervals,
not confidence intervals over independent people from a larger population.

The learned thresholds are close to 0.50 rather than 0.80. This is an important negative result:
confidence scores on this small controlled cohort do not support aggressive abstention with a reliable
cross-subject gain. The model bundle therefore stores a training-only OOF threshold, and the result
must be interpreted as a review signal rather than a safety guarantee.

## Class-level acceptance behavior

`Coverage` is the fraction of true-class windows that are accepted. `Correct coverage` is the fraction
of true-class windows that are both accepted and correctly classified. `Accepted precision` is the
precision among accepted predictions of that class.

| Mode | Class | Mean confidence | Coverage | Correct coverage | Accepted precision |
|---|---|---:|---:|---:|---:|
| Empirical priors | Baseline | 0.7482 | 0.9168 | **0.7826** | 0.7096 |
| Empirical priors | Stress | **0.8393** | **0.9319** | 0.6698 | **0.8787** |
| Empirical priors | Amusement | 0.7050 | 0.8830 | 0.1156 | 0.3961 |
| Equal priors | Baseline | 0.6269 | 0.8421 | 0.5170 | 0.5760 |
| Equal priors | Stress | **0.8069** | **0.8753** | **0.6953** | **0.8488** |
| Equal priors | Amusement | 0.6112 | 0.7847 | 0.3006 | 0.2782 |

The balanced operating mode improves Amusement correct coverage relative to empirical priors, but its
accepted Amusement precision remains low. The empirical mode has high acceptance coverage for all
classes but still accepts many incorrect Amusement predictions. This confirms that confidence alone
does not solve the underlying participant-domain and class-overlap problems.

## Reproduction

```powershell
python -m ecg_emotion.cli audit-abstention `
  --data data\processed\wesad_core_140hz_30s_robust.npz `
  --output artifacts\wesad-selective-audit-balanced `
  --model balanced-shrinkage-lda `
  --sample-rate 140 `
  --min-coverage 0.3 `
  --inner-folds 3 `
  --seed 42
```

Use `--model shrinkage-lda` and a different output directory to reproduce the empirical-prior audit.
The JSON output and fitted local bundles are ignored by Git; the raw WESAD data is never uploaded.
