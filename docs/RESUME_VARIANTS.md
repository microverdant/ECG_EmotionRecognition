# Targeted Resume Variants

This document turns the project evidence into role-specific resume material. It is a public, generic
content bank: add personal dates, institution names, competition names, and links only in the private
master resume.

## Positioning strategy

Use one master resume as the source of truth, then tailor the top third and selected project bullets to
the job description. Do not put every technology and every project on every version.

| Target direction | Lead with | Keep secondary | Avoid leading with |
|---|---|---|---|
| ML / Healthcare AI Engineer | ECG research system and evaluation rigor | 9900 dissertation, Streamlit demo | Unreal Engine details |
| Data / Applied Scientist | grouped evaluation, class behavior, uncertainty | ECG features, 9900 design | generic event-management language |
| Signal Processing Engineer | filtering, resampling, R-peaks, HRV | ECG model comparison, IoT | Jira clone as a major project |
| Software / Backend Engineer | reproducibility, CLI, tests, CI, repository design | 9900 system ownership, ECG demo | claiming ownership of the whole bootcamp product |
| Technical Project / Product roles | 9900 leadership and society operations | ECG delivery, IoT collaboration | presenting research metrics without context |

## Master project inventory

### ECG Emotion Recognition — individual dissertation project

**Recommended title:** Subject-Independent ECG Emotion Recognition | Python, SciPy, scikit-learn,
PyTorch, Streamlit

- Built an end-to-end ECG affect-recognition pipeline for WESAD, converting 700 Hz chest ECG into
  2,140 leakage-safe 140 Hz windows and extracting 22 morphology, spectral, heart-rate, and HRV
  features.
- Designed subject-independent evaluation with grouped cross-validation and Leave-One-Subject-Out
  robustness analysis; Shrinkage LDA achieved `0.5392 +/- 0.0227` Macro-F1 under five-fold grouped
  evaluation.
- Implemented and compared interpretable LDA baselines with Logistic Regression, Random Forest, RBF-SVM,
  compact 1D-CNN, and convolutional LSTM models, documenting overfitting and participant domain shift.
- Added an equal-prior operating mode that improved LOSO Amusement recall from `0.1326` to `0.4065`,
  while explicitly reporting the associated accuracy and subject-variation trade-off.
- Built leakage-safe OOF confidence and abstention audits, a reproducible experiment suite, and a
  Streamlit showcase with model selection, class probabilities, threshold provenance, and accept /
  abstain states.

**Use for:** ML Engineer, Healthcare AI, Data Scientist, Applied Scientist, Signal Processing.

### 9900 — Master dissertation

Use the following structure after inserting the actual project name and technical stack:

- Led the master dissertation project from problem definition through system design, owning the overall
  solution concept, architecture, and implementation plan.
- Translated an open-ended requirement into a structured technical proposal, defining the main modules,
  interfaces, evaluation criteria, and delivery milestones.
- Coordinated [team size / contributors] through [design reviews / implementation milestones / final
  demonstration], resolving technical dependencies and maintaining end-to-end scope ownership.

**Use for:** Technical Project Manager, Product Engineer, Systems Engineer, Software Engineer, ML
Engineer. Do not claim a specific technology or performance result until it is confirmed.

### IoT competition project

- Participated throughout the complete IoT competition lifecycle, contributing to requirements analysis,
  system design, implementation, integration, testing, and final presentation.
- Collaborated across hardware, software, and demonstration constraints to integrate the team solution
  into a coherent end-to-end prototype.
- [Add the verified device, protocol, platform, measurable result, or award here.]

**Use for:** IoT Engineer, Embedded Software, Systems Engineer, Technical Project roles.

### 9415 — Unreal Engine game project

- Developed an Unreal Engine game project, translating a gameplay concept into an interactive prototype
  through [level design / gameplay logic / UI / asset integration / testing].
- Iterated on player experience and technical implementation based on [playtesting / milestone reviews /
  course requirements].
- [Add the verified Unreal Engine version, Blueprints / C++, team role, and final deliverable here.]

**Use for:** Gameplay Programmer, Technical Designer, Interactive Software, general Software roles when
the target employer values real-time systems or rapid prototyping.

### marketing-simplified — bootcamp Jira-like project

- Contributed repository design for a Jira-like bootcamp project, organizing the codebase structure,
  module boundaries, and collaboration conventions for maintainable team development.
- [Add only the specific repository responsibilities you personally completed: branching strategy,
  package layout, API boundaries, documentation, or CI configuration.]

**Use for:** Software Engineer, Backend Engineer, Developer Tools, Technical Operations. Keep the wording
scoped to repository design; do not present the project as entirely self-developed.

### University society leadership

- Served as society president, managing a 300-member community and a 50-person core leadership team.
- Planned and delivered 10+ large-scale and multi-university events, coordinating people, timelines,
  stakeholders, and execution risks across multiple workstreams.
- Built operating structure and delegated ownership across the core team to maintain delivery quality at
  community scale.

**Use for:** Technical Project Manager, Product Operations, Graduate Leadership, Consulting, and as a
leadership differentiator on engineering applications.

## Version A — ML / Healthcare AI Engineer

### Selected projects

**Subject-Independent ECG Emotion Recognition — Individual Dissertation**

- Built a CPU-friendly ECG emotion-recognition pipeline for WESAD, processing 700 Hz chest ECG into
  2,140 leakage-safe 140 Hz windows and 22 morphology, spectral, heart-rate, and HRV features.
- Benchmarked LDA, classical machine-learning baselines, compact CNN, and convolutional LSTM models
  under subject-grouped validation; Shrinkage LDA reached `0.5392 +/- 0.0227` Macro-F1.
- Implemented LOSO robustness, class-balanced priors, and training-only OOF abstention analysis to expose
  class imbalance and participant domain shift rather than optimizing a single optimistic score.
- Delivered a Streamlit inference showcase and reproducible experiment suite with CI-tested code and
  strict separation of raw participant data from public source control.

**9900 — Master Dissertation | Project Lead**

- Led problem definition, solution design, architecture, and implementation planning for the master
  dissertation project; coordinated [team size] contributors through [verified deliverables].

**University Society President**

- Managed a 300-member community and 50-person core team while delivering 10+ large-scale and
  multi-university events.

## Version B — Data / Applied Scientist

### Selected projects

**Subject-Independent ECG Emotion Recognition — Individual Dissertation**

- Designed an evaluation-first study of ECG-based affect recognition using grouped cross-validation and
  LOSO analysis to measure generalization to unseen participants.
- Extracted 22 interpretable signal features and compared linear, nonlinear, CNN, and LSTM models;
  reported Macro-F1, balanced accuracy, per-class confusion, calibration, and subject-level variation.
- Quantified a class-prior trade-off: equal-prior LDA raised LOSO Amusement recall from `0.1326` to
  `0.4065`, while reducing overall accuracy and increasing participant variation.
- Built cross-fitted OOF threshold selection and bootstrap uncertainty reporting, making confidence and
  abstention claims auditable.

**9900 — Master Dissertation | Project Lead**

- Converted an open-ended research problem into a structured technical solution, defining hypotheses,
  system boundaries, evaluation criteria, and delivery milestones.

**IoT Competition Project**

- Contributed across requirements, system integration, testing, and final presentation in a constrained
  end-to-end IoT prototype.

## Version C — Signal Processing / Healthcare AI

### Selected projects

**Subject-Independent ECG Emotion Recognition — Individual Dissertation**

- Implemented ECG cleaning, anti-aliased resampling from 700 Hz to 140 Hz, fixed-window segmentation,
  R-peak detection, R-R validity checks, HRV computation, morphology features, and spectral features.
- Prevented leakage by normalizing windows independently, fitting learned scalers on training subjects,
  and keeping overlapping windows within evaluation partitions.
- Evaluated 22-feature representations under grouped CV and LOSO; documented that participant domain
  shift and Amusement / Baseline overlap dominate the remaining error.
- Exposed class-specific confidence, pooled confusion matrices, and abstention behavior through a local
  Streamlit demo.

**IoT Competition Project**

- Participated in the full sensing-to-system-integration workflow, including requirements, integration,
  testing, and demonstration.

**9415 — Unreal Engine Project**

- Developed an interactive real-time prototype in Unreal Engine, strengthening experience with event
  driven logic, system integration, iteration, and user-facing behavior.

## Version D — Software / Backend Engineer

### Selected projects

**ECG Emotion Recognition — Reproducible ML System**

- Built a modular Python project with CLI entry points for data preparation, training, evaluation,
  robustness audits, selective prediction, and local inference.
- Added 21 automated tests, Ruff quality checks, GitHub CI, deterministic seeds, JSON experiment records,
  and an end-to-end experiment-suite runner.
- Designed model bundles with preprocessing metadata, class labels, training-only threshold provenance,
  and explicit inference acceptance states.
- Kept raw data and generated artifacts outside version control through repository-level data contracts
  and ignore rules.

**marketing-simplified — Bootcamp Project**

- Contributed repository architecture for a Jira-like collaboration product, defining maintainable code
  organization and team development conventions within the scope of repository design.

**9900 — Master Dissertation | Project Lead**

- Owned system-level design and technical planning, coordinating [team size] contributors and aligning
  module boundaries with delivery milestones.

## Version E — Technical Project / Product

### Selected experience

**9900 — Master Dissertation | Project Lead**

- Led the end-to-end design of the master dissertation solution, translating an ambiguous problem into
  an actionable architecture, implementation plan, and evaluation framework.
- Coordinated [team size] contributors across [verified workstreams], managing dependencies, milestones,
  technical decisions, and final delivery.

**University Society President**

- Managed a 300-member community and 50-person core team, establishing ownership across workstreams and
  delivering 10+ large-scale and multi-university events.
- Balanced stakeholder communication, resource allocation, execution risk, and event quality across
  multiple concurrent initiatives.

**ECG Emotion Recognition — Individual Dissertation**

- Independently delivered a complete research software project from data contract and evaluation design
  through model benchmarking, robustness analysis, documentation, CI, and an interactive demo.

## ATS keyword bank

Select keywords that are genuinely supported by the final version of the resume:

`Python` · `NumPy` · `SciPy` · `scikit-learn` · `PyTorch` · `Pandas` · `ECG` · `signal processing` ·
`HRV` · `feature engineering` · `time-series classification` · `grouped cross-validation` · `LOSO` ·
`calibration` · `abstention` · `Streamlit` · `GitHub Actions` · `CI` · `CLI design` · `Unreal Engine` ·
`IoT` · `system design` · `project leadership` · `stakeholder coordination`

## What still needs personal verification

- exact dates and institution names;
- the official title and technical stack of the 9900 dissertation;
- team size and confirmed leadership scope for 9900;
- IoT competition name, technologies, award, and personal contribution;
- Unreal Engine version, Blueprints / C++ split, team scope, and final deliverable;
- the exact repository-design responsibilities in marketing-simplified;
- links to public demos, repositories, or portfolio pages.
