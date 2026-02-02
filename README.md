# DDESONN: Deep Dynamic Experimental Self-Organizing Neural Network

Mathew William Armitage Fok (<quiksilver67213@yahoo.com>)

**Note on multiple README files:**  
This repository intentionally contains a second README at:

`inst/dev/README.md`


That file is used for development notes, internal context, and in-progress documentation during active experimentation.  
The root `README.md` (this file) is the canonical public-facing README for users, CRAN, and external contributors.

---

## Table of contents
1. Project overview
2. Key capabilities
3. Architecture
4. Project timeline
5. Repository structure
6. Getting started
7. Running the examples
8. Datasets
9. Roadmap
10. To-Do (Active Work)
11. Contributing
12. License
13. Contact

---

## Project overview

DDESONN (Deep Dynamic Experimental Self-Organizing Neural Network) is an R-based research framework for adaptive neural network experimentation.

The project was initiated to build a fully custom neural network system that did not already exist, and to develop a deep, first-principles understanding of machine learning by necessity rather than by copying existing frameworks.

DDESONN blends self-organizing principles with modern deep-learning practices to support:

- Configurable single-layer or multi-layer architectures
- Dynamic ensemble learning with pruning and add-back mechanisms
- Full control of optimizer, regularization, and activation flows
- Reproducible evaluation and artifact reporting

The primary design objective of DDESONN is to provide a fully customizable, entirely R-native neural network codebase and framework, intentionally avoiding external deep-learning backend library dependencies to preserve full architectural control and transparency.

It is being prepared for public release on CRAN.

---

## Key capabilities

- Flexible architecture selection (SL or ML) with independent activations, dropout, and initialization
- Manual training loop with explicit forward and backward propagation
- Optimizers implemented: SGD, RMSProp, Adam, Lookahead (weights and biases handled separately)
- L1, L2, and mixed regularization for both weights and biases
- Automatic F1-optimized threshold tuning with precision and recall scoring
- Dynamic ensemble orchestration with metadata and relevance tracking
- Excel and plot reporting using writexl, openxlsx, ggplot2, and plotly
- High-level API helpers in R/api.R for external integration

---

## Architecture

Core implementation is modular and intentionally explicit:

- R/DDESONN.R  
  Central R6 class implementing SONN core logic, training, prediction, and orchestration

- R/activation_functions.R  
  Activation function library (ReLU, sigmoid, bent, and others)

- R/optimizers.R  
  Optimizer implementations and optimizer state handling

- R/update_weights_block.R  
  Weight update routines with optimizer routing

- R/update_biases_block.R  
  Bias update routines kept separate from weight logic

- R/performance_relevance_metrics.R  
  Accuracy, precision, recall, F1, and relevance metrics

- R/utils.R  
  Shared helper utilities

- R/api.R  
  High-level API-style wrapper for simplified consumption

- R/evaluate_predictions_report.R  
  Excel and plot-based evaluation reporting

Formal R vignettes for guided exploration and reproducible demonstrations are available in the vignettes directory.

Techila (distributed/parallel compute) support exists to scale heavier experiments across multiple servers/workers.  
This becomes relevant quickly when you start running large seed sweeps (e.g., hundreds to thousands of seeds across hundreds of epochs).

---

## Project timeline

- 2024-05-07 ??? Project origin  
  The project formally began in May 2024 as a research initiative to design and implement a novel self-organizing neural network framework in R, prioritizing explicit training logic, architectural transparency, and experimental flexibility.

- Initial intensive sprint (approximately 3 months)  
  Sustained day-in/day-out development. Learning machine learning from first principles was unavoidable in order to design the architecture manually, reason through layer interactions and dimensional flow, identify bottlenecks, and resolve bugs by tracing logic across layers.

- Iterative development (lax / intermittent)  
  Development continued at a more sustainable pace, refining architectural decisions and expanding functionality while preserving full transparency and custom control.

- Second intensive hardening phase (approximately 3???4 months)  
  Focused on correctness, stability, optimizer behavior, ensemble reliability, and reproducibility.

- Late 2025 to early 2026 hiatus  
  Development paused in late 2025 while focusing on two other high-intensity projects.

- 2026 to present  
  Work resumed with emphasis on maintainability, documentation, and long-term research viability.

---

## Repository structure

DDESONN/
?????? R/
??? ?????? DDESONN.R
??? ?????? activation_functions.R
??? ?????? api.R
??? ?????? optimizers.R
??? ?????? performance_relevance_metrics.R
??? ?????? update_biases_block.R
??? ?????? update_weights_block.R
??? ?????? utils.R
??? ?????? reports/
??? ?????? evaluate_predictions_report.R
???
?????? inst/
??? ?????? scripts/
??? ??? ?????? DDESONN_mtcars_example.R
??? ??? ?????? DDESONN_mtcars_A-D_examples*.R
??? ??? ?????? Heart_failure_ScenarioA.R
??? ??? ?????? LoadandPredict.R
??? ??? ?????? TestDDESONN.R
??? ?????? dev/
??? ?????? README.md
???
?????? data/
?????? vignettes/
?????? helpfulFiles/
?????? ideas/
?????? junk/
???
?????? DESCRIPTION
?????? NAMESPACE
?????? DDESONN.Rproj
?????? CHANGELOG.md
?????? README.md
?????? README_v*.md
?????? LICENSE/

---

## Getting started

---

### Prerequisites

- R version 4.1 or higher
- RStudio project file included (DDESONN.Rproj)
- Dependencies listed in DESCRIPTION

---

### Installation

Bash:

    git clone https://github.com/MatHatter/DDESONN.git
    cd DDESONN

Inside R:

    required_pkgs <- c(
      "R6","cluster","fpc","tibble","dplyr","tidyverse","ggplot2","plotly",
      "gridExtra","rlist","writexl","readxl","tidyr","purrr","pracma",
      "openxlsx","pROC","ggplotify"
    )

    missing <- setdiff(required_pkgs, rownames(installed.packages()))
    if (length(missing)) install.packages(missing)
    invisible(lapply(required_pkgs, library, character.only = TRUE))

To load for development (dev-only):

    devtools::load_all()

For installed packages:

    library(DDESONN)

Note: `source()` is development-only and not recommended for installed packages.

High-level API usage (training split is always `x`/`y`):

    res <- ddesonn_run(
      x = train_x,
      y = train_y,
      validation = list(x = valid_x, y = valid_y),
      test = list(x = test_x, y = test_y),
      training_overrides = list(num_epochs = 1, validation_metrics = TRUE)
    )

Evaluation plot toggles (ROC/PR/accuracy) can be enabled via `training_overrides`.
The PR curve includes AUPRC by default; set `show_auprc = FALSE` to suppress:

    res <- ddesonn_run(
      x = train_x,
      y = train_y,
      classification_mode = "binary",
      seeds = 1,
      validation = list(x = valid_x, y = valid_y),
      test = list(x = test_x, y = test_y),
      training_overrides = list(
        validation_metrics = TRUE,
        evaluate_predictions_report_plots = list(
          roc_curve = TRUE,
          pr_curve = TRUE,
          accuracy_plot = TRUE,
          accuracy_plot_mode = "both",
          show_auprc = TRUE
        )
      )
    )

---

### Prediction APIs: internal vs public

Bottom line: **`ddesonn_predict()` = internal prediction engine (raw forward pass /
ensemble aggregation; used internally in training/validation and internal evaluation
paths).** **`predict.ddesonn_model()` / `predict()` = public, canonical user-facing API
that wraps `ddesonn_predict()` and standardizes arguments + output shape + optional
thresholding.**

Why: internal code uses `ddesonn_predict()` because it???s a forward-pass primitive
that???s faster and easier to control inside training loops (no user-facing return
formatting). User-facing inference should use `predict()` because it provides a
stable contract (type/aggregate/threshold handling, return structure).
Multiclass note: For multiclass classification, y should be encoded as integer class indices 1..K (or a one-hot matrix whose columns follow the model???s class order), otherwise accuracy comparisons may be incorrect.

When `test = list(x = test_x, y = test_y)` is provided, the final run summary
always includes test loss and test accuracy computed once after training
completes, and the values are available at `res$test_metrics$loss` and
`res$test_metrics$accuracy`. If you want to independently reproduce test
accuracy, call `predict(res$model, test_x)$predicted_output`, apply the same
threshold printed in the final summary, and compare element-wise to `test_y`
(`mean(as.integer(pred >= thr) == test_y)`), which should match the reported
test accuracy when thresholds, aggregation, and preprocessing are identical.

API design notes (optional explicit splits):

- ddesonn_run(x, y, validation = list(x = , y = ), test = list(x = , y = ),
  x_valid = , y_valid = , x_test = , y_test = )
- Explicit `x_valid`/`y_valid` and `x_test`/`y_test` override the list inputs.
- Explicit pairs must be complete (no `x_valid` without `y_valid`).
- Backward compatibility is preserved.
- Run history: `res$history` mirrors the training metadata (including best
  train/validation losses) and, when a test split is supplied, adds
  `test_loss` alongside `result$test_metrics`.

---

### Model usage note (post-training)

Training and validation run inside `ddesonn_run()` and call the model???s R6
methods directly.

**Evaluation contract (test data):**

- When `test$x`/`test$y` (or `x_test`/`y_test`) are supplied, `ddesonn_run()` is the
  authoritative source for **test loss and test accuracy**. These metrics are computed
  once after training completes, are stored at `res$test_metrics$loss` and
  `res$test_metrics$accuracy`, and are returned/printed as part of the final run summary.
- If you want to reproduce test accuracy manually, call `predict(res$model, x_test)`
  and compute accuracy as *(number of correct predictions ?? total rows)* via an
  element-wise comparison against `y_test` using the same threshold shown in the
  final summary (and the same aggregation and preprocessing).
- Given the **same threshold and preprocessing**, this manually computed accuracy
  **should match** the `ddesonn_run()` test accuracy. Any mismatch indicates a
  threshold or data-handling difference (not a model inconsistency).
- `ddesonn_run()` is for **evaluation**, while `predict()` is for **inspection,
  custom metrics, and downstream logic**???neither replaces the other.
- `ddesonn_run()` does **not** return per-row predictions; per-row outputs are
  provided by `predict()` only.

After training completes, the returned model (`res$model`) supports standard
R workflows via `predict(model, newdata)`. This is enabled by a lightweight
S3 adapter that forwards `predict()` calls to the underlying R6 `$predict()`
method.

Training behavior and final summary output are unchanged; this only
standardizes post-training usage.

Notes on aggregation + split reports:

- Aggregated predictions just reuse the existing `ddesonn_predict(..., aggregate = ...)` output for each split; no new aggregation behavior is added.
- Aggregation controls how multiple ensemble members are combined (e.g., mean/median vs none), and test metrics use the same default aggregation as predict() unless overridden.
- The binary split report helper is only for formatting Keras-style output (classification report + AUC/AUPRC + confusion matrix) in one place so Train/Validation/Test can print consistently without duplicating logic; core F1/ROC/precision/recall calculations already exist elsewhere.

---

## Running the examples

Ready-to-run demos are available under inst/scripts:

- DDESONN_mtcars_example.R
- DDESONN_mtcars_A-D_examples*.R
- Heart_failure_ScenarioA.R
- LoadandPredict.R
- TestDDESONN.R

Run directly:

    source("inst/scripts/DDESONN_mtcars_example.R")

Artifacts and plots are written under a user-writable data directory resolved by
ddesonn_artifacts_root() (with plots under ddesonn_plots_dir()), preserving
the same subfolder layout used previously under artifacts/.

---

## Datasets

Bundled sample data in data:

- heart_failure_clinical_records.csv
- WMT_1970-10-01_2025-03-15.csv
- train_multiclass_customer_segmentation.csv
- test_multiclass_customer_segmentation.csv

Verify original dataset licensing if repurposed.

---

## Roadmap & Design Intent

> **Note on scope and intent**  
> The items below describe **current behavior**, **explicit design intent**, and
> **forward-looking considerations**.  
> They are documented to clarify direction and preserve future ideas.  
> They do **not** imply active development or any committed delivery timeline.

### R-01 ?? Structured hyperparameter experimentation  
**Status:** Design intent (future)  
**Related To-Do:** T-01

Add structured hyperparameter grid and sweep utilities to support controlled,
reproducible experimentation across model configurations.

### R-02 ?? Optional preprocessing utilities  
**Status:** Design intent (future)  
**Related To-Do:** T-02

Introduce optional preprocessing helpers, including:

- Capped + `log1p` transforms for heavy-tailed features  
  (e.g., `creatinine_phosphokinase`)
- Zero-preserving behavior for interpretability and safety

### R-03 ?? Evaluation contract and thresholding semantics  
**Status:** Current behavior (documented)  
**Related To-Do:** T-03

The evaluation pipeline follows a strict and intentional thresholding contract:

- `evaluate_predictions_report.R` selects and applies a tuned threshold (`best_thr`)
  when generating thresholded predictions.
- `DDESONN.R` records a single authoritative threshold value (`thr_used`), which may
  be either the tuned threshold or a user-provided override.
- Confusion matrix utilities operate only on **already-thresholded** binary
  predictions and return **counts only**.
- Accuracy, precision, recall, and F1 are derived from confusion-matrix counts so
  all reported metrics consistently reflect `thr_used` (not a fixed 0.5 default).

### R-04 ?? Single-run per-epoch diagnostics  
**Status:** Forward-looking consideration  
**Related To-Do:** T-04

Potential future diagnostic capability to track training and validation metrics
across epochs for a **single model run**.

**Design constraints:**

- Strictly diagnostic (non-summary)
- Reuses existing artifact helpers:
  - `ddesonn_artifacts_root()`
  - `ddesonn_plots_dir()`
- Output path:  
  `{artifacts_root}/plots/single_run_per_epoch/`
- Explicitly excluded from `process_performance()` and all ensemble summaries

### R-05 ?? Single-run vs ensemble contract decoupling  
**Status:** Forward-looking consideration  
**Related To-Do:** T-05

In single-run mode, ensemble orchestration is disabled, but ensemble slot objects
(e.g., `ensemble[[k]]`) and metadata contracts remain in use.

Decoupling this behavior would require a non-trivial architectural refactor and is
documented here for clarity and future consideration.

### R-06 ?? `validation_metrics` scope and stabilization checkpoint  
**Status:** Current behavior (documented) + forward-looking consideration  
**Related To-Do:** T-06, T-07

`validation_metrics` gates the validation-only evaluation report pipeline, including
plots, confusion-matrix-derived metrics, artifact exports, and tuned-threshold handling.
Despite its name, it does not represent generic metric computation.

**Stabilization decision (v1):**

- `validation_metrics` is retained as a v1 stabilization switch controlling whether
  validation-based evaluation and reporting are executed.
- Training data is explicitly excluded from this pathway to prevent information
  leakage, optimistic bias, and invalid threshold selection.

**Design intent (future):**

- Separate **threshold tuning** from the broader evaluation report pipeline so tuned
  thresholds can be computed independently (lower cognitive load, fewer dependencies).
- Revisit `validation_metrics` semantics with explicitness (e.g., tri-state control:
  `off | validation | train`) only after the tuning logic is modularized.

### R-07 ?? `viewTables` behavior consolidation  
**Status:** Forward-looking consideration  
**Related To-Do:** T-08

`viewTables` is camelCase and currently exists in sparse areas of the codebase.
It is not yet guaranteed to be consistently honored across all reporting paths,
artifacts, `.rds` outputs, or table/tibble/data-frame display points.

The most reliable way to observe table display behavior in v1 is via the scripts in:
`inst/scripts/` ??? especially `TestDDESONN.R`.

Future work may unify table emission so `viewTables` behaves predictably across:
- console printing
- exported artifacts
- `.rds` summaries
- data frames / tibbles

### R-08 ?? Vignettes expansion and optional interactive diagnostics  
**Status:** Forward-looking consideration  
**Related To-Do:** T-09

The project already includes a major comparative vignette:
`vignettes/DDESONNvKeras_1000Seeds.Rmd` (Heart Failure, 1000-seed summary).

Future releases may expand the vignette suite (more datasets, more experiments,
more reproducible walkthroughs) and optionally explore interactive diagnostics
(e.g., Shiny) as a non-core layer.

### R-09 ?? Techila-scale experimentation patterns  
**Status:** Forward-looking consideration  
**Related To-Do:** T-10

Techila exists to scale heavy experiments across multiple servers/workers for seed
sweeps and larger runs. This is particularly valuable when you want hundreds to
thousands of seeds without waiting on a single machine.

---

## To-Do (Design-Linked)

### T-01 ?? Hyperparameter sweep utilities  
Linked from: **R-01**

Implement structured grid and sweep tooling with explicit configuration,
clear artifacts, and reproducibility guarantees.

### T-02 ?? Preprocessing utility formalization  
Linked from: **R-02**

Define a clean, opt-in preprocessing interface without implicit transformations
or side effects.

### T-03 ?? Threshold usage hardening  
Linked from: **R-03**

- Confirm `best_thr` selection remains localized to
  `evaluate_predictions_report.R`
- Ensure `thr_used` is the single source of truth in summaries and metadata
- Ensure all derived metrics are computed from confusion matrices reflecting
  `thr_used`

### T-04 ?? Per-epoch diagnostic tracking  
Linked from: **R-04**

Prototype per-epoch metric capture for single runs only, with no impact on
ensemble aggregation or performance summaries.

### T-05 ?? Ensemble contract decoupling analysis  
Linked from: **R-05**

Assess architectural implications of separating single-run execution from
ensemble metadata and orchestration contracts.

### T-06 ?? `validation_metrics` contract clarification (post-v1)  
Linked from: **R-06**

- Clearly define what `validation_metrics` enables/returns (evaluation report
  pipeline + artifacts + tuned-threshold support)
- Identify and document the call sites that currently depend on the flag
- Reduce ???hidden behavior??? and ensure the name matches the behavior contract

### T-07 ?? Extract threshold tuning into a standalone utility  
Linked from: **R-06**

- Pull tuned-threshold computation into a dedicated function that can run without
  the full evaluation report artifacts/exports
- Ensure the tuned threshold can be stored/returned consistently (e.g., per-model
  `chosen_threshold`) while keeping reporting optional
- After extraction, consider explicit tri-state evaluation routing:
  `off | validation | train` (or separate `evaluation_report` + `evaluation_data`)

### T-08 ?? `viewTables` standardization and coverage expansion  
Linked from: **R-07**

- Confirm where `viewTables` is currently honored and where it is ignored
- Decide what ???table viewing??? means across:
  console, data frames/tibbles, `.rds` tables, and report artifacts
- Consolidate table emission so `viewTables` behavior is predictable across the project

### T-09 ?? Expand vignettes and research demos  
Linked from: **R-08**

- Add additional polished vignettes for guided exploration (beyond `DDESONNvKeras_1000Seeds.Rmd`)
- Keep demos reproducible and artifact-backed
- Treat vignettes as the primary ???user education layer??? for v1+ releases

### T-10 ?? Techila distributed experimentation hardening  
Linked from: **R-09**

- Provide a clean, documented Techila workflow for scaling seed sweeps
- Make it easier to run heavy experiments across multiple servers with minimal setup friction

---

## Contributing

1. Fork and branch from main
2. Run demos to confirm no regressions
3. Submit pull requests with clear descriptions and tests

Contributions are highly appreciated ??? especially those focused on:
- polishing and tightening documentation
- improving vignettes and reproducible demos
- reporting/diagnostics improvements (tables, plots, artifacts)
- helping implement or refine items in the Roadmap & Design Intent / To-Do list

If you???re interested in helping push the project toward a cleaner plateau, the
Roadmap & To-Do sections are the best place to pick a meaningful contribution.

---

## License

DDESONN is released for personal, educational, and research use only.  
Commercial use requires written authorization.

---

## Contact

Mathew William Armitage Fok