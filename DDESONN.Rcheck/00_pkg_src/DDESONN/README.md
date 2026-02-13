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
2. Core capabilities
3. Architecture
4. Project timeline
5. Repository structure
6. Getting started
7. Running the examples
8. Datasets
9. Reproducibility
10. Roadmap
11. To-Do (Active Work)
12. Contributing
13. License
14. Other work by the author
15. Contact

---

## Project overview

DDESONN - Deep Dynamic Experimental Self-Organizing Neural Network - is an R-based research framework for adaptive neural network experimentation.

The project was initiated to build a fully custom neural network system that did not already exist, and to develop a deep, first-principles understanding of machine learning by necessity rather than by copying existing frameworks.

DDESONN blends self-organizing principles with modern deep-learning practices to support:

- Configurable single-layer or multi-layer architectures
- Dynamic ensemble learning with pruning and add-back mechanisms
- Full control of optimizer, regularization, and activation flows
- Reproducible evaluation and artifact reporting

The primary design objective of DDESONN is to provide a fully customizable, entirely R-native neural network codebase and framework, intentionally avoiding external deep-learning backend library dependencies to preserve full architectural control and transparency.

#### What DDESONN is

DDESONN is a fully native R framework for constructing, training, evaluating,
and inspecting Deep Dynamic Ensemble Self-Organizing Neural Networks.

The package is designed for users who need direct control over model
architecture, optimization behavior, and training workflow details rather than
black-box abstractions. It exposes both high-level helpers and inspectable
low-level behavior for reproducible neural-network experimentation in R.

---

## Why DDESONN exists and why I built it this way

DDESONN exists because I wanted to understand machine learning at a deeper level than "use a library and hope it works."

I didn't want a neural network that was hidden behind abstractions. I wanted a neural network that people could actually look into layer by layer, error by error, update by update and see exactly what's happening. Most modern frameworks make it easy to train a model, but they also make it easy to never truly understand what the model is doing internally.

So I built DDESONN to be **inspectable**, **transparent**, and **architecturally explicit** and I intentionally avoided relying on external neural network or machine learning libraries. That wasn't because I couldn't use them. It was because I wanted to build the full machinery end-to-end and learn what "correct implementation" actually means.

### The honest story (trials, tribulations, and why it matters)

This package took an extreme amount of time and emotional energy to build.

There were long stretches where I thought it was correct, but still didn't fully trust it. And that uncertainty is hard because when you're building the full architecture from scratch, bugs aren't obvious. They can hide inside dimension handling, layer wiring, activation derivatives, error propagation, weight updates, and edge cases that only appear under certain random seeds or training paths.

I nearly gave up twice.

What kept me going was the belief that I was on the right track - even when the results didn't always look right. In a weird way, life events kept pulling me back onto this path. Every time I stepped away, I came back with more clarity. And every time I came back, I pushed the implementation closer to what it should be.

As I went deeper, it honestly got scarier - because there were moments where DDESONN looked better than the benchmark models, and other moments where it didn't. That inconsistency can mess with your head when you've invested everything into building it correctly.

But the turning point wasn't "one magic upgrade." It was the final phase of **clearing out the bugs** and **aligning the implementation to the mathematically correct behavior**. Once those last issues were resolved, the model became dramatically more stable.

### What "better" means here

When I say "better," I don't mean one cherry-picked run.

I mean repeated evaluation across large numbers of randomized initializations (seeds). In my testing, once the final correctness issues were resolved, DDESONN produced results that were:

- **more stable on average**
- with **lower standard deviation**
- and **less extreme worst-case error**
  across large seed sweeps (e.g., 1,000 seeds)

At that point, it stopped feeling like "maybe this works" and started feeling like "this is now a stable, correct implementation that competes."

### Transparency is the point

DDESONN is built to show you what it's doing.

Even in low-verbosity mode, it exposes the key structural diagnostics (layer dimensions, activation choices, error shapes, and sanity checks). High-verbosity mode expands that into full step-by-step tracing when you're debugging or studying behavior.

This is not just a model it's an implementation we can learn from.

---

## Logging / Verbosity levels

DDESONN supports two tiers of debug output

- **Low verbosity (default):** prints only the essential "trust" diagnostics (layer-by-layer)
  - layer dimensions per layer
  - activation name per layer
  - predicted output shape
  - error shape / key scalar metrics per layer
  - any shape/NA/overflow guards and recovery actions
- **High verbosity:** prints full tracing for debugging and research
  - per-layer forward values (summary stats)
  - per-layer backprop error stats
  - gradient shape + update sanity checks
  - detailed corrective actions when dimension alignment occurs

This design ensures that even low verbosity is still inspectable and scientifically meaningful, while high verbosity remains available for deep debugging.

---

## How to implement low verbose but still layer-by-layer without spamming

Use two principles:

1. **Print structure once per run** (or once per epoch if something changes)
2. **Print per-layer summaries, not full matrices** (dims + min/max/mean + NA counts)

Example template (R-style pseudo you can adapt):

- Always print: `dims`, `activation`, `dropout`, `label/pred dims`
- For errors: print `dim(error)`, `mean(abs(error))`, `max(abs(error))`
- For weights/gradients: print only `dim()` + `max(abs())` if high verbose

---

## Core Capabilities

- Fully native R deep learning framework — no external deep-learning backend.
- Object-oriented model engine implemented with R6.
- Flexible architecture selection (single-layer or deep multi-layer) with dimension-agnostic configuration.
- Independent activation functions, derivatives, dropout, and initialization per layer.
- Manual training loop with explicit forward and backward propagation.
- Transparent optimizer-state updates with full internal control.

#### Optimization & Regularization

- Optimizers implemented from scratch:
  - SGD  
  - RMSProp  
  - Adam  
  - Lookahead  
- Separate weight and bias update logic in dedicated update blocks.
- L1, L2, and mixed regularization for both weights and biases.
- Optional learning-rate scheduling via training overrides.
- User-controllable self-organization toggle (`self_org`) through:
  - `ddesonn_fit()`
  - `ddesonn_run(training_overrides = ...)`

#### Evaluation & Threshold Intelligence

- Automatic F1-optimized threshold tuning.
- Precision and recall scoring.
- ROC and Precision-Recall curve generation.
- AUC and AUPRC computation.
- Relevance tracking and custom performance metrics.

#### Ensemble & Orchestration

- Dynamic ensemble orchestration.
- Primary and temporary model promotion.
- Metadata tracking and structured relevance scoring.

#### Reporting & Integration

- Excel export and structured reporting via:
  - `writexl`
  - `openxlsx`
- Static and interactive visualization with:
  - `ggplot2`
  - `plotly`
- High-level API helpers in `R/api.R` for external integration.
- Artifact path management and debug-state utilities for reproducibility.

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
Use it optionally by guarding calls, for example: `if (requireNamespace("techila", quietly = TRUE)) { ... } else { ... }`.
This becomes relevant quickly when you start running large seed sweeps (e.g., hundreds to thousands of seeds across hundreds of epochs).

---

## Project timeline

DDESONN began as an exploratory research project and progressed through several architectural checkpoints as core ideas were validated and refined.

Subsequent iterations focused on formalizing the architecture, improving reproducibility, and restructuring the codebase to meet CRAN packaging standards.

- 2024-05-07 - Project origin  
  The project formally began in May 2024 as a personal research initiative to design and implement a novel self-organizing neural network framework in R, prioritizing explicit training logic, architectural transparency, and experimental flexibility.

- Initial intensive sprint (approximately 3 months)  
  Sustained day-in/day-out development. Learning machine learning from first principles was unavoidable in order to design the architecture manually, reason through layer interactions and dimensional flow, identify bottlenecks, and resolve bugs by tracing logic across layers.

- Iterative development (lax / intermittent)  
  Development continued at a more sustainable pace, refining architectural decisions and expanding functionality while preserving full transparency and custom control.

- May 2025 had to relearn and pick back up. Second intensive hardening phase (approximately 3-4 months)  
  Focused on correctness, stability, optimizer behavior, ensemble reliability, and reproducibility.

- End of Sept. 2025 
  Development paused in late 2025 while focusing on two other high-intensity projects.

- Jan. 2026 - Feb. 2026 (approximately 1.5 months) 
  Work resumed with emphasis on maintainability, documentation, and long-term research viability.

Earlier checkpoint versions and legacy research code may be published separately in a dedicated archival repository to document the project???s evolution, including early snapshots where some components were not fully retained.

---

## Repository structure

DDESONN/
├── R/
│   ├── DDESONN-package.R
│   ├── DDESONN.R
│   ├── activation_functions.R
│   ├── aliases.R
│   ├── api.R
│   ├── evaluate_predictions_report.R
│   ├── optimizers.R
│   ├── paths.R
│   ├── performance_relevance_metrics.R
│   ├── predict.R
│   ├── state.R
│   ├── update_biases_block.R
│   ├── update_weights_block.R
│   └── utils.R
│
├── inst/
│   ├── dev/
│   │   └── README.Rmd
│   ├── extdata/
│   │   ├── WMT_1970-10-01_2025-03-15.csv
│   │   ├── heart_failure_clinical_records.csv
│   │   ├── test_multiclass_customer_segmentation.csv
│   │   └── train_multiclass_customer_segmentation.csv
│   └── scripts/
│       ├── DDESONN_mtcars_A-D_examples.R
│       ├── DDESONN_mtcars_A-D_examples_regression.R
│       ├── Heart_failure_ScenarioA.R
│       ├── LoadandPredict.R
│       └── TestDDESONN.R
│
├── man/
├── vignettes/
├── DESCRIPTION
├── NAMESPACE
├── DDESONN.Rproj
├── README.md
├── LICENSE
└── LICENSE.md

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
      training_overrides = list(
        num_epochs = 1,
        validation_metrics = TRUE,
        self_org = FALSE  # set TRUE to enable self-organization
      )
    )

#### Which function should I use?

If `ddesonn_run()` already works for you, you're not doing anything wrong. It is the
"all-in-one" orchestrator and is the best default for most users.

Use this quick guide:

- **`ddesonn_run()`**: one-call workflow for train/validation/test orchestration,
  seed loops, optional ensemble scenarios, and summary outputs. Best for
  experiments and benchmark runs.
- **`ddesonn_model()`**: construct a model object only (architecture/setup stage).
  Use when you want explicit control before training.
- **`ddesonn_fit()`**: train an already-created model. Use when you want a
  custom loop, staged training, or fine-grained control over train calls.
- **`predict()` / `predict.ddesonn_model()`**: user-facing inference on new data
  after training.
- **`ddesonn_predict()`**: internal low-level prediction engine. Useful for
  package internals and advanced users, but most users should prefer `predict()`.
- **`ddesonn_training_defaults()`**: inspect the baseline training parameters used
  by wrappers.
- **`ddesonn_activation_defaults()` / `ddesonn_dropout_defaults()` /
  `ddesonn_optimizer_options()`**: helper utilities to inspect or build settings.

In short: think of `ddesonn_run()` as the convenient "driver", while the other
functions are modular building blocks that make the driver customizable,
testable, and reusable in advanced workflows.

Typical progression:

1. Start with `ddesonn_run()`.
2. Move to `ddesonn_model()` + `ddesonn_fit()` when you need custom training flow.
3. Use `predict()` for downstream inference and reporting.

Self-organization toggle (public API):

- In `ddesonn_fit()`, pass `self_org = TRUE` (or `FALSE`) directly.
- In `ddesonn_run()`, pass `training_overrides = list(self_org = TRUE)` (or `FALSE`).
- Default is OFF (`self_org = FALSE`) unless you explicitly enable it.

`self_organize()` is an unsupervised topology-adjustment phase that updates the
network using input-space neighborhood/organization error rather than
prediction-target residual error. In other words, it optimizes topographical
structure of the representation (input manifold organization), not the direct
supervised prediction-loss objective.

In exploratory experiments, enabling it may have positive implications for
topographical-analysis accuracy on some datasets/workflows, so it is useful to
benchmark both settings.


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

Why: internal code uses `ddesonn_predict()` because it-?s a forward-pass primitive
that-?s faster and easier to control inside training loops (no user-facing return
formatting). User-facing inference should use `predict()` because it provides a
stable contract (type/aggregate/threshold handling, return structure).

Multiclass note: For multiclass classification, y should be encoded as integer class indices 1..K (or a one-hot matrix whose columns follow the model-?s class order), otherwise accuracy comparisons may be incorrect.

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

Training and validation run inside `ddesonn_run()` and call the model-?s R6
methods directly.

**Evaluation contract (test data):**

- When `test$x`/`test$y` (or `x_test`/`y_test`) are supplied, `ddesonn_run()` is the
  authoritative source for **test loss and test accuracy**. These metrics are computed
  once after training completes, are stored at `res$test_metrics$loss` and
  `res$test_metrics$accuracy`, and are returned/printed as part of the final run summary.
- If you want to reproduce test accuracy manually, call `predict(res$model, x_test)`
  and compute accuracy as *(number of correct predictions - total rows)* via an
  element-wise comparison against `y_test` using the same threshold shown in the
  final summary (and the same aggregation and preprocessing).
- Given the **same threshold and preprocessing**, this manually computed accuracy
  **should match** the `ddesonn_run()` test accuracy. Any mismatch indicates a
  threshold or data-handling difference (not a model inconsistency).
- `ddesonn_run()` is for **evaluation**, while `predict()` is for **inspection,
  custom metrics, and downstream logic**-?neither replaces the other.
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

## Reproducibility

DDESONN supports reproducible experimentation through:

- Deterministic seed control (`set.seed(...)` and `seeds = ...` in `ddesonn_run()`)
- Explicit training defaults via `ddesonn_training_defaults()`
- Scriptable scenarios under `inst/scripts/`
- Vignettes for reproducible walkthroughs
- Artifact-root control via:
  - `ddesonn_artifacts_root(output_root = ...)`
  - `Sys.getenv("DDESONN_ARTIFACTS_ROOT")`
  - `options(DDESONN_OUTPUT_ROOT = ...)`
- Plot directory resolution via `ddesonn_plots_dir()`
- Debug inspection via `ddesonn_debug_state()`

These controls allow experiments to be rerun deterministically, inspected at multiple verbosity levels, and reproduced across systems without hidden state.

#### Vignettes

Start with these vignettes in `vignettes/`:

- `plot-controls_scenario1-2_single-run_scenarioA.Rmd`
- `plot-contols_scenario1_ensemble-runs_scenarioC-D.Rmd`
- `logs_scenarioD_ensemble_runs_temp_iterations.Rmd`
- `DDESONNvKeras_1000Seeds.Rmd`

These cover:

- Single-run flows  
- Ensemble scenarios  
- Logging and diagnostic analysis  
- Benchmark-oriented multi-seed reproducibility experiments  

#### Reproducibility Artifacts for 1000 Seeds Vignette

DDESONN includes precomputed `.rds` files under:

`inst/extdata/`

These files contain saved model outputs, metrics, and summaries used specifically for the `DDESONNvKeras_1000Seeds.Rmd` vignette to:

- Demonstrate large multi-seed experiments (1,000 randomized initializations)
- Avoid long runtimes during vignette builds
- Ensure deterministic, reproducible benchmark comparisons

These artifacts are:

- Not loaded automatically  
- Not part of the public API  
- Not intended for direct use outside the associated vignette  

They are provided solely to support reproducibility and documentation.

---

## Roadmap & Design Intent

> **Note on scope and intent**  
> The items below describe **current behavior**, **explicit design intent**, and
> **forward-looking considerations**.  
> They are documented to clarify direction and preserve future ideas.  
> They do **not** imply active development or any committed delivery timeline.

#### R-01 - Structured hyperparameter experimentation  
**Status:** Design intent (future)  
**Related To-Do:** T-01

Add structured hyperparameter grid and sweep utilities to support controlled,
reproducible experimentation across model configurations.

#### R-02 - Optional preprocessing utilities  
**Status:** Design intent (future)  
**Related To-Do:** T-02

Introduce optional preprocessing helpers, including:

- Capped + `log1p` transforms for heavy-tailed features  
  (e.g., `creatinine_phosphokinase`)
- Zero-preserving behavior for interpretability and safety

#### R-03 - Evaluation contract and thresholding semantics  
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

#### R-04 - Single-run per-epoch diagnostics  
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

#### R-05 - Single-run vs ensemble contract decoupling  
**Status:** Forward-looking consideration  
**Related To-Do:** T-05

In single-run mode, ensemble orchestration is disabled, but ensemble slot objects
(e.g., `ensemble[[k]]`) and metadata contracts remain in use.

Decoupling this behavior would require a non-trivial architectural refactor and is
documented here for clarity and future consideration.

#### R-06 - `validation_metrics` scope and stabilization checkpoint  
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

#### R-07 - `viewTables` table-emission standardization
**Status:** Partially implemented (v1) + scoped forward-looking refinement
**Related To-Do:** T-08

`viewTables` is now supported as an explicit, per-run handler and is routed
through a centralized table-emission helper (ddesonn_viewTables()).


As of the current implementation:
- `viewTables` can be passed explicitly to `ddesonn_run()` / `ddesonn_fit()`.
- Table-like outputs from:
  - final run summaries
  - Core Metrics: Final Summary: binary classification reports (classification report + confusion matrix)
  - evaluation reports (EvaluatePredictionsReport)
  - model selection helpers (e.g., find_best_model())
  - aggregation / fusion debug previews
  - selected prediction-evaluation debug paths are routed through ddesonn_viewTables()
- A legacy fallback lookup (get0("viewTables", inherits = TRUE)) is preserved for
backward compatibility when no explicit handler is supplied
- A run-level warning guard prevents repeated warnings when invalid handlers are passed

This establishes a top-level, consistent table-display contract for the most
visible and user-facing reporting paths, without breaking existing workflows.

Remaining work (documented, not urgent) involves auditing low-visibility or
rarely executed debug paths to ensure all table-like emissions route through
the same helper.

#### R-08 - Vignettes expansion and optional interactive diagnostics  
**Status:** Forward-looking consideration  
**Related To-Do:** T-09

The project already includes a major comparative vignette:
`vignettes/DDESONNvKeras_1000Seeds.Rmd` (Heart Failure, 1000-seed summary).

Future releases may expand the vignette suite (more datasets, more experiments,
more reproducible walkthroughs) and optionally explore interactive diagnostics
(e.g., Shiny) as a non-core layer.

#### R-09 - Techila-scale experimentation patterns  
**Status:** Forward-looking consideration  
**Related To-Do:** T-10

Techila exists to scale heavy experiments across multiple servers/workers for seed
sweeps and larger runs. This is particularly valuable when you want hundreds to
thousands of seeds without waiting on a single machine.

#### R-10 - Cross-language reference implementations  
**Status:** Forward-looking consideration  

Future releases may explore reference implementations of the DDESONN architecture in other programming languages (e.g., Python, MATLAB, C#, C++).  

The goal would not be to wrap existing deep-learning libraries, but to preserve the same architectural transparency and explicit training logic across languages.

---

## To-Do (Design-Linked)

#### T-01 - Hyperparameter sweep utilities  
Linked from: **R-01**

Implement structured grid and sweep tooling with explicit configuration,
clear artifacts, and reproducibility guarantees.

#### T-02 - Preprocessing utility formalization  
Linked from: **R-02**

Define a clean, opt-in preprocessing interface without implicit transformations
or side effects.

#### T-03 - Threshold usage hardening  
Linked from: **R-03**

- Confirm `best_thr` selection remains localized to
  `evaluate_predictions_report.R`
- Ensure `thr_used` is the single source of truth in summaries and metadata
- Ensure all derived metrics are computed from confusion matrices reflecting
  `thr_used`

#### T-04 - Per-epoch diagnostic tracking  
Linked from: **R-04**

Prototype per-epoch metric capture for single runs only, with no impact on
ensemble aggregation or performance summaries.

#### T-05 - Ensemble contract decoupling analysis  
Linked from: **R-05**

Assess architectural implications of separating single-run execution from
ensemble metadata and orchestration contracts.

#### T-06 - `validation_metrics` contract clarification (post-v1)  
Linked from: **R-06**

- Clearly define what `validation_metrics` enables/returns (evaluation report
  pipeline + artifacts + tuned-threshold support)
- Identify and document the call sites that currently depend on the flag
- Reduce hidden behavior and ensure the name matches the behavior contract

#### T-07 - Extract threshold tuning into a standalone utility  
Linked from: **R-06**

- Pull tuned-threshold computation into a dedicated function that can run without
  the full evaluation report artifacts/exports
- Ensure the tuned threshold can be stored/returned consistently (e.g., per-model
  `chosen_threshold`) while keeping reporting optional
- After extraction, consider explicit tri-state evaluation routing:
  `off | validation | train` (or separate `evaluation_report` + `evaluation_data`)

#### T-08 - `viewTables` coverage audit and completion pass
Linked from: **R-07**

- Perform a repository-wide audit for remaining direct `print()`, `View()`,
`head()`, or table-rendering calls on data frames/tibbles in reporting,
evaluation, or debug paths
- Route any remaining table-like output through `ddesonn_viewTables()` or
`emit_table()` (which delegates to it)

- Confirm `viewTables` behavior is consistent across:
  - console output
  - evaluation summaries
  - debug preview
- Keep changes minimal and non-breaking; this task is strictly a coverage and
consistency sweep, not a redesign

#### T-09 - Expand vignettes and research demos  
Linked from: **R-08**

- Add additional polished vignettes for guided exploration (beyond `DDESONNvKeras_1000Seeds.Rmd`)
- Keep demos reproducible and artifact-backed
- Treat vignettes as the primary -?user education layerfor v1+ releases

#### T-10 - Techila distributed experimentation hardening  
Linked from: **R-09**

- Provide a clean, documented Techila workflow for scaling seed sweeps
- Make it easier to run heavy experiments across multiple servers with minimal setup friction

#### T-11 - Cross-language feasibility assessment  
Linked from: **R-10**

Evaluate architectural portability and determine minimal core components required for a language-agnostic implementation.

---

## Contributing

Contributions are welcome and appreciated.

### Workflow

1. Fork the repository and branch from `main`.
2. Run existing demos and examples to confirm no regressions.
3. Submit a pull request with a clear description and, where applicable, tests or reproducible examples.

### For Substantive Changes

If your pull request introduces behavioral changes, architectural adjustments, or new functionality, please include:

- A clear problem statement  
- Reproducible scripts or minimal examples  
- Notes describing expected behavior vs. observed behavior  
- Any relevant performance or diagnostic output  

This helps ensure that changes remain scientifically traceable and consistent with the design philosophy of DDESONN.

### Areas Where Help Is Especially Valuable

Contributions are particularly appreciated in areas such as:

- Polishing and tightening documentation  
- Improving vignettes and reproducible demos  
- Reporting and diagnostics enhancements (tables, plots, artifacts)  
- Implementing or refining items in the Roadmap & Design Intent / To-Do list  

If you're interested in helping move the project toward a cleaner and more stable plateau, the Roadmap & To-Do sections are the best place to identify meaningful contribution opportunities.

---

## License

DDESONN is released for personal, educational, and research use only.  
Commercial use requires written authorization.

---

## Other work by the author

The author also maintains additional modeling projects in R and Python, including:

- **OLR - Optimal Linear Regression**  
  CRAN: [olr on CRAN](https://cran.r-project.org/web/packages/olr/index.html)

---

## Contact

If you found DDESONN useful, interesting, or thought-provoking, feel free to connect with me on **LinkedIn**.

If you send a connection request, please include a short note mentioning DDESONN so I know where you found it. I read those messages.

Questions about the architecture, implementation details, or research design are welcome. I’m happy to respond when I can.

**Mathew William Armitage Fok**