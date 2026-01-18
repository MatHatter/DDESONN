````md
# DDESONN: Deep Dynamic Experimental Self-Organizing Neural Network

Mathew William Armitage Fok (<quiksilver67213@yahoo.com>)

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

---

## Project timeline

- 2024-05-07 — Project origin  
  The project formally began in May 2024 as a research initiative to design and implement a novel self-organizing neural network framework in R, prioritizing explicit training logic, architectural transparency, and experimental flexibility.

- Initial intensive sprint (approximately 3 months)  
  Sustained day-in/day-out development. Learning machine learning from first principles was unavoidable in order to design the architecture manually, reason through layer interactions and dimensional flow, identify bottlenecks, and resolve bugs by tracing logic across layers.

- Iterative development (lax / intermittent)  
  Development continued at a more sustainable pace, refining architectural decisions and expanding functionality while preserving full transparency and custom control.

- Second intensive hardening phase (approximately 3–4 months)  
  Focused on correctness, stability, optimizer behavior, ensemble reliability, and reproducibility.

- Late 2025 to early 2026 hiatus  
  Development paused in late 2025 while focusing on two other high-intensity projects.

- 2026 to present  
  Work resumed with emphasis on maintainability, documentation, and long-term research viability.

---

## Repository structure

"DDESONN/"
- "R/"
  - "DDESONN.R"
  - "activation_functions.R"
  - "api.R"
  - "optimizers.R"
  - "performance_relevance_metrics.R"
  - "update_biases_block.R"
  - "update_weights_block.R"
  - "utils.R"
  - "reports/"
    - "evaluate_predictions_report.R"

- "inst/"
  - "scripts/"
    - "DDESONN_mtcars_example.R"
    - "DDESONN_mtcars_A-D_examples*.R"
    - "Heart_failure_ScenarioA.R"
    - "LoadandPredict.R"
    - "TestDDESONN.R"

- "data/"
- "vignettes/"
- "helpfulFiles/"
- "ideas/"
- "junk/"

- "DESCRIPTION"
- "NAMESPACE"
- "DDESONN.Rproj"
- "CHANGELOG.md"
- "README.md"
- "README_v*.md"
- "LICENSE/"

---

## Getting started

### Prerequisites

- R version 4.1 or higher
- RStudio project file included (DDESONN.Rproj)
- Dependencies listed in DESCRIPTION

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

API design notes (optional explicit splits):

- ddesonn_run(x, y, validation = list(x = , y = ), test = list(x = , y = ),
  x_valid = , y_valid = , x_test = , y_test = )
- Explicit `x_valid`/`y_valid` and `x_test`/`y_test` override the list inputs.
- Explicit pairs must be complete (no `x_valid` without `y_valid`).
- Backward compatibility is preserved.


### Model usage note (post-training)

Training and validation run inside `ddesonn_run()` and call the model’s R6
methods directly.

After training completes, the returned model (`res$model`) supports standard
R workflows via `predict(model, newdata)`. This is enabled by a lightweight
S3 adapter that forwards `predict()` calls to the underlying R6 `$predict()`
method.

Training behavior and final summary output are unchanged; this only
standardizes post-training usage.


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

## Roadmap

- Add structured hyperparameter grid and sweep utilities for controlled experimentation.

- Optional preprocessing utilities:
  - Capped + `log1p` transforms for heavy-tailed features (e.g., `creatinine_phosphokinase`)
  - Designed to reduce extreme outlier influence while preserving zeros.

- Evaluation contract / thresholding (documentation + hardening):
  - `evaluate_predictions_report.R` selects and applies a tuned threshold (`best_thr`) when generating
    thresholded predictions and the corresponding confusion matrix.
  - The final package-level summary/metadata in `DDESONN.R` uses `thr_used` as the authoritative
    threshold value that is recorded and reported (this may be `best_thr` or a user override).
  - Confusion matrix utilities operate on **already-thresholded** (binary) predictions and return counts only.
  - Accuracy/precision/recall/F1 are computed from confusion-matrix counts so reported metrics + metadata
    reflect `thr_used` (not a fixed 0.5 threshold) without introducing new metric helpers.

- Future diagnostic (not yet implemented): **Single-run per-epoch performance tracking**
  - Track training and validation metrics across epochs for a single model run.
  - Intended strictly for diagnostics (learning curves, overfitting detection, instability analysis).
  - Will reuse existing artifact and plot helpers:
    - `ddesonn_artifacts_root()`
    - `ddesonn_plots_dir()`
  - Output path:
    ```
    <artifacts_root>/plots/single_run_per_epoch/
    ```
  - Explicitly excluded from `process_performance()` and all ensemble summaries.

- Potential future change (non-trivial refactor):
  - In single-run mode, ensemble orchestration is disabled, but ensemble slot objects
    (`ensemble[[k]]`) and `Ensemble_Main_0_model_*_metadata` are still used.
  - Decoupling these contracts would require a major architectural change and may be revisited later.

---

## To-Do (Active Work)

- Refactor `DDESONN_predict_eval()` so all required variables are passed explicitly and handled locally,
  avoiding reliance on global or inherited environments.

- Review evaluation data mapping for thresholds (`best_thr` vs `thr_used`):
  - Confirm `best_thr` is selected/applied only within `evaluate_predictions_report.R`.
  - Confirm `thr_used` is the single source of truth for what is stored/reported in final summaries in `DDESONN.R`
    (including when overrides are used).
  - Ensure all derived metrics (accuracy/precision/recall/F1) are computed from confusion matrices that reflect
    `thr_used` (not a fixed 0.5 threshold).

- Continue decompartmentalization to slim the core codebase:
  - Initial step: move the SONN method into its own dedicated file.
  - Preserve the primary `DDESONN` R6 class in `R/DDESONN.R`.


## Contributing

1. Fork and branch from main
2. Run demos to confirm no regressions
3. Submit pull requests with clear descriptions and tests

---

## License

DDESONN is released for personal, educational, and research use only.  
Commercial use requires written authorization.

---

## Contact

Mathew William Armitage Fok
````
