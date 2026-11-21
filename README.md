# DDESONN: Deep Dynamic Experimental Self-Organizing Neural Network

Mathew William Armitage Fok (<quiksilver67213@yahoo.com>)

Learning notes, startup material and DDESONN modeling updates:
[Follow on Substack](https://substack.com/@alphalion).

See [changes since the latest CRAN release (7.1.11)](dev/release-summary-since-7.1.11.md)
for a summary of the updates available on GitHub.

**Research highlight: A win for DDESONN.** Its mean held-out accuracy was higher
than Keras's in a matched comparison across 100 seeds on one clean split. With summed gradients,
mean accuracy was **91.36% vs. 87.48%**. With averaged gradients, it was
**90.39% vs. 87.62%**.
[Read the 100-seed benchmark](https://github.com/MatHatter/DDESONN/blob/main/vignettes/clean-matched-100-seeds.Rmd).

Both files report the same held-out accuracy for all 400 seed-and-condition
combinations in the clean benchmark.

- [Benchmark results (CSV)](inst/shiny/benchmark/data/100SeedsDDESONNvsKerasCleanSplit.csv) lists the per-seed scores in a table.
- [Benchmark results (RDS)](dev/research/keras-clean-reference/100SeedsDDESONNvsKerasCleanSplit.rds) includes the same scores plus run details. Open it in R with `readRDS()`.

In the earlier [1,000-seed case study](https://github.com/MatHatter/DDESONN/blob/main/vignettes/heart-failure-standard-adagrad-1000-seeds.Rmd),
the archived results show that DDESONN averaged **99.92% test accuracy vs. 99.68%
for Keras**, with less variation across seeds. The results are useful as a
historical side-by-side comparison on the original split. Because **738 of 750
test rows** had inputs identical to training rows, the test scores are
unreliable estimates of either model's accuracy on unseen inputs. That overlap
motivated the 100-seed follow-up, which evaluated both frameworks on the same
split with no exact input matches across training, validation, and test sets. See the
[seed-count rationale](https://github.com/MatHatter/DDESONN/blob/main/dev/research/keras-clean-reference/SEED_COUNT_RATIONALE.md)
for why we reported 100 seeds per condition and how to choose a seed count for
future modeling experiments, tests, and case studies in DDESONN or other data
science projects.

The older 1,000-seed study is kept as a historical record. Its archived RDS
metrics files are [available on GitHub](https://github.com/MatHatter/DDESONN/tree/main/inst/extdata/heart_failure_runs)
and are excluded from the package to keep the download smaller.

## Recommended Shiny GUI: launch and first steps

DDESONN includes a Shiny graphical user interface (GUI) for preparing data,
training and evaluating models, using saved models, and browsing research results.
Install DDESONN and its optional workbench dependencies, then launch the app
from an interactive R session:

For dependency updates on Windows, close other R/RStudio sessions and install
from a fresh R session before loading development tools or starting the app.
This lets Windows replace package DLLs that would otherwise be in use.

```r
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("MatHatter/DDESONN")
install.packages(c("shiny", "callr"))
shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN",
                          mustWork = TRUE))
```

- **Train a new model:** start with Iris on **1. Data**, review the target,
  predictors and preprocessing on **2. Features**, click **Prepare data**, then
  configure and train on **3. Build & Train**. Review scores and plots on
  **4. Model / Results**, and export your model bundle when ready.
- **Use a saved model:** load a workbench `.rds` bundle and your CSV, then choose
  holdout evaluation (known answers) or production predictions (unknown answers).
- **Experiments:** browse saved results without launching training. **Current**
  shows the clean matched DDESONN/Keras study, with 100 seeds per condition and
  identical predictor inputs kept in separate partitions. It uses an experimental
  native float32 adapter on one fixed split. **Historical** pairs 1,000
  standard-AdaGrad DDESONN runs with archived legacy DDESONN observations and
  retains overlapping training/test inputs; Keras was not rerun for that view.
- **Help and R code:** use the information icons, **Vignettes**, and
  **Show full R code** for explanations and the corresponding workflow script.

Shiny handles its plots automatically. For a short installed guide, run:

```r
vignette("packaged-data-workbench", package = "DDESONN")
help("DDESONN", package = "DDESONN")
```

Stop the app with the RStudio Stop button or Ctrl+C in an R terminal.
See the [workbench guide](inst/shiny/benchmark/README.md) for more detail and
[NEWS](NEWS.md) for release changes.

<img src="https://raw.githubusercontent.com/MatHatter/ddesonn-assets/main/DDESONN_brain.png" width="800">

---

## Table of contents
1. Project links
2. Project overview
3. Core capabilities
4. Advanced Customization
5. Architecture
6. Project timeline
7. Repository structure
8. Getting started
9. Run terminology
10. Running the examples
11. Datasets
12. Reproducibility
13. Roadmap
14. To-Do
15. Contributing
16. License
17. Other work by the author
18. Contact

---

## Project links

[![GitHub Repo](https://img.shields.io/badge/GitHub-DDESONN-blue?logo=github)](https://github.com/MatHatter/DDESONN)

- **Source code:** [https://github.com/MatHatter/DDESONN](https://github.com/MatHatter/DDESONN)
- **Issue tracker:** [https://github.com/MatHatter/DDESONN/issues](https://github.com/MatHatter/DDESONN/issues)

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

### What DDESONN is

DDESONN is a fully native R framework for constructing, training, evaluating,
and inspecting Deep Dynamic Ensemble Self-Organizing Neural Networks.

The package is designed for users who need direct control over model
architecture, optimization behavior, and training workflow details rather than
black-box abstractions. It exposes both high-level helpers and inspectable
low-level behavior for reproducible neural-network experimentation in R.

### Native Implementation (No External Deep Learning Backends)

DDESONN is implemented entirely in R and does **not** rely on external
deep-learning computational backends (e.g., TensorFlow, Torch, or compiled
GPU runtimes). All forward propagation, backpropagation, optimizer state
updates, and ensemble orchestration are handled directly within the R codebase.

This design choice ensures:

- Full inspectability of internal model state
- Transparent dimensional flow and gradient behavior
- Reproducible numerical execution without hidden backend logic
- Architectural control at the layer, optimizer, and update-block level

The framework is intentionally explicit rather than abstracted behind
external engine calls.

---

## Why DDESONN exists and why I built it this way

DDESONN exists because I wanted to understand machine learning at a deeper level than "use a library and hope it works."

Neural networks first fascinated me during an Advanced Time Series Analysis course (before the current wave of AI hype), where I began to appreciate the mathematical structure behind prediction, stability, and model evaluation, and I knew early that understanding these systems deeply—not just using them—would matter long-term. I also remember telling classmates in a business science course that I aspired to publish another package beyond OLR - Optimal Linear Regression, and that commitment quietly stayed with me, eventually evolving into what became DDESONN.

I didn't want a neural network that was hidden behind abstractions. I wanted a neural network that people could actually look into layer by layer, error by error, update by update and see exactly what's happening. Most modern frameworks make it easy to train a model, but they also make it easy to never truly understand what the model is doing internally.

So I built DDESONN to be **inspectable**, **transparent**, and **architecturally explicit**, and I intentionally avoided relying on external neural network or machine learning libraries. That wasn't because I couldn't use them. It was because I wanted to build the full machinery end-to-end and learn what "correct implementation" actually means.

### The honest story (trials, tribulations, and why it matters)

This package took an extreme amount of time and emotional energy to build.

There were long stretches where I thought it was correct, but still didn't fully trust it, and that uncertainty is hard because when you're building the full architecture from the ground up, bugs aren't obvious. They can hide inside dimension handling, layer wiring, activation derivatives, error propagation, weight updates, and edge cases that only appear under certain random seeds or training paths.

I nearly gave up twice.

What kept me going was the belief that I was on the right track—even when the results didn't always look right. In a strange way, life events kept pulling me back onto this path. Every time I stepped away, I came back with more clarity, and every time I came back, I pushed the implementation closer to what it should be.

As I went deeper, it honestly got scarier, because there were moments where DDESONN looked better than benchmark models, and other moments where it didn't, and that inconsistency can mess with your head when you've invested everything into building it correctly.

An additional motivation along the way was to benchmark DDESONN against established deep-learning frameworks and push it toward competitive performance. Early on, I set an ambitious target around a 96.00% reference result, but I later realized that some comparison settings were not properly aligned, which forced me to revisit tuning assumptions, correct implementation details, and remove duplicated or misrouted update logic that had subtly distorted behavior. That effectively reset the target and turned the benchmark into a moving goalpost, because once the implementation was aligned correctly, the bar naturally shifted upward.

As performance improved into the high 99.8% range, the dynamic changed again, because at that level single-run comparisons stopped being meaningful and variance across random seeds began to dominate observed differences. What initially felt like a race toward peak accuracy evolved into a deeper investigation of stability, reproducibility, and distributional behavior across large seed sweeps, where mean performance, standard deviation, and worst-case outcomes mattered more than isolated best runs.

The turning point wasn't one magic upgrade. It was the final phase of clearing out the subtle bugs and aligning the implementation to mathematically correct behavior, eliminating duplicate logic, tightening update flows, and ensuring evaluation consistency. Once those last structural issues were resolved, the model became dramatically more stable.

### What "better" means here

When I say "better," I don't mean one cherry-picked run.

I mean repeated evaluation across large numbers of randomized initializations (seeds). In my testing, once the final correctness issues were resolved, DDESONN produced results that were:

- **more stable on average**
- with **lower standard deviation**
- and **less extreme worst-case error**
  across large seed sweeps (e.g., 1,000 seeds)

At that point, it stopped feeling like "maybe this works" and started feeling like "this is now a stable, correct implementation that competes."

The broader acceleration of AI made fully inspectable work built from the ground up feel even more important - not as a trend to follow, but as a way to understand what these systems are actually doing.

### Transparency is the point

DDESONN is built to show you what it's doing.

Even in low-verbosity mode, it exposes the key structural diagnostics (layer dimensions, activation choices, error shapes, and sanity checks), and high-verbosity mode expands that into full step-by-step tracing when you're debugging or studying behavior.

This is not just a model — it is an implementation we can learn from.

### AI-Assisted Iteration Disclosure

Artificial intelligence tools were used during development to support
iteration speed, debugging, refactoring, and documentation drafting.

The primary tools used were ChatGPT and Codex (sparingly), with Copilot and Blackbox AI used on a more limited basis.

While AI tools accelerated iteration, the completion of this project
required substantial sustained personal effort, discipline, and persistence.

DDESONN was designed with a flexible, research-oriented architecture that
enables structured ensemble workflows, temporary-to-main model promotion,
metric-driven refinement, customizable optimization strategies, and
configurable activation behavior. The innovative ensemble methodology,
experimental structure, validation logic, user-level customization depth,
and final implementation authority remained under my direct authorship,
review, and verification.

AI systems functioned as development accelerators and exploratory aids.
All architectural design decisions ultimately reflect deliberate human
direction and sustained independent effort.


---

## Logging / Verbosity levels

DDESONN supports structured diagnostics designed to keep runs scientifically inspectable without overwhelming console noise.

### Always-on output (independent of flags)

**CORE METRICS / Final Summary** output is emitted as part of the run summary path, independent of `verbose`, `verboseLow`, and `debug` settings. This keeps key end-of-run reporting consistent across executions.

### Diagnostic tiers

| Console setting for `ddesonn_run()` | Output |
|---|---|
| Default (both flags off) | Usual evaluation output, including the Train, Validation and Test tables when available |
| `verboseLow = TRUE` | Important progress messages and a compact run summary with dataset, split counts and percentages, epochs, checkpoint and overlap |
| `verbose = TRUE` | Detailed diagnostics and the full run report, including effective settings and per-model outcomes |

The returned `run_report` always includes the full details. The overlap check
compares exact tabular predictor vectors in the supplied splits.

### Trace the training methods from R

Top-level `verbose = TRUE` on `ddesonn_run()` prints the full run report
and split prediction traces. For a closer look inside the training methods,
set `verbose = TRUE` inside `training_overrides`. Add it to the list you
already pass for settings such as epochs or learning rate. You can use both
settings together.

A DDESONN coordinates its network members. Each SONN is one member.
Both R6 classes are defined in [`R/DDESONN.R`](R/DDESONN.R).
The table describes the begin/end markers selected by `trace_epochs`.

| Class and method | What it does | What the epoch trace times |
|---|---|---|
| `DDESONN$train()` | Coordinates the fit and calls each member's training method. | The container's batch preparation for that epoch. |
| `SONN$train_network()` | Runs one member's epochs, applies optimizer updates, and tracks checkpoints. | One full network epoch, including any learning and prediction calls inside it. |
| `SONN$learn()` | Runs the forward and backward calculations and returns gradients. | One gradient calculation. |
| `SONN$predict()` | Uses the current weights and biases to make predictions. | One prediction call made within a training epoch. |

Add `trace_epochs` to your existing `training_overrides` list:

```r
trace_epochs = list(
  all = FALSE,         # Omitted methods are off
  train = 5:10,         # Container batch preparation at epochs 5 through 10
  train_network = TRUE, # Every epoch of each network
  learn = 10:15,        # Learning steps at epochs 10 through 15
  predict = FALSE       # Omit prediction markers inside the epoch loop
)
```

Without `trace_epochs`, no epoch filter is applied. Turning on `verboseLow`,
`verbose`, or `debug` does not select any epochs. The filter applies to method
begin/end markers. All requested epochs run,
and predictions outside the epoch loop follow the usual verbosity settings.
Selected end markers show elapsed wall-clock time for that method or epoch.
These intervals can be nested, so adding them double-counts time. The full
run report lists fit time, measured prediction and scoring outside fits,
and the remaining run work separately. Per-network times are included in
their parent fit. The R console measurement ends before final evaluation
reporting. Shiny measures through the scoring for each training stage.
An explicit method setting overrides `all`. Use `all = TRUE` to enable
every omitted method at every epoch, or omit `all` to keep existing verbosity
for omitted methods. Epoch numbers start at 1 and restart for each fit.
Ranges include both endpoints, so `learn = 1:10` traces epochs 1 through 10.
Epoch 0 is invalid.

In Shiny, **Recent runs > Run details** shows the fit and network times.
The **Equivalent R code** for Custom training includes commented epoch trace
examples. Uncomment one when running the recipe in R.

| Setting | What it adds |
|---|---|
| Top-level `verboseLow = TRUE` | Important progress messages and the compact run summary. |
| Top-level `verbose = TRUE` | Full run report and split prediction traces. |
| `training_overrides = list(verbose = TRUE)` | Detailed diagnostics inside the training methods. |
| `training_overrides = list(trace_epochs = ...)` | Begin/end markers for the methods and epochs you select. Explicit selections work with both verbosity flags and debug off. |
| `debug = TRUE` | Extra debug messages. Some also depend on the verbosity flags, as described below. |

Use epoch traces to find where a fit slows down or fails, or to inspect
what runs around an epoch you are investigating while tuning. These markers
show when a method runs. For gradients and other detailed diagnostics, use
the training verbosity override. Keras has standard fit progress and
callbacks for observing training. DDESONN exposes these R6 method
boundaries and their epoch filters directly through its training API.

### Debug mode (`debug`)

`debug = TRUE` enables the public run API's debug diagnostics directly.
`verbose` and `verboseLow` stay independent. Some debug markers also need
one of those flags because they pass through the console logger.

| Trace | Settings and scope |
|---|---|
| `EVOKE-FIT-BEGIN` / `EVOKE-FIT-END` | Set `debug = TRUE` and `verboseLow = TRUE`. These markers bracket the MAIN and TEMP calls to `ddesonn_fit()` inside `ddesonn_run()`. |
| `EVOKE-PREDICT-BEGIN` / `EVOKE-PREDICT-END` | Set `verbose = TRUE` for split prediction traces. The extra prediction-data traces around MAIN and TEMP fits also require `debug = TRUE`. |
| R6 method traces | The verbosity received by the methods controls paired markers for `DDESONN$train()`, `SONN$train_network()`, `SONN$learn()`, self-organization, and performance/relevance updates. They appear when the corresponding methods run with detailed verbosity. |

For example, pass `debug = TRUE, verboseLow = TRUE` to `ddesonn_run()` to trace the fit calls.
Set `verbose = TRUE` as well to include detailed prediction traces.

### Table views (`viewTables`)

- `viewTables = TRUE` enables table-formatted output for supported sections
- table rendering uses `ddesonn_viewTables()` and requires `knitr` for polished formatting
- this supplements structured reporting; it does not remove core summary reporting

This design keeps low-verbosity runs practical, while still allowing deeper trace/debug modes when needed.

---

## Core Capabilities

- Fully native R deep learning framework — no external deep-learning backend.
- Object-oriented model engine implemented with R6.
- Flexible architecture selection (single-layer or deep multi-layer).
- Dimension-agnostic per-layer configuration: per-layer configuration vectors automatically align to network depth (replicate/truncate), supporting flexible architectures with reduced manual setup.
- Independent activation functions, derivatives, dropout, and initialization per layer.
- Manual training loop with explicit forward and backward propagation.
- Optional self-organization phase (`self_org`) for topology-oriented pre-adjustment during training.
- Transparent optimizer-state updates with full internal control.
- Structured ensemble orchestration: Main (Champion) vs Temporary (Challenger) ensembles with metric-driven promotion, pruning, and iterative refinement to converge toward a stronger primary ensemble.
- Run-level metadata persistence (`store_metadata()`): automatic recording of seeds, configuration, thresholds, metrics, ensemble transitions, and model identifiers for reproducible, auditable experimentation.

#### Optimization & Regularization

- Optimizers implemented from the ground up:
  - SGD  
  - RMSProp  
  - Adam  
  - Lookahead  
- Separate weight and bias update logic in dedicated update blocks.
- L1, L2, and mixed regularization for both weights and biases.
- Optional learning-rate scheduling via training overrides.
- Optimizer and activation selection is available through the public API
  (`ddesonn_model(...)`, `ddesonn_fit(...)`, and
  `ddesonn_run(training_overrides = list(optimizer = ..., activation_functions = ...))`).
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

DDESONN supports structured dynamic ensemble orchestration built around
a **Primary (Main / Champion) Ensemble** and one or more
**Temporary (Temp / Challenger) Ensembles**.

All Champion vs Challenger promotions and prunes are recorded as structured run
metadata so ensemble evolution is reproducible and fully auditable.

The workflow operates conceptually as:

- Temporary (Challenger) ensembles are trained and evaluated  
- Performance is measured under user-selected metrics  
- High-performing Challenger models may be promoted to the Main (Champion) ensemble  
- Underperforming Champion models may be pruned  
- The Champion ensemble evolves over iterations  

This architecture allows controlled model competition under a chosen
metric (e.g., loss, F1, AUC, or other user-selected evaluation criteria).

In practical terms:

- You can run multiple Challenger iterations  
- Select a metric to govern Champion promotion  
- Build a progressively refined Champion ensemble  
- Compare stability across seeds and iterations  

This design mirrors a strict **Champion vs Challenger** structure while remaining
fully metric-driven and reproducible.

Current vignettes demonstrate ensemble scenarios. A future vignette will
provide a focused walkthrough of Champion vs Challenger promotion logic,
metric-based pruning, and multi-iteration refinement.

- Dynamic ensemble orchestration.  
- Champion/Challenger metric-based replacement flow (remove weakest Champion, insert strongest Challenger using the resolved target metric and direction).  
- Deterministic promotion governed strictly by the selected metric (maximize or minimize).  
- Structured ensemble metadata tracking across iterations, including:
  - `main_log` (Champion log): iteration-level snapshots of the Champion ensemble state and metric values
  - `movement_log` (Champion/Challenger transitions): deterministic promotion/replacement events (what moved, from/to, delta, and why)
  - `change_log`: iteration-level update diagnostics and structural deltas for traceability

- Public API support for `classification_mode = "binary"`, `"multiclass"`, and `"regression"`.  
- Binary classification supports threshold tuning, ROC/AUC, precision-recall, and relevance-based evaluation.  
- Multiclass note: For multiclass classification, `y` should be encoded as integer class indices `1..K` (or a one-hot matrix whose columns follow the model's class order), otherwise accuracy comparisons may be incorrect.  
- Regression mode supports continuous target prediction with metric-driven evaluation and error diagnostics.  

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

### Dimension-agnostic behavior (exactly how it works)

DDESONN lets you define single-layer or deep multi-layer architectures with user-selected widths (no hardcoded depth limit in the public API flow).

For `hidden_sizes`, the current rules are:

- **`architecture = "single"`**: any supplied `hidden_sizes` are ignored (with a warning).
- **`architecture = "multi"`**: at least one positive hidden size is required.
- Non-positive hidden entries are removed during normalization.

For list/vector conformance elsewhere, DDESONN aligns by direct replicate/truncate logic:

1. **Activation specs at predict time** are normalized to match layer count `L`:
   - too short -> last provided activation is repeated to length `L`
   - too long -> extras are truncated
2. **Dropout-rate specs in training** are aligned to layer count:
   - too short -> padded with `NULL`
   - too long -> truncated
   - output-layer dropout is explicitly disabled
3. **Prediction vs target shape guards** (metric/evaluation paths):
   - if prediction columns are fewer than required, values are replicated to fill
   - if prediction columns are extra, columns are truncated

Structural conformance in this section is strictly **replicate/truncate only**. Values are copied or sliced to the required shape without additional transformation.

Architecture can also be set explicitly in user code, or auto-resolved by an API helper:

- **Explicit (user-facing, `ddesonn_model`)**
  - Single-layer: set `ML_NN = FALSE`.
  - Multi-layer: set `ML_NN = TRUE` and provide `hidden_sizes` (for example `c(32, 16)`).
- **Auto-detected helper (`R/api.R`)**
  - `normalize_architecture(architecture = "auto", hidden_sizes = ...)` resolves to single vs multi based on whether positive hidden sizes are present.

Minimal examples:

```r
# explicit single-layer
m_single <- ddesonn_model(input_size = ncol(x), output_size = 1, ML_NN = FALSE, hidden_sizes = integer())

# explicit multi-layer
m_multi  <- ddesonn_model(input_size = ncol(x), output_size = 1, ML_NN = TRUE, hidden_sizes = c(32, 16))

# API helper auto-detect (internal helper in R/api.R)
normalize_architecture(architecture = "auto", hidden_sizes = integer(0))  # -> single
normalize_architecture(architecture = "auto", hidden_sizes = c(32, 16))   # -> multi
```

---

### Prediction Aggregation & Grouped Metrics

`predict(..., aggregate = ...)` applies when a DDESONN object contains multiple ensemble members. In that case, each model produces a prediction matrix, and the aggregation rule combines those per-model outputs into one final prediction matrix.

Conceptually, this follows standard ensemble learning practice: combine outputs from multiple learners into a single decision surface for downstream use.

Common usage patterns:

- **Regression** (`n x 1` or `n x d` numeric outputs):
  - aggregate = `"mean"`: combines model outputs element-wise using the arithmetic mean.
  - aggregate = `"median"`: combines model outputs element-wise using the median, often useful for robustness to outlier models.

- **Binary classification** (`n x 1` probabilities):
  - model-level probability outputs are combined element-wise to form a final probability vector of shape `n x 1`.
  - if class labels are requested, thresholding is applied after aggregation.

- **Multiclass classification** (`n x K` class-probability matrices):
  - model outputs are combined element-wise across all `K` columns, preserving shape `n x K`.
  - predicted classes are selected from the aggregated matrix (for example by highest class score per row).

Expected shape behavior:

- If there are `M` models and each returns shape `n x K`, aggregation consumes `M` matrices and returns one `n x K` matrix.
- If `aggregate = "none"`, the workflow uses a single member output directly (no cross-model combining).

#### What grouped metrics are

Reachability in this repository:

- **API path (`R/api.R`)**: grouped metrics are reachable via training configuration (`grouped_metrics`) that is passed through `ddesonn_fit()`/`ddesonn_run()` into the model training call.
- **Script path (`inst/scripts/TestDDESONN.R`)**: grouped metrics are also directly toggled in the script workflow via `grouped_metrics <- ...`.

Grouped metrics are summary metrics computed across a set of models or runs, rather than from a single model only. They are useful when you want segmented evaluation views across experiment dimensions (for example, by run, seed, ensemble role, or model subset) to understand stability and behavior under variation.

In practice, grouped metrics support questions such as:

- How does typical performance change across seeds?
- Are temporary/challenger members consistently stronger or weaker than champion/main members?
- Does a chosen aggregation rule improve consistency across runs?

Example usage scenarios:

1. **Seed-sweep analysis**
   - Input: predictions from multiple models across many seeds, each with shape `n x 1` (binary) or `n x K` (multiclass).
   - Output: per-seed metric tables plus grouped summaries to compare central tendency and spread across seeds.

2. **Champion vs challenger iteration review**
   - Input: model outputs from main and temp ensembles during iterative replacement.
   - Output: grouped summaries by ensemble group/iteration to audit whether replacements improve the selected objective metric over time.

3. **Regression ensemble comparison**
   - Input: `M` model prediction vectors (`n x 1`) on the same test set.
   - Output: aggregated prediction vector (`n x 1`) and grouped error summaries to compare combined output quality versus per-model performance.

#### Relationship to high/low performance relevance boxplots

Grouped metrics and high/low performance relevance boxplots are complementary, but they are not generated from the same source objects.

- Grouped metrics summarize evaluation outcomes across model/run groupings.
- High/low relevance boxplots visualize distributional behavior from the relevance/performance plotting pipeline.

Because they are computed through different paths, values are not expected to match one-to-one. A practical workflow is to use grouped metrics for comparative selection/monitoring, then use high/low relevance boxplots for visual distribution diagnostics on the selected groups.

---

## Advanced Customization

While high-level workflows are provided through `ddesonn_run()`,
`ddesonn_model()`, and `ddesonn_fit()`, the project also includes an
experimentation script located at:

`inst/scripts/TestDDESONN.R`

This script reflects the original development workflow and provides
direct, low-level control over the training pipeline, including:

- Optimizer behavior
- Activation-function selection and derivatives
- Ensemble configuration
- Self-organization toggling
- Training overrides and metric selection
- Seed-loop experimentation

In this context, nearly every component of the training process can be
explicitly tuned and inspected.

The current public API exposes structured configuration for most common
use cases. Future releases may expand first-class API hooks to make
advanced customization more directly accessible through the public interface.

**Learning-rate scheduling observations**

During experimentation, learning-rate scheduling did not consistently
improve binary-classification performance and in several tests slightly
reduced accuracy relative to fixed learning-rate runs.  

However, the same scheduling logic appeared more beneficial in
regression settings, where gradual step-down learning rates improved
training stability and convergence behavior.

As a result, learning-rate scheduling should be considered
experiment-dependent rather than universally beneficial.

### Optional native-R sequence encoder

DDESONN includes an experimental native-R state-space model (SSM) encoder for
sequence inputs. Enable it with `sequence_encoder = "ssm"`. The default is
`sequence_encoder = "none"`.

In the exploratory NQ evaluation, adding the encoder produced small,
inconclusive changes relative to the baseline. Independent comparisons and
reproducible examples are welcome.
[SSM verification and benchmark notes](NEWS.md#verification-and-evidence)

Reproducible SSM evaluations, useful applications, fixes and documentation
improvements are strongly encouraged through [GitHub issues](https://github.com/MatHatter/DDESONN/issues)
and [pull requests](https://github.com/MatHatter/DDESONN/pulls).
You can also [email the author](mailto:quiksilver67213@yahoo.com) or contact him
on LinkedIn; see [Contact](#contact).

The engineered-feature workflow is unchanged by default (`sequence_encoder =
"none"`). To add causal temporal context, build aligned windows and enable the
native selective state-space encoder:

```r
seq <- ddesonn_sequence_data(market_matrix, sequence_length = 48)
x_aligned <- engineered_features[attr(seq, "end_index"), ]
model <- ddesonn_model(ncol(x_aligned), 1, hidden_sizes = 16,
  sequence_encoder = "ssm", sequence_data = seq,
  sequence_length = 48, ssm_state_dim = 16, ssm_conv = 4)
ddesonn_fit(model, x_aligned, y_aligned, sequence_data = seq)
probability <- ddesonn_predict(model, x_aligned,
  sequence_data = seq)$prediction
```

Windows use only observations at or before their end row. Training-channel
scaling and learned encoder parameters are stored in the model and reused after
save/reload; inference never refits them.

SSM models use joint training by default: every training epoch updates both the
prediction network and its encoder from the final prediction error. The encoder
has separate parameters and Adam state; the selected network optimizer is
unchanged. Each ensemble member retains its own encoder, including when a TEMP
member is promoted. Best-epoch restoration restores the matching network and
encoder together. Validation sequences are evaluated without training on them.

To compare with the earlier, separately pretrained and then frozen encoder, pass
`ssm_training = "pretrain"` to `ddesonn_fit()` (or include it in
`ddesonn_run(training_overrides = list(ssm_training = "pretrain", ...))`).
`ssm_lr` optionally controls the encoder learning rate; by default joint training
uses the network's current learning rate. Per-member `ssm_training_history`
attributes record completed epochs, encoder gradient norms and parameter changes.

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

- 2024-05-07 — Project origin  
  The project formally began as a personal research initiative to design and implement a novel self-organizing neural network framework in R, prioritizing explicit training logic, architectural transparency, and experimental flexibility.

- 2024-05 to 2024-08 — Initial intensive development phase (4 months)  
  Sustained day-in/day-out development. Machine learning concepts were studied from first principles in order to design the architecture manually, reason through dimensional flow, identify bottlenecks, and resolve deep implementation issues.

- 2024-09 to 2025-06 — Development pause (10 months)  
  Active development slowed significantly during this period due to full-time professional commitments.

- 2025-06 to 2025-08 — Iterative refinement and hardening phase (3 months)  
  Work resumed with renewed focus on correctness, optimizer stability, ensemble reliability, and reproducibility. Significant bug-clearing and mathematical alignment improvements were completed during this period.

- 2025-09 to 2025-10 — Transitional development and benchmark breakthrough (2 months)  
  A significant research milestone was the two-batch, 1,000-seed heart-failure comparison with Keras, later documented in the historical benchmark vignette. The results used the legacy custom AdaGrad path and do not establish current standard-AdaGrad superiority. Exact original execution settings are only partially documented.

- 2025-11 to 2025-12 — Reduced development activity (2 months)  
  Development intensity decreased substantially as two new parallel projects required priority. Work during this period was limited.

- 2026-01 to 2026-02 — Final packaging, vignette expansion, and CRAN preparation phase (2 months)  
  Focus shifted to converting the research framework into a structured, turnkey R package suitable for CRAN distribution. This included API stabilization, documentation alignment, artifact-path standardization, reproducibility controls, and the creation of formal vignettes for guided exploration.  
  Additional vignettes are planned to further expand structured demonstrations and ensemble deep-dive documentation.

Earlier checkpoint versions and legacy research code may be published separately in a dedicated archival repository to document the project's evolution, including early snapshots where certain components were not fully retained.

---

## Repository Structure

**Documentation structure:**
This repository also includes `inst/scripts/techila/README.Rmd`, which provides Techila/distributed-run notes and execution guidance.
The root `README.md` is the canonical public-facing README for users, CRAN, and external contributors.

```text
DDESONN/
├── R/
├── man/
├── vignettes/
│   ├── heart-failure-standard-adagrad-1000-seeds.Rmd
│   ├── logs_main-change-movement_ensemble_runs_scenarioD.Rmd
│   ├── plot-contols_scenario1_ensemble-runs_scenarioC-D.Rmd
│   └── plot-controls_scenario1-2_single-run_scenarioA.Rmd
│
├── inst/
│   ├── extdata/
│   │   ├── heart_failure_clinical_records.csv
│   │   ├── train_multiclass_customer_segmentation.csv
│   │   ├── test_multiclass_customer_segmentation.csv
│   │   ├── WMT_1970-10-01_2025-03-15.csv
│   │   └── heart_failure_runs/  (Git only; excluded from package)
│   │       ├── run1/
│   │       └── run2/
│   │
│   └── scripts/
│       ├── DDESONN_mtcars_A-D_examples.R
│       ├── DDESONN_mtcars_A-D_examples_regression.R
│       ├── Heart_failure_ScenarioA.R
│       ├── LoadandPredict.R
│       ├── TestDDESONN.R
│       ├── vsKeras/
│       │   └── 1000SEEDSRESULTSvsKeras/
│       └── techila/
│           ├── README.Rmd
│           ├── single_runner_local_mvp.R
│           └── single_runner_techila_mvp.R
│
├── DESCRIPTION
├── NAMESPACE
├── README.md
├── LICENSE
└── LICENSE.md
```

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

Install the development version directly from GitHub (optional):

```r
remotes::install_github("MatHatter/DDESONN")
```

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

**Epoch selection**

The optimal number of epochs is experiment-dependent and may vary
significantly across datasets, architectures, and seeds.

In practice, `num_epochs` should be treated as a tunable upper bound
rather than a fixed universal value. The objective is to identify a
suitable stopping point for each model configuration rather than rely
on a single epoch count across experiments.

### Console quickstart

Pass `verboseLow = TRUE` directly to `ddesonn_run()` to see a compact run summary
with the dataset, split sizes, epochs, and overlap check before the evaluation
tables. Use `verbose = TRUE` when you need the full report and prediction traces.

The training split is always supplied as `x` and `y`.

    res <- ddesonn_run(
      x = train_x,
      y = train_y,
      verboseLow = TRUE,
      validation = list(x = valid_x, y = valid_y),
      test = list(x = test_x, y = test_y),
      training_overrides = list(
        num_epochs = 1,
        validation_metrics = TRUE,
        self_org = FALSE  # set TRUE to enable self-organization
      )
    )
    
Many experiments in this repository use `num_epochs = 360` for the
heart-failure dataset used in the benchmark vignettes. This value was
empirically suitable for that dataset but should not be interpreted as
a recommended universal setting.

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

Why: internal code uses `ddesonn_predict()` because it's a forward-pass primitive
that's faster and easier to control inside training loops (no user-facing return
formatting). User-facing inference should use `predict()` because it provides a
stable contract (type/aggregate/threshold handling, return structure).

Multiclass note: For multiclass classification, y should be encoded as integer class indices 1..K (or a one-hot matrix whose columns follow the model's class order), otherwise accuracy comparisons may be incorrect.

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

Training and validation run inside `ddesonn_run()` and call the model's R6
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
  custom metrics, and downstream logic** - neither replaces the other.
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

## Run terminology

Single vs Ensemble:

- Use **"single run"** when referring to one run/mode.
  - Scenario A (`do_ensemble = FALSE`, `num_networks = 1`).
- Use **"single runs"** when referring to multiple single-run cases.
  - Scenario B (`do_ensemble = FALSE`, `num_networks > 1L`).
- Use **"ensemble run"** only when explicitly referring to one specific ensemble execution.
  - Scenario C (`do_ensemble = TRUE`, `num_temp_iterations = 0`).
- Use **"ensemble runs"** when referring to multiple ensemble executions.
  - Scenario D (`do_ensemble = TRUE`, `num_temp_iterations > 0`).

Important distinction:

- `length(seeds) > 1L` does **not** by itself mean "runs" in this terminology block.
- Here, plural wording is tied to model multiplicity (`num_networks > 1L`) and ensemble iteration structure, not to seed count alone.


Scenario-family note:

- **Scenario A/B/C/D** refers to run-orchestration families (`do_ensemble`, `num_networks`, `num_temp_iterations`).
- **Scenario 1/2** is a separate naming family used for **plot-controls wiring only** (not run-orchestration mode labels).
- In plot-controls docs, **Scenario 1** means three plot call sites are configured/called independently.
- In plot-controls docs, **Scenario 2** means the same three are configured via one `plot_controls` umbrella call.

What this repository already reflects:

- API/docs primarily describe the mode as **"single run"** (singular).
- Plural phrasing such as **"single runs"** appears when discussing broader scope/coverage.
- Workflow guidance uses **"ensemble runs"** (plural) for multi-execution contexts.
- Internal comments also use singular phrasing when pointing to one specific run (for example, "the single run lives at ...").

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

Bundled sample data in `inst/extdata/`:

- heart_failure_clinical_records.csv
- WMT_1970-10-01_2025-03-15.csv
- train_multiclass_customer_segmentation.csv
- test_multiclass_customer_segmentation.csv

Current multiclass usage is demonstrated in `inst/scripts/TestDDESONN.R`.
Standalone CRAN-friendly multiclass example scripts/vignettes are welcome via PR.

---

## Reproducibility

DDESONN includes a run-level metadata store that persists the critical inputs and
outputs needed to compare, trace, and reproduce experiments across iterations and
environments. This metadata is recorded automatically during training via the core
engine (`R/DDESONN.R`) and captures seeds, configuration, training flags, selected
metrics, thresholds used, and per-model identifiers so results are auditable rather
than dependent on console output.

In addition to artifact path controls, this metadata store retains structured fields
such as model serial IDs, ensemble iteration context, activation/dropout settings,
best-epoch summaries, and the resolved performance/relevance metric selections used
during evaluation and selection.

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

### Standard AdaGrad replaces legacy AdaGrad

The default `optimizer = "adagrad"` uses standard AdaGrad: accumulate squared
gradients, then apply **one update per parameter tensor per training step**.
The legacy multilayer path applied both a direct weight update and an adaptive
weight update to the same weights. That double-update behavior was removed in
7.3.0. We have retired it in favor of the standard recurrence, verified with
independent numerical reference tests; no legacy compatibility mode is retained.

Construct a new model for training rather than reusing legacy optimizer state.
Replace explicit `adagrad_x` configurations with `adagrad`; the removed name
raises an error rather than silently selecting another algorithm. Training
trajectories, bias updates and initialization can differ from earlier versions.

#### What the completed 1,000-seed reconstruction shows

The new standard-AdaGrad reconstruction completed all 1,000 fresh seeds at 360
epochs using frozen 7.3.0 source. Mean fixed-split test accuracy is 99.970533%,
versus 99.921867% for probability-recomputed archived legacy results. Repeated
seed-by-row errors total 221 versus 586. This descriptive comparison retains
substantial train/test feature overlap and does not isolate the optimizer's
causal effect or establish performance on new patients.
See the [consolidated report](dev/research/heart-failure-adagrad-new-seeds-1-1000/REPORT.md)
for every seed, all metrics, graphs, provenance and limitations. The earlier
100-seed intermediate report is superseded; it remains in Git history and the
standalone archive rather than as a separate current guide.

#### How this relates to the historical Keras comparison

The historical 1,000-seed Keras comparison used the legacy DDESONN path.
A separate [new 1,000-seed standard-AdaGrad reconstruction](dev/research/heart-failure-adagrad-new-seeds-1-1000/REPORT.md)
now records all fresh seed results. It does not rerun Keras or establish a fully
matched comparison. Correcting AdaGrad leaves the historical dataset's substantial
train/test feature overlap in place: both sides of the historical comparison
remain limited by that overlap. The scores do not establish performance on new
patients or isolate the optimizer's effect. A separate task will prevent identical
feature vectors from crossing splits and rerun both methods on the same new splits.

Historical results remain available. See [historical research and package-file
changes](#historical-1000-seed-research) for the affected paths and archive links.

### Per-seed test metrics and fused ensemble behavior

DDESONN run artifacts commonly include RDS outputs for train/validation and test metrics.
Depending on mode, per-seed test representation is built differently:

- **Ensemble mode (`is_ens = TRUE`)**
  - The per-seed table helper reads fused files from `RUN_DIR/fused/` matching
    `fused_run*_seed*_*.rds`.
  - It binds each file's `metrics` table, parses `seed` and `run_index` from the filename,
    then filters one fusion strategy as the canonical test view (default: `Ensemble_wavg`;
    alternatives may include `Ensemble_avg`, `Ensemble_vote_soft`, `Ensemble_vote_hard`).
  - The selected fused metrics are normalized to `test_acc`, `test_precision`,
    `test_recall`, and `test_f1` before joining to train/validation summaries.

- **Single-run mode (`is_ens = FALSE`)**
  - The helper reads the latest `SingleRun_Test_Metrics_*_seeds_*.rds` file.
  - It normalizes seed naming (`seed`/`SEED`) and metric columns (including `f1_score` -> `f1`),
    then keeps one row per seed (highest accuracy) for the final merged table.

In both modes, merged per-seed summaries are produced by combining train/validation seed-level
metrics with the mode-appropriate test representation.

`SingleRun_Pretty_Test_Metrics_*_seeds_*.rds` files are intended as readable/inspection-oriented
outputs (for example, predicted labels/probabilities aligned with outcome `y` and predictor context)
rather than as the canonical source used for the numeric per-seed summary merge above.

Reference helper scripts and related workflows currently include:

- `inst/scripts/vsKeras/TablesPerSeedMostRecentRunResults.R`
- `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/DDESONNproof.R`
- `inst/scripts/LoadandPredict.R`
- `R/predict.R`

Clarification on terminology: the per-seed fused rows `Ensemble_avg`, `Ensemble_wavg`,
`Ensemble_vote_soft`, `Ensemble_vote_hard` are **ensemble-style fused prediction outputs**
computed from model predictions for reporting/selection at the seed level. They are not, by themselves, the full 
training/orchestration process that builds and evolves ensembles; the Champion/Challenger promotion and pruning 
flow is handled in the run pipeline.

Availability note: historical benchmark outputs and scripts remain in Git at the verified archive revision linked below; they are excluded from CRAN installations.

### Vignettes

Start with these vignettes in `vignettes/`:

- `plot-controls_scenario1-2_single-run_scenarioA.Rmd`
- `plot-contols_scenario1_ensemble-runs_scenarioC-D.Rmd`
- `logs_main-change-movement_ensemble_runs_scenarioD.Rmd`
- `heart-failure-standard-adagrad-1000-seeds.Rmd`

Naming clarification: in these vignette filenames, "Scenario 1/2" indicates plot-control style only, while "Scenario A/B/C/D" indicates run orchestration family. Refer to section: Run terminology.

These cover:

- Single-run flows  
- Ensemble scenarios  
- Logging and diagnostic analysis  
- Historical multi-seed research and future benchmarking guidance

#### Historical 1,000-seed research

The two 500-seed heart-failure runs and Keras workbook remain available in Git,
but their generated RDS outputs and historical experiment folder are excluded
from the CRAN package. The historical vignette is self-contained and does not
require those files at installation or build time.

These results are associated with the legacy custom AdaGrad path, whose
multilayer implementation applied two weight updates. They are preserved as
research evidence, not as proof of standard AdaGrad outperforming Keras.
The committed scripts contain master/per-run seeds, but the exact original
execution configuration cannot be reconstructed from the saved outputs.

See [the provenance audit](dev/research/legacy-adagrad-benchmarks.md) for source
evidence, seed-history limitations, file hashes and archive retrieval commands.
The verified archive revision is
[`0d1b557098b56b39ed2357b30716dcddc25871b3`](https://github.com/MatHatter/DDESONN/tree/0d1b557098b56b39ed2357b30716dcddc25871b3).


#### Files excluded from the package starting in 7.3.0

This is a packaging change recorded for the 7.3 release series. The historical
outputs are associated with the retired optimizer; they were excluded to reduce
package size and keep archived experiments separate from current examples.
**They were not deleted from Git.** The verified archive below is the **7.1.11
submission-preparation revision**; it should not be confused with version 7.1.1.

The five generated RDS files and the workbook/four scripts listed here are
excluded from source-package builds and installations by `.Rbuildignore`:

| Path (relative to the repository root) | Available archive |
| --- | --- |
| `inst/extdata/heart_failure_runs/run1/SingleRun_Test_Metrics_500_seeds_20251025.rds` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/extdata/heart_failure_runs/run1/SingleRun_Test_Metrics_500_seeds_20251025.rds) |
| `inst/extdata/heart_failure_runs/run1/SingleRun_Train_Acc_Val_Metrics_500_seeds_20251025.rds` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/extdata/heart_failure_runs/run1/SingleRun_Train_Acc_Val_Metrics_500_seeds_20251025.rds) |
| `inst/extdata/heart_failure_runs/run2/SingleRun_Pretty_Test_Metrics_500_seeds_20251026.rds` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/extdata/heart_failure_runs/run2/SingleRun_Pretty_Test_Metrics_500_seeds_20251026.rds) |
| `inst/extdata/heart_failure_runs/run2/SingleRun_Test_Metrics_500_seeds_20251026.rds` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/extdata/heart_failure_runs/run2/SingleRun_Test_Metrics_500_seeds_20251026.rds) |
| `inst/extdata/heart_failure_runs/run2/SingleRun_Train_Acc_Val_Metrics_500_seeds_20251026.rds` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/extdata/heart_failure_runs/run2/SingleRun_Train_Acc_Val_Metrics_500_seeds_20251026.rds) |
| `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/1000seedsKeras.xlsx` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/1000seedsKeras.xlsx) |
| `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/DDESONNproof.R` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/DDESONNproof.R) |
| `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TablesPerSeedRunResults1000seedsResults.R` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TablesPerSeedRunResults1000seedsResults.R) |
| `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestDDESONN_1000seeds.R` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestDDESONN_1000seeds.R) |
| `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py` | [Git archive](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py) |

An additional run1 prediction file,
`inst/extdata/heart_failure_runs/run1/SingleRun_Pretty_Test_Metrics_500_seeds_20251025.rds`,
was already absent from the verified 7.1.11 archive. It remains recoverable at
[earlier revision 4bc9944](https://github.com/MatHatter/DDESONN/blob/4bc994462612ce868b90bc70885eb1635eddcdee/inst/extdata/heart_failure_runs/run1/SingleRun_Pretty_Test_Metrics_500_seeds_20251025.rds);
it is not an additional removal in 7.3.0.

Input datasets, current example scripts and reporting helpers remain in the
package. The [file inventory](dev/research/legacy-benchmark-inventory.csv) records
sizes and checksums; the [archive guide](dev/research/legacy-adagrad-benchmarks.md)
provides retrieval commands. This versioned record remains the reference for
these exclusions; future release notes need only describe new file changes.

---

## Roadmap & Design Intent

<small>

> **Note on scope and intent**  
> The items below describe **current behavior**, **explicit design intent**, and  
> **forward-looking considerations**.  
> They are documented to clarify direction and preserve future ideas.  
> They do **not** imply active development or any committed delivery timeline.

</small>

#### R-00 - Maintenance cleanup pass (non-breaking)
**Status:** Forward-looking consideration  

A future maintenance pass may perform light, non-breaking cleanup in shared utilities (especially `R/utils.R`), including removing legacy safety helpers that are no longer referenced, tightening comments, and reducing incidental duplication. This work would be scoped to readability and maintainability only, with no behavioral changes intended.

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
**Status:** Implemented in part, with further expansion planned

**Related To-Do:** T-09

The vignette suite includes the Shiny workbench guide, the clean matched
100-seed benchmark, and the historical 1,000-seed AdaGrad comparison. The
optional Shiny workbench provides interactive training and diagnostics.

Further vignette expansion could cover additional datasets, experiments, and
reproducible walkthroughs.

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

#### R-11 - Main vs Temporary Ensemble Deep-Dive Vignette  
**Status:** Planned documentation expansion  

A dedicated vignette will formally document:

- Champion vs challenger promotion logic
- Metric-based pruning and selection
- Iterative temporary ensemble sweeps
- Stability analysis across seeds
- Controlled main-ensemble refinement

This will provide a structured walkthrough of ensemble evolution mechanics
currently demonstrated in `TestDDESONN.R` and related scripts.

#### R-12 - Alternative Structural Alignment Strategies  
**Status:** Forward-looking consideration  
**Related To-Do:** T-12  

Structural conformance is currently limited to replicate/truncate alignment. Refer to subsection: Dimension-agnostic behavior (exactly how it works).

Future iterations may explore alternative alignment strategies (e.g., averaging,
weighted aggregation, or other reconciliation mechanisms), if empirical
evaluation supports their inclusion.

The current implementation intentionally avoids transformation during
shape alignment to preserve deterministic and explicit structural behavior.

#### R-13 - Neural Network Visualization / Equation Artwork Generator  
**Status:** Forward-looking concept  

Future versions may include a visualization utility capable of generating
neural-network artwork where equations, constants, and hyperparameters from
the DDESONN engine (e.g., learning rate, hidden sizes, activation functions,
optimizer parameters) are embedded along network pathways and synaptic
structures.

The visual output would intentionally follow the same aesthetic style as the
DDESONN neural-network brain artwork shown at the top of the README, keeping
equations and numbers integrated along synaptic paths rather than placed
randomly around the network.

---

## To-Do (Design-Linked)

See [changes since the 7.1.11 CRAN release](dev/release-summary-since-7.1.11.md)
for work completed since that release, including improvements that were never
listed below or in the [exploration guide](vignettes/community-and-next-steps.Rmd#ideas-i-still-want-to-explore).

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

- Continue expanding guided examples. The workbench guide and clean 100-seed
  benchmark are now available as package vignettes.
- Keep demos reproducible and artifact-backed
- Treat vignettes as the primary-user education layer for v1+ releases

#### T-10 - Techila distributed experimentation hardening  
Linked from: **R-09**

- Provide a clean, documented Techila workflow for scaling seed sweeps
- Make it easier to run heavy experiments across multiple servers with minimal setup friction

#### T-11 - Cross-language feasibility assessment  
Linked from: **R-10**

Evaluate architectural portability and determine minimal core components required for a language-agnostic implementation.

#### T-12 - Evaluate alternative alignment mechanisms  
Linked from: **R-12**

- Assess feasibility of averaging or weighted reconciliation during
  structural conformance
- Benchmark against replicate/truncate baseline
- Ensure deterministic behavior and reproducibility guarantees
- Avoid introducing implicit transformation into current alignment paths

---

## Contributing

Contributions are welcome and appreciated. For bugs, feature requests, and collaboration discussion, please use the GitHub issues page: [https://github.com/MatHatter/DDESONN/issues](https://github.com/MatHatter/DDESONN/issues).

Please contribute reproducible examples, benchmark scripts, bug reports and
fixes with tests. Independent evaluations of the experimental SSM are especially
welcome, including results that show no benefit. GitHub contributions keep the
evidence and discussion available for others to inspect and reproduce.

For comparative performance claims, use matched data splits and training
settings, repeat across seeds (10 or more where practical), and report variability
alongside averages. A single-seed run is an implementation check, not sufficient
evidence that one model or framework is better.

### Workflow

1. Fork the repository and create a branch from `main`.
2. Run existing demos and example scripts to confirm there are no regressions.
3. Submit a pull request with a clear description and, where applicable, tests or reproducible examples.

### For Substantive Changes

If your pull request introduces behavioral changes, architectural adjustments, or new functionality, please include:

- A clear problem statement  
- Reproducible scripts or minimal examples  
- Notes describing expected behavior versus observed behavior  
- Any relevant performance metrics or diagnostic output  

This ensures that changes remain scientifically traceable and consistent with the design philosophy of DDESONN.

### Optional Integrations (Techila)

Techila support is available for distributed experimentation and large-scale seed sweeps. As distributed environments can vary significantly, contributions and validation feedback related to Techila integration are especially welcome.

### Areas Where Help Is Especially Valuable

Contributions are particularly appreciated in areas such as:

- Polishing and tightening documentation  
- Improving vignettes and reproducible demos  
- Reporting and diagnostics enhancements (tables, plots, artifacts)  
- Implementing or refining items in the Roadmap & Design Intent / To-Do list  

If you are interested in helping move the project toward a cleaner and more stable plateau, the Roadmap & To-Do sections are the best place to identify meaningful contribution opportunities.

---

## License

DDESONN is released under the MIT License, which permits commercial use.
See [LICENSE.md](LICENSE.md) for the full license text.

---

## Other work by the author

The author also maintains additional modeling projects in R and Python, including:

- **OLR - Optimal Linear Regression**  
  CRAN: [olr on CRAN](https://CRAN.R-project.org/package=olr) | PyPI: [olr on PyPI](https://pypi.org/project/olr/)

---

## Contact

I'd be grateful to hear what you have built or achieved with DDESONN. Following
these recent updates, I strongly encourage contributions, useful applications,
reproducible experiments, fixes, documentation and suggestions. Please reach out
with good ideas, questions, or results—including things that did not work.

[Connect with me on LinkedIn](https://www.linkedin.com/in/mathewfok) or
[email me](mailto:quiksilver67213@yahoo.com). Please mention DDESONN in your
connection request. You can also share applications, issues and ideas through
[GitHub issues](https://github.com/MatHatter/DDESONN/issues) and contribute a pull
request. I appreciate hearing how people use the model and what we could build next.

I still want to explore the neural-network/equation visualization idea described
in the roadmap, and I plan to refresh my separate OLR modeling package. If you
try the [experimental SSM encoder](#optional-native-r-sequence-encoder), I would
welcome a reproducible example and the results of your baseline comparison.
Suggestions and help with these future directions are welcome.

See the [community and next-steps guide](vignettes/community-and-next-steps.Rmd).

**Mathew William Armitage Fok**
