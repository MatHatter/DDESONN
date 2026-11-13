# DDESONN model workbench

An optional native R interface for training, evaluation, saved models and
experiment comparisons. The benchmark viewer was introduced in 7.3.2; model
training pages were added in 7.3.3. The guided workflow was released in 7.3.4;
checkpoint selection, recipes, state-aware actions and improved training feedback
are the 7.3.5 follow-up.

From an installed package:

```r
install.packages(c("shiny", "callr")) # once, if needed
shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN"))
```

From the repository root:

```r
shiny::runApp("inst/shiny/benchmark")
```

The app opens on Data with an iris example ready to prepare. Experiments retains
the compact 100-seed benchmark with metric/seed filters, plots and downloads.
All UI assets are local. Shiny and callr are suggested packages, not imports.

## Experiment CSV format

Upload CSV files up to 10 MB with one row per seed: a unique positive integer
`seed` and at least one complete pair of `METRIC_legacy` / `METRIC_corrected`.
Supported metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `log_loss`,
`brier`. Values must be finite and nonnegative; all except log loss must be at
most 1. Missing values and partial pairs are rejected. Differences are always
recomputed. Upload replaces the dataset; a combined 1–1,000 CSV is needed to
view the full planned benchmark. Invalid uploads retain the previous dataset.
The data tab displays up to 100 rows; downloads contain all selected rows.

## Included data provenance

`data/heart-failure-100.csv` is an exact copy of
`dev/research/heart-failure-adagrad-seeds-1-100/paired-test-metrics.csv` at
repository revision `0b9f59c278638ec4a3a1892e90f5140f94ff0401`.
The detailed report at that revision records protocol, source fingerprints,
artifact checksums, initialization uncertainty and data-overlap limitations.
Training used version 7.3.0; the report was added for 7.3.1. This small CSV is
included so the installed dashboard works without raw models or development
folders. No legacy raw archives are restored to the package payload.

Do not interpret aggregate seed evaluations as independent patients or attribute
historical differences solely to an optimizer correction. The About tab retains
the included benchmark's limitations; uploaded studies need their own provenance.

## Brain artwork

Click the header thumbnail to open an enlarged view, then use Close or Escape
to return. The full-resolution link supports viewing the original image.
The artwork is bundled locally as `www/DDESONN_brain.png`, copied unchanged from
`MatHatter/ddesonn-assets` revision `1b6d45806009ae28cfaab5380ff6bfec045db162`
(path `DDESONN_brain.png`), the image linked in the repository README.
It is project artwork rather than a diagram of an individual model.

Artwork SHA-256: `751d6675a10c8318b69268eb9c5e013d94508ab4e4870f82005c46f4b75dc1e5`.

## Model workbench

The app opens on **1. Data**, followed by **2. Features**, **3. Build & Train**
and **4. Model / Results**. These four steps share a color treatment;
**Experiments** is separate. Model / Results contains **Performance** and
**Saved models & predictions** tabs. Both `shiny` and `callr` are optional dependencies:

```r
install.packages(c("shiny", "callr"))
shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN"))
```

For a first run:

1. Keep the included iris example on **Data**, inspect the preview, and click
   **Next: choose features**.
2. On **Features**, keep Species as the target, the four numeric predictors and
   the multiclass task. Choose the split and click **Prepare data**.
3. On **Build & Train**, configure the model and click **Train new model**.
4. Click **Next: review results** to inspect **Performance**. Use **Saved models &
   predictions** to save/load a bundle or predict on new rows.

The circled **i** beside a heading or control offers a short explanation. Hover
for the description, or click to open it. Keyboard users can Tab to the icon,
press Enter or Space to open it, and Escape to close it and return focus.
Clicking outside also closes the help. Help uses native HTML disclosures and
local styling, without another package dependency.

### What callr does

`callr` is used inside the Shiny app. When you click **Train new model** or
**Continue from saved weights**, it starts another R process and runs the public
DDESONN training API there. Shiny keeps handling the interface and reads the
worker's epoch logs to show progress. Cancel stops that worker; the previous
completed model remains available.

It is not a package-build monitor or another machine-learning framework. Both
processes run R, and training still uses DDESONN's native implementation. The
base package does not require callr; the optional model workbench does.

### Supported training workflow

- Binary and multiclass classification, and regression; finite numeric predictors.
  Class targets may be strings or numbers. Binary probabilities use the second
  sorted training class as positive and a fixed 0.5 threshold. Multiclass uses
  one-hot targets, softmax output and the highest probability for prediction.
- Deterministic random splits (stratified for classification) or ordered splits.
  All splits are disjoint row indices. Means/SDs are fitted on training rows only;
  constant features use scale 1. Grouped or duplicate inputs require an externally
  prepared split; random row splitting does not establish subject independence.
- Basic: a single model, zero to five hidden layers, task-appropriate default activations,
  AdaGrad/Adam/RMSprop/SGD and a fixed learning rate. Optional penalties are
  configured in Custom.
  Optional initial self-organization runs before a new model's supervised training and is off by
  default. Self-organization can destabilize training; smaller learning rates
  may help. Custom additionally exposes dropout and ensemble orchestration.
  Internal batch normalization, sequence encoding and automatic early stopping
  remain outside this interface.
- Basic uses one public `ddesonn_fit()` call per fit, in an isolated `callr` worker. The loss
  plot and progress use actual epoch log records; log delivery can lag training.
  The **Weights used for results** dropdown defaults to **Best validation checkpoint**, selecting the highest
  validation accuracy for classification or lowest validation loss for regression
  within that fit. Both weights and biases are restored before prediction and
  evaluation. Choose **Final epoch (last weights)** to use the last weights after the loop. Both modes run all requested
  epochs; automatic early stopping is not implemented. Test never selects a
  checkpoint. The selected epoch is recorded and marked on the loss chart.
- Cancel terminates the worker and discards its incomplete result. A previous
  completed model remains intact. Disconnecting the session stops its active
  worker and removes temporary job files. Session models are not shared.
- Evaluation includes accuracy/log loss/Brier and a confusion matrix for
  classification, or RMSE/MAE/R squared for regression. Test is the initial
  performance view; validation and training remain available in the selector. Multiclass Brier is the
  mean sum of squared class-probability errors; binary Brier uses the positive
  class. R squared is unavailable for a constant target.

Changing data or selections on Features invalidates preparation. Changing training controls affects
only the next new model. Existing evaluation always belongs to the completed
model's original split. A saved bundle contains the model, preprocessing, split
data, labels, configuration and evaluation. It can contain private source data.
Only trusted workbench bundles are supported for loading.

**Continue from saved weights** starts another API fit on the model's original
prepared data with its original settings and the requested extra epochs. Initial
self-organization is not repeated; the warm start keeps the learned weights. The
API creates new optimizer accumulators and restarts epoch schedules on each fit;
this is a warm start, not exact continuation of an interrupted optimizer state.
To change architecture, optimizer or data, train a new model.

Prediction CSVs must contain the saved predictor names; column order is restored
and training preprocessing is reused without refitting. New-data scoring is
synchronous; use the R API for large batch-scoring jobs. CSV uploads are limited
to 10 MB; trusted model-bundle uploads to 50 MB. This is a local development
workbench, not a multi-user training service with resource quotas.

### API scope

The UI calls `ddesonn_model()`, `ddesonn_fit()` and `ddesonn_predict()` directly.
`TestDDESONN.R` also orchestrates data preparation, plots, ensembles and research
runs; it is not equivalent to one API call. More advanced workflows remain
available through the R API and scripts rather than being silently approximated
in the interface. No optimizer implementation was changed for this workbench.

## Learn the equivalent R API

Expandable **Equivalent R code** panels accompany dataset loading, prepared
features/splits/scaling, the next new model, displayed evaluation and saved-model
prediction. Help for checkpoint selection, scaling and self-organization includes
API or base R examples. Code is selectable; the training/model panels also offer
full `.R` recipe downloads. No private workbench helpers are needed to run them.

**Next-model** code follows the current controls and prepared split. **Completed-
model** code stays attached to the model's saved configuration, even when you
change controls later. Use the same package version and original CSV in the same
row order; the example recipe uses `datasets::iris`. Recipes replay preprocessing,
training, checkpoint restoration and Test evaluation. Additional fits append to
the previous recipe; an older bundle without a recipe instead requires the
previous saved bundle as an explicit input. No prior weights are fabricated.

For best validation selection, the generated code enables `validation_metrics`
and `best_weights_on_latest_weights_off` in `ddesonn_fit()`, then explicitly
assigns the member's `best_weights` and `best_biases` back to its active weights
and biases before `ddesonn_predict()`. The reporting flag alone is not treated as
proof that later inference uses those tensors. Classification uses the first
strictly best validation accuracy; regression uses validation loss. Warm starts
retain the saved selection mode and select within the additional fit, rather
than comparing every checkpoint from every previous fit. Older bundles missing
the new setting keep their original final-epoch behavior.

Binary options use right-aligned on/off switches. Standardization starts on;
initial self-organization starts off. The switches support mouse, touch and the
Space key, and their text labels and circled-i help remain available. Weight
selection is a dropdown because both alternatives need an explicit name.


## Training feedback and available actions

Train becomes available after data preparation and is disabled while a job is
running. Cancel is available only during a job. Save and evaluated-prediction
downloads require a completed model; prediction downloads require successfully
scored new rows. An existing completed model remains available to evaluate, save
and predict with while the next model trains. Warm starts, model loading and
training-data replacement wait until the active job finishes or is cancelled.
Changing Features invalidates preparation, without discarding the completed model.

The checkpoint summary reports the selected epoch **within the latest fit**, its
requested epoch count, selection mode, validation criterion and score. For example,
`Epoch 12 of 50 · best validation accuracy: 92% · Best validation checkpoint`.
Regression reports validation loss as mean squared error (MSE). Final-epoch mode
reports the validation score of the final weights, without suggesting it was the
best score. All requested epochs run in either mode. Test is never used to select
weights. Older bundles without a recorded epoch say so rather than inventing one.

Hover near a training-loss point to inspect its epoch and logged loss. On touch
screens or with a keyboard, expand **Training loss values (epoch and loss)** to
read the same observations as a table. Neither inspection method selects weights
or changes predictions. The checkpoint dropdown affects the next fit only.

Training errors offer next steps, such as checking settings, lowering the learning
rate, preparing standardized data again or disabling initial self-organization.
Expand **Technical details** for the underlying error and worker diagnostics.
Failed and cancelled runs preserve the previous completed model and its recipe;
a later successful fit replaces it. Each browser session keeps its own model,
job, predictions and action states.

Known core limitation: a fit with only one predictor can fail in the model's
internal `colMeans()` preprocessing because a matrix dimension is dropped. Use
two or more numeric predictors as a workaround. The workbench preserves the
previous completed model and reports a shape-specific error; this UI change does
not alter the core preprocessing implementation.


## Equivalent R inside help (7.3.6)

Every circled lowercase-i control contains an **Equivalent R** section. The
examples cover data loading/inspection, target and predictor selection, task
encoding, stratified or ordered splits, fractions, training-only scaling, model
architecture, epochs, optimizer/rate, regularization, initial self-organization,
checkpoint restoration, evaluation, model bundles, prediction and additional fits.

Prepared-data examples use the last prepared split and preprocessing. Training
examples use the current valid fit controls. Before preparation, an explicitly
labeled iris example supplies context; invalid fit settings show labeled defaults.
Completed-model help uses its saved recipe, and additional-fit examples retain the
saved model's settings rather than the next-new-model controls. Examples name the
required Data/Features panels, trusted bundle or CSV/recipe files. Download the
complete recipe for a runnable end-to-end workflow; help snippets are contextual
pieces of that workflow. Basic recipes use the public API directly. Custom full recipes source the installed
workbench helpers to retain the complete multi-seed prediction workflow.

The small i remains a real lowercase letter, with centered glyph alignment and a
visible keyboard focus ring. Press Enter or Space to open help and Escape to close
it and return focus. Expanded code can scroll vertically and horizontally, and the
popover repositions on scrolling/resizing to stay within the viewport. Opening,
reading or selecting help code never trains a model or selects/restores weights.


## Focused snippets and full code (7.3.7)

Each help title now shows only its own operation or setting: Epochs shows the
epoch budget, Optimizer the update-rule argument, and Regularization strength
the model's penalty strength. The snippets are contextual fragments, not standalone
programs. Their required objects come from the complete workflow; they do not repeat
all preparation and training steps inside each icon.

Use **Show full R code** at the bottom of any workbench page to open a separate
viewer and download its displayed script. Choose the next model's prepared data
and current valid settings, the completed model's saved recipe, or loading and
predicting with a saved bundle. Before data preparation, the next-model view is
explicitly labeled as an iris illustration. Completed-model code keeps its saved
settings after controls change. Older bundles without a recipe do not fabricate a
training replay; their load/predict code remains available.

The icon uses an SVG circle inside a fixed square button, with a real lowercase i.
The circle is not a CSS border that can stretch or clip. The existing accessible
help labels, keyboard focus, Escape dismissal, scrolling and touch behavior remain.
The full-code dialog is read-only; opening it does not train or change weights.


## Learning-rate comments (7.3.8)

The focused Learning rate fragment and all newly generated fit recipes explain
that `lr_decay_rate` is a multiplicative retention factor. `1` keeps 100% of the
rate: `0.01 * 1 = 0.01`. Before the minimum floor is applied, `0.5` halves the
rate and `0.9` retains 90% at each interval (`lr_decay_epoch`). The public API
default is one epoch for classification and 20 epochs for regression.

The scheduled rate is `max(lr_min, lr * lr_decay_rate^floor(epoch / lr_decay_epoch))`.
The workbench sets `lr_decay_rate = 1` and `lr_min = lr`, so the rate stays fixed.
To enable decay through the R API, choose both a positive factor below 1 and a minimum
below the starting rate. Existing saved recipes remain historical snapshots;
newly generated recipes include the explanation. No scheduler values or behavior
change in this release.

### Readable examples (7.3.9)

Help fragments stay specific to their titles. Blank lines separate logical blocks,
and full scripts label data preparation, model fitting, checkpoint selection and
evaluation. Display and download use the same generated code.

**R example: load a saved model and predict new rows** uses the `.rds` file from
**Save model**. That bundle contains the trained model and preprocessing settings.
Replace the bundle and CSV paths; the CSV needs the original predictor columns,
not target labels. Reuse the saved feature order, means and SDs. No training runs:
classification produces class probabilities and regression produces numeric
predictions. Saving another bundle or exporting only the API model is optional;
the latter omits preprocessing. Historical completed recipes are preserved.

The intention comes from the original standalone script, written before the
package API existed: reload a saved model with its original settings and use it
again. The example here expresses that workflow through the current API.

### Completed runs and attempt history (7.3.10)

The original layout is retained: Build & Train on the left, Training progress at
the upper right and Current completed model below it. Radio buttons list each
successful run and loaded bundle for this session. Selecting one restores its
Build & Train controls and activates its saved chart, checkpoint summary,
evaluation, prediction, warm-start source, saved recipe and model downloads.
Changing controls configures the next **new** run on the currently prepared data;
it does not edit a completed model or change Features/data preparation. Warm
starts use the selected model's own saved data and configuration.

Latest attempt reports running/completed/failed/cancelled status with a run number.
Errors belong there, with guidance and expandable technical details. Starting a
new attempt clears the prior active error; later successes leave that warning in
Attempt history. Failed/cancelled attempts never replace completed models and are
not offered as usable model choices. Loaded bundles add completed entries too.
Models and history are per session and kept in memory; download bundles you need
to retain before closing the session. A bundle stores its selected model, not all
models in the session's history.

The chart and loss table show only the selected completed model, even while a
later job runs. With no completed model yet, live loss is shown during a job.
An older bundle without loss history has an explicit empty-state message.
Hover uses a normal cursor, enlarges the nearest dot and shows epoch/loss. It
never selects/restores weights; the selected checkpoint remains marked separately.
The loss table provides the same epoch/loss values without requiring hover.

### Failed-attempt curves and engine versions (7.3.11)

A failed/cancelled attempt can have useful loss measurements without yielding a
usable model. If no completed model is selected, retain that attempt's logged
curve and label its outcome explicitly. Save/predict stay unavailable until a
model completes. A selected completed model continues to show its own history.

Training uses the installed `DDESONN` package, including when you open `app.R`
from a checkout. Updating app files alone does not update that engine. Expand
Technical details after failure to see the exception and installed engine/R
versions. Check `packageVersion("DDESONN")`; install the updated source package
and restart R when the installed engine is older than the source you intend to
run. This installation does not require rendering vignettes.


### Basic / Custom foundation (7.3.12)

The Basic path retains its training defaults and numerical behavior. Its split
seed is under collapsed **Reproducibility** on Features. The browser remembers
only the Basic/Custom preference; data, completed models and workers are session
state. Custom settings are in keyboard-accessible expandable sections. In 7.3.13, scheduling,
dropout and regularization are off-by-default toggles; their detailed controls
appear only when enabled. Scheduling off means a fixed learning rate.

Custom supports Single, multiple members inside a model, MAIN, and MAIN with
TEMP iterations through the existing run orchestrator. Every training seed uses
exactly the prepared row split and scaling. The preview counts seeds × members ×
(1 + TEMP iterations) member fits, multiplied by epochs; self-organization adds
separate initial work. Architecture/activation, initialization, all supported
optimizers and penalties, dropout, and fixed/exponential scheduling are available.
Standardization remains on Features; internal batch normalization stays off.
The decay uses the formula above, including its epoch-one behavior. Both
checkpoint choices exhaust the epoch budget. Each best-validation member
restores both tensors independently, including members promoted from TEMP.
The chart counts cumulative member epochs; it does not imply one shared
checkpoint epoch. Warm starts fit each saved seed's MAIN members, restart their
schedule/optimizer state, and never repeat self-organization or TEMP searching.

**Ensemble prediction combination** applies first within each seed, then across
seeds. Average probabilities (numeric predictions for regression), componentwise
median and validation-weighted average are distinct choices. Weights use member
validation accuracy or inverse validation MSE (floor 1e-12), normalized within
seed; zero total accuracy falls back to equal weights. Saved weights are reused
for new data. Multiclass medians are renormalized; an all-zero row falls back to
the mean. Binary vote proportion counts probabilities >= 0.5. Binary majority
requires more than half, tying negative; multiclass uses plurality with ties in
saved class order. No quorum is exposed, and file-based multiclass fusion now
rejects explicit quorum settings rather than silently ignoring them. Binary
file-fusion quorum still specifies the required positive-vote count; its default
is strict majority. Multiclass average and legacy soft voting are duplicates, so
only average is offered. Regression has no voting control. Across seeds, average
or median applies to the combined outputs; binary classification then uses >=
0.5 (ties positive), multiclass the first maximum. Voting outputs are decisions
or vote proportions rather than calibrated probabilities.

The validation seed table and displayed weights explain the saved workflow.
Keep the whole bundle and use its full prediction recipe: a bare root model with
`ddesonn_predict()` does not implement the Custom member/seed fusion. New full
Custom recipes use the installed workbench helpers for training, prediction and
warm starts; historical Basic recipes remain intact. A Custom bundle can exceed
the existing 50 MB upload limit for large workloads, so choose a representative
configuration and retain reproducible recipes alongside the bundle.

Ordinary progress keeps epoch records and genuine warnings/errors. Enable
**Detailed training diagnostics** before a Custom run for tensor and optimizer
logs. The progress panel includes the most recent 2,000 log lines; completed
bundles retain those logs. Weighted majority voting and the optional experimental
sequence/SSM workbench encoder remain follow-up work.


### Contextual help and download previews (7.3.14)

The segmented Basic/Custom selector retains native radio keyboard semantics and
remembers only the browser mode. A short Custom briefing identifies the native R
engine and optional orchestration; both modes use the same engine. Workflow help
explains the distinction from the separate Keras API without performance claims.

Every Custom control has lowercase-i help and a focused equivalent-R fragment.
Run structure help follows the selected structure: single, members, MAIN or
MAIN/TEMP. MAIN is the incumbent/champion; TEMP supplies challengers, also called
candidates or competitors. These are terminology for existing roles, not extra
algorithms. Member combination and across-seed combination have separate help.
Long explanations live in the help rather than beneath the dropdowns. Custom
uses the label Custom advanced settings. Detailed diagnostics is a switch; its
help explains verbose tensor/optimizer output and the retained Training log.

Each download has help identifying its exact contents and a focused code example.
Bundle preview summarizes the selected run's actual fields, task, predictors,
member/seed counts, split row counts, preprocessing, results, logs and saved
recipe; the RDS retains complete tensors and split data/labels. It excludes the
session's other completed runs. Recipe preview opens the existing full-code
viewer on the corresponding next/saved script. The downloaded script matches
that viewer. Prediction CSV disclosures preview the first 20 rows; downloads
contain all corresponding rows. Experiment downloads likewise explain selected
rows versus metric summaries and use their existing table previews.


### Clearer labels and aligned actions (7.3.15)

The penalty-only disclosure is named Regularization. Help names L1/L2 and the
Custom L1_L2, Group_Lasso and Max_Norm variants, including the distinction
between a max-norm penalty and a hard constraint. Activation and initialization
help adds context for the listed options. The combined structure is labeled
Main ensemble with temporary iterations (MAIN + TEMP); its count is Temporary
iterations. Backend parameter names and training behavior are unchanged.

Next: choose features, Prepare data and Train new model align to the right of
their control panels. Evaluated predictions download/help and the CSV preview
summary align to the right of Model performance. The larger Show full R code
button remains in its green panel. That panel follows the visible control
container width on Data, Features, Build & Train and Experiments, the full Model
performance width, and the Save or restore width on Saved models & predictions.
The alignment updates on tab changes and responsive window resizing.


### Compact Basic controls and Custom separation (7.3.16)

Basic ends with its prepared-split epoch summary and a shared training action
row. Custom reveals an Advanced Custom settings divider below the shared
controls, followed by optional regularization and the remaining Custom sections.
Regularization is off by default and hidden in Basic. Switching a new run to
Basic removes any hidden Custom penalty; existing completed models and warm
starts continue to use their saved configurations.

The Basic/Custom pill is one-third narrower. Prepare data and Train new model
are slightly wider; Cancel and Train share one row with Train at the far right.
Show full R code is wider, using its original 14px font and standard button
height. Its green panel retains the page-specific widths introduced in 7.3.15.


### State-aware progress and empty states (7.3.17)

Idle/finished attempt status and errors use a separate callout above Training
progress. Active-run status and progress appear inside the card while running;
errors remain above it. The chart, loss table and checkpoint/epoch guidance
appear only with recorded loss. Hover instructions live in chart help; the
visible inspection text appears only over an actual point. Failed/cancelled
first-run curves remain visible when recorded. Loaded models without history
have one specific missing-history message. Current completed model has a single
initial empty-state message; model and checkpoint details appear after completion
or loading. Existing completed models remain available after unsuccessful runs.
# Vignette links (7.3.18)

The **Vignettes** tab links the new standard-AdaGrad 1,000-seed research record,
the historical DDESONN/Keras context, single-run and ensemble plot controls,
and ensemble logs. Installed HTML guides open locally in a new tab; a guide
without installed HTML links to its source on GitHub. Build/install package
vignettes to make all guides available locally. This tab does not launch training
or load demonstration models. Beginner examples and model-loading changes remain
separate work.

The new benchmark retains the historical overlap-heavy split and frozen 7.3.0
training source. The split correction and a matched Keras/DDESONN rerun are future
work for another PR; the current comparison does not establish generalization to
new patients or isolate the AdaGrad correction's causal effect.
