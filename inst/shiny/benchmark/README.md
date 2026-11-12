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
- A single model, zero to five hidden layers, task-appropriate default activations,
  AdaGrad/Adam/RMSprop/SGD, fixed learning rate, optional L1/L2 regularization.
  Optional initial self-organization runs before a new model's supervised training and is off by
  default. Self-organization can destabilize training; smaller learning rates
  may help. Dropout, batch normalization, sequence encoding, automatic early
  stopping and ensemble orchestration are not exposed in this interface.
- One public `ddesonn_fit()` call per fit, in an isolated `callr` worker. The loss
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
pieces of that workflow. No private workbench helper functions are required.

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
