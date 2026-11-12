# DDESONN model workbench

An optional native R interface for training, evaluation, saved models and
experiment comparisons. The benchmark viewer was introduced in 7.3.2; model
training pages were added in 7.3.3.

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

The app now opens on **Data**, with **Build & train**, **Evaluate**, **Models**
and **Experiments** pages. Both `shiny` and `callr` are optional dependencies:

```r
install.packages(c("shiny", "callr"))
shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN"))
```

For a first run, keep the included iris data and multiclass task, click
**Prepare data**, then **Train new model**. Evaluate the completed model, save a
model bundle, or upload new rows for prediction on Models.

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
  No self-organization, dropout, batch normalization, sequence encoder, early
  stopping or ensemble orchestration in this first interface.
- One public `ddesonn_fit()` call per fit, in an isolated `callr` worker. The loss
  plot and progress use actual epoch log records; log delivery can lag training.
  Final-epoch weights are evaluated. Test results never select checkpoints.
- Cancel terminates the worker and discards its incomplete result. A previous
  completed model remains intact. Disconnecting the session stops its active
  worker and removes temporary job files. Session models are not shared.
- Evaluation includes accuracy/log loss/Brier and a confusion matrix for
  classification, or RMSE/MAE/R squared for regression. Multiclass Brier is the
  mean sum of squared class-probability errors; binary Brier uses the positive
  class. R squared is unavailable for a constant target.

Changing Data settings invalidates preparation. Changing training controls affects
only the next new model. Existing evaluation always belongs to the completed
model's original split. A saved bundle contains the model, preprocessing, split
data, labels, configuration and evaluation. It can contain private source data.
Only trusted workbench bundles are supported for loading.

**Continue from saved weights** starts another API fit on the model's original
prepared data with its original settings and the requested extra epochs. The
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
