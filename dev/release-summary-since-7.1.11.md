# DDESONN CRAN release delta file Δ since 7.1.11

Each CRAN release has one delta file. This file uses CRAN release 7.1.11
as its baseline and summarizes subsequent GitHub changes recorded in tags,
merged pull requests, and the package NEWS. GitHub versions and CRAN
publication are separate.
[Package NEWS](../NEWS.md) · [GitHub tags](https://github.com/MatHatter/DDESONN/tags) ·
[CRAN Git mirror tags](https://github.com/cran/DDESONN/tags)

## 7.2 series

- Added an optional native R state-space sequence encoder. The R API exposes
  sequence inputs for training, validation, test and prediction.
- Improved joint learning from the final prediction error while retaining a
  separate pretraining option. Added fixes for training-only scaling, backward
  gradient dimensions, optimizer state, and encoder pairing through
  checkpoint selection and ensemble changes.
- The encoder remains off by default. Exploratory sequence results did not
  establish a predictive benefit.
  [SSM documentation](../NEWS.md#ddesonn-720)

## 7.3 series

- **7.3.0–7.3.2** shipped standard AdaGrad, corrected optimizer updates and
  state handling across network layouts, and added independent numerical
  checks. Historical benchmark records document the earlier custom AdaGrad
  behavior and its compatibility limits.
- **7.3.3–7.3.17** developed the Shiny workbench for preparing data, training
  models in background workers, selecting checkpoints, displaying loss
  curves, predicting, and saving models. Added cancellation and failure
  handling, Basic and Custom training, and combination rules for members and
  seeds.
- **7.3.18–7.3.29** documented the 1,000-seed heart-failure benchmark and its
  input overlap. Added packaged dataset inspection, reusable R recipes,
  chart exports, and preprocessing fitted on training rows only.
- **7.3.30–7.3.42** refined grouped binary splits and the saved-model workflow.
  Added Recent runs, separate labeled evaluation and production prediction,
  continued training on a copy, and a clearer display of reports and
  cancellation state.
  [Workbench guide](../inst/shiny/benchmark/README.md) ·
  [Historical benchmark vignette](../vignettes/heart-failure-standard-adagrad-1000-seeds.Rmd)

## 7.4 series

- **7.4.0–7.4.4** added the dense gradient average/sum switch, made sum the default,
  and kept older mixed-reduction models reproducible. Added clean grouped
  binary evaluation and matched DDESONN–Keras runs, with scripts and datasets
  for reproducing the experimental comparisons.
- **7.4.5–7.4.11** improved workbench labels, study notation, vignette links,
  Shiny launch guidance, and help pages. Corrected CRAN preflight issues and
  gave background-training tests longer to finish with logs on timeout.
- **7.4.12** added a vignette for the clean benchmark. It compares DDESONN and
  Keras under both matched sum and matched average settings, with 100 seeds
  per condition on one fixed split. These results and the historical
  1,000-seed comparison have separate scopes.
- **7.4.13–7.4.14** added reproducible run reports to R and Shiny, retained
  configuration across continued fits, and organized output by verbosity.
  Added pairwise exact predictor overlap checks across training, validation,
  and test inputs.
  [Clean 100-seed vignette](../vignettes/clean-matched-100-seeds.Rmd) ·
  [Run report and overlap PR](https://github.com/MatHatter/DDESONN/pull/77)

## 7.5 series

- **7.5.0**, merged into GitHub main, added optional, independent epoch filters
  for four R6 method traces and made `debug = TRUE` usable without a separate
  environment variable. [7.5.0 PR](https://github.com/MatHatter/DDESONN/pull/78)
- **7.5.1**, merged into GitHub main, removed Shiny's trace sliders, kept
  commented examples in Equivalent R code, and improved console summary
  spacing. [7.5.1 PR](https://github.com/MatHatter/DDESONN/pull/79)
- **7.5.2**, available from a GitHub branch, separates Sum and Average for
  weights and biases and keeps Sum / Sum as the default. The inspected
  `v7.5.2` tag still points to the 7.5.1 commit.
  [7.5.2 source](https://github.com/MatHatter/DDESONN/tree/codex/7.5.2-gradient-reduction) ·
  [7.5.2 tag commit](https://github.com/MatHatter/DDESONN/commit/5e6599f59f51aa549d45e7df15af16fe387c5afe)
- **7.5.3**, available from a GitHub branch, keeps Shiny's data help and split
  preparation working when the loaded runtime lacks the overlap helpers.
  [7.5.3 source](https://github.com/MatHatter/DDESONN/tree/codex/7.5.3-shiny-overlap)
- **7.5.4** adds elapsed time to selected method
  trace end markers and makes live timing consistent for new and continued
  training in Shiny. The run report highlights main fits and separates
  measured fit time, prediction and scoring outside fits, and other work.
  [7.5.4 PR](https://github.com/MatHatter/DDESONN/pull/81)
- **7.5.5** places the split overlap check in Local completed model and Recent
  runs. It shows the selected run's recorded overlaps for all three split
  pairs. Guides and examples now links to the Scenario B example for training
  multiple independent models.
  [7.5.5 PR](https://github.com/MatHatter/DDESONN/pull/82)
- **7.5.6** keeps the help page for both plot-directory functions when
  regenerating package documentation and links to this summary from the README.
- **7.5.10** brings the artwork and table of contents up below the benchmark
  link and Shiny quick start, and moves the detailed version entries to this
  page.

- **7.5.11** clarifies how to enable the experimental native-R SSM encoder
  and describes the exploratory NQ benchmark as inconclusive.

- **7.5.12** highlights the 100-seed benchmark, records vignette progress,
  and links to the release delta file from To-Do and to OLR on PyPI.
- **7.5.14** reduces the package download size by storing the Shiny brain artwork
  as lossless WebP with the original image dimensions.

The [100-seed benchmark reproduction results](research/keras-clean-reference/100SeedsDDESONNvsKerasCleanSplit.rds)
include per-seed scores, raw fits, and a split-overlap audit.

### Changes on GitHub after 7.5.14

- **Making predictions with a saved model.** Shiny reads your new CSV before
  making predictions with the saved weights. If the model learned `01` as a
  category, Shiny keeps the leading zero. If an entire numeric column is blank
  and the model learned a fill value during training, Shiny uses that value.
  Downloaded prediction recipes read the CSV the same way.
- **Checking how accurate a model is.** DDESONN compares predictions with
  the correct answers in your validation and test data. If the prediction is
  “manual” and the correct answer is “manual,” it counts as correct even if
  the categories are listed in a different order.
- `ddesonn_fit()` supports training without a validation split by default,
  and `ddesonn_run()` infers the output width from integer multiclass labels.
- The clean benchmark vignette clarifies that its frozen training loops use
  a single gradient reduction. Current DDESONN supports separate weight and
  bias reductions.
- The README links to the 100-seed benchmark scores in CSV and RDS and
  includes package installation in the Shiny quickstart.

## Detailed version history

### Version 7.4.0 - clean-split evaluation preparation

Add clean-split heart-failure evaluation that keeps identical predictor inputs in
one partition. [Section Three](research/section-three/README.md) retains a
100-seed results table and scripts for reproducing or extending it.
No per-seed model artifacts are included. Historical overlapping scores remain
separate; this is not a replacement 1,000-seed or strict framework-parity study.

#### Gradient aggregation: an explicit Boolean choice

`average_gradients <- FALSE` is the 7.4.0 default: sum both dense weight and bias
batch gradients. Set it to `TRUE` to average both. The API, the main
`inst/scripts/TestDDESONN.R` script and Shiny's **Average weight and bias gradients**
switch use this same Boolean choice. In Shiny, off means summed and on means
averaged. This changes training before tensor clipping and regularization; no
accuracy advantage is assumed for either choice.

| Verified training path | Weight gradients | Bias gradients |
| --- | --- | --- |
| 7.1.11-7.2.0 legacy multilayer AdaGrad | Summed | Summed layer errors |
| 7.3.0-7.3.42 standard AdaGrad | Summed | Averaged |
| 7.4.0, `average_gradients = FALSE` (default) | Summed | Summed |
| 7.4.0, `average_gradients = TRUE` | Averaged | Averaged |
| Usual Keras mean binary-cross-entropy data loss | Averaged | Averaged |

The historical rows describe the verified AdaGrad path, not every old optimizer.
The 7.1.11 learning function calculated means, but its AdaGrad bias update replaced
them with summed layer errors. Version 7.3.0 removed that override. Summing in
7.4.0 does not restore the old double weight-update behavior or legacy optimizer
state. The separate SSM encoder's gradient contract is unchanged.

```r
model <- ddesonn_model(12, 1, average_gradients = FALSE)
ddesonn_fit(model, x_train, y_train, average_gradients = FALSE) # sum both
# Use average_gradients = TRUE to average both instead.
```

Models and workbench bundles retain their selected mode. With no fit-time override,
`ddesonn_fit()` inherits it. Older 7.3-era bundles without this field retain the
historical summed-weight/averaged-bias mode for continuation. To reproduce that
mixed reduction explicitly with a current model, use the compatibility override
`gradient_reduction = "legacy_mixed"`; a single Boolean cannot express that mode.

In Shiny, choose **Sum** or **Average** separately for weight and bias gradients.
R callers can pass `gradient_reduction = "mean_weights"` to average weights
and sum biases, or `gradient_reduction = "legacy_mixed"` to sum weights and
average biases. In `ddesonn_run()`, pass it in `training_overrides`.

Older serialized training methods may require a newly created model to switch
modes. New generated recipes record the Boolean, or the explicit compatibility
mode for an old bundle. Previously downloaded recipes need an explicit setting
when replayed against a new default.

[Section Three](research/section-three/README.md) records the archived-settings
comparison on a clean split. Across 100 seeds, mean test accuracy was 89.45% for
DDESONN sum, 81.83% for DDESONN average, 87.62% for Keras average, and 87.48% for
Keras sum. Seeds 1–10 are retained; seeds 11–100 extend the same protocol. Framework settings still differ. Only the small results table and
rerunnable scripts are retained; no per-seed model files are included.
See the [session notes](SESSION_HANDOFF.md) for continuing with more seeds.

A corrected [matched-settings, no-overlap comparison](research/keras-clean-reference/README.md)
now contains 100 completed seeds per condition. It keeps the original training settings,
including batch size 16, and changes only the split to keep identical predictor
vectors together. The earlier overlapping-split comparison is superseded for
this purpose; its near-perfect scores do not answer the requested clean comparison.

### Version 7.3.42 - upload progress and additional-training controls

Restored the standard full-field model-upload progress bar. Additional-training
loss values and logs show 25 items per page. The graph matches the main chart
with points, hover details and checkpoint information. Choose last-epoch or
best-validation weights, set epochs with 10/100/1000/Clear shortcuts, and use the
Cancel button with more surrounding space.

### Version 7.3.41 - saved-model training stays separate

Additional training stays under the saved-model Train further tab, with live loss,
logs, cancellation and results. The original model and new-model training workspace
stay unchanged. Recent runs identifies model origins.

### Version 7.3.40 - compact saved-model upload controls

Model and CSV Browse fields and the model dropdown use compact widths. The italic
model status sits close below Browse, aligned right. Upload explanations appear
before the model upload control.

### Version 7.3.39 - simpler saved-model evaluation

Removed the redundant labeled customer-example section and download button from
the saved-model workflow. Upload your evaluation CSV directly.

### Version 7.3.38 — clearer model loading, exporting and extra training

Read the two-file upload instructions before Browse, then see an italic loaded-
model status directly below it. Carets identify expandable trained-model details
and saved-model preparation help. Save & export suggests an editable `.rds`
filename. **Train further — optional** now explicitly starts with an already-trained
model, a small additional-epochs field and one **Start additional training** button.

### Version 7.3.37 — recent runs and clearer workspace boundaries

**Recent runs** retains the latest 10 completed models and CSV runs in this browser
session. Restore an item, remove it individually, or clear recent history. Clearing
the saved-model workspace preserves Recent runs and training; clearing history
preserves the active workspace. This is in-memory history, not automatic disk
storage: download a model bundle to keep it after the session ends.

A new [community guide](../vignettes/community-and-next-steps.Rmd) invites you to share
what you built, contribute examples or ideas, and connect with the author on
[LinkedIn](https://www.linkedin.com/in/mathewfok). It also covers the planned
visualization idea, separate OLR work and the current experimental SSM findings.

### Version 7.3.36 — explain holdout testing and production prediction

The CSV choices now say **Evaluate a holdout sample (known answers)** and
**Make production predictions (answers unknown)**. Small circled lowercase-i
controls explain each choice, including the bank loan example: use saved weights
to estimate repayment before it is known, then compare saved predictions with
actual outcomes when available. Opening help does not change your selection.

### Version 7.3.35 — clear saved models and start fresh

Use **Clear saved-model workspace** before browsing if you want to start blank.
It resets loaded models, the CSV, file-picker captions and displayed results;
your training workspace and files on disk stay unchanged. Switching or reloading
a model also clears the previous CSV. Identical uploads refresh one entry rather
than adding duplicates, and invalid replacements keep the last valid model with
an error beside upload. A separately downloadable Compact 16x8 customer model
provides a distinctly named second model to try. Its ZIP contains only the .rds
model and one labeled CSV: upload the model, then the CSV to evaluate it.

### Version 7.3.34 — evaluate labeled CSVs or predict unknown answers

Choose **Evaluate the model on labeled data** to measure accuracy against known
answers, or **Predict answers for unlabeled data** to obtain predictions without
an accuracy claim. Evaluation leads with CSV accuracy and a confusion matrix
(or RMSE and an actual/predicted plot for regression). Matching-input counts are
optional supporting checks, not performance scores. Reviewing the original model's
saved training/validation/test results does not require uploading another CSV.

### Version 7.3.33 — separate training and saved-model workflows

Choose **Train a new model** (light blue) or **Use a saved model** (light teal).
Saved-model uploads stay separate from the training tabs. Their summary shows
original training, validation and test scores plus any saved training history.
Upload your prediction CSV, run it, and read its separately labeled chart/table.
An overlap report identifies inputs matching saved splits. CSV accuracy appears
only when compatible target labels are supplied; demonstration rows are not
presented as an independent evaluation. Extra training starts only through an
explicit transition that works on a copy and preserves the uploaded original.

### Version 7.3.32 — load a model, upload rows, view predictions

Saved models & predictions now guides you through a named model, a separate
prediction CSV and one **Run predictions** action. A dedicated graphing area and
results table appear below the CSV; downloading comes afterward as an option.
The selected filename stays in sync with Current completed model. Export controls
and extra training are separate tools, with a clear explanation of what additional
epochs mean for uploaded models. Packaged customer rows are a collapsed alternative
to your own CSV and use the same prediction action.

### Version 7.3.31 — help preparing and loading a model

The Saved models & predictions tab now includes **Need help preparing a model?**
with numbered steps and a customer-segmentation prompt. Use ChatGPT separately,
reference this GitHub repository and your installed version, and request a trained
compatible workbench bundle or a reproducible R script if execution is unavailable.
The guide explains how to load the bundle, upload new predictor rows, review the
model evaluation and download predictions. ChatGPT is not built into the app.

### Version 7.3.29 — visible profiling and clearer data preparation

Data opens its descriptive summary by default and labels profiling as optional
exploration: it never transforms training data. Choose **Show all columns** or
select the columns to inspect; an empty custom selection prompts for a choice.

Features explains when preprocessing is needed: encode categorical inputs,
handle missing predictors, and review identifiers or answer-derived columns.
Customer segmentation illustrates these needs; complete numeric predictors may
need neither encoding nor filling. The existing scaling switch now explicitly
says **Scale inputs using training means and SDs**. Scaling was already available
before the Auto/Manual section and remains independently enabled by default for
binary classification, multiclass classification and regression. Click
**Prepare data** before training: split first, fit rules on training rows, then
reuse those rules for validation, test and future predictions.

### Version 7.3.28 — separate fit curves, chart exports and clearer help

Training charts label separate member/seed/stage fits and leave their curves
unconnected. Download PNG, SVG or loss records, or copy the chart image in
supported browsers. Every preprocessing dropdown now has explanatory help;
inherited Auto/Manual choices name the actual default treatment.

### Version 7.3.27 — Auto/Manual preprocessing

On Features, review a suggested preprocessing plan, override any step or column,
and prepare the data. Auto handles customer categories and missing predictors,
and proposes excluding ID. Encoding, imputation and scaling are fitted on
training rows only and saved with the model for later predictions. Full R code
reproduces the choices. Manual mode requires explicit treatments for incomplete
or categorical predictors; standardization remains independently controlled.

### Version 7.3.26 — align Features with the loaded dataset

Both Features navigation paths show the loaded dataset's target, task and all
non-target predictors. Loading a new example refreshes these defaults while
ordinary navigation preserves manual choices. WMT defaults to ordered splits;
other packaged training examples default to random splits.

### Version 7.3.24 — browse and profile your dataset

The inspector follows the selected example immediately. Browse head, tail or
specific rows and columns, view matching R code, and explore full-dataset
statistics and numeric box plots before preparing your split.

### Version 7.3.23 — one dataset for training, validation and test

Data now offers five labeled examples. Shiny splits your selected dataset on
Features. The optional unlabeled customer file is available under Predict on
new rows after training or loading a compatible model; it is not needed for
training or evaluation. All packaged CSVs are retained unchanged.

### Version 7.3.22 — packaged examples, guide links and blue workflow selector

Choose iris, mtcars, heart failure, WMT prices or the customer train/test CSVs
directly from Data. Existing package files are read locally, with suggested
settings, previews, downloads and reproducible R source code. The unlabeled
customer file can use a compatible completed model for prediction. Contextual
links open Vignettes, including a new packaged-data guide. The centered
Basic/Custom pill now has a blue color scheme.

### Version 7.3.21 — explicit training runs and epochs

Shiny distinguishes the Data-split seed from Training-run seed values. A seed
value of 42 means one training run, not 42 runs. For Single model, seeds 42,43
with 100 epochs train run 1 for 100 epochs, then run 2 for 100: 200 total.
Visible examples and the workload preview explain runs, model fits and epochs.
This clarifies the existing behavior without changing training.

### Version 7.3.20 — shared regularization and centered workflow selector

Optional regularization is available in both Basic and Custom. The selected
penalty and strength stay selected when changing workflows; off applies no
penalty. Both workflows offer the existing native optimizer list, without
changing the training engine. The compact Basic/Custom pill is centered on
Build & Train.

### Version 7.3.19 — consolidated research and full-seed explorer

The completed 1,000-seed report is now the current research entry point. The
superseded standalone 100-seed report and historical Keras/DDESONN vignette have
been removed from the active tree; Git history and the standalone archive retain
them. Shiny now explores all 1,000 new standard-AdaGrad seeds against archived
legacy observations and links the four current vignettes. The historical
statistics and overlap caveats remain in the consolidated report. Correcting
AdaGrad does not fix the historical split; the split fix and matched rerun remain
separate future work.

### Version 7.3.18 — new 1,000-seed research record and vignette links

The new reconstructed heart-failure experiment completed all 1,000 fresh seeds
at 360 epochs using frozen **7.3.0** standard-AdaGrad code. Mean fixed-split test
accuracy was **99.970533%**, with **221 errors** across repeated seed-by-row
predictions (188 false positives and 33 false negatives). Substantial overlap
between training and test inputs limits conclusions about unseen cases; this
does not establish a matched Keras comparison or isolate the optimizer's effect.

Read the [new research record](research/heart-failure-adagrad-new-seeds-1-1000/README.md)
and [benchmark vignette](../vignettes/heart-failure-standard-adagrad-1000-seeds.Rmd).
The Shiny workbench now has a **Vignettes** tab for all package guides, opening
installed HTML when available and online source otherwise. The existing
100-seed record is preserved in Git history and the archive; raw models and the large reproducibility
bundle remain outside the distributed R package. The no-overlap matched study
is future work.

### Version 7.3.14 — Custom help and download previews

Custom controls now have focused lowercase-i help and equivalent R fragments,
including all four run structures, MAIN/TEMP roles and separate member/seed
combination. A segmented Basic/Custom selector, a short Custom briefing and a
clear diagnostics switch keep the form compact. Download help identifies exact
contents; bundle, recipe and prediction previews make exports inspectable.

### Version 7.3.12 — Basic / Custom workbench foundation

Basic keeps the familiar workflow. Custom adds expandable controls for run
structure, members, training seeds, activation/initialization, regularization,
dropout and learning-rate scheduling. The browser remembers only the mode;
data and models stay in their own Shiny session. Prepare the split once on
Features: all training seeds reuse its rows and training-only scaling.

Custom uses `ddesonn_run()` for new MAIN/TEMP orchestration and `ddesonn_fit()`
for warm starts, without repeating self-organization or TEMP searches on a
warm start. Schedules restart on each additional fit. The workload preview
counts new member fits and requested member epochs, excluding self-organization.
All checkpoint modes run the requested epochs; each member restores its own
best weights **and biases**. Cumulative member epochs form the Custom chart.

#### Ensemble prediction combination

Members combine within each training seed; the resulting seed predictions then
combine by average or median. Binary seed combinations classify at `>= 0.5`
(including ties); multiclass combinations choose the first maximum.
Validation comparisons show individual seeds.
The weighted option uses normalized validation accuracy for classification or
inverse validation MSE for regression (floor `1e-12`). Zero total classification
accuracy falls back to equal weights. Test labels never choose weights.

Average and validation-weighted average return probabilities for classification
and numeric predictions for regression. Median uses componentwise medians;
multiclass rows are renormalized, falling back to the mean if all medians are
zero. Binary vote proportion counts member probabilities `>= 0.5`. Binary
majority requires strictly more than half the votes (ties negative). Multiclass
voting uses plurality, with ties choosing the first saved class; there is no
quorum control. Voting produces decisions/vote proportions rather than calibrated
probabilities. Multiclass `avg` and legacy `vote_soft` are the same calculation,
so Custom offers one average option. Regression voting is unavailable.

Keep the whole downloaded model bundle. Custom prediction and warm-start recipes
use the installed workbench helpers so all saved seeds, member weights and
combination settings apply. A bare root API model alone is not the complete
Custom prediction workflow. Detailed diagnostics are opt-in; epoch progress and
warnings remain visible.

Weighted majority voting and the optional experimental sequence/SSM workbench
encoder remain planned follow-ups. Internal batch normalization is off; use the
training-only standardization on Features. No automatic early stopping is added.

### Version 7.3.11 — retain unsuccessful training curves

Logged loss stays visible after a failed or cancelled attempt, even before a
model has completed. The chart identifies the attempt and remains inspectable.
Training error details also identify the installed engine and R versions.

### Version 7.3.10 — completed-run selection

Select successful training runs and loaded models in the existing completed-model
panel. Their settings, loss chart, results and downloads follow the selection.
Latest-attempt status and attempt history distinguish failures from usable saved
models. Hover enlarges an epoch dot and shows its loss without changing weights.

### Version 7.3.9 — readable R examples

R snippets now separate logical steps with blank lines and explanatory comments.
The saved-model example explains the downloaded bundle, required CSV columns,
prediction output and optional exports. It uses an existing model without training
again. Code viewers also use more comfortable line spacing.

### Version 7.3.8 — explicit learning-rate comments

Version 7.3.8 explains the learning-rate multiplier and minimum floor directly
in the focused help and generated R recipes. A multiplier of 1 keeps 100% of
the rate; it does not subtract 1 or reduce the rate to 1. The workbench continues
to use a fixed learning rate.

### Version 7.3.7 — focused help and full R code

Version 7.3.7 gives each help title its own short R fragment. The **Show full R
code** button at the bottom of the workbench opens the complete workflow for
viewing or downloading, instead of repeating it in each help icon. A fixed SVG
circle keeps the lowercase i's outline round and fully visible.

### Version 7.3.6 — centered help icons and equivalent R examples

Version 7.3.6 expands every workbench help popover with equivalent R code and
improves the alignment of the small circled lowercase i. Examples follow prepared
settings and saved models where available, with clearly labeled illustrations
before preparation. Wider, scrollable help stays within desktop and mobile views.

### Version 7.3.5 — Shiny training feedback and checkpoint controls

Version 7.3.5 refines the guided workbench released in 7.3.4 with state-aware
buttons, explicit best-validation or final-epoch checkpoint selection, checkpoint
summaries, actionable training errors, loss inspection and reproducible R recipes.
Both checkpoint modes run all requested epochs. Initial self-organization is
optional for new models and is not repeated during warm starts.

Version 7.3.4 introduced the numbered Data, Features, Build & Train, and Model /
Results workflow and contextual help. The training workbench introduced in 7.3.3
remains native R. Version 7.3.0 introduced standard AdaGrad; 7.3.1 documented the
100-seed follow-up using that unchanged implementation. The SSM fixes from 7.2
are carried forward; SSM remains experimental and off by default.
See [release notes](../NEWS.md) for the version-by-version changes.
