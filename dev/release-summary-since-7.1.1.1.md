# DDESONN changes since 7.1.1.1

This summary uses 7.1.1.1 as the requested baseline. It covers the changes
recorded in GitHub tags, merged pull requests, and the package NEWS. GitHub
versions and CRAN publication are separate. The CRAN Git mirror lists
7.1.9 and 7.1.11 in the 7.x series. There is no exact 7.1.1.1 tag in either
repository. If 7.1.11 was the intended baseline, the changes that follow it
start in the 7.2 section.
[Package NEWS](../NEWS.md) · [GitHub tags](https://github.com/MatHatter/DDESONN/tags) ·
[CRAN Git mirror tags](https://github.com/cran/DDESONN/tags)

## 7.1 series

- **7.1.2–7.1.5** refined documentation, examples, package structure, and CRAN
  preflight checks.
- **7.1.6–7.1.7** addressed vignette rendering and index feasibility, and
  clarified metadata in the logs.
- **7.1.8–7.1.9** reduced default diagnostic output and hardened artifact path
  handling, with a temporary directory as the default artifact location.
- **7.1.10–7.1.11** added reproducible artwork assets and made further changes to
  the README, vignettes, and package documentation for submission. Some of
  these Git tags point to a source tree with a different DESCRIPTION version.

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
- **7.5.4 candidate**, in development, adds elapsed time to selected method
  trace end markers and makes live timing consistent for new and continued
  training in Shiny. The run report highlights main fits and separates
  measured fit time, prediction and scoring outside fits, and other work.
