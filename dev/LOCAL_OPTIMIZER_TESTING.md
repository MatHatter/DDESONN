# Test standard Adagrad against Adagrad X locally

Keep `adagrad_x` as the default while evaluating the corrected standard
`adagrad` implementation. The comparison driver is
[compare-adagrad-mtcars.R](compare-adagrad-mtcars.R).

It trains actual models, predicts transmission class for the repository's mtcars
example, and saves accuracy and error metrics against the observed labels.
It runs both implementations with matching initial weights and data splits.
The output directory contains metrics.csv, predictions.csv, summary.csv and
per-fit logs/result snapshots. Historical mode also saves original-seed-flow.csv.

## Install this checkout first

From a terminal at the repository root, with R and the package dependencies
already installed:

```sh
R CMD INSTALL .
```

In RStudio, the equivalent is:

```r
install.packages(".", repos = NULL, type = "source")
```

Using an older installed package will not test the changes in this PR.

## Short check: 3 epochs

From the repository root:

```sh
Rscript --vanilla dev/compare-adagrad-mtcars.R ../ddesonn-comparison-short
```

This runs five paired model seeds. It is a quick execution/accuracy check,
not a reproduction of the older high-accuracy 300-epoch configuration.

## Recovered historical settings: 300 epochs

On macOS/Linux:

```sh
DDESONN_COMPARISON_PROFILE=historical Rscript --vanilla dev/compare-adagrad-mtcars.R ../ddesonn-comparison-300-epochs
```

On Windows PowerShell:

```powershell
$env:DDESONN_COMPARISON_PROFILE = "historical"
Rscript --vanilla dev/compare-adagrad-mtcars.R ../ddesonn-comparison-300-epochs
Remove-Item Env:DDESONN_COMPARISON_PROFILE
```

This runs 12 fits: one pair with the recovered script's original RNG sequence,
plus five additional pairs with model seeds 1 through 5. Each fit uses 300 epochs,
LR 0.02 and decay by 0.5 every 20 epochs. It is substantially slower than the
short check. The driver uses up to four independent workers, with one BLAS
thread per fit; PSOCK workers support Windows as well as macOS/Linux.

Historical mode reads the original script from Git commit
b026343b30f80ba5a23bc0cef9f77ff958b811a6. Git must be available on PATH and that
commit must exist locally; a shallow clone may need its history fetched first.
It uses the currently installed optimizer implementations with the recovered
settings, not the entire historical package version.

## Read the tables separately

- `original-seed-flow.csv`: the older script sets seed 42, samples the split,
  then constructs the model without another seed reset.
- `summary.csv`: averages only the five additional model seeds. These share
  the fixed data split but reset initialization to seeds 1 through 5.
- `predictions.csv`: actual labels, probabilities, predicted classes, and
  actual-minus-predicted errors for every evaluated row.
- `metrics.csv`: per-fit training/validation/test accuracy, error counts,
  log loss and Brier scores.

Always use a fresh output directory after changing code, example data or
settings. Valid cached results may be reused for the same inputs, but the
driver rejects caches whose driver/example/installed-package fingerprint
differs, rather than silently reporting old numbers.

The held-out set has only seven cars. Inspect several seeds and data splits,
and use larger datasets before selecting a general default.

## Correctness tests and known package-check failures

Run the relevant tests against the newly installed package:

```sh
Rscript --vanilla -e 'library(DDESONN); testthat::test_dir("tests/testthat", filter="^(adagrad-x|optimizers)$", stop_on_failure=TRUE)'
```

The optimizer and compatibility suites pass. The full package suite still has
two baseline SSM failures: the convolution-gradient mismatch and an unsupported
`info` argument in `expect_length()`. Those SSM files were not changed here.
