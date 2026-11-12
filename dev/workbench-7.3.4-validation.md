# DDESONN 7.3.4 workbench validation

## Source and scope

The review branch is `codex/workbench-readiness-7.3.4`, based on
`codex/smaller-help-icons` at `36a3afac6fd1203ea6952419e6848c1cfb7492bf`.
At inspection and push time, that branch's four commits were not in `main`
(`ef2ec7b7abfd25191987c3550504a3067d063618`). The existing 7.3.4 checkpoint,
recipe, guided-workflow, switch and help work is retained.

Version remains 7.3.4. No merges, publication or tag writes were performed.
Remote tag objects remained unchanged:

- `v7.3.3`: `1dc0583c7f95870647b8b8f06a6491dc3c0d57da`, peeled to `196cf529ab18f94c36b646fb1b0e4f25725ae8fc`.
- `v7.3.4`: `51f9a6dadfe434e6f371f140c6c34db7ad3f1a6e`, peeled to `ef2ec7b7abfd25191987c3550504a3067d063618`.

## Runtime and tests

R 4.5.3 on Debian 13, with all DESCRIPTION Imports and Suggests installed in
the user-owned conda-forge environment `/workspace/.setup/r-env`. BLAS and OpenMP
were limited to one thread for long builds and checks. No application credentials,
Python deep-learning backend, GPU or database was required.

The final-source suite passed **72 tests and 12,580 assertions**, with **zero
failures, errors, warnings or skips**. This includes numerical optimizer tests,
SSM tests, all three workbench task types, checkpoint tensor restoration,
preprocessing, recipe replay, serialization, real callr cancellation/failure,
per-session action states and error classification. The prior 12,531-assertion
result was treated only as historical evidence.

Installed-package test command (with the installed review package first in
`R_LIBS_USER`):

```sh
export PATH=/workspace/.setup/r-env/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
Rscript -e 'library(DDESONN); testthat::test_dir("/workspace/DDESONN/tests/testthat", reporter="summary", stop_on_failure=TRUE)'
```

Results and the serialized current-run test results are under
`/workspace/.setup/734-full/verified-tests.log` and `verified-tests.rds`.
The test-generated `tests/testthat/Rplots.pdf` was removed after testing.

## Installed-app browser checks

Shared-desktop control tools were not offered for managed environment
`ccarenv_b64_Y2NhcmVudl8zODU0MzM4M2VlOWM4MTkxYWQ2NzJhNzY5OWNkYzEwOA`.
Its desktop capability was declared enabled with unknown state. Validation used
installed Chromium through Playwright against the installed package's Shiny app,
with real WebSocket sessions, uploads, downloads and callr jobs.

Desktop (1366 x 1000) and mobile/touch (390 x 844) checks passed:

- Initial Train/Cancel/Save states; training and warm-start availability during a job.
- Keyboard help opening, Escape dismissal/focus return, and Space-operated switches.
- Best-validation and final-epoch classification fits, requested epoch budgets,
  selected-epoch summaries and Test-first evaluation.
- Loss-chart hover showing epoch/loss and a keyboard-accessible values table.
  Enter opens/closes the four-epoch table. Downloaded model bytes were identical
  before and after hovering.
- New-data prediction and a 150-row iris prediction download.
- Friendly invalid-setting errors and expandable technical details.
- Cancellation retaining the completed model byte-for-byte.
- Final-epoch training with initial self-organization, followed by a warm start
  with the same selection mode and no repeated self-organization.
- Save/reload restoring the model and its checkpoint.
- A second independent mobile session starting without prepared data or a model,
  loading its own model, and predicting without replacing the desktop model.
- Touch switches/help, help staying within the viewport, preparation and no
  horizontal page overflow.
- No browser JavaScript errors.

Both regression checkpoint choices additionally passed on the mobile viewport
using two numeric predictors. Summaries correctly reported best validation MSE
or final-epoch validation MSE. An actual one-predictor worker failure displayed
shape-specific guidance and preserved an existing completed model byte-for-byte.

Browser scripts, logs, screenshots and downloaded bundles/recipes are under
`/workspace/.setup/734-full/`. In particular, see `final-browser-validation.log`,
`final-browser/results.json`, `browser-regression.log`, `browser-shape-error.log`,
`final-browser/desktop.png`, `final-browser/mobile.png`, `regression-mobile.png`
and `shape-error-mobile.png`.

The saved histories independently confirmed all requested epochs: best mode logged
4 and selected 1, final mode logged 4 and selected 4, and the warm start logged
2 additional epochs and selected 2.

The three recipes downloaded from the browser (best-validation, final-epoch and
warm-start) were executed independently with the public R API. Each reproduced
its saved model's selected epoch, Test predictions and metrics within 1e-12.
The final model recorded initial self-organization as applied; the warm-start
model recorded it as not repeated. See `final-replay.log`.

## Full package build and check

A normal `R CMD build /workspace/DDESONN` completed with all four vignettes
rendered, including the real 360-epoch examples. No vignette skip flag or manual
skip flag was used. A further normal build and check of the final source are
recorded under `/workspace/.setup/734-full/verified/`.

The final-source normal build completed successfully. The archive contains all
four rendered HTML vignettes, remains version 7.3.4, and its seven checked
source/documentation/test files match the working tree byte-for-byte (DESCRIPTION
has normal build-generated metadata). Archive SHA-256:
`13963bbb97579a1d4f3381b6df91b809b06e656c17fcad98450e8c3e5578c3ef`.

The first default-font full check completed with **one WARNING**: the environment
lacks `inconsolata.sty` for the PDF manual. Vignette rebuilding and all tests passed;
this was a font dependency problem, not an Rd syntax failure. A complete indexed
31-page manual then built successfully using R's supported
`R_RD4PDF=times,hyper` setting (installed Times/Courier fonts). Neither vignette
execution nor manual generation is skipped. The final check uses this font
setting and is recorded under `verified/font-check/`.

**Final font-configured check: `Status: OK`**, with zero errors, warnings or
notes. Built-package tests reported 12,580 passes, zero failures/warnings/skips.
Vignette output rebuilding and full PDF-manual generation both reported OK.
The check command completed with exit status 0. The checked final archive is also
installed in the default environment R library, including its rendered documentation.

External CRAN/Bioconductor index probes are blocked by the environment's network
policy; all declared dependencies are installed locally. Rendering also emits a
Pandoc `--mathjax` deprecation message. These log messages are distinguished from
the package check's final error/warning/note counts.

Commands, from a directory outside the checkout:

```sh
export PATH=/workspace/.setup/r-env/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
R CMD build /workspace/DDESONN
R_RD4PDF=times,hyper R CMD check DDESONN_7.3.4.tar.gz
```

## Known core limitation and GitHub access

One-predictor training reaches an existing core preprocessing expression,
`colMeans(Rdata[, numeric_columns])`, where R drops the matrix dimension. This
was reproduced through the installed app. Two or more numeric predictors work
for the validated regression workflow. The core implementation was left
unchanged; the workbench now explains the shape problem instead of misleadingly
blaming callr installation, and preserves the prior completed model.

Git fetch, read and push work through the existing platform authentication.
The GitHub open-PR page showed no open PRs before creation was attempted.
`gh pr list` and `gh pr create` both received `Forbidden` from
`https://api.github.com/graphql`. No PR was created. The review branch is pushed;
`dev/PR_7_3_4_DESCRIPTION.md` contains the proposed title/body for a PR against
`main`. No token values were printed or new credentials requested.
