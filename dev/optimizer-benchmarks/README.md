# mtcars accuracy and performance benchmark

Start with [RESULTS.md](RESULTS.md) for the current tables and their differences.

This is an **empirical accuracy / implementation-performance experiment**.
It is separate from `tests/testthat/test-optimizers.R`, which checks the optimizer
formulas and structure. High accuracy here does not prove an implementation is
correct; low accuracy does not automatically prove a bug.

Run from the repository root with DESCRIPTION dependencies and `pkgload` installed:

```sh
Rscript --vanilla dev/optimizer-benchmarks/mtcars-accuracy.R 1 /tmp/mtcars-seed-1
# Run another seed separately, after reviewing the first:
Rscript --vanilla dev/optimizer-benchmarks/mtcars-accuracy.R 2 /tmp/mtcars-seed-2
# Optional subset:
Rscript --vanilla dev/optimizer-benchmarks/mtcars-accuracy.R 3 /tmp/mtcars-seed-3 adagrad adam
```

Each invocation uses exactly one model seed. All selected optimizers run in
sequence, each in a fresh R process, with a 300-epoch training budget. The script
loads this checkout, so it cannot accidentally benchmark an older installed
DDESONN. Use a new output directory for each invocation; existing directories
are rejected to prevent mixing old and new results.

The built-in mtcars dataset predicts manual transmission (`am`) from `disp`,
`hp`, `wt`, `qsec` and `drat`, using hidden layers of 32 and 16 units. A fixed,
stratified split (seed 2026) contains 20 training cars, five validation cars,
and seven held-out test cars. Only the model seed changes between invocations.
All methods use the same initial weights, biases, split, and common learning
rate (.02). Saved snapshots verify identical initialization across methods.
Features are standardized with training-set means and standard deviations,
reused unchanged for validation and test. Automatic feature normalization,
self-organization, dropout, batch normalization and regularization are disabled;
the learning rate stays fixed. This measures the native baseline, with SSM off.
Validation selects the model snapshot. The benchmark explicitly restores the
saved best weights and biases before evaluation; the reporting flag alone does
not restore the live baseline network. The evaluated weights/biases checkpoint is saved for audit. Test labels never select settings.
Evaluation uses a fixed probability threshold of .5.

Outputs:

- `metrics.csv`: train/validation/test accuracy, mistake counts, log loss,
  Brier score, training-set majority-class baseline, and training time.
- `predictions.csv`: actual labels and unrounded probabilities for every car.
- `status.csv`: completed and failed optimizers, including failures instead of
  silently dropping them. Any failed optimizer causes a nonzero script exit.
- `protocol.txt`, `source-fingerprint.csv`, `session-info.txt`: settings,
  source identities and runtime versions for interpreting the run.
- Per-optimizer folders: logs and initialization/split snapshots.

Training seconds cover fitting, not R startup, source loading, or prediction.
The epoch count is the requested budget, not the best snapshot's epoch.
A finite-prediction check detects broken runs, but there is deliberately no
accuracy pass/fail threshold.

These shared settings are not tuned separately for each optimizer. In
particular, adaptive optimizers can have different learning-rate requirements.
Seven test cars make accuracy move in 14.3 percentage-point steps. Treat one
seed as a diagnostic, retain poor results, and compare repeated seeds before
making broad performance claims. A ranking on this dataset is not evidence of
universal optimizer superiority or of SSM performance.

## Equal-budget tuning for all ten optimizers

The common-rate table and tuned table are different experiments. Every optimizer
now receives the same four-rate search; the initial two-optimizer study is historical. FTRL's earlier
85.7% result used final weights and was superseded by the corrected shared-rate
checkpoint result (57.1%). Do not mix those rows with tuning results.

Run a separate validation-only search, one seed at a time:

```sh
Rscript --vanilla dev/optimizer-benchmarks/tune-mtcars.R 1 /tmp/mtcars-tuning-seed-1
```

The fixed grid is learning rates .001, .02, .1 and 1 for every supported optimizer.
Each candidate uses 300 epochs, identical initialization and the same data split.
Candidates evaluate training and validation only. The lowest validation log loss
selects a rate; the chosen rate is rerun before evaluating test predictions.
All candidates, including failed/nonfinite runs, remain in `validation-candidates.csv`; final chosen-model results
are in `selected-metrics.csv`. A selected rate is a seed-specific benchmark
setting, not a new package default or proof of general optimizer superiority.
The baseline runner supports `DDESONN_BENCHMARK_LR` for explicit rate overrides;
its protocol and metric rows record the rate used.

## Future matched DDESONN/Keras studies

Use 10–30 seeds for exploratory comparisons and consider 100 for a more
substantial stability assessment. Increase the count when estimated variability
or desired precision warrants it, rather than treating any count as proof.
Match datasets, train/validation/test splits, training-only preprocessing,
architectures, training and tuning budgets, metrics and model-selection rules.
Use corrected implementations and record every seed, software version, source
revision, optimizer control, stopping rule and failed run. Select settings using
validation data, never test performance. Report means, sample standard deviations,
uncertainty intervals and distributions; state the interval method and whether
variation reflects initialization, split resampling, or both. A small number of
seeds does not justify a statistical-superiority claim. Legacy 500/1,000-seed
results remain historical evidence; see dev/research/legacy-adagrad-benchmarks.md.
Parallel execution is deferred beyond 7.3.0; this release adds none.
