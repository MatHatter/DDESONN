# Comparison under recovered historical mtcars settings

This rerun uses the recovered mtcars example from commit
b026343b30f80ba5a23bc0cef9f77ff958b811a6 (v6.6.1), which immediately followed
the v6.6.0 commit reporting 100% test accuracy. It evaluates the current
standard Adagrad and preserved Adagrad X implementations with that script's
recorded settings. It does not load or claim to reproduce the entire historical
v6.6.0 package implementation.

## Original script seed flow

These runs keep the recovered script's original RNG flow: set.seed(42), draw
the train/validation/test indices, then construct the model without resetting
the RNG to a separate model seed. Each optimizer runs in a fresh R process;
initial weights, bias values, split data and controls match between the pair.

| Split | Standard Adagrad | Adagrad X |
| --- | ---: | ---: |
| Training (19 cars) | 100.0% | 100.0% |
| Validation (6 cars) | 100.0% | 100.0% |
| Held-out test (7 cars) | 71.4% | 100.0% |

Standard Adagrad misclassified two test cars; Adagrad X misclassified none.
The 100% result here is an actual measured held-out result under the recovered
settings with the current Adagrad X code, separate from the older commit's claim.

## Five additional matched model seeds

For model seeds 1 through 5 (with the same fixed split but an explicit seed reset
before constructing each model), mean accuracies are:

| Split | Standard Adagrad | Adagrad X |
| --- | ---: | ---: |
| Training | 100.0% | 100.0% |
| Validation | 100.0% | 100.0% |
| Held-out test | 85.7% | 82.9% |

Both implementations ranged from 71.4% to 100% test accuracy across these five
seeds. The five-seed means exclude the original-script seed-flow runs above.
Mean test log loss was 0.2762 for standard Adagrad and 0.3023 for Adagrad X;
mean Brier scores were 0.0972 and 0.1120, respectively. These small-sample
results do not establish a general winner.

All twelve fits completed. Initial parameters, data splits and settings were
checked for equality within every optimizer pair. A separate Python calculation
recomputed all 36 split/seed metric rows from saved labels and probabilities,
confirming accuracy, misclassification counts, log loss and Brier scores.

## Settings

- Data: mtcars, transmission am as the binary target; all other columns as features.
- Fixed split: 60/20/20 with rounding to 19/6/7 observations.
- Scaling fit on training rows, then applied to all splits.
- Hidden sizes c(32,16); relu/relu/sigmoid; one network.
- 300 epochs, LR 0.02, LR decay 0.5 every 20 epochs, minimum LR 1e-5.
- L1 regularization, model lambda 0.00028.
- Hidden-layer dropout 0.1; self-organization off; batch normalization enabled.
- Validation enabled, best-weight restoration requested.
- Classification threshold 0.5 for these measured predictions.

The recovered script sources R/api.R. The driver substitutes the current
installed package for that source call and explicitly supplies the v6.6.1 LR
schedule. The preceding v6.6.0 API had different fit-time defaults, so its commit
message alone does not establish identical optimizer settings.

The earlier three-epoch comparison is retained separately in
[optimizer-comparison-mtcars.md](optimizer-comparison-mtcars.md). Its lower
accuracy figures were not results from this 300-epoch configuration.

## Reproduction

Run the comparison driver with the historical profile:

```sh
DDESONN_COMPARISON_PROFILE=historical \
  /workspace/.ddesonn-r/bin/Rscript --vanilla \
  /workspace/DDESONN/dev/compare-adagrad-mtcars.R \
  /workspace/ddesonn-setup/adagrad-mtcars-historical-settings
```

The driver runs the original RNG flow and five additional explicitly matched
model seeds. It keeps the original-flow results separate from five-seed means.
Up to four independent worker processes run concurrently, with one BLAS thread
each. All accuracies are calculated from raw predicted probabilities and actual
labels, not copied from console messages.

Results, per-row predictions, errors and per-fit logs are retained under
/workspace/ddesonn-setup/adagrad-mtcars-historical-settings.
The original-script table is in original-seed-flow.csv; summary.csv contains
only the five additional seeds; metrics.csv and predictions.csv contain both.

Seven test observations remain a very small comparison. Neither a single 100%
run nor the seed averages establish general optimizer superiority.
