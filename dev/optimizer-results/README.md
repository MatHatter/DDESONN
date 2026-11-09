# Recorded mtcars comparison outputs

These are measured outputs from this review, retained so the reported tables
can be inspected without the original cloud workspace. They compare the current
standard Adagrad implementation with Adagrad X; they do not establish a general
optimizer ranking or reproduce the entire historical package version.

- `mtcars-3-epochs/`: five matched model seeds using the current small example.
- `mtcars-300-epochs/`: recovered v6.6.1 settings, with the original script RNG
  flow plus five separately matched initialization seeds.
- `metrics.csv`: per-fit/split accuracy, error count, log loss and Brier score.
- `predictions.csv`: observed labels, raw probabilities/classes and errors.
- `summary.csv`: five-seed averages, excluding the original script RNG flow.
- `original-seed-flow.csv`: only the original RNG-flow pair in historical mode.

All runs use the fixed seed-42 data split with 19 training, 6 validation and
7 held-out test cars. The five additional runs reset model initialization to
seeds 1 through 5; the original script does not reset it after sampling the split.
Training uses the optimizer code in this PR, based on repository commit
5dd541a7fe5c6fc56ddc5b7d4a3e8b0ec7d55373, under R 4.5.3.

The recovered historical script is from commit
b026343b30f80ba5a23bc0cef9f77ff958b811a6. The reports document the settings
and the limits of comparison:

- [Three-epoch comparison](../optimizer-comparison-mtcars.md)
- [Historical-settings comparison](../optimizer-comparison-historical-mtcars.md)
- [Run comparisons locally](../LOCAL_OPTIMIZER_TESTING.md)

Accuracies and probability error metrics were independently recomputed from the
saved predictions and labels. Fresh local runs write to a separate output
directory; these recorded CSVs are evidence, not test fixtures or targets that
future training must match.
