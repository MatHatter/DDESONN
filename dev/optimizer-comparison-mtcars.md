# Adagrad versus Adagrad X: small repository example

Historical result subsequently located: commit
[8f3a346](https://github.com/MatHatter/DDESONN/commit/8f3a34631100fbfab08919e5f726effc5fe09adf)
(v6.6.0, October 8, 2025) explicitly reports **100% held-out TEST accuracy and
100% validation accuracy** for DDESONN_mtcars_example.R. This is a documented
historical result, not a result independently reproduced by this comparison.

The immediately following commit, b026343, first tracks that example script.
It specifies **300 epochs**, LR 0.02, hidden sizes c(32,16), relu/relu/sigmoid,
one network, split seed 42, train-only scaling, a 60/20/20 split, validation
enabled and best-weight restoration enabled. No separate model seed is reset
after the split. The v6.6.1 API used by that tracked script specifies LR decay 0.5 every
20 epochs; current binary defaults use fixed LR unless overridden. The preceding
v6.6.0 API had different fit-time defaults, including Adam and a 0.9/50 schedule,
so this recovered v6.6.1 script is not proof of the exact settings of the earlier
100% claim.

The current comparison below uses the current script's **3 epochs** and five
explicit model seeds. It therefore does **not** reproduce the historical
300-epoch configuration, and its lower test accuracies must not be read as
evidence that the historical run never achieved 100%.

This comparison was executed after the optimizer review. It measures predictions
against observed labels, rather than only checking optimizer formulas or matching
the original implementation.

The driver is [compare-adagrad-mtcars.R](compare-adagrad-mtcars.R). It runs the
existing [mtcars example](../inst/scripts/DDESONN_mtcars_A-D_examples.R) through
its first fit, validation evaluation and held-out test evaluation, stopping
before its larger A-D scenario runner. The original example file is not edited
or replaced by a synthetic training example.

## Matched setup

- Target: mtcars transmission class, am.
- Original split seed 42: 19 training, 6 validation and 7 test rows.
- Original features and scaling fit on the training rows only.
- Original architecture: two hidden layers with 32 and 16 units.
- Original fit: 3 epochs, learning rate 0.02, validation enabled, best-weight
  restoration requested; remaining training defaults retained.
- Model seeds 1 through 5, each paired between adagrad and adagrad_x.
- Each fit runs in a fresh R process. Initial parameters, split data and controls
  are asserted identical between each pair.
- Both use the original example's threshold extraction, 0.5 in these runs.

This compares the implementations as they are exposed to users, including their
different regularization/update semantics. Adagrad X intentionally retains the
original initialization RNG draws, so subsequent dropout masks can differ even
when initial model weights match. This is not an experiment isolating only one
mathematical term with identical random draws throughout training.

## Results

Accuracy below is averaged across five model seeds on the same fixed split:

| Split | Standard adagrad | Original adagrad_x |
| --- | ---: | ---: |
| Training | 88.4% | 80.0% |
| Validation | 86.7% | 83.3% |
| Held-out test | 74.3% | 71.4% |

The test-accuracy range was 71.4–85.7% for adagrad and 71.4–71.4% for adagrad_x.
Mean test log loss was 0.4570 versus 0.5138, respectively; mean test Brier score
(mean squared probability error) was 0.1525 versus 0.1806. The class predicted by
the training-set majority alone achieved 42.9% on this test split.

These seven held-out observations are too few to establish general superiority.
Changing one test classification changes accuracy by 14.3 percentage points.
These results do not reproduce or invalidate the repository's separate
heart-failure or 1,000-seed benchmark results. No performance-based default
change was made: adagrad_x remains the preserved default.

## What passes, and what is measured

The driver requires matching initial parameters, splits and controls, finite
probabilities with correct dimensions/range, and agreement between independently
computed accuracies and the existing example's reported accuracies. It does not
require an arbitrary high accuracy to declare its execution valid.

It saves every observed label, predicted probability/class, actual-minus-
predicted probability error and class error. Accuracy, misclassification counts,
log loss, Brier scores and a majority-class baseline are calculated for all
three splits. A successful run means those checks completed; it is not a
certificate of good generalization.

Run with the installed package:

```sh
/workspace/.ddesonn-r/bin/Rscript --vanilla \
  /workspace/DDESONN/dev/compare-adagrad-mtcars.R \
  /workspace/ddesonn-setup/adagrad-mtcars-comparison
```

Current-run outputs:

- /workspace/ddesonn-setup/adagrad-mtcars-comparison/metrics.csv
- /workspace/ddesonn-setup/adagrad-mtcars-comparison/predictions.csv
- /workspace/ddesonn-setup/adagrad-mtcars-comparison/summary.csv
- Per-fit logs and result snapshots in the same directory.
