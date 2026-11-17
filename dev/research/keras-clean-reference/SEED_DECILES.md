# Seed-block and cumulative analysis

Source: `per-seed-test-accuracy.csv`, SHA-256 `96c7824943b329942a494023d48d46140497dcdbbc598525d2b21258b76b21d3`. All 400 records validated: four conditions, seeds 1–100 exactly once, 802 test rows, 360 epochs, 78,840 optimizer updates, final checkpoint. No training runs added.

Positive differences favor the experimental DDESONN float32 matched-protocol adapter. Sum is compared with sum (FALSE), mean with mean (TRUE). Same frozen overlap-free split and original training settings in every condition.

## Separate blocks of ten

| Seeds | Sum advantage (pp) | Mean advantage (pp) | Sum W/T/L | Mean W/T/L |
|---|---:|---:|---:|---:|
| 1–10 | +2.145 | +0.948 | 7/0/3 | 5/0/5 |
| 11–20 | +3.242 | +4.738 | 8/0/2 | 10/0/0 |
| 21–30 | +5.885 | +3.529 | 10/0/0 | 9/0/1 |
| 31–40 | +1.933 | +1.758 | 8/0/2 | 8/0/2 |
| 41–50 | +5.374 | +3.529 | 10/0/0 | 8/0/2 |
| 51–60 | +4.676 | +4.613 | 8/0/2 | 9/0/1 |
| 61–70 | +3.267 | +1.870 | 7/0/3 | 7/0/3 |
| 71–80 | +4.177 | +2.531 | 10/0/0 | 6/0/4 |
| 81–90 | +3.404 | +3.005 | 8/0/2 | 8/0/2 |
| 91–100 | +4.663 | +1.172 | 9/0/1 | 5/0/5 |

## Cumulative paired differences

| First N seeds | Sum advantage, 95% CI (pp) | Mean advantage, 95% CI (pp) |
|---|---:|---:|
| 10 | +2.145 [-1.711, +6.000] | +0.948 [-3.395, +5.290] |
| 20 | +2.693 [+0.600, +4.787] | +2.843 [+0.407, +5.279] |
| 30 | +3.757 [+2.149, +5.366] | +3.071 [+1.317, +4.826] |
| 40 | +3.301 [+2.054, +4.549] | +2.743 [+1.309, +4.177] |
| 50 | +3.716 [+2.646, +4.785] | +2.900 [+1.670, +4.131] |
| 60 | +3.876 [+2.879, +4.873] | +3.186 [+2.108, +4.263] |
| 70 | +3.789 [+2.838, +4.739] | +2.998 [+2.028, +3.967] |
| 80 | +3.837 [+2.986, +4.688] | +2.940 [+1.970, +3.909] |
| 90 | +3.789 [+2.959, +4.620] | +2.947 [+2.050, +3.844] |
| 100 | +3.877 [+3.108, +4.646] | +2.769 [+1.919, +3.620] |

## Precision planning

Approximate total seeds per condition needed for a two-sided 95% paired-t interval with the stated half-width, holding the observed paired-difference SD fixed. These are planning estimates, not guarantees or a sequential stopping rule.

| Reduction | Observed SD (pp) | Current half-width (pp) | Projected half-width at 250 (pp) | Total N for ±1 pp | Total N for ±0.5 pp |
|---|---:|---:|---:|---:|---:|
| sum | 3.876 | 0.769 | 0.483 | 61 | 234 |
| mean | 4.287 | 0.851 | 0.534 | 74 | 285 |

## Interpretation limits

Intervals are ordinary pointwise paired Student-t intervals across training seeds, approximate for these bounded differences. They are descriptive after inspecting the completed cohort; they are not time-uniform confidence sequences and have no adjustment for repeated looks or multiple comparisons. Matching seed labels gives matched initial tensors, but framework-native shuffling/dropout streams differ. Blocks follow the original numerical seed order; no reordering or favorable selection was used. Ten-seed blocks can be noisy and are not independent datasets.

All runs share one training/validation/test split. These intervals describe training randomness conditional on that split; they exclude uncertainty from sampling new test cases, new splits, datasets, or patients. More seeds cannot resolve those uncertainties. A stable cumulative curve does not prove convergence or universal superiority. Precision targets should be chosen based on useful accuracy differences before any further extension.
