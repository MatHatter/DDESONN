# Equal-budget tuning: all ten optimizers, seed 1

Each optimizer received the same four rates (.001, .02, .1, 1), 300 epochs per candidate, identical initialization, train-only scaling and a fixed 20/5/7 split. The lowest validation log loss selected the rate; candidate runs never evaluated test predictions. Selected configurations were rerun and evaluated after restoring the validation-selected checkpoint.

| Optimizer | Selected rate | Baseline test accuracy (.02) | Tuned test accuracy | Correct / 7 | Test log loss | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adam | 0.1 | 85.7% | 100.0% | 7/7 | 0.0004 | 4.93 |
| lookahead | 0.1 | 85.7% | 100.0% | 7/7 | 0.0953 | 5.08 |
| sgd | 0.1 | 85.7% | 100.0% | 7/7 | 0.1647 | 5.62 |
| adadelta | 1 | 57.1% | 85.7% | 6/7 | 0.3414 | 5.08 |
| ftrl | 1 | 57.1% | 85.7% | 6/7 | 0.4228 | 6.46 |
| sgd_momentum | 0.1 | 85.7% | 85.7% | 6/7 | 0.4432 | 6.19 |
| rmsprop | 0.1 | 85.7% | 85.7% | 6/7 | 0.5785 | 5.45 |
| adagrad | 1 | 85.7% | 85.7% | 6/7 | 0.8679 | 5.20 |
| nag | 0.1 | 85.7% | 85.7% | 6/7 | 3.5819 | 6.30 |
| lamb | 1 | 85.7% | 71.4% | 5/7 | 2.5308 | 5.42 |

Completed 40/40 candidates and all ten selected evaluations. Failed candidates: 0. All 320 selected per-car predictions were checked against the reported accuracy and Brier scores. Selected rates and validation scores were verified against all candidates. The launcher verified identical initialization and preprocessing across candidates.

This is an exploratory accuracy/performance comparison, not structural verification. Equal tuning can improve some methods and worsen others on held-out cars: LAMB dropped from 6/7 to 5/7. AdaGrad retained 6/7 but its test log loss worsened. Selection used validation results, not test results. Do not select a different rate afterward to improve this table.

One seed, five validation cars and seven repeatedly inspected test cars provide limited evidence. Use further seeds and a larger independent evaluation before making broad performance claims. No optimizer formulas or global defaults changed.

Files: validation-candidates.csv (including statuses), selected-metrics.csv, selected-predictions.csv, protocol.txt and script fingerprints. Full candidate logs and checkpoint/initialization snapshots remain at /workspace/.setup/mtcars-all-tuned-seed-1. Baseline package-source fingerprints were checked against current sources; scripts.csv records the current benchmark scripts.
