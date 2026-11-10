> Latest comparison: [all ten optimizers with equal tuning budgets](../seed-1-all-tuned/README.md).

# Corrected mtcars seed-1 benchmark

**Common-rate baseline, not the tuned run.** See [the tuned report](../seed-1-tuned/README.md) for validation-selected FTRL/AdaDelta rates.

300-epoch budget; identical initialization, training-only scaling and 20/5/7 splits. Evaluation explicitly restores each network’s saved validation-selected weights and biases. Earlier final-weight results are superseded.

| Optimizer | Test correct | Test accuracy | Log loss (lower is better) | Brier | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| rmsprop | 6/7 | 85.7% | 0.2619 | 0.0808 | 4.80 |
| adagrad | 6/7 | 85.7% | 0.2796 | 0.0815 | 5.46 |
| lamb | 6/7 | 85.7% | 0.3108 | 0.0941 | 5.85 |
| sgd | 6/7 | 85.7% | 0.3375 | 0.0941 | 5.26 |
| adam | 6/7 | 85.7% | 0.3701 | 0.1203 | 5.19 |
| lookahead | 6/7 | 85.7% | 0.3701 | 0.1203 | 5.05 |
| nag | 6/7 | 85.7% | 0.3771 | 0.1187 | 6.14 |
| sgd_momentum | 6/7 | 85.7% | 0.4941 | 0.1626 | 5.20 |
| ftrl | 4/7 | 57.1% | 0.6891 | 0.2480 | 5.23 |
| adadelta | 4/7 | 57.1% | 0.9477 | 0.3178 | 5.18 |

This is an empirical accuracy/performance comparison, not structural verification. Seven test cars and a single seed do not establish general optimizer superiority. Common hyperparameters are not tuned separately per method. The majority-class baseline gets 4/7 correct.

Corrections: training-set scaling is reused for all splits; saved checkpoint weights are explicitly restored before evaluation. Enabling the best-snapshot reporting flag alone does not restore the live baseline network. No optimizer formulas were changed in this correction.

Full metrics and all 320 per-car predictions accompany this report. Full logs, matching-initialization/scaling snapshots and evaluated model RDS files are at `/workspace/.setup/mtcars-accuracy-restored-seed-1`. Source fingerprints and runtime versions are preserved here.
