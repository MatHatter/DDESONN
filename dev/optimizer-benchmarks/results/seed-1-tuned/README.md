> Latest comparison: [all ten optimizers with equal tuning budgets](../seed-1-all-tuned/README.md).

# Seed 1: validation-selected training rates

These are tuned results, separate from the common-rate baseline. Both methods
used seed 1, a 300-epoch budget, the same 20/5/7 split and restored validation
checkpoints. Four rates (.001, .02, .1, 1) were tried on training/validation only.
Lowest validation log loss selected rate 1 for each method; only then was the
selected configuration evaluated on test cars. Optimizer formulas were unchanged.

| Optimizer | Common rate .02: test correct | Selected rate | Tuned test correct | Tuned test log loss |
| --- | ---: | ---: | ---: | ---: |
| FTRL | 4/7 (57.1%) | 1 | 6/7 (85.7%) | 0.4228 |
| AdaDelta | 4/7 (57.1%) | 1 | 6/7 (85.7%) | 0.3414 |

The fix is a better training configuration, not a changed formula or a new global
package default. One seed on seven test cars does not establish general accuracy.
The original common-rate table remains available at ../seed-1/README.md.
Full candidate/selected metrics and selected per-car predictions accompany this
report. Candidate logs, initialization/checkpoint snapshots and selected logs
remain at /workspace/.setup/mtcars-tuning-seed-1.
