# Current optimizer accuracy/performance results

**Latest: [equal-budget tuning of all ten optimizers](results/seed-1-all-tuned/README.md).**
Every method received the same four-rate search on seed 1 with 300 epochs per candidate.
Validation selected each rate, then test accuracy was measured.

Earlier experiments are retained separately:

- [Corrected common-rate baseline (.02 for all ten)](results/seed-1/README.md)
- [Earlier tuning of only FTRL/AdaDelta](results/seed-1-tuned/README.md)

Different settings can produce different FTRL scores. The earliest 85.7% chat
score used final weights and was superseded. Restoring the validation-selected
checkpoint at common rate .02 gave 57.1%; validation-selected rate 1 gave 85.7%.
The latest table uses consistent checkpoint evaluation and equal tuning budgets.

No package defaults or optimizer formulas changed during these benchmark runs.
