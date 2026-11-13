# New reconstructed standard-AdaGrad heart-failure benchmark: seeds 1–1000

All 1,000 seeds ran fresh at 360 epochs with frozen 7.3.0 source revision `a95bc8dda84c4eb35633ee5a5bb456169c1c54b8`. This is a new reconstructed experiment, not a verified continuation of the 100-seed record. Earlier records and raw artifacts are preserved in Git history and the standalone archive; current documentation is consolidated here. No release was published and no optimizer code changed.

Mean fixed-split test accuracy: **0.999705333333333**; sample SD: **0.000631588309487722**. Total repeated seed-by-row errors: **221** (188 FP, 33 FN); 15 of 750 test rows were misclassified at least once. Full precision/recall/F1/ROC AUC/log-loss/Brier summaries, every seed, row errors, graphs, pilot evidence and source are in the standalone report and bundle identified in ARTIFACTS.json.

Freeze SHA256: `09199bb31be1388778afe5de3609cd52e731a7b7a5afeedf6148d23eb86a3553`. All 20 archived source/dataset checksums matched. Two fresh seed-1 pilots exactly matched tensors, RNG, validation history, checkpoints and predictions. Checkpoint perturbation/restoration and fresh-process saved-model reload passed. Frozen-source optimizer arithmetic and independent-reference tests passed 12,162 assertions, zero failures/errors/warnings/skips. The supervisor rejected tampered predictions and independently verified all 1000 completion manifests before reporting.

The native member trainer avoids additional fit-wrapper normalization. All 12 predictors; train rows 1–3500, validation 3501–4250, test 4251–5000; training mean/sample-SD and maximum-absolute standardized training divisor. One 12→64→32→1 ReLU/ReLU/sigmoid network, He custom_scale 1.04349, AdaGrad .125 fixed, L1 .00028, dropout list(.1,NULL,NULL), epsilon 1e-7, native class weights 2/1. All 360 epochs; first strict best validation accuracy at .5; both weights and biases restored. No test tuning or early stopping.

The archived legacy multilayer path applied a direct weight update followed by another adaptive update. Standard AdaGrad removes this double-update behavior. Descriptive historical mean accuracy is 0.999218666666667, SD 0.00125609022428344; new-minus-legacy mean difference is 0.04866667 percentage points. Historical initial tensors/RNG/software/complete execution conditions are not fully verified, so differences cannot be attributed solely to the AdaGrad correction. Keras was not rerun; archived workbook statistics do not establish a fully matched comparison.

The recomputed overlap audit finds 1319 distinct feature vectors among 5000 rows, 734/750 validation and 738/750 test inputs matching training, and only 12 test rows absent from training inputs. The cause is unestablished; no patient identities or duplicate causes are inferred. The same 750 test rows recur across seeds; 750000 predictions are not independent clinical cases, and seed SD is not a patient-population confidence interval.

“We identified substantial overlap between training and test inputs, but its cause has not been established. After this iteration, we will rerun the 1,000-seed benchmark for both Keras and DDESONN using new splits that prevent identical feature vectors from appearing across training, validation, and testing. Both methods will use the same splits for a fair comparison.”

That no-overlap matched study is future work and was not launched here. See the standalone REPRODUCE.md for exact reproduction/resume commands and the report for limitations and all results. Local retrieval paths do not establish durable external storage; no external upload/hosting capability was available.

## Browse the results

- [Full report](REPORT.md), including every seed and the historical comparison.
- [Per-seed test metrics](per-seed-test-metrics.csv), [paired explorer data](paired-test-metrics.csv) and [metric summaries](metric-summary.csv).
- [Errors by test row](test-row-errors.csv), [graphs](graphs/) and [validation evidence](evidence/).
- [Reproduce or resume](REPRODUCE.md); [bundle size, checksum and current retrieval location](ARTIFACTS.json).

The corrected **standard AdaGrad** (not automatic differentiation) is compared descriptively with archived legacy DDESONN and Keras evidence from the same overlap-heavy historical heart-failure dataset. The shared defect does not become a clean evaluation just because the optimizer is corrected: repeated feature vectors across training and testing compromise both sides of that historical comparison. Historical Keras execution conditions are not fully verified, and Keras was not rerun here. This is not a leakage-free or fully matched comparison. The overlap fix and a new matched evaluation belong to a separate PR/task.

The superseded standalone 100-seed directory and historical Keras/DDESONN vignette were retired from the current tree in 7.3.19. Their historical evidence remains in Git history and the verified standalone bundle.

## 7.3.19 explorer consolidation

The paired CSV is byte-identical to `inst/shiny/benchmark/data/heart-failure-1000.csv`:
SHA256 `7809854ff9f322576aaa046135a2d84486246a9c61987910e4b2e3f086f779a9`.
It joins all 1,000 seed IDs from the validated archived legacy probabilities and
new standard-AdaGrad test results. `training_seconds` retains the export schema
and contains the new worker's elapsed status time, not pure optimizer time.

The 7.3.19 package installs/builds successfully. All 37 targeted dashboard and
vignette-link assertions pass. A fresh installed-package Shiny process serves
the four current guides and the rendered consolidated guide with HTTP 200.
Package inspection verifies the exact full-seed CSV, four vignette sources,
absence of the retired CSV/vignette, and exclusion of dev/raw experiment files.
No training, optimizer, historical split or frozen reproducibility-bundle changes
were made for this consolidation.
