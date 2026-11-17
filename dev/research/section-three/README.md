# Section Three: archived settings, sum versus average

[100-seed results](RESULTS.md). All 400 condition/seed pairs are complete and
validated: seeds 1–10 are retained from the prior PR, and seeds 11–100 add 360
new fits. See [the accuracy CSV](per-seed-test-accuracy.csv) for individual results.
Mean test accuracy is 89.45% for DDESONN sum, 81.83% for DDESONN average,
87.62% for Keras mean, and 87.48% for Keras sum.

All four conditions use 360 epochs on the same
clean split (seed 42): 3,500 train, 698 validation and 802 test rows, with no exact
predictor overlap across partitions. The corrected single-update engine is kept.

DDESONN uses its archived native settings: full batches, best-validation checkpoint,
class weights 2/1, train sample-SD/max-abs scaling, dropout .1 then zero. Its input-BN
flag is retained; the learning matrix is unchanged. Configured LR decay does not
reach the optimizer, whose measured LR stays .125. Keras retains batches of 16,
hidden BN, StandardScaler, no class weights, and the final checkpoint. Both use
12–64–32–1, AdaGrad, LR .125, L1 .00028, scale 1.04349 and threshold .5.
Keras sum changes only BCE reduction to `sum`; kernel L1 stays applied once per batch.
These comparisons isolate reduction within each framework, not full framework parity.

Install this checkout's R package, R package `callr`, and the Python packages in `requirements.txt`.
Run from the repository root; output stays outside the checkout:

```sh
Rscript dev/research/section-three/section_three.R /tmp/section-three-next prepare
Rscript dev/research/section-three/section_three.R /tmp/section-three-next FALSE 11 100
Rscript dev/research/section-three/section_three.R /tmp/section-three-next TRUE 11 100
python dev/research/section-three/keras.py /tmp/section-three-next mean 11 100
python dev/research/section-three/keras.py /tmp/section-three-next sum 11 100
```

The extension adds seeds 11–100. Use 1 10 in a separate output directory to
reproduce the original table. Scripts write local CSV metrics, row IDs/data and R text logs, not
model checkpoints. Append new result rows without overwriting seeds 1–10.
The Keras script reads the existing archived builder from `inst/scripts/vsKeras`.

`dev` is excluded from R package builds by `.Rbuildignore`. No per-seed models,
RDS/NPZ/Keras artifacts, predictions, histories or raw logs are included here.
Temporary raw cloud artifacts and their cleanup status are noted in
[the session notes](../../SESSION_HANDOFF.md).

## Retaining and validating results

`seed-1-10-retained.csv` preserves all forty percentages from the previous PR.
With 802 test rows, each two-decimal percentage uniquely identifies an integer
correct count. The summary uses those exact counts without rerunning or replacing
the original ten seeds. Their original selected epochs are unknown and remain blank.

After the four ranges finish, validate every result and build the 100-seed table:

```sh
python dev/research/section-three/summarize.py /tmp/section-three-next
```

This rejects missing/duplicate seeds, incomplete split metrics, wrong epoch or
Keras update counts, and overlapping predictor groups. `--partial` writes the
accuracy CSV for completed fits but leaves the ten-seed table unchanged until
all 400 results exist. Raw logs and intermediate artifacts stay outside Git.

The completed cloud run used `/workspace/section-three-100`. `status.json` records
per-condition counts and worker PIDs; `COMPLETE` is written only after all 400
results validate. A `FAILED` file records a stopped/incomplete run. The supervisor
refreshed the compact CSV and generated `RESULTS.md` after completion. No model checkpoints are saved.
