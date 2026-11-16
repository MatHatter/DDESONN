# Section Three: archived settings, sum versus average

[Ten-seed results](RESULTS.md). All four conditions ran 360 epochs on the same
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
Rscript dev/research/section-three/section_three.R /tmp/section-three-next FALSE 11 20
Rscript dev/research/section-three/section_three.R /tmp/section-three-next TRUE 11 20
python dev/research/section-three/keras.py /tmp/section-three-next mean 11 20
python dev/research/section-three/keras.py /tmp/section-three-next sum 11 20
```

The range is an example: choose the next count with the user. Use 1 10 to rerun
this table. Scripts write local CSV metrics, row IDs/data and R text logs, not
model checkpoints. Append new result rows without overwriting seeds 1–10.
The Keras script reads the existing archived builder from `inst/scripts/vsKeras`.

`dev` is excluded from R package builds by `.Rbuildignore`. No per-seed models,
RDS/NPZ/Keras artifacts, predictions, histories or raw logs are included here.
Temporary raw cloud artifacts and their cleanup status are noted in
[the session notes](../../SESSION_HANDOFF.md).
