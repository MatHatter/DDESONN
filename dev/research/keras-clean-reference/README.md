# Original training settings, with predictor overlap removed

**Complete: all 400 fresh fits validated.** The requested experiment is 100 training seeds in
each condition, 400 fresh fits total:

1. Original Keras mean.
2. Equivalent DDESONN `average_gradients=FALSE` (sum).
3. Equivalent DDESONN `average_gradients=TRUE` (mean).
4. Equivalent Keras sum.

**Only the data split changes from the original Keras training protocol.** The
previous overlapping-split run does not satisfy this request. No measurements
from it, or from the earlier differently configured Section Three study, are
reused in this table.

[Results](RESULTS.md) · [Current accuracy CSV](per-seed-test-accuracy.csv) ·
[Protocol](protocol.json) · [Independent overlap audit](independent-overlap-audit.json)

## Shared clean split and enforced checks

Identical vectors of all twelve predictors, excluding the outcome, stay together
in one partition. The frozen group-disjoint assignment from split seed 42 is
versioned in `split-rows.csv`. It contains every original row ID exactly once.

| Partition | Rows | Distinct predictor vectors |
|---|---:|---:|
| Train | 3,500 | 923 |
| Validation | 698 | 197 |
| Test | 802 | 199 |

Keeping duplicate groups intact changes the validation/test counts from 750/750.
It does not change the training settings. The same row IDs, training-derived
StandardScaler transform, labels and initial tensors are shared across all four
conditions.

**All three pairwise intersections are zero:** train/validation, train/test and
validation/test. This is checked for both raw predictor vectors and the actual
float32 model inputs, so preprocessing/casting cannot silently reintroduce exact
input collisions. The outcome is excluded from duplicate-group identities.
Duplicate rows within a partition are retained; identical predictors never cross
partitions. This is an exact-predictor grouping audit, not proof of independent
patient cohorts.

`reference.inputs` and the separate standard-library `audit_split.py` independently
check overlap. R workers verify the audited split/input/label/dataset hashes;
Keras workers recompute the audit and verify the common input bytes. Each fit
records the split and dataset fingerprints. The summarizer rechecks the overlap
and all fingerprints, rejects old-cohort results, and refuses a complete table
unless all 400 unique condition/seed pairs validate. Deliberately introducing a
train/test duplicate is a tested failure case.

## Original training configuration retained

- Original dataset and twelve predictors; same binary outcome.
- StandardScaler fitted on training rows only, population variance, no max-abs
  rescaling or extra input normalization.
- 12 → Dense64/ReLU → BN → Dropout(.10) → Dense32/ReLU → BN → Dropout(0) →
  Dense1/sigmoid, zero dense biases.
- Trainable hidden BN: momentum .9, epsilon 1e-6, gamma/beta initialized .6,
  moving mean/variance initially 0/1 and used at inference.
- Original VarianceScaling(scale=1.04349, fan_in, truncated_normal), with the
  original Keras per-seed initial tensors shared with DDESONN.
- AdaGrad: fixed LR .125, initial accumulator .1, epsilon 1e-7 inside the square
  root; no clipping, decay, class weighting or sample weighting.
- L1 .00028 on dense kernels only, once per update, unchanged between reductions.
- Training batch **16**, shuffle enabled, retain the short final batch; all
  **360 epochs**, **78,840 updates**, **final checkpoint**, no early stopping.
- Validation every epoch in batches **16**: 44 batches per epoch / 15,840 per fit.
  Final evaluation/prediction in batches **32**, separately within each partition.
- Original BinaryAccuracy threshold convention: probability **> .5**.

The Keras harness executes the original script's builder/compile/fit AST, with
only the supplied partitions changed. Sum changes only BCE data-loss reduction.
DDESONN uses the previously introduced opt-in native float32 reference adapter,
now supplied explicit clean partition boundaries. Its forward/backward/optimizer
math is unchanged, and the ordinary DDESONN trainer is unchanged. These are
adapter results, not results for the unchanged default trainer.

Both backends use float32 arithmetic and shared initial tensors. Shuffle/dropout
use framework-native RNG engines with the same distributions; numeric seeds do
not imply bitwise-identical trajectories.

The independent TensorFlow/native suite passed **146 checks** for both reductions:
forward output, loss, all parameter gradients/updates/accumulators, moving BN
state and multi-batch inference. Both reductions also passed the clean-split
end-to-end smoke check (219 training updates, 44 validation batches and batched
inference for one epoch). Evidence: `native-equivalence-check.json`.

## Reproduce and resume

Current tools: `/workspace/ddesonn-python/bin/python` and
`/workspace/ddesonn-env/bin/Rscript`. Dependencies are the pinned Python packages
in `../section-three/requirements.txt`, R packages DDESONN/Rcpp/jsonlite/digest,
g++, make and BLAS. From the repository root, for a **new** output directory:

```sh
export PATH=/workspace/ddesonn-env/bin:/workspace/ddesonn-python/bin:$PATH
mkdir -p /workspace/keras-clean-100
cp dev/research/keras-clean-reference/split-rows.csv /workspace/keras-clean-100/split-rows.csv
python -c 'import sys; from pathlib import Path; sys.path.insert(0,"dev/research/keras-clean-reference"); from reference import prepare; prepare(Path("/workspace/keras-clean-100"))'
python dev/research/keras-clean-reference/audit_split.py /workspace/keras-clean-100
python dev/research/keras-clean-reference/check_tensorflow.py
Rscript dev/research/keras-clean-reference/check_native.R
Rscript dev/research/keras-clean-reference/run_ddesonn.R /workspace/keras-clean-100 FALSE 1 100
Rscript dev/research/keras-clean-reference/run_ddesonn.R /workspace/keras-clean-100 TRUE 1 100
python dev/research/keras-clean-reference/run_keras.py /workspace/keras-clean-100 mean 1 100
python dev/research/keras-clean-reference/run_keras.py /workspace/keras-clean-100 sum 1 100
python dev/research/keras-clean-reference/summarize.py /workspace/keras-clean-100
```

Check `/workspace/keras-clean-100/status.json`, `workers.json`, `COMPLETE` and
`FAILED` before restarting anything. Eight initial processes (six Keras, two
native) checkpoint results atomically per seed; Keras uses file locks. Do not run
two native workers for the same reduction. Native completed files are skipped.
No earlier results are copied in. The supervisor updates the partial CSV and
writes the final table and completion marker only after all 400 fits validate.
Raw logs, initial tensors and compiled caches stay outside Git. No trained-model
artifacts are committed. This fixed clean split addresses exact predictor overlap;
100 seeds still do not constitute independent data-split replications.

## Seed-count decision and diagnostics

We retain 100 seeds per condition (400 fits). The [seed-count rationale](SEED_COUNT_RATIONALE.md)
reports the retrospective precision assessment, why extending to 250 is optional,
and how a future precision-based design should be specified. See the
[blocks of ten and cumulative intervals](SEED_DECILES.md) and
[observed/synthetic precision figure](seed-precision.png). No further fits were launched.

## Reproduce or contribute extra seeds

The [four public entry points and contribution guide](CONTRIBUTING_SEEDS.md)
prepare additional shared initializations, retain the frozen split/settings,
run independent numerical checks, and export complete four-condition seed bundles.
Extensions remain separate from the published 100-seed cohort.
