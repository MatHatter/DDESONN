> **Superseded for the requested comparison.** This completed run retained the
> historical predictor overlap by mistake. The intended experiment preserves
> original training settings and removes the overlap; see
> [the corrected run](../keras-clean-reference/README.md).

# Original Keras reference: 100 seeds × four conditions

This archived run mistakenly retained the overlapping original split. The **original Keras protocol** is the
fixed reference for every condition, including batch size **16**, the original
row-order split, and final-epoch evaluation:

| Condition | Data-loss reduction | DDESONN switch |
| --- | --- | --- |
| Original Keras | Original mean | — |
| DDESONN FALSE | Sum | `average_gradients=FALSE` |
| DDESONN TRUE | Mean | `average_gradients=TRUE` |
| Keras sum | Sum | — |

**Complete: all 400 fresh fits validated.** All 400 fits are fresh; the earlier Section Three
100-seed study had different settings and contributes no results here.
[Results](RESULTS.md) and [the accuracy CSV](per-seed-test-accuracy.csv) are generated
from validated fits. The table is marked complete only after every condition has
exactly seeds 1–100.

## Frozen settings

Source: `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py`, kept
unchanged. The harness executes its actual builder/compile/fit AST, supplying the
repository's dataset path and requested seed range instead of the archived local
Windows path and empty `range(1, 1)` loop.

- Original CSV, original row order, `dropna()`, all twelve original predictors.
- Original contiguous split: rows 1–3500 / 3501–4250 / 4251–5000. No clean-split
  substitution. This historical partition has known exact-predictor overlap.
- StandardScaler fitted only on training rows, population variance; no extra
  max-abs scaling. The native adapter receives the same float32 inputs seen by
  the Keras model after its input cast.
- 12 → Dense64/ReLU → BN → Dropout(.10) → Dense32/ReLU → BN → Dropout(0) →
  Dense1/sigmoid, zero dense biases.
- Trainable hidden BN: momentum .9, epsilon 1e-6, initial gamma/beta .6;
  moving mean/variance initially 0/1 and used for inference.
- Original VarianceScaling(scale=1.04349, fan_in, truncated_normal), seeded per
  run. Original Keras initial tensors are shared with DDESONN for each seed.
- AdaGrad: fixed LR .125, initial accumulator .1, epsilon 1e-7 inside the square
  root; no clipping, decay, class weighting or early stopping.
- L1 .00028 on the three dense kernels only, once per optimizer step; no bias/BN
  regularization. Sum changes only the BCE batch reduction.
- Batch size **16**, shuffle enabled; retain the last 12-row batch. **219 updates
  per epoch, 360 epochs, 78,840 updates per fit**.
- Validation runs every epoch, in batches of **16**, as the original `fit`
  defaults specify: 47 validation batches per epoch, 16,920 per fit. It changes
  neither optimizer nor BN state and does not select weights.
- **Final checkpoint** for all conditions; evaluation/prediction batches of **32**
  separately for train, validation and test, including short final batches. Accuracy uses probability **> .5**,
  matching the original Keras BinaryAccuracy convention, including exact ties.

All four use float32 model arithmetic. Shuffling/dropout use their respective
framework-native RNG engines with identical distributions, not identical random
streams. Same seeds do not imply bitwise-identical learning trajectories.

## DDESONN implementation and numerical verification

The old DDESONN training path cannot implement this protocol just by changing
flags. `R/keras_reference.R` now provides an **opt-in native protocol adapter** on
a DDESONN SONN member; `native_reference.cpp` implements its float32 dense/BN
forward and backward passes and AdaGrad updates. It reads/writes the member's
weights/biases and retains BN and optimizer state. It uses no TensorFlow/Keras
training backend. Ordinary DDESONN `train`/`learn` defaults are unchanged.

Consequently these are results for **DDESONN with the new matched-protocol
adapter**, not claims that the unchanged default DDESONN trainer has these
settings or scores. Use the adapter's BN-aware inference, not the old member's
BN-free predictor, for these models.

`check_tensorflow.py` produces independent TensorFlow fixtures using shared
inputs, initial tensors and dropout masks. `check_native.R` checks both reduction
modes over two consecutive optimizer steps (16-row and 12-row batches): forward
probabilities, loss, all ten gradient/parameter/accumulator tensors, four moving
BN tensors, and multi-batch inference (84 rows, evaluated in batches of 32). **146 checks passed**, maximum scaled error
4.68e-5 against a declared float32 tolerance of 2e-4. See
[numerical evidence](native-equivalence-check.json). Existing gradient-reduction
and optimizer regression suites also passed.

## Run and resume

Dependencies are the pinned packages in `../section-three/requirements.txt`, R
packages DDESONN/Rcpp/jsonlite, g++, make and system BLAS. Current local tools are
`/workspace/ddesonn-env/bin/Rscript` and `/workspace/ddesonn-python/bin/python`.
Run from the repository root:

```sh
export PATH=/workspace/ddesonn-env/bin:/workspace/ddesonn-python/bin:$PATH
# Once: prepare shared original initializations and StandardScaler inputs.
python -c 'import sys; from pathlib import Path; sys.path.insert(0,"dev/research/keras-original-reference"); from reference import prepare; prepare(Path("/workspace/keras-original-100"))'
python dev/research/keras-original-reference/check_tensorflow.py
Rscript dev/research/keras-original-reference/check_native.R
Rscript dev/research/keras-original-reference/run_ddesonn.R /workspace/keras-original-100 FALSE 1 100
Rscript dev/research/keras-original-reference/run_ddesonn.R /workspace/keras-original-100 TRUE 1 100
python dev/research/keras-original-reference/run_keras.py /workspace/keras-original-100 mean 1 100
python dev/research/keras-original-reference/run_keras.py /workspace/keras-original-100 sum 1 100
python dev/research/keras-original-reference/summarize.py /workspace/keras-original-100
```

Check `/workspace/keras-original-100/status.json` before launching duplicates.
The live supervisor updates per-condition counts, reports stopped workers, and
writes `COMPLETE` only after all 400 results validate. Results are checkpointed
atomically per seed; Keras workers coordinate through file locks. Run only one
DDESONN worker per reduction; its runner skips existing per-seed results.
`--partial` on the summarizer writes the CSV without claiming completion.
Initial tensors, compiled caches and raw logs stay outside Git; compact metrics,
protocol hashes and numerical evidence are retained. Processes require this
cloud machine to stay running. No trained-model artifacts are committed.

## Historical baseline

The original `1000seedsKeras.xlsx` exists, but its retained audit reports 1,001
rows, 1,000 unique seeds and duplicate seed 100. It is preserved as historical
evidence, not silently repaired or reused as a verified baseline. Both Keras
conditions are rerun here from the original script.

Before acceptance, 200 initial native fits with mathematically row-independent
unbatched inference were superseded and rerun from the ground up to match the original
evaluation batch size 32 exactly. Those files are preserved outside Git in
`/workspace/keras-original-100/superseded-unbatched-inference` and excluded from
the summarizer. Training settings were unchanged.

The native adapter also mirrors the original every-epoch validation passes
(batch size 16 during fit). Preliminary final-only-validation native results
are preserved in `superseded-final-only-validation` and excluded. Both reductions
passed an end-to-end smoke check covering training updates, validation batches,
validation accuracy and final batched inference before acceptance runs started.
