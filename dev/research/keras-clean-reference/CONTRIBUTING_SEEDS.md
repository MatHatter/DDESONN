# Running and contributing additional seeds

The published cohort remains seeds 1–100 in four conditions. Community extensions are optional, separate cohorts; none have been run as part of this addition. Use the frozen group-disjoint split and original settings. Never mix the historical overlapping experiments into this table. These results concern the experimental float32 DDESONN adapter, not the unchanged default trainer.

## Environment

Use Linux with Python 3.12, R 4.5.0, a C++ compiler supporting Rcpp, and a BLAS library exporting `cblas_sgemm`. Install Python packages with:

```sh
python -m pip install -r dev/research/keras-clean-reference/community/requirements.txt
```

The recorded R package versions are DDESONN 7.4.0 (install from this checkout with `R CMD INSTALL .`), Rcpp 1.0.14, jsonlite 1.9.1, and digest (for SHA-256 validation). Install the recorded dependency versions using your preferred R environment manager or CRAN archives. See `protocol.json` for the original compiler and BLAS fingerprint. Native runners check package versions; the independent numerical checks must also pass locally. Different hardware/compiler libraries may produce different training trajectories despite matching versions, so include environment records and disclose differences. GPU execution has not been validated for this protocol; use a CPU environment.

Run commands from the repository root. Select the Python environment through `python` and optionally set `RSCRIPT` to your Rscript executable. Specify a compatible BLAS library; on the original Debian machine:

```sh
export DDESONN_REFERENCE_BLAS=/usr/lib/x86_64-linux-gnu/libblas.so.3
export CUDA_VISIBLE_DEVICES=-1
```

## Prepare one fresh extension

Choose an agreed, unused contiguous range, for example 101–110, and an output directory outside the repository. This generates shared original-Keras initial tensors for every requested seed and verifies the frozen raw/float32 split fingerprints. It does not train models. The completed 1–100 files are untouched.

```sh
python dev/research/keras-clean-reference/community/extension.py prepare /tmp/ddesonn-extra-101-110 101 110
python dev/research/keras-clean-reference/community/check_tensorflow.py /tmp/ddesonn-extra-101-110
Rscript dev/research/keras-clean-reference/community/check_native.R /tmp/ddesonn-extra-101-110
```

The last two commands compare native probabilities, gradients, updates, AdaGrad accumulators, and BN state against TensorFlow (146 checks). They must pass before native training. Do not edit the prepared directory's protocol or manifest. If preparation fails, investigate the error and use a fresh directory after resolving it.

## The four entry points

```sh
python dev/research/keras-clean-reference/community/01_keras_mean.py /tmp/ddesonn-extra-101-110
python dev/research/keras-clean-reference/community/02_ddesonn_false.py /tmp/ddesonn-extra-101-110
python dev/research/keras-clean-reference/community/03_ddesonn_true.py /tmp/ddesonn-extra-101-110
python dev/research/keras-clean-reference/community/04_keras_sum.py /tmp/ddesonn-extra-101-110
```

These are original Keras mean, DDESONN FALSE/sum, DDESONN TRUE/mean, and Keras sum respectively. Run sequentially for simplicity. Each resumes by skipping saved seed files; a per-condition process lock prevents concurrent writers through these entry points. Do not use the old direct runners concurrently against the same directory. Settings remain 360 epochs, batch size 16, final checkpoint, train-only standardization, 3,500/698/802 rows, and zero exact predictor overlap. Every seed must have all four conditions; report failures without discarding unfavorable results. For deliberate reruns of an existing seed range, use a separately identified replication rather than counting duplicate seeds as new observations.

## Validate and contribute

```sh
python dev/research/keras-clean-reference/community/extension.py export /tmp/ddesonn-extra-101-110 /tmp/ddesonn-contribution-101-110
```

Export refuses missing conditions, incorrect training metadata, mismatched fingerprints, invalid accuracy counts, changed initial tensors, or changed protocol code. It includes raw per-fit metrics, a derived CSV, initialization hashes, numerical-check evidence and environment records. This is provenance and consistency checking, not proof against fabricated results. Retain the local prepared tensors and logs for review; no model artifacts are required in Git.

Open a GitHub pull request adding the exported bundle under `dev/research/keras-clean-reference/contributions/<unique-run-name>/`, plus a short README with seed range, machine/OS/compiler/BLAS, commands, runtime, any failures and environment differences. Coordinate seed ranges in an issue first. Reviewers should verify no seed/condition duplicates across accepted contributions, inspect provenance and numerical evidence, and independently reproduce a sample where practical. Contributions from different software/hardware should remain separately identified. Do not overwrite the frozen `per-seed-test-accuracy.csv`, `RESULTS.md`, `protocol.json`, or the 100-seed decile analysis. Any pooled analysis should be a separate, reproducible artifact with explicit inclusion rules and duplicate rejection.

## Why 100 is the published endpoint

See [the block/cumulative table](SEED_DECILES.md) and [precision rationale](SEED_COUNT_RATIONALE.md). All ten blocks show positive average differences for both matched reductions; the final paired estimates are +3.88 and +2.77 percentage points with interval half-widths below 0.86 pp. We retain 100 as a retrospective resource/precision decision. Block agreement supports the interpretation but is not a universal threshold or a calibrated stopping test. Future studies should specify useful precision and practical-effect margins in advance, or use valid sequential methods. More seeds on this split cannot establish performance on other datasets.
