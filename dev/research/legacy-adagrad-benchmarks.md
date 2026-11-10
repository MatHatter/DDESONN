# Legacy AdaGrad and the heart-failure/Keras research record

This audit is for the focused 7.3.0 cleanup. Historical raw files are retained
unchanged in the repository; `.Rbuildignore` excludes them from the R source
package and installations. No existing commits, tags or observations are removed
or rewritten. Standard `adagrad` remains the active optimizer; `adagrad_x` is not
restored. No parallel-processing feature is added.

## Evidence and archive

Use immutable revision **0d1b557098b56b39ed2357b30716dcddc25871b3** (v7.1.11
submission preparation), which is already an ancestor of this branch:

- [Legacy update block](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/R/update_weights_block.R#L112): the multilayer path subtracts the direct gradient/regularization update before optimizer dispatch.
- [Legacy AdaGrad dispatch](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/R/update_weights_block.R#L248) calls `apply_optimizer_update()` afterward.
- [Second weight subtraction](https://github.com/MatHatter/DDESONN/blob/0d1b557098b56b39ed2357b30716dcddc25871b3/R/optimizers.R#L710): the adaptive update is also subtracted from the same weights.
- The same archived code reconstructs some update matrices row-wise and the
  AdaGrad bias path uses summed layer errors. The double-update finding refers
  specifically to the verified multilayer path, not every historical model mode.
- The archived `TestDDESONN_1000seeds.R` selects `optimizer <- "adagrad"` and
  `ML_NN <- TRUE`. This supports associating the study with that legacy path;
  it does not prove the exact executable version for every saved output.

The archive contains all ten files in `legacy-benchmark-inventory.csv`, totaling
**2,214,321 bytes**: five RDS files, one workbook and four research scripts. The
inventory records sizes, SHA-256 hashes and Git blob IDs. All ten current files
were compared byte-for-byte with this archive and match.

To extract without switching or resetting your current checkout:

```sh
mkdir -p ../ddesonn-legacy-research
git archive 0d1b557098b56b39ed2357b30716dcddc25871b3 \
  inst/extdata/heart_failure_runs \
  inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras \
  vignettes/DDESONNvsKeras_1000Seeds.Rmd \
  | tar -x -C ../ddesonn-legacy-research
Rscript dev/research/summarize-legacy.R ../ddesonn-legacy-research
```

This reproduces the **summary of saved results**, not the original training.
The per-observation prediction file for run1 referenced by `DDESONNproof.R` is
absent from this compact archive, but is recoverable from earlier revision
`4bc994462612ce868b90bc70885eb1635eddcdee` at
`inst/extdata/heart_failure_runs/run1/SingleRun_Pretty_Test_Metrics_500_seeds_20251025.rds`.
That file was read successfully: 375,000 rows, 500 distinct seeds, range 1–500.
Use `git show REVISION:PATH > run1-predictions.rds` to retrieve it without adding
it back to the package. Both prediction-level batches therefore remain in Git. The recovered run1 file
has 1,807,742 bytes, Git blob `822ace3ea1fcc508e2b8c731f2e970575e559bbf`, and
SHA-256 `f63d78cc71a039a3002661b0793eb480433335153325cb1800d94de64c180a83`.
The proof and table scripts also expect preloaded session objects. They are
archived research utilities, not installed public package entry points.

## What the two 500-seed batches establish

The experiment aimed to study stability and predictive-performance variation
across random initializations of a native-R network, using Keras as a reference.
The two stored train/test metric pairs each have 500 rows with unique seed IDs:
run1 covers 1–500 and run2 covers 501–1,000. Combining them yields 1,000 distinct
seed labels. These are research milestones and remain available as evidence of
that development process.

The two batches account for the composite's size. The records inspected do not
explain **why two separate jobs** were chosen over one 1,000-seed job. We do not
invent a hardware/time limitation or infer statistical independence solely from
nonoverlapping seed labels. The user describes independent executions; the saved
IDs substantiate disjoint batches but not their complete execution conditions.
Filename dates are labels, not verified execution timestamps.

The legacy DDESONN test-metric files summarize mean accuracy 0.999218731, sample
SD 0.001256087794 and range 0.992–1. The workbook's Sheet2 reports Keras count
1,000, mean 0.996808557443, SD 0.003569492812 and range 0.978667–1. These values
are historical observations, not a claim of correctly implemented AdaGrad beating
Keras. The package vignette retains a compact, clearly qualified table.

The old aggregation helper uses the maximum test accuracy per seed group. Each
retained test table has exactly one row per seed, so that selection does not
change these files, but future comparisons must never select models by test
accuracy. Do not attribute the reported differences solely to the optimizer;
matching all preprocessing, data splits, tuning and model-selection conditions
has not been established from the artifacts.

## Seed discrepancy: evidence, chronology and uncertainty

- Following renames/copies, the dedicated `TestDDESONN_1000seeds.R` path first
  appears at `1a41c9fe0c8e74190137370a2d507a7865b6bcae` under
  `helpfulFiles/vKeras/1000SEEDSRESULTSvkeras/`, already containing
  `set.seed(111)`, `seeds <- 501:1000`, and `set.seed(s)`. The general
  `inst/scripts/TestDDESONN.R` ancestor inspected at `36e0b16132e07aa4986dcd01bed6dcbe1883dca6`
  also has master/per-run seeds. Copy detection is evidence of source similarity,
  not proof that either exact file produced the historical artifacts.
- Relocations subsequently moved that dedicated script to `dev/helpfulFiles/`
  (`08c321c`), `inst/dev/helpfulFiles/` (`80e41d1`), `inst/extdata/vsKeras/`
  (`4bc9944`), then `inst/scripts/vsKeras/` (`b1a069a`). Its seed controls were
  inspected at the first dedicated snapshot and these later archived locations.
- The older vignette (`DDESONNvKeras_1000Seeds.Rmd` at `b1a069a`) says
  the original runs had no master seed and the scripts were updated later.
  The renamed vignette at 0d1b557 repeats that narrative. The available history
  does **not** show an unseeded-to-seeded transition for this benchmark script.
- The RDS files store seed labels but no saved RNG state, full run configuration,
  session information or source commit. We cannot map the exact original execution
  configuration to a verified script revision, or settle whether the master seed
  was present when the recorded jobs actually ran.
- The Python script already contains `MASTER_SEED = 111` and per-run seeds at its
  first dedicated snapshot (`1a41c9f`), but `range(1, 1)` is empty. Its local Windows data path
  and empty loop require configuration. The committed script, as written, does
  not regenerate the workbook. The workbook is not proof of its launch settings.

Additional per-seed metadata is preserved at `1a41c9f` under
`inst/extdata/sample_runs/heart_failure_single_runs/`. Samples for run1 seed 1
and run2 seed 501 were inspected: they record learning rate .125, 360 epochs and
one network. Fields including hidden sizes, ML mode and several sample counts
are NULL; optimizer identity, master RNG state and source commit are absent.
Their predictor functions refer to the global environment, not a self-contained
saved execution configuration. These records support partial reconstruction,
not a complete claim about the original RNG/configuration. No replacement raw
metadata files are added to the repository or installation.

Consequently we remove the unsupported categorical "no master seed" assertion.
We do not claim that the original runs were unseeded, irreproducible, or precisely
reproducible from today's scripts. Later script edits cannot establish what was
executed earlier, and there is no basis for asserting a precise historical
configuration that the artifacts do not record.

## Current mathematical behavior

Current AdaGrad uses one update per tensor: accumulate `r <- r + g^2`, then
subtract `lr * g / (sqrt(r) + epsilon)`. Each weight and bias has separate state.
Bias updates use the actual bias gradients, not summed raw layer errors. Coupled
regularization, if requested, enters the gradient once before this update.
The double-subtraction compatibility path and its aliases are not active.

Existing targeted regression coverage checks independent two-step numerical
answers for weights and biases in both model modes, accumulator values, bias
batch-duplication invariance, regularization, shapes and end-to-end dispatch.
The separate 60-digit reference fixture checks 40-step trajectories and state,
including zero/tiny/signed gradients, rate changes and state serialization. No
optimizer implementation changes were required for this release cleanup.

## Distribution decisions

Exclude only these two historical subtrees:

- `inst/extdata/heart_failure_runs/`: four train/test metric tables plus run2's
  375,000-row prediction table. They are generated observations, not input data.
- `inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/`: the historical workbook and four
  experiment/session-analysis scripts. Their consumer was the old benchmark
  vignette; the replacement is self-contained and does not load excluded files.

Retain the input CSV datasets (heart failure, stock and segmentation data),
current example scripts and generic reporting helpers. Their use is separate from
these historical results; removing them for a size target would be inappropriate.
The source audit found no tracked generated vignette image/HTML/PDF payload to
remove. Normal vignette outputs are generated at build time. Existing ignore
rules continue to exclude local caches, plot dumps and render intermediates.

See `package-size-and-validation.md` for actual before/after build and installed
sizes, excluded-content verification, test outcomes and remaining limitations.
