# Gradient aggregation validation for 7.4.0 preparation

## Behavior

The public Boolean is `average_gradients`: FALSE (new default) sums both dense
weight and bias batch gradients; TRUE averages both. Reduction happens before
existing per-tensor norm handling and regularization. SSM input/encoder gradient
contracts are unchanged. The main `inst/scripts/TestDDESONN.R` script passes the
Boolean to every MAIN/TEMP training call and records it in metadata. Shiny exposes
an off-by-default switch in Build & Train, records settings in bundles and
recipes, and identifies aggregation in completed-model summaries.

Current models save both the Boolean and reduction metadata. Fit-time omission
inherits the saved mode. Pre-toggle 7.3-era workbench models retain the previous
summed-weight/averaged-bias behavior. Actual older serialized methods can continue
unchanged, but cannot accept a newly requested reduction without constructing a
current model; this is an explicit error, not a silent mode switch.

The README version table is scoped to the verified multilayer AdaGrad path.
The bias-update block is unchanged between v7.1.11 (`0d1b557`) and the 7.2.0
parent of `67bab6c`; that 7.3.0 change removed the summed-error bias override.
The new summed default does not restore the old double-update bug or establish
an accuracy advantage.

## Passed checks

- Fresh package installation and standalone experiment-script syntax.
- 77 assertions in the new gradient tests: independent binary BCE derivatives;
  duplication invariants for binary/multiclass/regression in SL and ML layouts;
  strict Boolean validation; serialization; fit-time inheritance and overrides;
  explicit historical mixed compatibility; Shiny switch state and generated code.
- Browser training with AdaGrad/off (summed) and Adam/on (averaged), with matching
  saved summaries and explicit Boolean recipes. Model download/reload, repeatable
  CSV predictions, cancellation, further training, keyboard and 390-pixel layout
  checks passed. The downloaded bundle retained TRUE and mean-reduction metadata.
- An actual serialized pre-toggle model continued in its historical mode.
  An unsupported explicit Boolean change failed without changing its weights.
- Regression checks reproduce and prevent a polling/cancellation race by ensuring
  both new-model and saved-model jobs detach before process shutdown waits.
- Pilot graphics cleanup preserves existing devices and closes only newly
  created devices; the chart-export/clean-pilot test-order reproduction passes.
- A fresh uninterrupted full test-directory run passed with no failures,
  warnings or skips, after the final package installation. JavaScript syntax
  and `git diff --check` passed.

Validation logs and raw benchmark artifacts remain local, outside the checkout;
see `dev/SESSION_HANDOFF.md` for temporary paths and cleanup notes.

The completed ten-seed archived-settings comparison is recorded in
[Section Three](research/section-three/RESULTS.md). Earlier modified-protocol
pilots are separate studies. Only scripts, a small results table and short notes
are retained; no raw model artifacts are committed. A fresh CRAN-style package
check and release are still outstanding.
