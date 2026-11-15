# Current Shiny regression validation

Validated against DDESONN 7.3.42 with R 4.5.3 and Chromium.

- Added R tests: 22 tests, 163 assertions, zero failures/warnings/skips.
  Each of the 10 optimizers trains and predicts silently with both single-layer
  and multilayer models. Every optimizer also produces ordinary workbench epoch
  progress without tensor/debug diagnostics. Separate-process package loading
  leaves Shiny unloaded, and absent optional dependencies produce guidance.
- Chromium trained all 10 optimizers with clean progress logs while alternating
  final and best-validation checkpoint selections.
- A focused final browser run with AdaDelta passed the complete workflow:
  cancellation preserves the completed model; model download/reload; two-row
  CSV prediction repeated with matching results and a chart; separate-copy
  additional training; blank/valid epoch controls; desktop and 390-pixel mobile
  layout; keyboard focus; no application JavaScript or console errors.
- Chromium's automatic request for the undeclared optional favicon returns 404;
  the runner records that request separately and still rejects other errors.
- Browser testing exposed a stale desktop width on the full-code panel after
  entering the saved-model view. Resetting its inline width and margin when no
  reference panel exists fixes the reproduced horizontal overflow.
- Package reinstall, JavaScript syntax check and `git diff --check` passed.

The existing full R suite passed before these new tests and the panel-width fix.
Only the two new R test files were rerun after the width fix; no new full-suite
or CRAN-style package check is claimed here. See README.md in this directory
for commands to reproduce these checks. Browser tests stay outside the CRAN
source package through the existing `^dev$` build exclusion.

Cloud artifacts are outside the checkout under `/workspace/.onboarding/`:
`added-tests-final.log`, `browser-final.log` (all-optimizer training and workflow
results), and `browser-passed/` plus `browser-passed.log` (the passing focused
workflow, screenshots and machine-readable result).
