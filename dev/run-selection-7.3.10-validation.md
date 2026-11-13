# DDESONN 7.3.10 validation

Base: origin/main at 234b69c (7.3.9, PR #21). Review branch:
`codex/run-selection-7.3.10`. No release tags are created or moved.

## Implemented behavior

- Existing panel placement is retained. The completed-model panel now lists
  successful training, warm-start and loaded-model snapshots with radio buttons.
- Selection restores Build & Train controls and activates that model's loss,
  checkpoint summary, evaluation, predictions, warm-start source and downloads.
  Editing controls configures the next new model on the currently prepared data;
  selecting a model does not replace Features or data preparation.
- Latest attempt has its own numbered status/error area. Attempt history retains
  outcomes and details. Starting another attempt clears the prior active error;
  failure/cancellation never replaces any completed model.
- Selected loss history remains visible during subsequent jobs. Bundles lacking
  history show an explicit empty state. Hover enlarges a dot with a normal
  cursor and tooltip, without redrawing the plot or changing selection/weights.
- Radio-list rebinding cannot reset a user's draft by echoing the selected value.
  Mobile row margins avoid the previous three-pixel horizontal overflow.
- Missing validation checkpoints receive specific guidance; no numerical training
  changes or automatic checkpoint fallback are introduced. The original user's
  failure cannot be diagnosed from a screenshot alone.

## Tests and installed-app checks

- R CMD INSTALL succeeded. The final installed app was tested in Chromium using
  Playwright at 1366×1100 and 390×844; desktop/mobile screenshots were inspected.
- Focused workbench tests: 372 assertions, zero failures/errors/warnings/skips.
  Dashboard tests: 26 assertions, zero failures/errors/warnings/skips.
- Final run-selection tests: 49 assertions, zero failures/errors/warnings/skips.
  They use actual models and callr workers for failure/success/cancellation,
  check older snapshots byte-for-byte, chart ownership, hover immutability,
  selection echoes, old bundles without history and session isolation.
- Browser pass: invalid attempt followed by success clears the active warning;
  both best-validation and final-epoch training; radio selection restores
  hidden layers, epoch budget, learning rate and checkpoint mode; chart and
  results identify the selected run; hover enlarges the point while the plot
  image and selection remain unchanged; cancellation preserves existing runs;
  selected-model download, reload, prediction and a successful warm start;
  keyboard selection, mobile selection without horizontal overflow and a
  separate session with no shared models. No JavaScript errors occurred.
- Existing executable-recipe tests reproduce preprocessing, predictions, metrics
  and selected epochs for all three tasks and both checkpoint modes.
- `git diff --check` passed.

Evidence: `/workspace/.setup/runs-7310/` contains installation logs, `tests.log`,
`selection-final.log`, `browser-final-checked.cjs`, `browser-final-checked.log`,
`desktop.png`, `mobile.png`, the downloaded bundle and build/check logs.

## Full package build and check

Pending. The normal build enables vignette rendering. The subsequent check will
include tests, vignette rebuilding and the full PDF manual. This environment's
previously established `R_RD4PDF=times,hyper` font configuration uses installed
Times/Courier fonts; the default Inconsolata font dependency is unavailable.
Nothing is skipped by that font configuration.

## Review status

The public open-PR page was inspected and showed no existing PR for this branch.
The exact title/body are in `dev/PR_7_3_10_DESCRIPTION.md`. Push and PR creation
status are pending. No merge, publication or release-tag changes are performed.
