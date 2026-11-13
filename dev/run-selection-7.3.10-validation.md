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
- Final run-selection tests: 50 assertions, zero failures/errors/warnings/skips.
  They use actual models and callr workers for failure/success/cancellation,
  check older snapshots byte-for-byte, chart ownership, hover immutability,
  selection echoes, preparation without relabeling the latest attempt, old bundles
  without history and session isolation.
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
`selection-checked.log`, `browser-final-checked.cjs`, `browser-final-checked.log`,
`desktop.png`, `mobile.png`, the downloaded bundle and build/check logs.

## Full package build and check

The normal vignette-enabled build completed successfully and produced
`DDESONN_7.3.10.tar.gz`. Archive comparison verified 11 final source,
documentation and test files byte-for-byte. SHA-256:
`0951e5c140c98c599fbf4334f1253eb1f58290a7d48cb54ef09e9b060893063e`.

The user clarified that vignette building was not requested for this UI change.
The full package check was therefore stopped at their direction during the full
test-suite stage. Installation, loading, code, Rd/documentation and examples had
reported OK. No completed full-check or full-suite pass is claimed. The focused
448 assertions and installed-app browser validation above passed independently.

The check used the existing `R_RD4PDF=times,hyper` font configuration. CRAN and
Bioconductor index probes were blocked by the environment; declared dependencies
were available locally.

## Review status

The public open-PR page was inspected and showed no existing PR for this branch.
The exact title/body are in `dev/PR_7_3_10_DESCRIPTION.md`. Branch
`codex/run-selection-7.3.10` is pushed. `gh pr create` failed with
`Post "https://api.github.com/graphql": Forbidden`; no PR was created.
No merge, publication or release-tag changes are performed.
