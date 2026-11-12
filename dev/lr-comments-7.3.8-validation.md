# Learning-rate comment validation: 7.3.8

Base: released main/7.3.7 at `33576a6108879284d87b7fc4064f78475854e071`.
Branch: `codex/lr-comments-7.3.8`.

## Documented behavior

The focused Learning rate code and all newly generated fit recipes/full scripts
now explain the multiplicative retention factor, the 1 = 100% example using the
current rate, before-floor examples for 0.5/0.9, the interval, minimum floor and
scheduler formula. The API default interval is 1 epoch for classification and
20 for regression. Actual decay requires a positive factor below 1 and a minimum
below the starting rate. Workbench settings remain factor 1 and minimum = start.
Existing saved recipes are retained as historical snapshots.

## Checks performed

- R CMD INSTALL succeeds and package metadata is 7.3.8.
- The complete scheduler source parses to an identical executable syntax tree
  before/after its inline-comment additions.
- Across three task modes, three learning rates, two checkpoint modes and
  new/warm fits, all **36 fit scripts** have identical executable syntax trees.
- All **18 learning-rate help variants** have identical executable syntax trees;
  the only changes in those outputs are explanatory comments.
- Actual scheduler checks confirm the 1, 0.5 and 0.9 factors at an explicitly set
  20-epoch interval, its repeated application at epoch 40, unchanged rate before
  the interval, and the fixed-rate effect of setting the minimum equal to start.
- Public API defaults were checked independently for binary, multiclass and
  regression modes to avoid confusing internal scheduler defaults with API defaults.
- Chromium/Playwright checks on the installed app confirm the explanation in
  the Learning rate popover and full-code viewer/download. The example updates
  from 0.01 to 0.0025 when that control changes, and the downloaded script matches
  the displayed code exactly. No model/job or JavaScript error is introduced.

Evidence is in `/workspace/.setup/lr-738/verify.log`, `verify.R`, `browser.log`,
`browser.cjs`, `help.png`, `full.R`, and installation logs. No new full numerical,
research-vignette, PDF-manual or package-check run is claimed for this comment and
documentation revision. No executable scheduler or generated-program changes occur.

## Review status

The exact 7.3.8 title/body are in `dev/PR_7_3_8_DESCRIPTION.md`.
Push and PR-creation status: pending. Existing tags are preserved; no merge,
release tag or publication is performed.
