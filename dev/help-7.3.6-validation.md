# 7.3.6 help validation

Base: released main/7.3.5 at `2de93a6fa087bd54e720110883cff16c37b02b60`.
Review branch: `codex/help-code-7.3.6`. This starts from the squash-merged release,
so the old branch's eight original commits are not replayed into the new PR.

## Changes and automated checks

Package version is 7.3.6. All 31 help topics have substantial equivalent R code.
The public-API/base-R snippets reuse the existing recipe generators and explicitly
identify prerequisites and illustration versus prepared/completed-model context.
R training/optimizer implementations are untouched.

The relevant installed-package workbench/dashboard suite passed **16 tests and
342 assertions**, with **zero failures, errors, warnings or skips**. It verifies
parsing for every topic and iris illustration, unique output IDs, absence of
private helper calls in snippets, RNG/model immutability, and executable
preparation/fit/evaluation equivalence across all three tasks and both checkpoint
modes. Existing recipe, persistence, background-job, cancellation and checkpoint
checks in those test files also passed.

The package installs successfully with `R CMD INSTALL`. Full research-vignette
rendering and PDF-manual package checks were not repeated for this UI patch;
the earlier 7.3.4/7.3.5 full-check records remain historical evidence, not fresh
7.3.6 validation.

## Installed-app browser evidence

Chromium/Playwright exercised real installed-package Shiny sessions at desktop
1366 x 1000 and mobile/touch 390 x 844 (plus 844 x 390 landscape):

- All 31 help controls load substantial R code and remain within the desktop viewport.
- Opening every help topic before preparation creates neither a job nor a model.
- Fit help follows current epochs/rate and includes restoration of both best tensors.
- Completed-model help retains its saved fit recipe after controls change.
- Additional-fit help uses the requested extra epochs and disables repeat self-organization.
- Downloaded completed bundles are byte-for-byte identical before/after opening help.
- Enter opens help; Escape closes it and returns focus. Mobile taps and scrolling work.
- Popovers remain inside the viewport after portrait-to-landscape rotation.
- No JavaScript errors were observed.

At device scale factor 4, the rendered ring and lowercase-i ink bounding boxes
have coincident centers at screenshot coordinates (36, 32). This checks the
visible glyph rather than just its CSS line box.

Final installed-app keyboard checks passed: Tab reaches the help region and then
the code region; PageDown scrolls the long popover; arrow keys scroll wide code;
Escape closes help and returns focus to its summary. Final landscape bounds also
passed after the focusability change.

Evidence is retained under `/workspace/.setup/help-736/`: `tests-final.log`,
`tests-final.rds`, `browser-verified.log`, `browser-results.json`,
`icon-centered.png`, `desktop-help.png`, `mobile-help.png`, and `keyboard.log`.
No localhost preview links are published.

## Review/publication

The existing release tags, including v7.3.5, are preserved. This is a separate
7.3.6 review branch; no merge, release tag or release publication is performed.
PR creation will be attempted after pushing; GitHub API access was previously
Forbidden. The exact proposed title/body are in `dev/PR_7_3_6_DESCRIPTION.md`.
