# Focused help and full-code viewer: 7.3.7 validation

Base: released 7.3.6/main at `a8c600d5563755b9dca5542d08a0764c2dea66e3`.
Branch: `codex/focused-help-7.3.7`. Package version is 7.3.7.

## Relevant R validation

The installed-package workbench/dashboard suite passed **16 tests and 398
assertions**, with **zero failures, errors, warnings or skips**. It checks that
all 31 topics have distinct parsed R fragments, each at most 14 lines for the
representative iris settings, and no actual training calls. Parameter-help
regression checks prevent Epochs or Optimizer from accumulating unrelated fit
settings. Model and RNG state remain unchanged by code generation.

Complete next-model scripts are tested separately across binary, multiclass and
regression tasks in best-validation and final-epoch modes. Their preprocessing,
selected epoch, predictions and metrics match the workbench. Completed recipes
remain exact saved snapshots after controls change; older bundles cannot invent
missing replay recipes. Existing workbench recipe, background-job, cancellation,
serialization, prediction, checkpoint and dashboard tests also passed.

`R CMD INSTALL` succeeds. R training/optimizer implementation files are unchanged
from released 7.3.6. Full research-vignette rendering, PDF-manual generation and
full package checks were not repeated for this UI change. Earlier results remain
historical evidence rather than new 7.3.7 checks.

## Installed-app browser checks

Chromium/Playwright exercised desktop (1366 x 1000, device scale 4), mobile/touch
(390 x 844, device scale 2) and landscape (844 x 390) sessions:

- SVG circle has equal width/height and uniform screen scaling; its text is a real i.
- At device scale 4, ring and glyph ink centers both measure (44, 40) in the icon screenshot.
- Full code opens before preparation with an explicitly labeled iris illustration.
- All 31 help titles load distinct short fragments inside the viewport without creating a job or model.
- Epochs and Learning rate show only their own settings and follow valid current values.
- The next-model script download exactly matches its displayed full code.
- The completed recipe retains two epochs after the next-model control changes to nine.
- Source switching shows next/completed/saved-prediction code in the dedicated viewer.
- Invalid next-model settings show guidance and disable script downloading.
- Completed model downloads are byte-for-byte identical before/after viewing full code.
- Keyboard focus, Escape return/dismissal, touch and portrait-to-landscape placement pass.
- No JavaScript errors were observed.

Both downloaded full-viewer scripts (`next.R`, `completed.R`) were executed
independently with the public API and reproduce the saved browser model's epoch,
Test predictions and metrics within 1e-12.

Evidence is under `/workspace/.setup/help-737/`: `tests.log`, `tests.rds`,
`browser.log`, `browser-results.json`, `replay.log`, `icon-verified.png`,
`focused-help.png`, `full-viewer.png`, and downloaded script/model files.
No localhost previews are published.

## Review status

The exact 7.3.7 PR title/body are in `dev/PR_7_3_7_DESCRIPTION.md`.
The branch is pushed. The public open-PR list was empty before creation was
attempted. Both GitHub GraphQL (`gh pr create`) and REST POST creation returned
Forbidden, so no PR was created. The branch and exact title/body are ready for
a PR against main. Existing tags are preserved; no merge, release tag or release
publication is performed.
