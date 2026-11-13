# DDESONN 7.3.11 validation

Base: main at f756364 (7.3.10, PR #23).
Branch: `codex/iris-checkpoint-7.3.11`.

## Evidence

- Source installation succeeded; no vignette rendering or package build/check.
- `test-run-selection.R`: 65 passing assertions, zero failures/errors/warnings/skips.
  This includes retained failed/cancelled first-attempt history without a completed
  model, correct labels and disabled model saving, alongside existing selection,
  immutability, failure/success, cancellation and session-isolation checks.
- Installed-app Chromium test: inject a worker failure after three logged loss
  rows; graph, loss table and failure message remain visible without creating a
  model. Then genuine Iris runs with defaults (16,8; 50 epochs; AdaGrad; .01)
  complete in best-validation and final-epoch modes. Selection restores the
  correct controls/chart; mobile has no horizontal overflow; no JavaScript errors.
- Default Iris reproduction: best validation selected epoch 16 with validation
  accuracy 90%; final epoch selected 50 with validation accuracy 80%.
- Additional runs covering standardization on/off, self-organization on/off and
  both checkpoint choices all completed (eight combinations). Numerical engine
  code is unchanged between the reproduction engine (7.3.10) and this patch;
  the installed 7.3.11 browser also completed genuine default runs in both modes.
- Final error-metadata probe verifies installed engine 7.3.11, R version and
  unchanged friendly guidance in the expanded details.
- `git diff --check` passed.

The user's specific Iris exception is not reproduced or diagnosed from the
collapsed summary alone. An earlier screenshot labeled its saved model 7.1.7,
which suggests checking the installed engine version, but does not prove the
cause of the current failure. The exact technical exception is still needed.

Evidence directory: `/workspace/.setup/iris-7311/` (`reproduce.log`, `grid.log`,
`tests.log`, `browser.log`, `failed-curve.png`, installation logs, `details.log`).
The browser failure is explicitly injected for regression validation; genuine
Iris runs are separate. No full-suite, full-check or vignette result is claimed.

## Review

No existing open PR was visible on the public PR page. The title/body are in
`dev/PR_7_3_11_DESCRIPTION.md`. Push/PR status pending. No merge, publication or
release-tag changes are performed.
