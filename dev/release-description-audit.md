# Release-description audit: 7.1.1 through 7.3.5

Current package version: **7.3.5**. PR #17 should be titled **Finish 7.3.5 Shiny training feedback and checkpoint controls**. The current package summary now explicitly describes the optional workbench while keeping checkpoint selection scoped to that workbench. `RoxygenNote: 7.3.3` is the documentation-generator version, not the package version, and is intentionally unchanged.

## What was checked

Compared all 21 tag targets from v7.1.1 through v7.3.4 with their root DESCRIPTION files, successive tagged source diffs, tagged NEWS where present, commit history and public GitHub tag pages. The visible descriptions on those pages are annotated-tag messages (`commit-desc`), not editable GitHub release bodies. No tag object, target or historical source file was rewritten. Historical claims of CRAN acceptance or zero-warning checks were not rerun and are not asserted as fresh validation.

## Tagged versions and accurate summaries

| Tag | Package Version in tagged DESCRIPTION | Source commit | Accurate scope |
|---|---|---|---|
| v7.1.1 | 7.1.0 | `abc650a73c9f` | CRAN documentation cleanup and NAG argument correction. |
| v7.1.2 | 7.1.2 | `4bc994462612` | Package documentation and structural cleanup. |
| v7.1.3 | 7.1.3 | `850949f38455` | README structure and Techila documentation. |
| v7.1.4 | 7.1.4 | `e39edad44333` | DESCRIPTION wording and submission documentation. |
| v7.1.5 | 7.1.5 | `98ed101aca79` | Description and vignette naming cleanup. |
| v7.1.6 | 7.1.6 | `e78d63c6d7dd` | Prebuilt vignette documentation. |
| v7.1.7 | 7.1.7 | `17e99904bf4a` | Vignette index and log documentation. |
| v7.1.8 | 7.1.9 | `6866cbc1c7fe` | Quieter output and artifact-path hardening. |
| v7.1.9 | 7.1.9 | `6866cbc1c7fe` | Session-temp artifact roots and quieter diagnostics. |
| v7.1.10 | 7.1.9 | `eff3963a8d81` | Reproducible artwork generator. |
| v7.1.11 | 7.1.11 | `0d1b557098b5` | Submission-preparation documentation and vignettes. |
| v7.2.0 | 7.1.11 | `c1fc01f83b77` | Optional native R SSM sequence encoder. |
| v7.2.1 | 7.1.11 | `20e5a8e39799` | Per-split sequence bridge. |
| v7.2.2 | 7.1.11 | `17da2e693619` | Retain the SSM training scaler. |
| v7.2.3 | 7.1.11 | `53aa8ede2d76` | SSM recurrent backward dimensions. |
| v7.2.4 | 7.1.11 | `5dd541a7fe5c` | Preserve SSM Adam parameter dimensions. |
| v7.3.0 | 7.3.0 | `a95bc8dda84c` | Standard optimizers, SSM fixes and historical artifact cleanup. |
| v7.3.1 | 7.3.1 | `0b9f59c27863` | 100-seed heart-failure research report. |
| v7.3.2 | 7.3.2 | `1496983f7833` | Optional Shiny experiment dashboard. |
| v7.3.3 | 7.3.3 | `196cf529ab18` | Interactive Shiny model training workbench. |
| v7.3.4 | 7.3.4 | `ef2ec7b7abfd` | Guided Shiny workflow and contextual help. |

The eight mismatches are v7.1.1, v7.1.8, v7.1.10 and v7.2.0–v7.2.4. v7.1.8 and v7.1.9 also resolve to the exact same commit. Their metadata cannot be changed retroactively without changing historical tags. The table preserves both labels instead of silently pretending they agree.

## Reviewable replacement descriptions

These descriptions can be used as release-page text without rewriting tags, if a separate editable GitHub release is created or updated in a later authorized publishing task. They are proposals, not published edits. The label/package-version differences are explicit.

### v7.1.1 — CRAN documentation cleanup and NAG argument correction

Contains the spelling word list, package global-variable declarations and internal erf helper; corrects the NAG call to use beta1. Historical CRAN-check claims are not new validation results.

Metadata note: tag **v7.1.1** contains package **7.1.0**. Preserve the tag; do not describe this as a correctly versioned 7.1.1 package archive.

### v7.1.2 — Package documentation and structural cleanup

Refines the package documentation sentinel, API help links and aliases, example documentation and Techila script layout. Includes reporting-format changes; it is not an optimizer-algorithm release.

### v7.1.3 — README structure and Techila documentation

Updates the repository-layout documentation and canonical README guidance; removes bundled development/Techila copies. Historical check/submission statements are not independently revalidated here.

### v7.1.4 — DESCRIPTION wording and submission documentation

Refines package-description wording, optional Techila documentation, word list, build exclusions and package help. Removes Techila from Suggests; does not introduce a new training algorithm.

### v7.1.5 — Description and vignette naming cleanup

Corrects description wording and the Keras-comparison vignette/README naming. No new model feature is introduced.

### v7.1.6 — Prebuilt vignette documentation

Adds prebuilt inst/doc HTML and its index and adjusts the comparison vignette. This is vignette packaging work.

### v7.1.7 — Vignette index and log documentation

Adjusts build exclusions, vignette index/generated HTML, package help and README/log metadata documentation.

### v7.1.8 — Quieter output and artifact-path hardening

Contains verbosity-gated diagnostics and the session-temp artifact-root/path resolver changes. This tag points to the same source commit as v7.1.9; it is not a separate earlier code state.

Metadata note: tag **v7.1.8** contains package **7.1.9**. Preserve the tag; do not describe this as a correctly versioned 7.1.8 package archive.

### v7.1.9 — Session-temp artifact roots and quieter diagnostics

Uses a session-temp default artifact root, hardens path handling and gates informational output by verbosity. The tagged source is identical to v7.1.8.

### v7.1.10 — Reproducible artwork generator

Adds inst/scripts/artwork/create_ddesonn_brain_art.py relative to v7.1.9. Does not change the R training implementation or bump its package metadata.

Metadata note: tag **v7.1.10** contains package **7.1.9**. Preserve the tag; do not describe this as a correctly versioned 7.1.10 package archive.

### v7.1.11 — Submission-preparation documentation and vignettes

Bumps package metadata to 7.1.11 and updates package/artifact-root help, README, artwork script and vignette names/content. This is the historical archive later referenced by the research documentation.

### v7.2.0 — Optional native R SSM sequence encoder

Adds sequence-data helpers, the optional native R state-space encoder, API/prediction integration, help, examples and encoder tests. The engineered-feature path remains the default.

Metadata note: tag **v7.2.0** contains package **7.1.11**. Preserve the tag; do not describe this as a correctly versioned 7.2.0 package archive.

### v7.2.1 — Per-split sequence bridge

Extends the sequence bridge across training, validation, test and prediction, with supporting API/encoder changes, examples and bridge tests.

Metadata note: tag **v7.2.1** contains package **7.1.11**. Preserve the tag; do not describe this as a correctly versioned 7.2.1 package archive.

### v7.2.2 — Retain the SSM training scaler

Fixes retention of the fitted training scaler in the sequence bridge and adds regression coverage.

Metadata note: tag **v7.2.2** contains package **7.1.11**. Preserve the tag; do not describe this as a correctly versioned 7.2.2 package archive.

### v7.2.3 — SSM recurrent backward dimensions

Corrects the recurrent backward-pass dimensions and expands encoder tests, including non-square shapes.

Metadata note: tag **v7.2.3** contains package **7.1.11**. Preserve the tag; do not describe this as a correctly versioned 7.2.3 package archive.

### v7.2.4 — Preserve SSM Adam parameter dimensions

Preserves parameter dimensions through SSM Adam updates and adds shape-invariant coverage.

Metadata note: tag **v7.2.4** contains package **7.1.11**. Preserve the tag; do not describe this as a correctly versioned 7.2.4 package archive.

### v7.3.0 — Standard optimizers, SSM fixes and historical artifact cleanup

Carries the optimizer implementation/reference-test work and completed SSM fixes, defaults to standard AdaGrad and rejects adagrad_x, documents historical research limitations and excludes large research artifacts from the package payload. The tag includes more than its final artifact-cleanup commit.

### v7.3.1 — 100-seed heart-failure research report

Adds the paired legacy-versus-standard AdaGrad research report and supporting summaries/artifact provenance, using the unchanged 7.3.0 implementation. Does not rerun the historical 1,000-seed Keras comparison.

### v7.3.2 — Optional Shiny experiment dashboard

Adds the optional native R benchmark-results viewer, bundled paired CSV and artwork, input validation, filters/downloads and dashboard tests. Model-training pages arrive in 7.3.3.

### v7.3.3 — Interactive Shiny model training workbench

Adds data preparation, public-API training/evaluation, background callr jobs and cancellation, saved bundles, prediction and warm starts. The numbered guided workflow and contextual help arrive in 7.3.4.

### v7.3.4 — Guided Shiny workflow and contextual help

Adds the numbered Data, Features, Build & Train and Model / Results workflow, contextual circled-i help, navigation guidance and the callr explanation. Explicit checkpoint choices, recipes, smaller help controls, switches and improved training feedback are the 7.3.5 follow-up, not part of this tag.

## 7.3.5 changes and validation

Updated DESCRIPTION, the README current-version summary, the workbench guide, the proposed PR title/body and NEWS. NEWS from the 7.3.4 heading onward is restored byte-for-byte from the actual v7.3.4 source; the unreleased follow-up features are under 7.3.5. The prior 7.3.4 validation report remains historical evidence for the identical runtime code. It is not relabeled as a new 7.3.5 full package check.

Post-bump checks passed: R CMD INSTALL completed; the installed package and a newly trained model bundle both report 7.3.5; a two-epoch iris training/checkpoint/evaluation smoke check passed. All R implementation, workbench runtime and test files are unchanged from the previously fully validated source. The entire NEWS history from the 7.3.4 heading onward matches the tagged 7.3.4 source exactly. The 7.3.5 metadata/documentation revision does not rerun the expensive full vignette/check suite; that earlier result remains explicitly pre-bump evidence. No R implementation, workbench runtime code or tests changed in this version/documentation-only revision.

## External editing status

PR #17 exists for codex/workbench-readiness-7.3.4. Keep that branch so its commits update the existing PR. Both `gh pr edit 17` (GraphQL) and a direct REST PATCH of PR #17 returned Forbidden for the prepared 7.3.5 title/body. The live PR title and empty description could not be changed from this environment; the exact replacement title/body are saved in `dev/PR_7_3_5_DESCRIPTION.md`. A REST read of release records also returned Forbidden. Public tag pages remain readable, and their existing annotated descriptions are unchanged. No duplicate PR, release publication or tag rewrite is authorized or performed.
