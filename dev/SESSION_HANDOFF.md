# Session notes for the next Codex session

- Working version: 7.4.0, unreleased. Use this PR's latest commit as the checkpoint.
- MIT consistency and the initial workbench regression fixes are already merged
  in #62. This PR adds the Boolean gradient switch and subsequent fixes/studies.
- `average_gradients=FALSE` sums both dense weight/bias gradients (default); TRUE
  averages both. API, main TestDDESONN script and Shiny record the setting.
  Old 7.3-era models retain their sum-weight/mean-bias compatibility mode.
- 7.1.11 (`0d1b557`) effectively used sum/sum AdaGrad AND duplicate updates.
  The duplicate updates were fixed in 7.3.0 (`67bab6c`). Do not restore that bug.
- Section Three preserves archived per-framework settings with the clean split.
  One hundred seeds in four conditions completed: mean accuracy 89.45% DDESONN sum,
  81.83% DDESONN average, 87.62% Keras average, 87.48% Keras sum.
  See [the table](research/section-three/RESULTS.md) and
  [run instructions](research/section-three/README.md).
- Keep each framework's settings fixed when adding seeds. Keras batches are 16,
  DDESONN batches 3,500; normalization, class weights and checkpoint selection
  differ. This is promising pilot evidence, not proof of general superiority.
- Earlier modified-protocol results (93.06/82.46/82.17%) and historical overlapping
  thousand-seed runs are separate studies; do not mix them into Section Three.
- Full R tests, focused gradient/pilot tests (104 assertions), browser regressions
  and all forty benchmark artifact checks passed. Lightweight scripts matched the earlier
  runners in short R and Keras sum/average smoke comparisons. Fresh CRAN-style
  check/release is pending.

## Temporary files (not committed or packaged)

- `/workspace/ddesonn-section-three` and `/workspace/ddesonn-section-three-ten-seeds`
  contain the original local model artifacts and logs used to verify the forty
  fits. Keep only if needed for investigation; safe to delete after accepting the
  summarized results. A future session must not depend on these directories.
- `/tmp/ddesonn-light-check*` and `/tmp/section-three-*-light*` are temporary
  script-comparison/smoke-check files; safe to delete after validation.
- Other `/workspace/heart-failure-*` directories and `/workspace/.onboarding`
  hold earlier local experiment/test outputs. They are outside the repository;
  retain only if investigating those earlier studies.

## Extension to 100 training seeds (2026-11-16)

- User requested retaining seeds 1–10 and adding 11–100 in all four Section Three
  conditions, with the archived settings unchanged (360 epochs per fit).
- Completed work and raw logs: `/workspace/section-three-100`; inspect `status.json`,
  `COMPLETE`, `FAILED`, and `supervisor.log` before starting any duplicate fits.
- R: `/workspace/ddesonn-env/bin/Rscript`; Python pinned dependencies:
  `/workspace/ddesonn-python/bin/python`.
- `seed-1-10-retained.csv` retains the original forty percentages;
  `per-seed-test-accuracy.csv` contains all 400 validated pairs. The summary script
  generated the final 100-seed `RESULTS.md`. All original percentages are unchanged.
- Fixed direct execution of `keras.py` shadowing the installed Keras package;
  no architecture, optimizer, shuffle, batching, epoch, split or loss settings changed.
- These remain training seeds on one clean partition, not independent splits.

- Completion checks: all four conditions cover seeds 1–100 exactly once; all new
  fits have three split metrics, 360 requested epochs, and valid checkpoint epochs;
  Keras has 78,840 optimizer updates per fit. Split sizes and zero predictor overlap
  revalidated; incomplete/duplicate-result rejection and clean-split regression checks passed.

## Corrected benchmark requirement (2026-11-16)

The user clarified that the original Keras script is the fixed reference for all
four conditions (original Keras mean, DDESONN FALSE, DDESONN TRUE, Keras sum),
including batch size 16 and the original split/settings. The completed Section
Three study does not meet that requirement and must remain a separate study.
See `research/keras-original-reference/README.md` for the source-derived contract,
historical workbook caveat, and native implementation gaps. No matched seed sweep
has been launched. Clarification was requested about whether “JSON” meant literal
configuration files or DDESONN (speech transcription); no answer received yet.

## Matched original-Keras run launched (2026-11-16)

- User explicitly requested the corrected four-condition, 100-seed comparison.
- New opt-in native float32 DDESONN protocol adapter implemented and verified
  against TensorFlow (146 checks); default trainer unchanged. Numerical evidence
  and original settings are in `research/keras-original-reference/README.md`.
- Active output root: `/workspace/keras-original-100` (NOT the older
  `/workspace/section-three-100`). Check status.json, workers.json, supervisor.log,
  COMPLETE and FAILED before starting/resuming workers.
- All 400 fits are fresh, same original 3500/750/750 split, batch 16, 360 epochs,
  78,840 updates and final checkpoint. No previous clean-split results are reused.
- Six Keras workers use per-seed file locks; two DDESONN workers use one worker
  per reduction. All checkpoint per seed. Supervisor generates the complete table
  only after all 400 per-condition/seed results validate.
- Initial tensors are shared from the original Keras builder. Both implementations
  are float32; shuffle/dropout RNG streams differ by framework, distributions match.
- Adapter inference uses hidden BN state; do not use the old BN-free predictor
  on these adapter-trained members.

- Original-protocol review caught native inference using one full matrix. The
  native adapter now predicts in original per-partition batches of 32. All 200
  initial native fits were preserved in `superseded-unbatched-inference` and
  rerun from scratch, not mixed into accepted metrics. TensorFlow/native checks
  were rerun with multi-batch inference; all 146 passed.

## Completed matched original-Keras comparison (2026-11-16)

- All 400 accepted fits completed and validated: exactly seeds 1–100 in each
  of original Keras mean, DDESONN FALSE/sum, DDESONN TRUE/mean, and Keras sum.
- Final mean test accuracies in that order: 99.80%, 99.96%, 99.91%, 99.93%.
- Canonical outputs: `research/keras-original-reference/RESULTS.md`,
  `per-seed-test-accuracy.csv`, `protocol.json`, and numerical-check evidence.
- Original 3500/750/750 split retained, including known predictor overlap. All
  training batches 16; native per-epoch validation batches 16 (16,920 per fit);
  final evaluation batches 32. All fits 360 epochs / 78,840 optimizer updates,
  final checkpoint. Shared original Keras initial tensors; framework-native
  RNG streams for shuffle/dropout. No earlier Section Three rows reused.
- DDESONN scores are for the new opt-in native float32 reference adapter, NOT
  unchanged default trainer scores. Standard gradient/optimizer regressions
  passed, 146 independent TensorFlow/native checks passed, and both reductions
  passed end-to-end protocol smoke checks.
- `/workspace/keras-original-100/COMPLETE` and status.json confirm completion;
  no active workers remain. Superseded native attempts are archived/excluded.
  Rerunning the strict summarizer independently validated all 400 accepted fits.

## Corrected objective: original settings EXCEPT overlap (2026-11-16)

- The user clarified that the original training scripts were to be retained
  EXCEPT the predictor overlap. Restoring the original overlapping split was
  the assistant's mistake, not a user requirement. Its 99.8–99.96% table is
  superseded for the intended comparison. Do not present it as the answer.
- Corrected output root: `/workspace/keras-clean-100`; code/results live in
  `research/keras-clean-reference`. All 400 fits are fresh. Check status.json,
  COMPLETE, FAILED and workers.json there before starting duplicates.
- Reused the frozen group-disjoint split seed 42 row assignment: 3500 train,
  698 validation, 802 test; 923/197/199 distinct predictor groups. Independent
  audits found ZERO overlap across all three pairs, both raw predictors and
  actual float32 inputs. Outcome excluded from group identity.
- Original training settings unchanged: StandardScaler/train-only, batch16,
  hidden BN, same architecture/dropout/initialization/AdaGrad/L1/no weighting,
  360 epochs, final checkpoint, validation batch16, final inference batch32.
  Training updates remain 78,840; validation batches now 15,840 due to 698 rows.
- Generalized only native partition boundaries; no forward/backward/optimizer
  changes. Kept the opt-in adapter; ordinary DDESONN trainer is unchanged.
- 146 TensorFlow/native checks and both one-epoch clean-protocol smoke cases
  passed. The report independently recomputes overlap, checks split/input/data
  fingerprints and rejects results from the overlapping cohort.

## Completed: original training settings with overlap removed

- Canonical answer is now `research/keras-clean-reference/RESULTS.md`.
  All 400 fresh fits validated (100 per condition), no active workers remain.
- Mean test accuracies: Keras mean 87.62%, DDESONN FALSE/sum 91.36%,
  DDESONN TRUE/mean 90.39%, Keras sum 87.48%.
- Independent final audit: ZERO common predictor vectors for train/validation,
  train/test, validation/test, both raw and actual float32 model inputs.
- Shared partitions: 3500 train / 698 validation / 802 test, all 5000 original
  rows assigned exactly once. Same frozen predictor-group assignment in every
  condition, same processed inputs and per-seed original Keras initial tensors.
- Only split membership changes from the original training protocol. Shared
  training settings were independently compared against the original-protocol
  manifest and are equal. Batch16, 360 epochs, 78,840 updates, final checkpoint;
  validation batch16 (15,840 batches per fit), final inference batch32.
- DDESONN uses the opt-in native float32 matched-protocol adapter, not the
  unchanged ordinary trainer. 146 numerical checks, both protocol smoke cases,
  deliberate-overlap rejection, old-cohort rejection and missing-results rejection
  passed. Strict summarizer independently reran after completion.
- The earlier 99.8–99.96% overlapping-split table is superseded for the user's
  request. It was retained by mistake; do not treat it as the clean comparison.
- Root: `/workspace/keras-clean-100`; COMPLETE and status.json confirm completion.
  Reproducible settings, split row IDs, audit evidence and CSV are versioned in
  the canonical directory. These remain training seeds on ONE fixed partition;
  no claim of independent patient-cohort replication.

## Optional community seed extensions prepared

The published endpoint remains 100 seeds per condition; no extra full training fits launched.
`dev/research/keras-clean-reference/CONTRIBUTING_SEEDS.md` documents four entry points,
frozen inputs/settings, portable compilation, numerical checks and validated export of separate
community contribution bundles. `SEED_COUNT_RATIONALE.md`, `SEED_DECILES.md` and the precision
figures document the retrospective resource/precision rationale, without claiming a prospective
stopping rule. `PR_DESCRIPTION.md` is a local draft for the next PR; no PR has been opened.
See `COMMUNITY_VALIDATION.md` for smoke-check coverage and remaining limits.
