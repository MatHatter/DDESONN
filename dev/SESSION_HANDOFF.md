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
  Ten seeds in four conditions completed: mean accuracy 91.27% DDESONN sum,
  80.69% DDESONN average, 87.93% Keras average, 89.21% Keras sum.
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
