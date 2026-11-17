# Community extension workflow validation

- Fresh seed-101 preparation succeeded using the pinned Python environment. Its raw predictor, float32 input, label and split fingerprints matched the frozen cohort.
- Portable TensorFlow/native numerical checks passed all 146 comparisons using a temporary output directory and explicit system BLAS path.
- All four public entry points dispatched the correct framework/reduction and seed range. Subprocess training was mocked for these dispatch checks; no additional full fits ran.
- Export rejected an incomplete four-condition cohort. Input verification rejected an intentionally changed initialization file.
- The portable native runner's complete input-reading, training and metric block was compared byte-for-byte with the frozen native runner and matched. Its additional preamble records/checks packages and changes compilation paths only.
- Frozen adapter fingerprints still match the completed cohort. `git diff --check` passed.

Full end-to-end training/export for a new cohort was not exercised because that would consume additional training compute. Cross-machine reproducibility and additional community seed results remain untested. The existing completed 400-fit cohort and original independent numerical/completion evidence are preserved.

## Shiny reproduction and session-code follow-up

- Generated sum, mean and selected-row historical R analysis scripts executed and reproduced the expected values without fitting models.
- Packaged ZIP tests compare every frozen adapter source byte-for-byte with the canonical study files and verify all four launchers are present.
- Extracted-bundle preparation generated original seeds 1–100 and reproduced the frozen input audit in a fresh temporary directory; no full fits ran.
- Headless Chromium checks passed desktop/mobile layout, reserved dropdown-arrow spacing, study switching, correct experiment-code routing, reproduction ZIP, four individual launchers, the selected-pair launcher, and analysis downloads. Downloaded clean/historical analysis scripts also ran successfully in separate R processes.
