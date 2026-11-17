# Community extension workflow validation

- Fresh seed-101 preparation succeeded using the pinned Python environment. Its raw predictor, float32 input, label and split fingerprints matched the frozen cohort.
- Portable TensorFlow/native numerical checks passed all 146 comparisons using a temporary output directory and explicit system BLAS path.
- All four public entry points dispatched the correct framework/reduction and seed range. Subprocess training was mocked for these dispatch checks; no additional full fits ran.
- Export rejected an incomplete four-condition cohort. Input verification rejected an intentionally changed initialization file.
- The portable native runner's complete input-reading, training and metric block was compared byte-for-byte with the frozen native runner and matched. Its additional preamble records/checks packages and changes compilation paths only.
- Frozen adapter fingerprints still match the completed cohort. `git diff --check` passed.

Full end-to-end training/export for a new cohort was not exercised because that would consume additional training compute. Cross-machine reproducibility and additional community seed results remain untested. The existing completed 400-fit cohort and original independent numerical/completion evidence are preserved.
