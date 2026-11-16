Add gradient reduction switch and lightweight ten-seed results

Add `average_gradients` across the API, main experiment script and Shiny: FALSE
sums weights/biases, TRUE averages both. Preserve old-model mixed-mode behavior.
Include cancellation/graphics cleanup regressions and unreleased 7.4.0 notes.

Record Section Three's ten-seed, four-condition results with two rerunnable
scripts and short continuation notes. Keep archived per-framework settings and
the clean split; retain the corrected single-update optimizer. No per-seed
models, binary artifacts, prediction dumps, histories, manifests or raw test logs
are included. The research directory is excluded from package builds.

Validation: full R suite, focused gradient/pilot tests, browser regressions, and
all forty benchmark fits passed. Fresh CRAN-style check and release remain pending.
MIT and initial workbench regressions are already merged in #62.
