# Shiny browser regression tests

These optional tests use Playwright with Chromium against a locally running
DDESONN workbench. They are excluded from the CRAN source package by the existing
`^dev$` rule in `.Rbuildignore`. Normal package tests remain in `tests/testthat`.

Prerequisites: installed DDESONN, Shiny, callr, Node.js, Playwright and Chromium.
If Playwright is not provided by the environment, install it outside the checkout:

```sh
npm install --prefix /tmp/ddesonn-browser-deps playwright
export NODE_PATH=/tmp/ddesonn-browser-deps/node_modules
```

From the repository root, start the app in one terminal:

```sh
Rscript -e 'shiny::runApp("inst/shiny/benchmark", host="127.0.0.1", port=3838, launch.browser=FALSE)'
```

In another terminal:

```sh
node dev/browser-tests/workbench.cjs
```

The script exits nonzero on failure. It trains with every supported optimizer,
alternates final/best-validation checkpoint selection, checks progress logs for
debug leakage, cancels a background fit while preserving the completed model,
downloads and reloads a model, uploads a CSV and repeats predictions, runs
additional training as a separate copy, checks epoch controls, verifies mobile
horizontal sizing and keyboard focus, and captures JavaScript/console errors.
Chromium's optional `/favicon.ico` 404 is recorded separately in the result;
JavaScript errors and all other console errors fail the run.
Screenshots and `result.json` go to `/tmp/ddesonn-browser-tests` by default;
failures save `failure.png` and print diagnostics.

Overrides:

- `DDESONN_APP_URL`: running app URL (default `http://127.0.0.1:3838/`).
- `CHROMIUM_PATH`: Chromium executable (default `/usr/bin/chromium`).
- `DDESONN_BROWSER_OUTPUT`: artifact directory outside the checkout.
- `DDESONN_OPTIMIZERS`: optional comma-separated subset for focused diagnosis;
  omit this variable for the complete browser run.

In the prepared cloud environment, first add `/workspace/r-env/bin` to `PATH`.
The focused R regression checks can be run with:

```sh
Rscript -e 'library(DDESONN); testthat::test_dir("tests/testthat", filter="^(optimizer-output|workbench-startup)$", reporter="summary", stop_on_failure=TRUE)'
```
