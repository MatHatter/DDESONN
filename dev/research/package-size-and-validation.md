# Package size and release-cleanup validation

The baseline is optimizer-preparation commit `17ca10f` (version 7.3.0), before
this focused cleanup. The comparison uses real built source archives and fresh
installations of those archives in the same environment, with all four rendered
vignettes present. MB below means 1,000,000 bytes. Installed sizes sum regular
file bytes, not filesystem allocation blocks.

| Measurement | Before | After | Reduction |
|:--|--:|--:|--:|
| Source `.tar.gz` | 6,663,752 bytes | 4,860,733 bytes | 1,803,019 bytes (27.06%) |
| Installed package | 13,058,007 bytes | 10,812,440 bytes | 2,245,567 bytes (17.20%) |

The ten excluded tracked files total 2,214,321 bytes: five RDS outputs
(1,909,394), the workbook (161,607), three R scripts (137,295) and one Python
script (6,025). Their paths, archive commit, Git blobs and SHA-256 hashes are
in `legacy-benchmark-inventory.csv`. Compression and generated documentation
mean archive savings need not equal the excluded raw-file total.

All ten remain byte-for-byte unchanged in Git. No large replacement artifact is
added: the installed historical summary is a short, self-contained vignette.
The source and installed package no longer contain either excluded historical
subtree. Current CSV datasets and supported example/reporting scripts remain.
All `R/`, `tests/` and NAMESPACE files in the before/after source archives match
byte-for-byte; the current SSM and optimizer implementations are unchanged.
DESCRIPTION was already 7.3.0 following the earlier validated release preparation
and remains at that target version; this cleanup does not publish it.

## Commands and outcomes

- `R CMD build` completed both before and after cleanup, executing/building all
  four vignettes. These existing 360-epoch examples took several minutes; one
  BLAS thread was used. No parallel benchmark implementation was added.
- After the provenance wording was finalized, the changed historical vignette
  was rendered again and packaged with `R CMD build --no-build-vignettes`, reusing
  the three unchanged, successfully built vignette outputs. The final tarball
  was checked to contain the exact current source and all three archive revision
  pointers in the rendered historical page.
- `R CMD INSTALL` of both complete tarballs succeeded. The historical vignette
  was then rendered from the cleaned installation after asserting that the
  excluded directories were absent.
- Source tests passed: **58 cases, 12,392 expectations, 0 failures/errors/skips**.
  This includes AdaGrad weight/bias and state regressions, independent optimizer
  reference checks, end-to-end dispatch and the SSM suite.
- `R CMD check --no-manual --no-vignettes DDESONN_7.3.0.tar.gz` completed with
  **Status: OK (zero errors, warnings or notes)**. Installation, examples,
  installed tests, vignette files, dependencies and metadata checks passed.
  The installed test log reports `[ FAIL 0 | WARN 0 | SKIP 0 | PASS 12392 ]`.
  Vignette code/rebuilding was not repeated during check because it had already
  been executed/built separately as described above.
- `verify-package.py` confirmed exclusions, preserved example inputs/helpers,
  unchanged R/tests/exports, four rendered vignettes, and the actual size figures.
- The documented Git archive extraction was executed; all ten recovered hashes
  and the recomputed historical summary match the retained files. The older
  run1 prediction RDS was retrieved and inspected successfully too.
- `git diff --check` passes. No current optimizer, public API or architecture
  change was necessary for this cleanup.

Reproduce the content/size check from the repository root:

```sh
python dev/research/verify-package.py BEFORE_TAR AFTER_TAR \
  BEFORE_LIBRARY/DDESONN AFTER_LIBRARY/DDESONN
Rscript dev/research/summarize-legacy.R EXTRACTED_ARCHIVE_ROOT
```

Run logs and archives from this session are under `/workspace/.setup/73-cleanup/`,
including `tests.log`, `size-results.json`, `before-build/build.log`,
`after-build/build.log`, `final-build/build.log`, `final-build/check.log` and
`final-build/DDESONN.Rcheck/00check.log`.

## Limitations and follow-up

- PDF-manual generation was not run because LaTeX is unavailable. This was not a
  CRAN submission or an `--as-cran` check. Remote CRAN/Bioconductor index probes
  printed access warnings in the console; installed dependencies were available
  and the package dependency check completed OK. Online incoming checks remain
  for an environment with the required network access.
- Pandoc prints a deprecation message for `--mathjax` through the installed
  rmarkdown/Pandoc combination; rendering succeeds. No TLS or verification
  bypass was used.
- The largest retained generated documentation is the plot-example HTML:
  6,875,877 bytes, 148 embedded images (144 distinct), approximately 6,822,176
  bytes of base64 image content. These are legitimate example figures, not
  tracked stale outputs; they were retained to preserve documentation behavior.
  Further figure-size work can be considered separately, not by deleting input
  datasets or public features to hit an arbitrary target.
- The original master-seed/execution configuration and reason for two jobs
  rather than one are not fully established by the archive. Partial metadata
  and the verified seeded script history are documented, without declaring the
  historical work unseeded, irreproducible or evidence of current superiority.
- No original 500/1,000-seed training or Keras training was rerun. Only saved
  observations were summarized; future comparisons should use corrected,
  matched implementations and report uncertainty.
- No tags were created or changed, no Git history rewritten, no release
  published, and no pull request merged.
