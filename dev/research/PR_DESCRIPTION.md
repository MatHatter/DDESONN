The historical heart-failure/Keras vignette loaded large experiment outputs from every DDESONN installation and presented legacy custom-AdaGrad measurements as a current comparison. This focused 7.3.0 cleanup preserves the research record while separating it from corrected optimizer claims and CRAN runtime contents.

Changes:

- Keep standard `adagrad` and the existing 7.2 SSM fixes unchanged. No active `adagrad_x` alias or legacy double-update implementation is restored.
- Explain the verified historical multilayer double weight update, the two disjoint 500-seed batches and their research purpose, and why the observations do not establish current standard-AdaGrad superiority over Keras.
- Trace scripts through earlier paths: master/per-run seed controls already appear in the dedicated archived snapshot. Remove the unsupported categorical claim that original runs lacked a master seed; document the partial metadata and unresolved execution details instead.
- Preserve all ten current historical files unchanged in Git, with hashes and immutable retrieval revisions. Also locate the older run1 prediction file, which is missing from the newer compact archive. No historical commits or tags are rewritten.
- Exclude the five historical RDS outputs and the workbook/session-script folder via two scoped `.Rbuildignore` rules. Make the historical vignette self-contained and retain verified, qualified summary values. Current datasets, supported examples, exports and reporting helpers remain intact.
- Update 7.3.0 NEWS/README and recommend matched 10–30-seed exploration, larger stability studies when justified, and reporting distributions and uncertainty. No parallel-processing feature is added.

Measured results (full source archives and their installations):

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Source archive | 6,663,752 bytes | 4,860,733 bytes | 27.06% |
| Installed package | 13,058,007 bytes | 10,812,440 bytes | 17.20% |

Validation:

- 58 test cases / 12,392 expectations passed with no failures, errors or skips, including optimizer arithmetic, AdaGrad weights/biases and SSM regressions.
- Before/after package builds executed and built all four vignettes. The final historical text was rerendered separately while valid unchanged outputs were reused for final packaging.
- `R CMD check --no-manual --no-vignettes DDESONN_7.3.0.tar.gz`: Status OK; examples and installed tests passed. Vignette execution was not duplicated during check. PDF-manual and online CRAN incoming checks remain unrun.
- Installed historical vignette renders with excluded artifacts absent. Tarball/install verification confirms all ten exclusions, retained input datasets/examples, four rendered vignettes, and byte-identical R code, tests and NAMESPACE compared with the baseline.
- Archive recovery commands, hashes and historical summary reproduction were executed successfully. Full audit, inventory and check limitations are in `dev/research/`.

This PR is stacked on the optimizer-preparation branch and changes documentation/packaging only. DESCRIPTION was already 7.3.0 after prior validation and is retained. It does not publish a CRAN release, create tags, merge a PR or rerun the historical training study.
