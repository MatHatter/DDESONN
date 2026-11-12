# DDESONN experiment explorer

Introduced in DDESONN 7.3.2. An optional native R Shiny app for comparing paired legacy/standard AdaGrad
benchmark results. It reads saved CSV results; it does not train models.

From an installed package:

```r
install.packages("shiny") # once, if needed
shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN"))
```

From the repository root:

```r
shiny::runApp("inst/shiny/benchmark")
```

The app opens with the compact 100-seed benchmark, supports seed-range and metric
filters, plots individual values and paired differences, and downloads selected
rows or calculated summaries. All UI assets are local. Shiny is a suggested
package, not a required import.

Upload CSV files up to 10 MB with one row per seed: a unique positive integer
`seed` and at least one complete pair of `METRIC_legacy` / `METRIC_corrected`.
Supported metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `log_loss`,
`brier`. Values must be finite and nonnegative; all except log loss must be at
most 1. Missing values and partial pairs are rejected. Differences are always
recomputed. Upload replaces the dataset; a combined 1–1,000 CSV is needed to
view the full planned benchmark. Invalid uploads retain the previous dataset.
The data tab displays up to 100 rows; downloads contain all selected rows.

## Included data provenance

`data/heart-failure-100.csv` is an exact copy of
`dev/research/heart-failure-adagrad-seeds-1-100/paired-test-metrics.csv` at
repository revision `0b9f59c278638ec4a3a1892e90f5140f94ff0401`.
The detailed report at that revision records protocol, source fingerprints,
artifact checksums, initialization uncertainty and data-overlap limitations.
Training used version 7.3.0; the report was added for 7.3.1. This small CSV is
included so the installed dashboard works without raw models or development
folders. No legacy raw archives are restored to the package payload.

Do not interpret aggregate seed evaluations as independent patients or attribute
historical differences solely to an optimizer correction. The About tab retains
the included benchmark's limitations; uploaded studies need their own provenance.

## Brain artwork

Click the header thumbnail to open an enlarged view, then use Close or Escape
to return. The full-resolution link supports viewing the original image.
The artwork is bundled locally as `www/DDESONN_brain.png`, copied unchanged from
`MatHatter/ddesonn-assets` revision `1b6d45806009ae28cfaab5380ff6bfec045db162`
(path `DDESONN_brain.png`), the image linked in the repository README.
It is project artwork rather than a diagram of an individual model.

Artwork SHA-256: `751d6675a10c8318b69268eb9c5e013d94508ab4e4870f82005c46f4b75dc1e5`.
