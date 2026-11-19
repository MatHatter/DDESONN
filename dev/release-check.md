# Release check

From the root of a fresh checkout, run this in the RStudio Console:

```r
source("dev/check-release.R")
check_ddesonn_release()
```

Use a current patch release of R and dependencies installed for that R version.
The command builds the source tarball, including vignettes, and runs
`R CMD check --as-cran --no-manual`. It prints the output directory and leaves
the tarball and `00check.log` there. Use the optional `output` argument to select
another directory outside the checkout. Checks that finish with notes return
the log path. Warnings and errors still fail the release check.

The check uses a fresh temporary directory outside the RStudio session's
temporary directory. On Windows, R copies packages into its restricted check
library instead of creating NTFS junctions. This keeps the dependency checks
and background-worker tests enabled while avoiding junction-backed libraries.

If the check reports `unable to verify current time`, it could not reach its
time verification service. Run it with that service reachable for a check
without the note.
