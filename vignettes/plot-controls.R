## ----setup, include=FALSE-----------------------------------------------------
# ============================================================
# FILE: vignettes/plot-controls.Rmd
# FULL WORKING — VIGNETTE SHOWS PLOTS (Scenario 1 & 2)
#
# GOAL:
# - Demonstrate TWO supported user-facing interfaces for plot control:
#   (1) Scenario 1: training_overrides knobs (minimal integration)
#   (2) Scenario 2: plot_controls umbrella (recommended)
#
# IMPORTANT (binary eval):
# - To emit BOTH confusion matrix heatmaps (fixed + tuned),
#   users must set:
#     accuracy_plot = TRUE
#     accuracy_plot_mode = "both"
#
# This vignette shows that wiring explicitly.
# ============================================================

knitr::opts_chunk$set(
  echo = TRUE,
  message = TRUE,
  warning = FALSE,
  fig.width = 8,
  fig.height = 5,
  fig.align = "center",
  out.width = "900px",
  fig.path = "plot-controls_files/figure-html/"
)

if (!requireNamespace("DDESONN", quietly = TRUE)) {
  if (requireNamespace("devtools", quietly = TRUE)) {
    devtools::load_all(".", quiet = TRUE)
  } else {
    stop("DDESONN is not installed and devtools is unavailable.")
  }
}

library(DDESONN)


# ============================================================
# Output root for vignette artifacts
# - We anchor under knitr's figure path so PNGs ship with the HTML.
# ============================================================
.fig_root <- normalizePath(knitr::opts_chunk$get("fig.path"), winslash = "/", mustWork = FALSE)
if (!nzchar(.fig_root)) .fig_root <- "figure/"
out1 <- file.path(.fig_root, "DDESONN_plots_s1")
out2 <- file.path(.fig_root, "DDESONN_plots_s2")

dir.create(out1, recursive = TRUE, showWarnings = FALSE)
dir.create(out2, recursive = TRUE, showWarnings = FALSE)

# Make the package helpers resolve to a writable location under the vignette figure folder.
options(DDESONN_OUTPUT_ROOT = out1)
Sys.setenv(DDESONN_ARTIFACTS_ROOT = out1)

# Helper to print/include saved PNGs from a run.
include_saved_plots <- function(output_root, header) {
  plot_dir <- ddesonn_plots_dir(output_root)
  if (!dir.exists(plot_dir)) {
    cat("\n\n[plot-controls] plots dir does not exist:", plot_dir, "\n")
    return(invisible(NULL))
  }

  pngs <- list.files(plot_dir, pattern = "\\.png$", recursive = TRUE, full.names = TRUE)
  pngs <- pngs[order(pngs)]
  if (!length(pngs)) {
    cat("\n\n[plot-controls] no PNGs found under:", plot_dir, "\n")
    return(invisible(NULL))
  }

  cat("\n\n")
  cat(paste0("## ", header, "\n\n"))

  # include_graphics must be returned/printed by the chunk.
  knitr::include_graphics(pngs)
}

## ----data---------------------------------------------------------------------
set.seed(111)

# Prefer package extdata; falls back to an inferred binary target if needed.
ext_dir <- system.file("extdata", package = "DDESONN")
if (!nzchar(ext_dir)) stop("Could not find DDESONN extdata folder.", call. = FALSE)

hf_path <- file.path(ext_dir, "heart_failure_clinical_records.csv")
if (!file.exists(hf_path)) {
  csvs <- list.files(ext_dir, pattern = "\\.csv$", full.names = TRUE)
  if (!length(csvs)) stop("No .csv files found in extdata.", call. = FALSE)
  hf_path <- csvs[[1]]
}

df <- read.csv(hf_path)

target_col <- if ("DEATH_EVENT" %in% names(df)) {
  "DEATH_EVENT"
} else {
  cand <- names(df)[vapply(df, function(col) {
    v <- suppressWarnings(as.numeric(col))
    if (all(is.na(v))) return(FALSE)
    u <- unique(v[is.finite(v)])
    length(u) <= 2 && all(sort(u) %in% c(0, 1))
  }, logical(1))]
  if (!length(cand)) stop("Could not infer a binary target column.", call. = FALSE)
  cand[[1]]
}

y_all <- matrix(as.integer(df[[target_col]]), ncol = 1)
x_df  <- df[, setdiff(names(df), target_col), drop = FALSE]
x_all <- as.matrix(x_df)
storage.mode(x_all) <- "double"

n <- nrow(x_all)
idx <- sample.int(n)

n_train <- floor(0.70 * n)
n_valid <- floor(0.15 * n)

i_tr <- idx[1:n_train]
i_va <- idx[(n_train + 1):(n_train + n_valid)]
i_te <- idx[(n_train + n_valid + 1):n]

x_train <- x_all[i_tr, , drop = FALSE]
y_train <- y_all[i_tr, , drop = FALSE]

x_valid <- x_all[i_va, , drop = FALSE]
y_valid <- y_all[i_va, , drop = FALSE]

x_test  <- x_all[i_te, , drop = FALSE]
y_test  <- y_all[i_te, , drop = FALSE]

# Scale train-only (no leakage)
x_train_s <- scale(x_train)
ctr <- attr(x_train_s, "scaled:center")
scl <- attr(x_train_s, "scaled:scale")
scl[!is.finite(scl) | scl == 0] <- 1

x_valid_s <- sweep(sweep(x_valid, 2, ctr, "-"), 2, scl, "/")
x_test_s  <- sweep(sweep(x_test,  2, ctr, "-"), 2, scl, "/")

mx <- suppressWarnings(max(abs(x_train_s)))
if (!is.finite(mx) || mx == 0) mx <- 1

x_train <- x_train_s / mx
x_valid <- x_valid_s / mx
x_test  <- x_test_s  / mx

cat(sprintf("[split] train=%d valid=%d test=%d\n", nrow(x_train), nrow(x_valid), nrow(x_test)))

