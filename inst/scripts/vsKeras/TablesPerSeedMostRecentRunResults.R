# ===============================================================
# DDESONN — Per-Seed Summary Script
# ---------------------------------------------------------------
# Purpose:
#   - Summarize metrics across many seeds for a DDESONN run
#   - Works for EnsembleRuns OR SingleRuns
#   - Prints:
#       (1) Distribution summary (mean/std/quantiles/min/max)
#       (2) Per-seed table (train_acc, val_acc, test_acc)
#
# Notes for new users:
#   - This script does NOT require 1000 seeds. It works with any count
#     (e.g., 5 seeds for a quick comparison run).
#   - You can point it at a specific run folder via RUN_DIR, or let it
#     auto-pick the most recent run folder under the expected root.
#
# How "test" is defined:
#   - Ensemble mode: uses the "fused" metrics as the test-equivalent.
#     ("fused" = combined ensemble predictions/metrics, e.g. avg/wavg/vote)
#   - Single-run mode: uses SingleRun_Test_Metrics_*.rds
# ===============================================================

# -----------------------------
# 0) Choose run type
# -----------------------------
is_ens <- TRUE  # TRUE = EnsembleRuns, FALSE = SingleRuns

# -----------------------------
# 1) Optional: manually set RUN_DIR
# -----------------------------
# If you set RUN_DIR to a specific run folder, the script will use it
# (as long as it exists and looks valid). Otherwise it auto-selects the
# latest run folder under the computed root.
#
# Example:
# RUN_DIR <- "C:/Users/you/Desktop/DDESONN/artifacts/EnsembleRuns/Run_2026-02-17_..."
#
RUN_DIR <- get0(".BM_DIR", inherits = TRUE, ifnotfound = NULL)

suppressPackageStartupMessages({
  library(dplyr)
})

# -----------------------------
# 2) Small helper: vector summary stats
# -----------------------------
summarize_stats <- function(v) c(
  count = sum(!is.na(v)),
  mean  = mean(v, na.rm = TRUE),
  std   = sd(v, na.rm = TRUE),
  min   = suppressWarnings(min(v, na.rm = TRUE)),
  `25%` = suppressWarnings(quantile(v, 0.25, na.rm = TRUE, names = FALSE)),
  `50%` = suppressWarnings(quantile(v, 0.50, na.rm = TRUE, names = FALSE)),
  `75%` = suppressWarnings(quantile(v, 0.75, na.rm = TRUE, names = FALSE)),
  max   = suppressWarnings(max(v, na.rm = TRUE))
)

# ===============================================================
# 3) Decide root folder for runs
# ===============================================================
# If your DDESONN environment provides ddesonn_artifacts_root(),
# we use it. Otherwise, you can set `root` manually.
#
# - When ddesonn_artifacts_root() exists, it typically comes from your
#   DDESONN package utilities (often wired via a paths helper).
# - If it DOES NOT exist in the current session, this script will stop
#   and ask you to set `root` manually.
# ===============================================================

if (exists("ddesonn_artifacts_root", mode = "function")) {
  root <- file.path(ddesonn_artifacts_root(NULL), if (is_ens) "EnsembleRuns" else "SingleRuns")
} else {
  # ----------------------------------------
  # Manual fallback if helper is unavailable:
  # ----------------------------------------
  # Set this to wherever your run folders live.
  #
  # Example:
  # root <- "C:/Users/wfky1/Desktop/DDESONN/inst/artifacts/EnsembleRuns"
  #
  stop(
    "ddesonn_artifacts_root() was not found in this session.\n",
    "Please set `root` manually to your runs folder (EnsembleRuns or SingleRuns)."
  )
}

root_norm <- normalizePath(root, winslash = "/", mustWork = FALSE)

# -----------------------------
# 4) Utilities: pick latest run folder safely
# -----------------------------
use_latest_subdir <- function(root_path) {
  runs <- list.dirs(root_path, full.names = TRUE, recursive = FALSE)
  if (!length(runs)) stop("No run folders under: ", root_path)
  runs[order(file.info(runs)$mtime, decreasing = TRUE)][1]
}

is_under <- function(path, parent) {
  if (is.null(path) || !nzchar(path) || !dir.exists(path)) return(FALSE)
  startsWith(normalizePath(path, winslash = "/", mustWork = FALSE), parent)
}

# If RUN_DIR is missing/invalid OR outside the expected root, auto-pick latest.
if (!is_under(RUN_DIR, root_norm)) {
  RUN_DIR <- use_latest_subdir(root)
}

cat("[SUMMARY] Using RUN_DIR = ", RUN_DIR, "\n", sep = "")

# ===============================================================
# 5) Load Train/Val metrics (latest matching file in RUN_DIR)
# ===============================================================
train_pat <- if (is_ens) {
  "^Ensembles_Train_Acc_Val_Metrics_\\d+_seeds_.*\\.rds$"
} else {
  "^SingleRun_Train_Acc_Val_Metrics_\\d+_seeds_.*\\.rds$"
}

train_file <- {
  fs <- list.files(RUN_DIR, pattern = train_pat, full.names = TRUE)
  if (!length(fs)) stop("Train/val metrics RDS not found in ", RUN_DIR)
  fs[order(file.info(fs)$mtime, decreasing = TRUE)][1]
}

tv <- readRDS(train_file)

# Strip any prefix like "performance_metric." or "relevance_metric."
names(tv) <- sub("^(performance_metric|relevance_metric)\\.", "", names(tv))

# Normalize seed column (accept seed or SEED)
if (!"seed" %in% names(tv)) {
  if ("SEED" %in% names(tv)) {
    tv$seed <- tv$SEED
  } else {
    stop("No 'seed' or 'SEED' column present in: ", train_file)
  }
}

# Ensure required columns exist
needed_cols <- c("best_train_acc", "best_val_acc")
missing_cols <- setdiff(needed_cols, names(tv))
if (length(missing_cols)) {
  stop("Missing required column(s) in train/val table: ", paste(missing_cols, collapse = ", "))
}

# Reduce to per-seed best train/val
tv_seed <- tv %>%
  group_by(seed) %>%
  summarise(
    train_acc = suppressWarnings(max(best_train_acc, na.rm = TRUE)),
    val_acc   = suppressWarnings(max(best_val_acc,   na.rm = TRUE)),
    .groups   = "drop"
  )

# ===============================================================
# 6) Load "test" metrics
# ===============================================================
# Ensemble mode:
#   - Reads RUN_DIR/fused/fused_run*_seed*_*.rds
#   - Picks one fusion method (method_pick) and treats its accuracy as test_acc
#
# Single-run mode:
#   - Reads latest SingleRun_Test_Metrics_*.rds in RUN_DIR
#   - Picks best row per seed (highest accuracy) as test_acc
# ===============================================================

if (is_ens) {
  
  # -----------------------------
  # 6A) Ensemble: use fused metrics
  # -----------------------------
  fdir <- file.path(RUN_DIR, "fused")
  if (!dir.exists(fdir)) stop("Expected fused dir not found for ensemble: ", fdir)
  
  ffiles <- list.files(fdir, pattern = "^fused_run\\d+_seed\\d+_.*\\.rds$", full.names = TRUE)
  if (!length(ffiles)) stop("No fused RDS files in ", fdir)
  
  fused_rows <- do.call(rbind, lapply(ffiles, function(f) {
    z <- readRDS(f)
    m <- z$metrics
    if (is.null(m) || !is.data.frame(m)) return(NULL)
    
    bn <- basename(f)
    
    # Extract seed/run index from filename
    m$seed      <- suppressWarnings(as.integer(sub(".*_seed(\\d+)_.*", "\\1", bn, perl = TRUE)))
    m$run_index <- suppressWarnings(as.integer(sub("^fused_run(\\d+).*", "\\1", bn, perl = TRUE)))
    
    m
  }))
  
  if (is.null(fused_rows) || !nrow(fused_rows)) stop("Fused metrics were empty under ", fdir)
  
  # Choose which fusion kind to use as the "test" representation for ensembles
  method_pick <- "Ensemble_wavg"
  # Alternatives that may exist in your fused metrics:
  #   "Ensemble_avg", "Ensemble_vote_soft", "Ensemble_vote_hard"
  
  fused_best <- fused_rows %>% dplyr::filter(kind == method_pick)
  if (!nrow(fused_best)) stop("No fused rows with kind == ", method_pick, " in ", fdir)
  
  # Normalize expected metric column names (if some are absent, fill NA)
  for (nm in c("accuracy", "precision", "recall", "f1")) {
    if (!nm %in% names(fused_best)) fused_best[[nm]] <- NA_real_
  }
  
  fused_seed <- fused_best %>%
    select(seed, accuracy, precision, recall, f1) %>%
    rename(
      test_acc       = accuracy,
      test_precision = precision,
      test_recall    = recall,
      test_f1        = f1
    )
  
} else {
  
  # -----------------------------
  # 6B) Single-run: use test metrics RDS
  # -----------------------------
  test_pat <- "^SingleRun_Test_Metrics_\\d+_seeds_.*\\.rds$"
  
  test_file <- {
    fs <- list.files(RUN_DIR, pattern = test_pat, full.names = TRUE)
    if (!length(fs)) stop("Single-run test metrics RDS not found in ", RUN_DIR)
    fs[order(file.info(fs)$mtime, decreasing = TRUE)][1]
  }
  
  test_df <- readRDS(test_file)
  
  # Strip any prefix like "performance_metric." or "relevance_metric."
  names(test_df) <- sub("^(performance_metric|relevance_metric)\\.", "", names(test_df))
  
  # Normalize seed column (accept seed or SEED)
  if (!"seed" %in% names(test_df)) {
    if ("SEED" %in% names(test_df)) test_df$seed <- test_df$SEED
    else stop("No 'seed' or 'SEED' column in test metrics: ", test_file)
  }
  
  # Ensure metrics exist (fill NA if missing)
  for (nm in c("accuracy", "precision", "recall", "f1_score")) {
    if (!nm %in% names(test_df)) test_df[[nm]] <- NA_real_
  }
  
  # Unify f1 column name
  if ("f1_score" %in% names(test_df) && !"f1" %in% names(test_df)) {
    test_df$f1 <- test_df$f1_score
  }
  
  # Reduce to one row per seed: highest accuracy per seed
  fused_seed <- test_df %>%
    group_by(seed) %>%
    arrange(dplyr::desc(accuracy)) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      seed = as.integer(seed),
      test_acc       = as.numeric(accuracy),
      test_precision = as.numeric(precision),
      test_recall    = as.numeric(recall),
      test_f1        = as.numeric(f1)
    )
}

# ===============================================================
# 7) Merge & print
# ===============================================================

merged <- tv_seed %>%
  inner_join(fused_seed %>% select(seed, test_acc, test_precision, test_recall, test_f1), by = "seed") %>%
  arrange(seed)

# Drop extra test columns post-merge (keep only test_acc)
merged <- merged %>%
  select(-test_precision, -test_recall, -test_f1)

summary_all <- sapply(merged[c("seed", "train_acc", "val_acc", "test_acc")], summarize_stats)
summary_all <- round(as.data.frame(summary_all), 4)

cat("=== Summary (per seed; fused used as test when ensemble) ===\n")
print(summary_all, row.names = TRUE)

cat("\n=== Per-seed table ===\n")
print(merged, n = 31)
