# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Experimental Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# 
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

# ===============================================================
# DDESONN — Summary (per-seed; uses fused as "test" for ensembles)
# ===============================================================

is_ens <- TRUE #or FALSE

suppressPackageStartupMessages({
  library(dplyr)
})

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

# ---------- decide run root ----------
root   <- file.path("artifacts", if (is_ens) "EnsembleRuns" else "SingleRuns")

# ---------- choose RUN_DIR: honor .BM_DIR if valid; else pick latest ----------
RUN_DIR <- get0(".BM_DIR", inherits = TRUE, ifnotfound = NULL)
root_norm <- normalizePath(root, winslash = "/", mustWork = FALSE)

use_latest_subdir <- function(root_path) {
  runs <- list.dirs(root_path, full.names = TRUE, recursive = FALSE)
  if (!length(runs)) stop("No run folders under: ", root_path)
  runs[order(file.info(runs)$mtime, decreasing = TRUE)][1]
}

is_under <- function(path, parent) {
  if (is.null(path) || !nzchar(path) || !dir.exists(path)) return(FALSE)
  startsWith(normalizePath(path, winslash = "/", mustWork = FALSE), parent)
}

if (!is_under(RUN_DIR, root_norm)) {
  RUN_DIR <- use_latest_subdir(root)
}

cat("[SUMMARY] Using RUN_DIR = ", RUN_DIR, "\n", sep = "")

# ---------- load latest train/val metrics RDS ----------
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

# strip any prefix like "performance_metric." or "relevance_metric."
names(tv) <- sub("^(performance_metric|relevance_metric)\\.", "", names(tv))

# normalize seed column
if (!"seed" %in% names(tv)) {
  if ("SEED" %in% names(tv)) {
    tv$seed <- tv$SEED
  } else {
    stop("No 'seed' or 'SEED' column present in: ", train_file)
  }
}

# ensure the best_* columns exist (they should if wired from training)
needed_cols <- c("best_train_acc", "best_val_acc")
missing_cols <- setdiff(needed_cols, names(tv))
if (length(missing_cols)) {
  stop("Missing required column(s) in train/val table: ", paste(missing_cols, collapse = ", "))
}

# ---------- reduce to per-seed best train/val ----------
tv_seed <- tv %>%
  group_by(seed) %>%
  summarise(
    train_acc = suppressWarnings(max(best_train_acc, na.rm = TRUE)),
    val_acc   = suppressWarnings(max(best_val_acc,   na.rm = TRUE)),
    .groups   = "drop"
  )

# ---------- fused "test" metrics (ensemble only) ----------
if (is_ens) {
  fdir <- file.path(RUN_DIR, "fused")
  if (!dir.exists(fdir)) stop("Expected fused dir not found for ensemble: ", fdir)
  ffiles <- list.files(fdir, pattern = "^fused_run\\d+_seed\\d+_.*\\.rds$", full.names = TRUE)
  if (!length(ffiles)) stop("No fused RDS files in ", fdir)
  
  fused_rows <- do.call(rbind, lapply(ffiles, function(f) {
    z <- readRDS(f)
    m <- z$metrics
    if (is.null(m) || !is.data.frame(m)) return(NULL)
    bn <- basename(f)
    m$seed      <- suppressWarnings(as.integer(sub(".*_seed(\\d+)_.*", "\\1", bn, perl = TRUE)))
    m$run_index <- suppressWarnings(as.integer(sub("^fused_run(\\d+).*", "\\1", bn, perl = TRUE)))
    m
  }))
  if (is.null(fused_rows) || !nrow(fused_rows)) stop("Fused metrics were empty under ", fdir)
  
  # choose one representation method for the ensemble
  method_pick <- "Ensemble_wavg"  # alternatives: "Ensemble_avg", "Ensemble_vote_soft", "Ensemble_vote_hard"
  fused_best <- fused_rows %>% dplyr::filter(kind == method_pick)
  if (!nrow(fused_best)) stop("No fused rows with kind == ", method_pick, " in ", fdir)
  
  # normalize expected metric column names
  for (nm in c("accuracy","precision","recall","f1")) {
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
  # ---------- single-run "test" metrics (from SingleRun_Test_Metrics_*.rds) ----------
  test_pat <- "^SingleRun_Test_Metrics_\\d+_seeds_.*\\.rds$"
  test_file <- {
    fs <- list.files(RUN_DIR, pattern = test_pat, full.names = TRUE)
    if (!length(fs)) stop("Single-run test metrics RDS not found in ", RUN_DIR)
    fs[order(file.info(fs)$mtime, decreasing = TRUE)][1]
  }
  
  test_df <- readRDS(test_file)
  
  # strip "performance_metric." / "relevance_metric." prefixes
  names(test_df) <- sub("^(performance_metric|relevance_metric)\\.", "", names(test_df))
  
  # normalize seed
  if (!"seed" %in% names(test_df)) {
    if ("SEED" %in% names(test_df)) test_df$seed <- test_df$SEED
    else stop("No 'seed' or 'SEED' column in test metrics: ", test_file)
  }
  
  # ensure metrics exist (untuned)
  for (nm in c("accuracy","precision","recall","f1_score")) {
    if (!nm %in% names(test_df)) test_df[[nm]] <- NA_real_
  }
  # unify f1 column name
  if ("f1_score" %in% names(test_df) && !"f1" %in% names(test_df)) {
    test_df$f1 <- test_df$f1_score
  }
  
  # reduce to one row per seed (highest accuracy per seed)
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

# ---------- merge and summarise ----------
merged <- tv_seed %>%
  inner_join(fused_seed %>% select(seed, test_acc, test_precision, test_recall, test_f1), by = "seed") %>%
  arrange(seed)

# drop extra test columns post-merge (keep only test_acc)
merged <- merged %>%
  select(-test_precision, -test_recall, -test_f1)

summary_all <- sapply(merged[c("seed","train_acc","val_acc","test_acc")], summarize_stats)
summary_all <- round(as.data.frame(summary_all), 4)

cat("=== Summary (per seed; fused used as test when ensemble) ===\n")
print(summary_all, row.names = TRUE)

cat("\n=== Per-seed table ===\n")
print(merged, n = 31)
