## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment  = "#>",
  message  = FALSE,
  warning  = FALSE
)

## ----ddesonn-summary, message=FALSE-------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(knitr)
})

# Locate package data root
hf_root <- system.file(
  "extdata", "sample_runs", "heart_failure_single_runs",
  package = "DDESONN"
)

# ---- Define RDS paths ----
train_run1_path <- file.path(
  hf_root, "20251025_175155__m1__wSeed",
  "SingleRun_Train_Acc_Val_Metrics_500_seeds_20251025_175155.rds"
)
test_run1_path <- file.path(
  hf_root, "20251025_175155__m1__wSeed",
  "SingleRun_Test_Metrics_500_seeds_20251025_175155.rds"
)
train_run2_path <- file.path(
  hf_root, "20251026_111537__m1__wSeed",
  "SingleRun_Train_Acc_Val_Metrics_500_seeds_20251026_111537.rds"
)
test_run2_path <- file.path(
  hf_root, "20251026_111537__m1__wSeed",
  "SingleRun_Test_Metrics_500_seeds_20251026_111537.rds"
)

# Read the four files
train_run1 <- readRDS(train_run1_path)
test_run1  <- readRDS(test_run1_path)
train_run2 <- readRDS(train_run2_path)
test_run2  <- readRDS(test_run2_path)

# Combine
train_all <- dplyr::bind_rows(train_run1, train_run2)
test_all  <- dplyr::bind_rows(test_run1,  test_run2)

# Per-seed tables
train_seed <- train_all %>%
  group_by(seed) %>%
  slice_max(order_by = best_val_acc, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    seed, 
    train_acc = best_train_acc,
    val_acc   = best_val_acc
  )

test_seed <- test_all %>%
  group_by(seed) %>%
  slice_max(order_by = accuracy, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    seed,
    test_acc = accuracy
  )

merged <- inner_join(train_seed, test_seed, by = "seed") %>%
  arrange(seed)

# Helper to summarize a numeric vector
summarize_column <- function(x) {
  pct <- function(p) stats::quantile(x, probs = p, names = FALSE, type = 7)
  data.frame(
    count = length(x),
    mean  = mean(x),
    std   = sd(x),
    min   = min(x),
    `25%` = pct(0.25),
    `50%` = pct(0.50),
    `75%` = pct(0.75),
    max   = max(x),
    check.names = FALSE
  )
}

summary_train <- summarize_column(merged$train_acc)
summary_val   <- summarize_column(merged$val_acc)
summary_test  <- summarize_column(merged$test_acc)

summary_all <- data.frame(
  stat = c("count","mean","std","min","25%","50%","75%","max"),
  train_acc = unlist(summary_train[1,]),
  val_acc   = unlist(summary_val[1,]),
  test_acc  = unlist(summary_test[1,]),
  check.names = FALSE
)

round4 <- function(x) if (is.numeric(x)) round(x, 4) else x
pretty_summary <- as.data.frame(lapply(summary_all, round4))

knitr::kable(pretty_summary, caption = "DDESONN — 1000-seed summary (train/val/test)")

## ----ddesonn-merged-preview---------------------------------------------------
knitr::kable(head(merged, 10), caption = "First 10 seeds — per-seed train/val/test accuracies")

## ----keras-summary, message=FALSE---------------------------------------------
suppressPackageStartupMessages(library(readxl))

keras_pkg_path <- system.file(
  "extdata", "vKeras", "1000SEEDSRESULTSvkeras", "1000seedsKeras.xlsx",
  package = "DDESONN"
)

keras_local_fallback <- file.path(
  "helpfulFiles", "vKeras", "1000SEEDSRESULTSvkeras", "1000seedsKeras.xlsx"
)

keras_path <- if (nzchar(keras_pkg_path)) keras_pkg_path else keras_local_fallback

if (file.exists(keras_path)) {
  keras_stats <- readxl::read_excel(keras_path, sheet = 2)
  knitr::kable(keras_stats, caption = "Keras — 1000-seed summary imported from Excel (Sheet 2)")
} else {
  cat("Keras Excel not found; expected either the packaged path or local fallback.\n")
}

