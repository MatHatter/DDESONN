#!/usr/bin/env Rscript
# ================================================================
# DDESONN — Scenario A (Heart Failure Clinical Records)
# Two tracks only:
#   - Techila track: uses Techila + foreach
#   - Local track:  strictly serial for-loop (NO foreach, NO parallel)
# No fallbacks. Tracks are fully independent.
# ================================================================

suppressPackageStartupMessages(library(DDESONN))

options(stringsAsFactors = FALSE)

# ========= USER TOGGLE (pick exactly one track) ==================
USE_TECHILA <- FALSE  # TRUE => Techila track | FALSE => Local serial track
# ================================================================

`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x
pick_first_existing <- function(paths) {
  p <- paths[file.exists(paths)]
  if (length(p) == 0) stop("❌ None of the candidate paths exist: ", paste(paths, collapse = " | "))
  p[[1]]
}
require_pkg <- function(pkg, lib.loc = NULL) {
  ok <- suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE, lib.loc = lib.loc))
  if (!ok) stop("❌ Required package missing for selected track: '", pkg, "'")
}

cat("\n=== ENV DEBUG ===\n")
cat("getwd():", getwd(), "\n")
cat("R.version.string:", R.version.string, "\n")
cat("libPaths():", paste(.libPaths(), collapse = " | "), "\n")
cat("Selected track:", if (USE_TECHILA) "Techila (foreach)" else "Local (serial for-loop)", "\n\n")

# ------------------------------------------------------------------------------
# Track init (NO CROSS-WIRING)
# ------------------------------------------------------------------------------
if (USE_TECHILA) {
  require_pkg("techila")
  require_pkg("foreach")
  message("⚙ Initializing Techila track ...")
  techila::init()
  techila::registerDoTechila()
  message("✅ Techila foreach backend registered.")
} else {
  message("⚙ Initializing Local track (serial) ...")
  # Intentionally DO NOT load foreach/doParallel/parallel here.
  message("✅ Local serial runner ready (no foreach).")
}

# ------------------------------------------------------------------------------
# Dataset: Heart Failure Clinical Records (binary target = DEATH_EVENT)
# ------------------------------------------------------------------------------
data_path <- pick_first_existing(c(
  file.path("data", "heart_failure_clinical_records.csv"),
  file.path(getwd(), "data", "heart_failure_clinical_records.csv")
))
cat("📄 Using dataset:", data_path, "\n")

hf <- read.csv(data_path, stringsAsFactors = FALSE)
target <- "DEATH_EVENT"
features <- setdiff(colnames(hf), target)

# Basic dataset sanity checks + debug
cat("\n=== DATA DEBUG ===\n")
cat("nrow(hf):", nrow(hf), "  ncol(hf):", ncol(hf), "\n")
cat("target:", target, "\n")
cat("feature count:", length(features), "\n")
cat("head(hf):\n"); print(utils::head(hf, 3))
cat("str(hf):\n"); utils::str(hf)
if (!target %in% names(hf)) stop("❌ target column not found: ", target)

# Ensure target is numeric (0/1)
to01 <- function(v) {
  if (is.numeric(v)) return(as.integer(v))
  if (is.logical(v)) return(as.integer(v))
  if (is.factor(v)) return(as.integer(as.character(v)))
  if (is.character(v)) {
    vv <- tolower(trimws(v))
    m <- ifelse(vv %in% c("1","true","yes","y"), 1L,
                ifelse(vv %in% c("0","false","no","n"), 0L, NA_integer_))
    return(as.integer(m))
  }
  as.integer(v)
}
hf[[target]] <- to01(hf[[target]])
if (any(is.na(hf[[target]]))) {
  cat("⚠ NA found in target after coercion. Showing counts:\n")
  print(table(hf[[target]], useNA = "ifany"))
  stop("❌ Target contains NA after coercion; please clean labels.")
}

set.seed(42)
n <- nrow(hf)
idx <- sample(seq_len(n))
train_idx <- idx[1:floor(0.6 * n)]
valid_idx <- idx[(floor(0.6 * n) + 1):floor(0.8 * n)]
test_idx  <- idx[(floor(0.8 * n) + 1):n]

train_x <- hf[train_idx, features, drop = FALSE]
train_y <- hf[train_idx, target,  drop = FALSE]
valid_x <- hf[valid_idx, features, drop = FALSE]
valid_y <- hf[valid_idx, target,   drop = FALSE]
test_x  <- hf[test_idx,  features,  drop = FALSE]
test_y  <- hf[test_idx,  target,    drop = FALSE]

cat("\n=== SPLIT DEBUG ===\n")
cat(sprintf("Train: %d rows | Valid: %d | Test: %d\n", nrow(train_x), nrow(valid_x), nrow(test_x)))
cat("train_y summary:\n"); print(summary(train_y[[1]]))
cat("valid_y summary:\n"); print(summary(valid_y[[1]]))
cat("test_y  summary:\n"); print(summary(test_y[[1]]))

# ------------------------------------------------------------------------------
# Scale numeric columns (fit on TRAIN only)
# ------------------------------------------------------------------------------
scale_fit <- function(X) {
  num_cols <- names(which(sapply(X, is.numeric)))
  if (!length(num_cols)) stop("No numeric columns to scale.")
  mu <- vapply(X[num_cols], function(col) mean(col, na.rm = TRUE), numeric(1))
  sd <- vapply(X[num_cols], function(col) stats::sd(col, na.rm = TRUE), numeric(1))
  sd[is.na(sd) | sd == 0] <- 1
  list(mu = mu, sd = sd, num_cols = num_cols)
}
scale_apply <- function(X, s) {
  Xs <- X
  for (nm in s$num_cols) Xs[[nm]] <- (Xs[[nm]] - s$mu[[nm]]) / s$sd[[nm]]
  Xs
}

cat("\n=== SCALER DEBUG ===\n")
scaler  <- scale_fit(train_x)
cat("Numeric columns scaled:", paste(scaler$num_cols, collapse = ", "), "\n")
train_x <- scale_apply(train_x, scaler)
valid_x <- scale_apply(valid_x, scaler)
test_x  <- scale_apply(test_x,  scaler)
cat("Post-scale head(train_x):\n"); print(head(train_x[, scaler$num_cols, drop = FALSE], 3))

# ------------------------------------------------------------------------------
# Activation functions for DDESONN
# ------------------------------------------------------------------------------
activation_functions <- list(relu, relu, sigmoid)
activation_functions_predict <- list(relu, relu, sigmoid)

# ------------------------------------------------------------------------------
# Helper: safe threshold extraction + prediction debugging
# ------------------------------------------------------------------------------
extract_threshold <- function(pred) {
  thr <- pred$chosen_threshold %||% pred$best_threshold %||% pred$threshold %||% NA_real_
  if (!is.null(thr) && length(thr) == 1 && is.finite(as.numeric(thr))) {
    return(as.numeric(thr))
  }
  NA_real_
}
pred_debug <- function(stage, probs, y_true, thr) {
  cat(sprintf("\n=== PRED DEBUG (%s) ===\n", stage))
  cat("length(probs):", length(probs), "\n")
  if (length(probs)) {
    cat("range(probs):", paste(range(probs, na.rm = TRUE), collapse = " to "), "\n")
    cat("head(probs):", paste(head(round(probs, 4), 6), collapse = ", "), "\n")
  }
  if (!is.na(thr)) cat("threshold:", thr, "\n") else cat("threshold: <NA> (will fallback to 0.5)\n")
  cat("y_true table:\n"); print(table(as.integer(y_true), useNA = "ifany"))
}
cmatrix <- function(pred, truth) {
  tab <- table(Pred = pred, True = truth)
  print(tab)
  acc <- mean(pred == truth)
  cat("acc:", round(acc, 4), "\n")
  invisible(acc)
}

# ------------------------------------------------------------------------------
# Scenario A Runners — Techila vs Local (strictly separate)
# ------------------------------------------------------------------------------
run_scenario_A_techila <- function(seeds = 1:5) {
  foreach::foreach(
    seed = seeds,
    .combine = rbind,
    .export = c(
      "%||%", "scale_fit", "scale_apply", "extract_threshold", "pred_debug", "cmatrix", "to01",
      "activation_functions", "activation_functions_predict",
      "train_x", "train_y", "valid_x", "valid_y", "test_x", "test_y"
    ),
    .packages = character(0)
  ) %dopar% {
    if (!exists("ddesonn_model")) source("R/api.R")
    set.seed(seed)
    
    trX <- train_x; trY <- train_y
    vaX <- valid_x; vaY <- valid_y
    teX <- test_x;  teY <- test_y
    
    model <- ddesonn_model(
      input_size = ncol(trX),
      output_size = 1,
      hidden_sizes = c(64, 32),
      classification_mode = "binary",
      num_networks = 1,
      init_method = "he",
      lambda = 0.00028,
      custom_scale = 1.04349
    )
    
    fit <- ddesonn_fit(
      model,
      trX, trY,
      validation = list(x = vaX, y = vaY),
      num_epochs = 360,
      lr = 0.125,
      reg_type = "L1",
      validation_metrics = TRUE,
      verbose = FALSE
    )
    
    pred_val  <- ddesonn_predict(model, vaX, aggregate = "mean")
    probs_val <- suppressWarnings(as.numeric(pred_val$prediction))
    thr_val   <- extract_threshold(pred_val)
    if (!length(probs_val) || all(is.na(probs_val))) probs_val <- rep(0.5, nrow(vaX))
    pred_debug(sprintf("VALID (seed=%d)", seed), probs_val, vaY[[1]], thr_val)
    thr_eff <- if (is.finite(thr_val)) thr_val else 0.5
    pred_class_val <- as.integer(probs_val >= thr_eff)
    val_acc <- tryCatch(mean(pred_class_val == as.integer(vaY[[1]])), error = function(e) NA_real_)
    
    pred_test  <- ddesonn_predict(model, teX, aggregate = "mean")
    probs_test <- suppressWarnings(as.numeric(pred_test$prediction))
    if (!length(probs_test) || all(is.na(probs_test))) probs_test <- rep(0.5, nrow(teX))
    pred_debug(sprintf("TEST (seed=%d)", seed), probs_test, teY[[1]], thr_eff)
    pred_class_test <- as.integer(probs_test >= thr_eff)
    test_acc <- tryCatch(mean(pred_class_test == as.integer(teY[[1]])), error = function(e) NA_real_)
    
    data.frame(
      seed    = as.integer(seed),
      thr     = as.numeric(thr_eff),
      val_acc = suppressWarnings(as.numeric(val_acc)),
      test_acc= suppressWarnings(as.numeric(test_acc))
    )
  }
}

run_scenario_A_local <- function(seeds = 1:5) {
  # STRICTLY SERIAL — no foreach/parallel calls
  out <- vector("list", length(seeds))
  ii <- 0L
  for (seed in seeds) {
    ii <- ii + 1L
    if (!exists("ddesonn_model")) source("R/api.R")
    set.seed(seed)
    
    trX <- train_x; trY <- train_y
    vaX <- valid_x; vaY <- valid_y
    teX <- test_x;  teY <- test_y
    
    model <- ddesonn_model(
      input_size = ncol(trX),
      output_size = 1,
      hidden_sizes = c(64, 32),
      classification_mode = "binary",
      num_networks = 1,
      init_method = "he",
      lambda = 0.00028,
      custom_scale = 1.04349
    )
    
    fit <- ddesonn_fit(
      model,
      trX, trY,
      validation = list(x = vaX, y = vaY),
      num_epochs = 360,
      lr = 0.125,
      reg_type = "L1",
      validation_metrics = TRUE,
      verbose = FALSE
    )
    
    pred_val  <- ddesonn_predict(model, vaX, aggregate = "mean")
    probs_val <- suppressWarnings(as.numeric(pred_val$prediction))
    thr_val   <- extract_threshold(pred_val)
    if (!length(probs_val) || all(is.na(probs_val))) probs_val <- rep(0.5, nrow(vaX))
    pred_debug(sprintf("VALID (seed=%d)", seed), probs_val, vaY[[1]], thr_val)
    thr_eff <- if (is.finite(thr_val)) thr_val else 0.5
    pred_class_val <- as.integer(probs_val >= thr_eff)
    val_acc <- tryCatch(mean(pred_class_val == as.integer(vaY[[1]])), error = function(e) NA_real_)
    
    pred_test  <- ddesonn_predict(model, teX, aggregate = "mean")
    probs_test <- suppressWarnings(as.numeric(pred_test$prediction))
    if (!length(probs_test) || all(is.na(probs_test))) probs_test <- rep(0.5, nrow(teX))
    pred_debug(sprintf("TEST (seed=%d)", seed), probs_test, teY[[1]], thr_eff)
    pred_class_test <- as.integer(probs_test >= thr_eff)
    test_acc <- tryCatch(mean(pred_class_test == as.integer(teY[[1]])), error = function(e) NA_real_)
    
    out[[ii]] <- data.frame(
      seed    = as.integer(seed),
      thr     = as.numeric(thr_eff),
      val_acc = suppressWarnings(as.numeric(val_acc)),
      test_acc= suppressWarnings(as.numeric(test_acc))
    )
  }
  do.call(rbind, out)
}

# ------------------------------------------------------------------------------
# Run (choose track explicitly)
# ------------------------------------------------------------------------------
seeds <- 1:5
cat("\n🚀 Running Scenario A via ", if (USE_TECHILA) "Techila (foreach)" else "Local (serial)", " ...\n", sep = "")

res <- if (USE_TECHILA) {
  run_scenario_A_techila(seeds)
} else {
  run_scenario_A_local(seeds)
}

cat("\n=== RESULTS (per-seed) ===\n")
print(res)
cat("str(res):\n"); utils::str(res)
cat("\nval_acc summary:\n"); print(summary(res$val_acc))
cat("\ntest_acc summary:\n"); print(summary(res$test_acc))

mean_val_acc  <- suppressWarnings(mean(res$val_acc,  na.rm = TRUE))
mean_test_acc <- suppressWarnings(mean(res$test_acc, na.rm = TRUE))
if (!is.finite(mean_val_acc))  mean_val_acc  <- NA_real_
if (!is.finite(mean_test_acc)) mean_test_acc <- NA_real_

cat("\nMean validation accuracy:", ifelse(is.na(mean_val_acc),  "NA", paste0(round(mean_val_acc  * 100, 2), "%")), "\n")
cat("Mean test accuracy:      ", ifelse(is.na(mean_test_acc), "NA", paste0(round(mean_test_acc * 100, 2), "%")), "\n")