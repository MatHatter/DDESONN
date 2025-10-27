#!/usr/bin/env Rscript
# Example workflow using the high-level DDESONN API.
source("R/api.R")

suppressPackageStartupMessages(source("R/api.R"))

`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

# ------------------------------------------------------------------------------
# Dataset & splits (mtcars → binary classification on 'am')
# ------------------------------------------------------------------------------
data <- mtcars
target <- "am"
features <- setdiff(colnames(data), target)

set.seed(42)
n <- nrow(data)
all_idx <- seq_len(n)

# 60/20/20 split
idx_train <- sample(all_idx, floor(0.6 * n))
remain1   <- setdiff(all_idx, idx_train)
idx_valid <- sample(remain1, floor(0.2 * n))
idx_test  <- setdiff(remain1, idx_valid)

# Build X/Y frames BEFORE scaling
train_x <- data[idx_train, features, drop = FALSE]
train_y <- data[idx_train, target,   drop = FALSE]
valid_x <- data[idx_valid, features, drop = FALSE]
valid_y <- data[idx_valid, target,   drop = FALSE]
test_x  <- data[idx_test,  features, drop = FALSE]
test_y  <- data[idx_test,  target,   drop = FALSE]

# ------------------------------------------------------------------------------
# Scaling helpers (fit on TRAIN only)
# ------------------------------------------------------------------------------
scale_fit <- function(X) {
  num_cols <- names(which(sapply(X, is.numeric)))
  if (length(num_cols) == 0) stop("No numeric columns to scale.")
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

# Fit scaler on TRAIN, apply to all splits
scaler  <- scale_fit(train_x)
train_x <- scale_apply(train_x, scaler)
valid_x <- scale_apply(valid_x, scaler)
test_x  <- scale_apply(test_x,  scaler)

# Quick sanity check
# str(train_x); str(valid_x); str(test_x)

# ------------------------------------------------------------------------------
# Single-model example fit (kept as a simple, clear baseline)
# ------------------------------------------------------------------------------
model <- ddesonn_model(
  input_size = ncol(train_x),
  output_size = 1,
  hidden_sizes = c(32, 16),
  classification_mode = "binary",
  activation_functions = c("relu", "relu", "sigmoid"),
  activation_functions_predict = c("relu", "relu", "sigmoid"),
  num_networks = 1
)

# Restore best weights toggle (from upgraded API)
RESTORE_BEST_WEIGHTS <- TRUE  # set FALSE to keep final-epoch weights

ddesonn_fit(
  model,
  train_x,
  train_y,
  validation = list(x = valid_x, y = valid_y),
  num_epochs = 3,
  lr = 0.02,
  validation_metrics = TRUE,
  verbose = TRUE,
  best_weights_on_latest_weights_off = RESTORE_BEST_WEIGHTS
)

# ------------------------------------------------------------------------------
# VALIDATION EVALUATION
# ------------------------------------------------------------------------------
pred <- ddesonn_predict(model, valid_x, aggregate = "mean")

# Flatten predicted probs
probs <- as.numeric(pred$prediction)

# Threshold: prefer tuned/best if present; else 0.5
thr <- if (!is.null(pred$chosen_threshold)) {
  pred$chosen_threshold
} else if (!is.null(pred$best_threshold)) {
  pred$best_threshold
} else {
  0.5
}

# Derived classes and actuals
predicted_class <- as.integer(probs >= thr)
actual          <- as.integer(valid_y[[1]])

stopifnot(length(probs) == length(actual), length(predicted_class) == length(actual))

comparison <- data.frame(
  actual = actual,
  predicted_class = predicted_class,
  predicted_prob = round(probs, 3)
)

print(tail(comparison, 10))

acc <- mean(comparison$actual == comparison$predicted_class)
cat("Validation accuracy:", round(acc * 100, 2), "% (thr =", thr, ")\n")
print(table(Actual = comparison$actual, Predicted = comparison$predicted_class))

cat("First few probability predictions:\n");  print(head(pred$prediction))
cat("Summary probability predictions:\n");     print(summary(pred$prediction))
if (!is.null(pred$class)) {
  cat("Predicted classes:\n"); print(head(pred$class))
}

# ------------------------------------------------------------------------------
# TEST EVALUATION (true hold-out, 20% split)
# ------------------------------------------------------------------------------
pred_test <- ddesonn_predict(model, test_x, aggregate = "mean")
probs_test <- as.numeric(pred_test$prediction)
pred_class_test <- as.integer(probs_test >= thr)   # use validation threshold
actual_test <- as.integer(test_y[[1]])

stopifnot(length(probs_test) == length(actual_test))

comparison_test <- data.frame(
  actual = actual_test,
  predicted_class = pred_class_test,
  predicted_prob = round(probs_test, 3)
)

test_acc <- mean(comparison_test$actual == comparison_test$predicted_class)
cat("TEST accuracy (true hold-out):", round(test_acc * 100, 2), "% (thr =", thr, ")\n")
test_cm <- table(Actual = comparison_test$actual, Predicted = comparison_test$predicted_class)
print(test_cm)

# Handy globals for quick review after run
THRESHOLD_USED <- thr
VALID_ACC      <- acc
TEST_ACC       <- test_acc
VALID_CM       <- table(Actual = comparison$actual,       Predicted = comparison$predicted_class)
TEST_CM        <- test_cm
COMPARISON_VALID <- comparison
COMPARISON_TEST  <- comparison_test

# ------------------------------------------------------------------------------
# DDESONN Scenario Runner (A–D) — clear, minimal orchestration
# ------------------------------------------------------------------------------
scenario_presets <- list(
  A = list(label="Scenario A", do_ensemble=FALSE, num_networks=1L, num_temp_iterations=0L, aggregate="mean",  prediction_type="class", seeds = 1L),
  B = list(label="Scenario B", do_ensemble=FALSE, num_networks=4L, num_temp_iterations=0L, aggregate="none",  prediction_type="response", seeds = 1:4),
  C = list(label="Scenario C", do_ensemble=TRUE,  num_networks=5L, num_temp_iterations=0L, aggregate="mean",  prediction_type="class", seeds = 1:5),
  D = list(label="Scenario D", do_ensemble=TRUE,  num_networks=3L, num_temp_iterations=2L, aggregate="mean",  prediction_type="class", seeds = c(11, 22, 33))
)

# Default output_root now uses the REPO ROOT so persistence adds "artifacts" exactly once.
run_scenario <- function(scn = c("A","B","C","D"), output_root = .ddesonn_find_root()) {
  scn <- match.arg(scn)
  cfg <- scenario_presets[[scn]]
  
  cat("\n==============================\n",
      cfg$label, "\n",
      "==============================\n", sep = "")
  
  run <- ddesonn_run(
    x = train_x,
    y = train_y,
    classification_mode = "binary",
    hidden_sizes = c(32, 16),
    seeds = cfg$seeds,
    do_ensemble = cfg$do_ensemble,
    num_networks = cfg$num_networks,
    num_temp_iterations = cfg$num_temp_iterations,
    validation = list(x = valid_x, y = valid_y),
    training_overrides = list(num_epochs = 2, lr = 0.05, validation_metrics = TRUE, verbose = FALSE),
    temp_overrides = if (cfg$num_temp_iterations > 0L) list(num_epochs = 1, lr = 0.02, validation_metrics = TRUE, verbose = FALSE) else NULL,
    prediction_data = valid_x,
    prediction_type = cfg$prediction_type,
    aggregate = cfg$aggregate,
    seed_aggregate = "none",
    output_root = output_root,  # persistence will add artifacts/SingleRuns|EnsembleRuns
    save_models = TRUE
  )
  
  # Mirror the same guard used in persistence to print the actual artifacts root once.
  art_root <- {
    nr <- normalizePath(output_root, winslash = "/", mustWork = FALSE)
    if (basename(nr) == "artifacts") nr else file.path(output_root, "artifacts")
  }
  cat("Artifacts root:", normalizePath(art_root, winslash = "/", mustWork = FALSE), "\n")
  
  # Tiny validation preview (aggregate or first per-model)
  if (!is.null(run$predictions$aggregate)) {
    cat("Validation preview (aggregate head):\n"); print(utils::head(run$predictions$aggregate))
  } else if (length(run$runs) && !is.null(run$runs[[1]]$main$predictions$per_model)) {
    cat("Validation preview (first model head):\n"); print(utils::head(run$runs[[1]]$main$predictions$per_model[[1]]))
  } else {
    cat("[no predictions found in run object]\n")
  }
  
  invisible(run)
}

## =========================
## DDESONN Runner – Modes
## =========================
## Pick ONE to run now by uncommenting the call(s) below.

## SCENARIO A: Single-run only (no ensemble, ONE model)
invisible(run_scenario("A"))

## SCENARIO B: Single-run, MULTI-MODEL (no ensemble)
# invisible(run_scenario("B"))

## SCENARIO C: Main ensemble only (no TEMP/prune-add)
# invisible(run_scenario("C"))

## SCENARIO D: Main + TEMP iterations (prune/add enabled)
# invisible(run_scenario("D"))
