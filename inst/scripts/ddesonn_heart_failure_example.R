#!/usr/bin/env Rscript
# ============================================================
# DDESONN Runner – Four Scenarios (A–D) in One Walkthrough
# ============================================================
# Scenario A: Single-run only (no ensemble, ONE model)
#   do_ensemble         <- FALSE
#   num_networks        <- 1L
#   num_temp_iterations <- 0L   # ignored when do_ensemble = FALSE
#
# Scenario B: Single-run, MULTI-MODEL (no ensemble aggregation)
#   do_ensemble         <- FALSE
#   num_networks        <- 4L
#   num_temp_iterations <- 0L
#
# Scenario C: Main ensemble only (no TEMP/prune-add loop)
#   do_ensemble         <- TRUE
#   num_networks        <- 5L
#   num_temp_iterations <- 0L
#
# Scenario D: Main pass + TEMP iterations (prune/add enabled)
#   do_ensemble         <- TRUE
#   num_networks        <- 3L
#   num_temp_iterations <- 2L
#
# This script shows how each scenario maps onto the high-level
# DDESONN API.  Every section is heavily commented so that a
# first-year university student could follow along.

suppressPackageStartupMessages(library(DDESONN))

# ------------------------------------------------------------------
# 1. Prepare a toy classification task (the classic 'mtcars' dataset)
# ------------------------------------------------------------------
# We predict whether a car has automatic (0) or manual (1) transmission.
data <- mtcars
target <- "am"
features <- setdiff(colnames(data), target)

# Split 75% / 25% into training and validation sets.
set.seed(42)
idx <- sample.int(nrow(data), floor(0.75 * nrow(data)))
train_x <- data[idx, features]
train_y <- data[idx, target, drop = FALSE]
valid_x <- data[-idx, features]
valid_y <- data[-idx, target, drop = FALSE]

label_vec <- function(y) {
  if (is.data.frame(y)) {
    as.numeric(y[[1]])
  } else if (is.matrix(y)) {
    as.numeric(y[, 1])
  } else {
    as.numeric(y)
  }
}

build_model <- function(num_networks) {
  ddesonn_model(
    input_size = ncol(train_x),
    output_size = 1,
    hidden_sizes = c(32, 16),
    classification_mode = "binary",
    num_networks = num_networks
  )
}

train_model <- function(model, seed, epochs = 2, lr = 0.05, sample_weights = NULL, verbose = FALSE) {
  set.seed(seed)
  ddesonn_fit(
    model,
    train_x,
    train_y,
    validation = list(x = valid_x, y = valid_y),
    num_epochs = epochs,
    lr = lr,
    sample_weights = sample_weights,
    validation_metrics = TRUE,
    verbose = verbose
  )
  invisible(model)
}

summarise_predictions <- function(pred, label = NULL, heading = NULL) {
  if (!is.null(heading)) {
    cat("\n", heading, "\n", sep = "")
    cat(strrep("-", nchar(heading)), "\n", sep = "")
  }
  cat("First few probability predictions:\n")
  print(head(pred$prediction))
  if (!is.null(pred$class)) {
    cat("First few class predictions:\n")
    print(head(pred$class))
  }
  if (!is.null(label)) {
    cat("First few ground-truth labels:\n")
    print(head(label))
  }
}

hard_example_weights <- function(model, top_fraction = 0.3) {
  probs <- ddesonn_predict(model, train_x, aggregate = "mean")$prediction[, 1]
  truth <- label_vec(train_y)
  errors <- abs(truth - probs)
  weights <- rep(1, length(errors))
  if (length(errors) == 0L) {
    return(weights)
  }
  top_k <- max(1L, ceiling(length(errors) * top_fraction))
  hardest <- order(errors, decreasing = TRUE)[seq_len(top_k)]
  weights[hardest] <- 5
  weights
}

# ---------------------------------------------
# Scenario A – a single SONN trained end-to-end
# ---------------------------------------------
cat("\n==============================\n")
cat("Scenario A: One network, no ensemble\n")
cat("==============================\n")
scenario_a <- build_model(num_networks = 1)
train_model(scenario_a, seed = 1001)
pred_a <- ddesonn_predict(scenario_a, valid_x, aggregate = "mean", type = "class")
summarise_predictions(pred_a, label_vec(valid_y), "Scenario A results")

# ----------------------------------------------------------------------
# Scenario B – train multiple networks but inspect them individually.
# ----------------------------------------------------------------------
# This mirrors running several SONNs without averaging their outputs.
cat("\n==============================\n")
cat("Scenario B: Multiple networks, analysed separately\n")
cat("==============================\n")
scenario_b <- build_model(num_networks = 4)
train_model(scenario_b, seed = 2002)
pred_b <- ddesonn_predict(scenario_b, valid_x, aggregate = "none")

# Each entry in pred_b$per_model is a matrix of predictions for one SONN.
for (i in seq_along(pred_b$per_model)) {
  cat(sprintf("\nModel %d probability head:\n", i))
  print(head(pred_b$per_model[[i]]))
}
# If you still want class labels per model, apply your own threshold (0.5 here).
per_model_classes <- lapply(pred_b$per_model, function(mat) ifelse(mat >= 0.5, 1L, 0L))
cat("\nExample: class predictions from the first model in Scenario B:\n")
print(head(per_model_classes[[1]]))

# ---------------------------------------------------------
# Scenario C – classic ensemble: average the SONN members.
# ---------------------------------------------------------
cat("\n==============================\n")
cat("Scenario C: Main ensemble (mean aggregation)\n")
cat("==============================\n")
scenario_c <- build_model(num_networks = 5)
train_model(scenario_c, seed = 3003)
pred_c <- ddesonn_predict(scenario_c, valid_x, aggregate = "mean", type = "class")
summarise_predictions(pred_c, label_vec(valid_y), "Scenario C results")

# -------------------------------------------------------------------
# Scenario D – ensemble + two TEMP passes that focus on hard samples.
# -------------------------------------------------------------------
cat("\n==============================\n")
cat("Scenario D: Ensemble with TEMP-style fine-tuning\n")
cat("==============================\n")
scenario_d <- build_model(num_networks = 3)
train_model(scenario_d, seed = 4004)

# After the main run we look for hard training cases and up-weight them.
weights <- hard_example_weights(scenario_d, top_fraction = 0.3)
cat("TEMP setup: emphasising", sum(weights > 1), "hard training rows out of", length(weights), "total.\n")

# Run two lightweight TEMP iterations. We use a smaller learning rate so the
# fine-tuning nudges the ensemble instead of completely retraining it.
for (iter in seq_len(2)) {
  cat(sprintf("Starting TEMP iteration %d...\n", iter))
  train_model(
    scenario_d,
    seed = 5000 + iter,
    epochs = 1,
    lr = 0.02,
    sample_weights = weights,
    verbose = FALSE
  )
}

pred_d <- ddesonn_predict(scenario_d, valid_x, aggregate = "mean", type = "class")
summarise_predictions(pred_d, label_vec(valid_y), "Scenario D results after TEMP passes")

cat("\nWalkthrough complete!\n")
