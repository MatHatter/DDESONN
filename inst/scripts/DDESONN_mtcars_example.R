#!/usr/bin/env Rscript
# Example workflow using the high-level DDESONN API.

suppressPackageStartupMessages(library(DDESONN))

# Use the built-in 'mtcars' data set for a lightweight binary classification task.
# The transmission column (am) is treated as the target label.
# Data + basic setup
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

# --- Scaling helpers (train-only stats) ---
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
  for (nm in s$num_cols) {
    Xs[[nm]] <- (Xs[[nm]] - s$mu[[nm]]) / s$sd[[nm]]
  }
  Xs
}

# Fit scaler on TRAIN, apply to all splits
scaler  <- scale_fit(train_x)
train_x <- scale_apply(train_x, scaler)
valid_x <- scale_apply(valid_x, scaler)
test_x  <- scale_apply(test_x,  scaler)

# Quick sanity check
str(train_x); str(valid_x); str(test_x)





train_y <- data[idx_train, target, drop = FALSE]

valid_y <- data[idx_valid, target, drop = FALSE]

test_y  <- data[idx_test,  target, drop = FALSE]

model <- ddesonn_model(
  input_size = ncol(train_x),
  output_size = 1,
  hidden_sizes = c(32, 16),
  classification_mode = "binary",
  activation_functions = c("relu", "relu", "sigmoid"),
  activation_functions_predict = c("relu", "relu", "sigmoid"),
  num_networks = 1
)

# Explicit toggle for best-weight restore behavior from the upgraded API
RESTORE_BEST_WEIGHTS <- TRUE  # set FALSE to keep final-epoch weights

ddesonn_fit(
  model,
  train_x,
  train_y,
  validation = list(x = valid_x, y = valid_y),
  num_epochs = 300,
  lr = 0.02,
  validation_metrics = TRUE,
  verbose = TRUE,
  best_weights_on_latest_weights_off = RESTORE_BEST_WEIGHTS
)

# ----------------------------
# VALIDATION EVALUATION (kept)
# ----------------------------
pred <- ddesonn_predict(model, valid_x, aggregate = "mean")

# --- Build actual vs predicted safely ---

# 1) Flatten predicted probs to a numeric vector
probs <- as.numeric(pred$prediction)

# 2) Pick a threshold (prefer tuned/best if present; else 0.5)
thr <- if (!is.null(pred$chosen_threshold)) {
  pred$chosen_threshold
} else if (!is.null(pred$best_threshold)) {
  pred$best_threshold
} else {
  0.5
}

# 3) Derive classes from probabilities
predicted_class <- as.integer(probs >= thr)

# 4) Actual labels as a simple vector
actual <- as.integer(valid_y[[1]])

# 5) Sanity check lengths
stopifnot(length(probs) == length(actual), length(predicted_class) == length(actual))

# 6) Side-by-side comparison
comparison <<- data.frame(
  actual = actual,
  predicted_class = predicted_class,
  predicted_prob = round(probs, 3)
)

print(tail(comparison, 10))

# Quick metrics
acc <- mean(comparison$actual == comparison$predicted_class)
cat("Validation accuracy:", round(acc * 100, 2), "% (thr =", thr, ")\n")

# Confusion matrix
print(table(Actual = comparison$actual, Predicted = comparison$predicted_class))












cat("First few probability predictions:\n")
print(head(pred$prediction))

cat("Summary probability predictions:\n")
print(summary(pred$prediction))



if (!is.null(pred$class)) {
  cat("Predicted classes:\n")
  print(head(pred$class))
}

# -----------------------------------------
# TEST EVALUATION (true hold-out, 20% split)
# -----------------------------------------
pred_test <- ddesonn_predict(model, test_x, aggregate = "mean")

probs_test <- as.numeric(pred_test$prediction)
pred_class_test <- as.integer(probs_test >= thr)   # use threshold from validation
actual_test <- as.integer(test_y[[1]])

stopifnot(length(probs_test) == length(actual_test))

comparison_test <<- data.frame(
  actual = actual_test,
  predicted_class = pred_class_test,
  predicted_prob = round(probs_test, 3)
)

test_acc <<- mean(comparison_test$actual == comparison_test$predicted_class)
cat("TEST accuracy (true hold-out):", round(test_acc * 100, 2), "% (thr =", thr, ")\n")

test_cm <- table(Actual = comparison_test$actual, Predicted = comparison_test$predicted_class)
print(test_cm)

# Handy globals for quick review after run
THRESHOLD_USED <<- thr
VALID_ACC <<- acc
TEST_ACC <<- test_acc
VALID_CM <<- table(Actual = comparison$actual, Predicted = comparison$predicted_class)
TEST_CM <<- test_cm
COMPARISON_VALID <<- comparison
COMPARISON_TEST <<- comparison_test