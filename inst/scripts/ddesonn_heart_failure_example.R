#!/usr/bin/env Rscript
# Example workflow using the high-level DDESONN API.

suppressPackageStartupMessages(library(DDESONN))

# Use the built-in 'mtcars' data set for a lightweight binary classification task.
# The transmission column (am) is treated as the target label.
data <- mtcars
target <- "am"
features <- setdiff(colnames(data), target)

set.seed(42)
idx <- sample.int(nrow(data), floor(0.75 * nrow(data)))
train_x <- data[idx, features]
train_y <- data[idx, target, drop = FALSE]
valid_x <- data[-idx, features]
valid_y <- data[-idx, target, drop = FALSE]

model <- ddesonn_model(
  input_size = ncol(train_x),
  output_size = 1,
  hidden_sizes = c(32, 16),
  classification_mode = "binary",
  num_networks = 1
)

ddesonn_fit(
  model,
  train_x,
  train_y,
  validation = list(x = valid_x, y = valid_y),
  num_epochs = 2,
  lr = 0.05,
  validation_metrics = TRUE,
  verbose = FALSE
)

pred <- ddesonn_predict(model, valid_x, aggregate = "mean")
cat("First few probability predictions:\n")
print(head(pred$prediction))

if (!is.null(pred$class)) {
  cat("Predicted classes:\n")
  print(head(pred$class))
}
