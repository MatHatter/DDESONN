# Dense-network reduction is chosen before tensor clipping and regularization.
.ddesonn_gradient_reduction <- function(average_gradients = NULL, gradient_reduction = "sum") {
  if (!is.null(average_gradients)) {
    if (!is.logical(average_gradients) || length(average_gradients) != 1L || is.na(average_gradients))
      stop("average_gradients must be TRUE or FALSE.", call. = FALSE)
    return(if (average_gradients) "mean" else "sum")
  }
  match.arg(gradient_reduction, c("sum", "mean", "legacy_mixed", "mean_weights"))
}

.ddesonn_reduce_gradients <- function(inputs, delta, reduction) {
  weights <- crossprod(inputs, delta)
  biases <- matrix(colSums(delta), nrow = 1L)
  if (reduction %in% c("mean", "mean_weights")) weights <- weights / nrow(delta)
  if (reduction %in% c("mean", "legacy_mixed")) biases <- biases / nrow(delta)
  list(weights = weights, biases = biases)
}

.ddesonn_gradient_choices <- function(reduction) {
  c(weights = if (reduction %in% c("mean", "mean_weights")) "mean" else "sum",
    biases = if (reduction %in% c("mean", "legacy_mixed")) "mean" else "sum")
}

.ddesonn_gradient_label <- function(reduction) {
  choices <- .ddesonn_gradient_choices(reduction)
  labels <- c(sum = "Sum", mean = "Average")
  sprintf("Weights: %s | Biases: %s", labels[choices[1L]], labels[choices[2L]])
}
