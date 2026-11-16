# Dense-network reduction is chosen before tensor clipping and regularization.
.ddesonn_gradient_reduction <- function(average_gradients = NULL, gradient_reduction = "sum") {
  if (!is.null(average_gradients)) {
    if (!is.logical(average_gradients) || length(average_gradients) != 1L || is.na(average_gradients))
      stop("average_gradients must be TRUE or FALSE.", call. = FALSE)
    return(if (average_gradients) "mean" else "sum")
  }
  match.arg(gradient_reduction, c("sum", "mean", "legacy_mixed"))
}

.ddesonn_reduce_gradients <- function(inputs, delta, reduction) {
  weights <- crossprod(inputs, delta)
  biases <- matrix(colSums(delta), nrow = 1L)
  if (reduction == "mean") weights <- weights / nrow(delta)
  if (reduction != "sum") biases <- biases / nrow(delta)
  list(weights = weights, biases = biases)
}
