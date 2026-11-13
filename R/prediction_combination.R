# Shared inference calculation for the workbench. Legacy file-based fusion has
# task-specific result formats; this helper always returns an N x K matrix.
.ddesonn_combine_predictions <- function(predictions, mode, method = "mean", weights = NULL) {
  mode <- match.arg(mode, c("binary", "multiclass", "regression"))
  method <- match.arg(method, c("mean", "weighted", "median", "majority", "proportion"))
  if (!is.list(predictions) || !length(predictions)) stop("Supply member predictions.")
  predictions <- lapply(predictions, as.matrix)
  shape <- dim(predictions[[1]])
  if (any(vapply(predictions, function(p) !identical(dim(p), shape) || any(!is.finite(p)), logical(1))))
    stop("Member predictions must have identical dimensions and finite values.")
  if (mode != "regression" && any(vapply(predictions, function(p) any(p < 0 | p > 1), logical(1))))
    stop("Classification probabilities must be within 0–1.")
  if (method %in% c("majority", "proportion")) {
    if (mode == "regression" || (method == "proportion" && mode != "binary"))
      stop("This vote method does not apply to this task.")
    if (mode == "binary") {
      votes <- Reduce(`+`, lapply(predictions, function(p) (p >= .5) * 1)) / length(predictions)
      return(if (method == "proportion") votes else (votes > .5) * 1)
    }
    counts <- matrix(0, shape[1], shape[2])
    for (p in predictions) {
      idx <- cbind(seq_len(shape[1]), max.col(p, ties.method = "first"))
      counts[idx] <- counts[idx] + 1
    }
    out <- counts * 0
    out[cbind(seq_len(shape[1]), max.col(counts, ties.method = "first"))] <- 1
    return(out)
  }
  if (method == "weighted") {
    if (length(weights) != length(predictions) || any(!is.finite(weights) | weights < 0) || sum(weights) <= 0)
      stop("Supply one finite nonnegative weight per member, with a positive total.")
    weights <- weights / sum(weights)
    out <- Reduce(`+`, Map(function(p, w) p * w, predictions, weights))
  } else if (method == "median") {
    out <- apply(array(unlist(predictions), c(shape, length(predictions))), c(1, 2), stats::median)
    out <- matrix(out, shape[1], shape[2])
  } else out <- Reduce(`+`, predictions) / length(predictions)
  if (mode == "multiclass") {
    total <- rowSums(out)
    # Componentwise medians can all be zero for disjoint one-hot members.
    empty <- total == 0
    if (any(empty)) out[empty, ] <- (Reduce(`+`, predictions) / length(predictions))[empty, , drop = FALSE]
    out <- out / rowSums(out)
  }
  out
}
