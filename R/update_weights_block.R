#' Update weights block (ML + SL) for DDESONN
#' 
#' Apply one standard optimizer update per tensor, or dispatch to the preserved
#' custom Adagrad X implementation.
#' 
#' @param self R6 model (DDESONN/SONN) – modified in-place
#' @param update_weights logical
#' @param optimizer character (adam, rmsprop, sgd, sgd_momentum, nag, ftrl, lamb, lookahead, adagrad, adagrad_x, adadelta)
#' @param optimizer_params_weights list of per-layer optimizer params (will be returned/updated)
#' @param weight_gradients list of per-layer dL/dW from learn()
#' @param lr numeric learning rate
#' @param reg_type NULL or one of c("L2","L1","L1_L2","Group_Lasso","Max_Norm")
#' @param beta1,beta2,epsilon numerics
#' @param epoch integer
#' @param lookahead_step integer
#' @param verbose logical
#' 
#' @return list(updated_optimizer_params = optimizer_params_weights)
#' @details Relies on existing helpers available in your namespace:
#'   - initialize_optimizer_params()
#'   - clip_gradient_norm()
#'   - apply_optimizer_update()
#' @keywords internal
#' @noRd
update_weights_block <- function(
    self, update_weights, optimizer, optimizer_params_weights, weight_gradients,
    lr, reg_type, beta1, beta2, epsilon, epoch, lookahead_step, verbose = FALSE
) {
  if (!update_weights) return(list(updated_optimizer_params = optimizer_params_weights))
  if (identical(optimizer, "adagrad_x")) {
    return(.adagrad_x$weights(self, update_weights, "adagrad",
      optimizer_params_weights, weight_gradients, lr, reg_type, beta1, beta2,
      epsilon, epoch, lookahead_step, verbose))
  }
  .update_parameter_block(self, "weights", optimizer, optimizer_params_weights,
                          weight_gradients, lr, reg_type, beta1, beta2, epsilon,
                          epoch, lookahead_step, verbose)
}

# One shared route for SL/ML weights and biases. learn() already supplies the
# per-parameter gradients, including activation derivatives for the biases.
.update_parameter_block <- function(self, target, optimizer, states, gradients,
                                     lr, reg_type, beta1, beta2, epsilon, epoch,
                                     lookahead_step, verbose) {
  parameters <- self[[target]]
  if (!is.list(parameters)) parameters <- list(parameters)
  if (is.null(states)) states <- vector("list", length(parameters))
  length(states) <- length(parameters)
  if (!is.list(gradients)) gradients <- list(gradients)
  if (length(gradients) != length(parameters)) stop("Gradient layer count mismatch.", call. = FALSE)
  # NULL retains the ordinary gradient-descent fallback.
  method <- if (is.null(optimizer)) "sgd" else optimizer
  for (layer in seq_along(parameters)) {
    p <- as.matrix(parameters[[layer]])
    g <- gradients[[layer]]
    # learn() emits 1 x units bias gradients; initialized ML biases are units x
    # 1 and SL biases may be vectors. These are equivalent unit orderings.
    if (target == "biases" && is.numeric(g) && length(g) == length(p) &&
        (is.null(dim(g)) || any(dim(g) == 1L)) && any(dim(p) == 1L)) {
      g <- matrix(g, nrow(p), ncol(p))
    }
    g <- .optimizer_matrix(g, p)
    g <- clip_gradient_norm(g, max_norm = 5)
    # Coupled regularization belongs to the gradient, before the sole update.
    g <- g + .optimizer_regularization(self, p, reg_type, target, layer)
    if (is.null(states[[layer]])) {
      states[[layer]] <- initialize_optimizer_params(method, list(dim(p)),
                          lookahead_step, layer, parameter = p)
    }
    out <- apply_optimizer_update(method, states, g, lr, beta1, beta2, epsilon,
                                   epoch, self, layer, target, verbose = verbose)
    states[[layer]] <- out$updated_optimizer_params
  }
  list(updated_optimizer_params = states)
}

.optimizer_regularization <- function(self, p, reg_type, target, layer) {
  zero <- p * 0
  if (is.null(reg_type)) return(zero)
  strength <- if (is.null(self$lambda)) 0 else self$lambda
  if (reg_type == "L2") return(strength * p)
  if (reg_type == "L1") return(strength * sign(p))
  if (reg_type == "L1_L2") return(strength * (.5 * sign(p) + .5 * p))
  if (reg_type == "Group_Lasso") {
    groups <- if (target == "weights" && isTRUE(self$ML_NN)) {
      if (is.null(self$groups) || is.null(self$groups[[layer]])) list(seq_len(ncol(p))) else self$groups[[layer]]
    } else list(seq_len(ncol(p)))
    for (group in groups) {
      block <- p[, group, drop = FALSE]
      zero[, group] <- block / (sqrt(sum(block^2)) + 1e-8)
    }
    return(strength * zero)
  }
  if (reg_type == "Max_Norm") {
    clipped <- p
    if (target == "weights" && isTRUE(self$ML_NN)) {
      norms <- sqrt(colSums(p^2))
      for (j in seq_along(norms)) if (norms[j] > 1) clipped[, j] <- p[, j] / norms[j]
    } else {
      norm <- sqrt(sum(p^2))
      if (norm > 1) clipped <- p / norm
    }
    return(strength * (p - clipped))
  }
  zero
}

clip_gradient_norm <- function(gradient, min_norm = 1e-3, max_norm = 5) {
  if (any(is.na(gradient)) || all(gradient == 0)) return(gradient)
  
  grad_norm <- sqrt(sum(gradient^2, na.rm = TRUE))
  
  if (is.na(grad_norm)) return(gradient)  # added line
  
  if (grad_norm > max_norm) {
    gradient <- gradient * (max_norm / grad_norm)
  } else if (grad_norm < min_norm && grad_norm > 0) {
    gradient <- gradient * (min_norm / grad_norm)
  }
  
  return(gradient)
}
