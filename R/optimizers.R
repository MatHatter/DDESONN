# Standard optimizer helpers return signed deltas; the caller subtracts them once.
# State belongs to one parameter tensor, never to a layer's weights AND biases.

.optimizer_matrix <- function(x, parameter, name = "gradient") {
  if (is.list(x) && length(x) == 1L) x <- x[[1L]]
  if (!is.numeric(x) || length(x) != length(parameter) ||
      (!is.null(dim(x)) && !identical(dim(x), dim(parameter)))) {
    stop(name, " must have the same shape as the parameter.", call. = FALSE)
  }
  if (any(!is.finite(x))) stop(name, " must be finite.", call. = FALSE)
  matrix(x, nrow(parameter), ncol(parameter), dimnames = dimnames(parameter))
}

initialize_optimizer_params <- function(optimizer, dim, lookahead_step, layer,
                                        verbose = FALSE, parameter = NULL) {
  if (!optimizer %in% setdiff(ddesonn_optimizer_options(), "adagrad_x"))
    stop("Optimizer ", optimizer, " not supported.", call. = FALSE)
  shape <- if (is.list(dim)) dim[[1L]] else dim
  if (length(shape) != 2L || any(!is.finite(shape)) || any(shape <= 0) ||
      any(shape != as.integer(shape))) stop("Invalid parameter dimensions.", call. = FALSE)
  # Dimension-only callers can initialize accumulators without consuming RNG.
  # Parameter-dependent methods must bind to real values before their first step.
  if (!is.null(parameter)) parameter <- .optimizer_matrix(parameter, matrix(0, shape[1], shape[2]), "parameter")
  zero <- matrix(0, shape[1], shape[2])
  state <- switch(optimizer,
    adam =, lamb =, adadelta = list(m = list(zero), v = list(zero)),
    rmsprop = list(v = list(zero)),
    adagrad = list(r = list(zero)),
    sgd = list(),
    sgd_momentum =, nag = list(momentum = list(zero)),
    ftrl = list(z = list(zero), n = list(zero)),
    lookahead = list(m = list(zero), v = list(zero), slow_weights = parameter,
                     lookahead_counter = 0L, lookahead_step = lookahead_step))
  c(list(param = parameter, t = 0L), state)
}

.optimizer_state <- function(params, name, grads) {
  state <- params[[name]]
  if (is.null(state)) return(lapply(grads, function(g) g * 0))
  if (!is.list(state)) state <- list(state)
  if (length(state) != length(grads)) stop("Optimizer state length mismatch: ", name, call. = FALSE)
  Map(function(s, g) {
    if (is.null(s)) return(g * 0)
    .optimizer_matrix(s, g, paste("optimizer state", name))
  }, state, grads)
}

.optimizer_step <- function(params, grads, optimizer, lr, beta1 = .9,
                            beta2 = .999, epsilon = 1e-8, t = NULL,
                            alpha = lr, beta = 1, lambda1 = .01, lambda2 = .01) {
  if (!is.list(grads)) grads <- list(grads)
  grads <- lapply(grads, function(g) {
    if (is.null(dim(g))) g <- matrix(g, nrow = 1L)
    if (!is.matrix(g) || !is.numeric(g) || any(!is.finite(g)))
      stop("Gradients must be finite numeric matrices.", call. = FALSE)
    g
  })
  if (length(lr) != 1L || !is.finite(lr) || lr < 0) stop("Invalid learning rate.", call. = FALSE)
  if (any(!is.finite(c(beta1, beta2, epsilon))) || beta1 < 0 || beta1 >= 1 ||
      beta2 < 0 || beta2 >= 1 || epsilon <= 0) stop("Invalid optimizer controls.", call. = FALSE)
  step <- if (is.null(t)) (if (is.null(params$t)) 0L else params$t) + 1L else t
  if (length(step) != 1L || !is.finite(step) || step < 1) stop("Invalid optimizer step.", call. = FALSE)
  params$t <- step
  updates <- vector("list", length(grads))
  if (optimizer %in% c("adam", "lamb", "adadelta")) {
    params$m <- .optimizer_state(params, "m", grads)
    params$v <- .optimizer_state(params, "v", grads)
  }
  if (optimizer == "rmsprop") params$v <- .optimizer_state(params, "v", grads)
  if (optimizer == "adagrad") params$r <- .optimizer_state(params, "r", grads)
  if (optimizer %in% c("sgd_momentum", "nag")) params$momentum <- .optimizer_state(params, "momentum", grads)
  if (optimizer == "ftrl") {
    params$z <- .optimizer_state(params, "z", grads)
    params$n <- .optimizer_state(params, "n", grads)
    if (!is.finite(alpha) || alpha <= 0 || any(c(beta, lambda1, lambda2) < 0))
      stop("Invalid FTRL controls.", call. = FALSE)
  }
  actual <- params$param
  if (optimizer %in% c("lamb", "ftrl")) {
    if (is.null(actual)) stop(optimizer, " requires the current parameter.", call. = FALSE)
    if (!is.list(actual)) actual <- list(actual)
    if (length(actual) != length(grads)) stop("Parameter list length mismatch.", call. = FALSE)
    actual <- Map(function(p, g) .optimizer_matrix(p, g, "parameter"), actual, grads)
  }
  for (i in seq_along(grads)) {
    g <- grads[[i]]
    if (optimizer == "sgd") {
      u <- lr * g
    } else if (optimizer %in% c("sgd_momentum", "nag")) {
      # Velocity uses the subtractive convention (including learning rate).
      params$momentum[[i]] <- beta1 * params$momentum[[i]] + lr * g
      u <- if (optimizer == "nag") lr * g + beta1 * params$momentum[[i]] else params$momentum[[i]]
    } else if (optimizer == "adagrad") {
      params$r[[i]] <- params$r[[i]] + g^2
      u <- lr * g / (sqrt(params$r[[i]]) + epsilon)
    } else if (optimizer == "rmsprop") {
      params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * g^2
      u <- lr * g / (sqrt(params$v[[i]]) + epsilon)
    } else if (optimizer %in% c("adam", "lamb")) {
      params$m[[i]] <- beta1 * params$m[[i]] + (1 - beta1) * g
      params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * g^2
      direction <- (params$m[[i]] / (1 - beta1^step)) /
        (sqrt(params$v[[i]] / (1 - beta2^step)) + epsilon)
      if (optimizer == "lamb") {
        direction <- direction + lambda2 * actual[[i]]
        pnorm <- sqrt(sum(actual[[i]]^2))
        unorm <- sqrt(sum(direction^2))
        ratio <- if (pnorm > 0 && unorm > 0) pnorm / unorm else 1
        direction <- ratio * direction
      }
      u <- lr * direction
    } else if (optimizer == "adadelta") {
      # v = EMA of squared gradients; m = EMA of squared unscaled deltas.
      # Use beta2 as rho for both accumulators; no Adam bias correction.
      params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * g^2
      delta <- sqrt(params$m[[i]] + epsilon) / sqrt(params$v[[i]] + epsilon) * g
      params$m[[i]] <- beta2 * params$m[[i]] + (1 - beta2) * delta^2
      u <- lr * delta
    } else if (optimizer == "ftrl") {
      p <- actual[[i]]
      n <- params$n[[i]] + g^2
      sigma <- (sqrt(n) - sqrt(params$n[[i]])) / alpha
      z <- params$z[[i]] + g - sigma * p
      next_p <- -(z - sign(z) * lambda1) / ((beta + sqrt(n)) / alpha + lambda2)
      next_p[abs(z) <= lambda1] <- 0
      u <- p - next_p
      params$z[[i]] <- z
      params$n[[i]] <- n
    } else stop("Unsupported optimizer: ", optimizer, call. = FALSE)
    updates[[i]] <- u
  }
  if (!is.null(params$param)) {
    p <- if (is.list(params$param)) params$param else list(params$param)
    if (length(p) == length(updates)) {
      p <- Map(function(p, u) .optimizer_matrix(p, u, "parameter") - u, p, updates)
      params$param <- if (is.list(params$param)) p else p[[1L]]
    }
  }
  list(params = params, weights_update = updates, biases_update = updates)
}

apply_optimizer_update <- function(optimizer, optimizer_params, grads_matrix, lr, beta1, beta2, epsilon, epoch, self, layer, target,
                                   alpha = NULL, lambda1 = NULL, lambda2 = NULL, verbose = FALSE) {
  if (!target %in% c("weights", "biases")) stop("Unknown optimizer target.", call. = FALSE)
  container <- self[[target]]
  parameter <- if (is.list(container)) container[[layer]] else container
  parameter <- as.matrix(parameter)
  gradient <- .optimizer_matrix(grads_matrix, parameter)
  state <- optimizer_params[[layer]]
  if (is.null(state)) stop("Missing optimizer state for layer ", layer, call. = FALSE)
  # Refresh parameter-dependent state from the model, never from random copies.
  state$param <- parameter
  if (optimizer == "lookahead") {
    state <- lookahead_update(state, list(gradient), lr, beta1, beta2, epsilon,
                              state$lookahead_step, "adam_update", epoch, 0)
    updated <- state$param
  } else {
    result <- .optimizer_step(state, list(gradient), optimizer, lr, beta1, beta2, epsilon,
                              alpha = if (is.null(alpha)) lr else alpha,
                              lambda1 = if (is.null(lambda1)) .01 else lambda1,
                              lambda2 = if (is.null(lambda2)) .01 else lambda2)
    state <- result$params
    updated <- parameter - result$weights_update[[1L]]
  }
  updated <- .optimizer_matrix(updated, parameter, "updated parameter")
  state$param <- updated
  if (is.list(container)) {
    container[[layer]] <- updated
    self[[target]] <- container
  } else self[[target]] <- updated
  optimizers_log_update(optimizer, epoch, layer, target, gradient, parameter, updated, verbose = verbose)
  list(updated_weights_or_biases = updated, updated_optimizer_params = state)
}

# Retain internal helper signatures and return fields used by existing callers.
adam_update <- function(params, grads, lr, beta1, beta2, epsilon, t) {
  out <- .optimizer_step(params, grads, "adam", lr, beta1, beta2, epsilon, t)
  c(out$params, list(weights_update = out$weights_update, biases_update = out$biases_update))
}
rmsprop_update <- function(params, grads, lr, beta2 = .999, epsilon = 1e-8) {
  out <- .optimizer_step(params, grads, "rmsprop", lr, beta2 = beta2, epsilon = epsilon)
  c(out$params, list(updates = out$weights_update))
}
adagrad_update <- function(params, grads, lr, epsilon) {
  .optimizer_step(params, grads, "adagrad", lr, epsilon = epsilon)
}
adadelta_update <- function(params, grads, lr, beta1 = .9, beta2 = .999, epsilon = 1e-8, t = 1) {
  .optimizer_step(params, grads, "adadelta", lr, beta1, beta2, epsilon)
}
sgd_momentum_update <- function(params, grads, lr, momentum) {
  .optimizer_step(params, grads, "sgd_momentum", lr, beta1 = momentum)
}
sgd_update <- function(params, grads, lr) .optimizer_step(params, grads, "sgd", lr)
nag_update <- function(params, grads, lr, beta1 = .9) {
  .optimizer_step(params, grads, "nag", lr, beta1 = beta1)
}
ftrl_update <- function(params, grads, lr, alpha = .1, beta = 1, lambda1 = .01, lambda2 = .01) {
  .optimizer_step(params, grads, "ftrl", lr, alpha = alpha, beta = beta, lambda1 = lambda1, lambda2 = lambda2)
}
lamb_update <- function(params, grads, lr, beta1, beta2, eps, lambda) {
  .optimizer_step(params, grads, "lamb", lr, beta1, beta2, eps, lambda2 = lambda)
}
