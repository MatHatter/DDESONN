# Adagrad X compatibility implementation.
# Numerical expressions retained from commit 5dd541a7fe5c6fc56ddc5b7d4a3e8b0ec7d55373.
# Only unreachable non-Adagrad dispatch branches were removed.
# Intentional legacy behavior: ML manual-plus-adaptive updates, row-major
# reconstruction, error sums for ML biases, and initialization RNG draws.
# Keep this isolated from the corrected standard optimizer implementations.
.adagrad_x <- local({
  initialize_optimizer_params <- function (optimizer, dim, lookahead_step, layer, verbose = FALSE)
  {
      if (length(dim) == 2 && is.null(layer)) {
          dim <- list(dim)
      }
      layer_dim <- dim[[1]]
      if (length(layer_dim) != 2 || any(is.na(layer_dim)) || any(layer_dim <= 0)) {
          if (verbose)
              message("Invalid dimensions detected. Setting default dimension [1, 1].")
          layer_dim <- c(1, 1)
      }
      nrow_dim <- layer_dim[1]
      ncol_dim <- layer_dim[2]
      current_layer <- if (!is.null(layer))
          layer
      else 1
      if (verbose) {
          message("Layer ", current_layer, " dimensions: nrow = ", nrow_dim, ", ncol = ", ncol_dim)
      }
      param_init <- matrix(rnorm(nrow_dim * ncol_dim), nrow = nrow_dim, ncol = ncol_dim)
      entry <- list(param = param_init, r = matrix(0, nrow = nrow_dim, ncol = ncol_dim))
      if (verbose) {
          message("Layer ", current_layer, " optimizer tracking params initialized:")
          for (line in utils::capture.output(str(entry))) message(line)
      }
      return(entry)
  }

  adagrad_update <- function (params, grads, lr, epsilon)
  {
      if (!is.list(params$r)) {
          params$r <- vector("list", length(grads))
      }
      updates <- vector("list", length(grads))
      for (i in seq_along(grads)) {
          grad_dims <- dim(grads[[i]])
          if (is.null(grad_dims)) {
              grad_dims <- c(1)
              grads[[i]] <- array(grads[[i]], dim = grad_dims)
          }
          if (is.null(params$r[[i]]) || !all(dim(params$r[[i]]) == grad_dims)) {
              params$r[[i]] <- array(0, dim = grad_dims)
          }
          params$r[[i]] <- params$r[[i]] + grads[[i]]^2
          updates[[i]] <- lr * grads[[i]]/(sqrt(params$r[[i]]) + epsilon)
      }
      return(list(params = params, weights_update = updates, biases_update = updates))
  }

  apply_optimizer_update <- function (optimizer, optimizer_params, grads_matrix, lr, beta1, beta2, epsilon, epoch, self, layer, target,
      alpha = NULL, lambda1 = NULL, lambda2 = NULL, verbose = FALSE)
  {
      is_sl <- !self$ML_NN
      if (is_sl) {
          orig_weights <- self$weights
          orig_biases <- self$biases
          on.exit({
              if (!is.null(orig_weights)) self$weights <- orig_weights
              if (!is.null(orig_biases)) self$biases <- orig_biases
          }, add = TRUE)
          if (target == "weights" && !is.list(self$weights))
              self$weights <- list(as.matrix(self$weights))
          if (target == "biases" && !is.list(self$biases))
              self$biases <- list(as.matrix(self$biases))
      }
      param_now <- if (identical(target, "weights"))
          as.matrix(self$weights[[layer]])
      else as.matrix(self$biases[[layer]])
      if (is.null(dim(param_now)))
          param_now <- matrix(as.numeric(param_now), 1, 1)
      target_dim <- dim(param_now)
      gm <- grads_matrix
      if (is.list(gm))
          gm <- gm[[1]]
      gm <- as.matrix(gm)
      if (is.null(dim(gm)))
          gm <- matrix(as.numeric(gm), nrow = target_dim[1], ncol = target_dim[2])
      if (nrow(gm) != target_dim[1] || ncol(gm) != target_dim[2]) {
          if (length(gm) == prod(target_dim)) {
              gm <- matrix(as.numeric(gm), nrow = target_dim[1], ncol = target_dim[2])
          }
          else if (length(gm) == 1L) {
              gm <- matrix(rep(as.numeric(gm), prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
          }
          else {
              gm <- matrix(rep(as.numeric(gm)[1], prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
          }
      }
      grads_matrix <- gm
      {
          Wpre <- if (target == "weights")
              self$weights[[layer]]
          else self$biases[[layer]]
          layer_boost <- if (layer == self$num_layers)
              1
          else 1
          grads_input <- if (is.list(grads_matrix)) {
              grads_matrix
          }
          else if (is.null(dim(grads_matrix))) {
              list(matrix(grads_matrix, nrow = 1, ncol = 1))
          }
          else if (length(dim(grads_matrix)) == 1) {
              list(matrix(grads_matrix, nrow = 1))
          }
          else {
              list(grads_matrix)
          }
          adagrad_result <- adagrad_update(params = optimizer_params[[layer]], grads = grads_input, lr = lr *
              layer_boost, epsilon = epsilon)
          optimizer_params[[layer]] <- adagrad_result$params
          update <- if (target == "weights") {
              adagrad_result$weights_update[[1]]
          }
          else {
              adagrad_result$biases_update[[1]]
          }
          target_matrix <- if (target == "weights")
              self$weights[[layer]]
          else self$biases[[layer]]
          target_dim <- dim(as.matrix(target_matrix))
          if (target == "biases") {
              if (length(update) == prod(target_dim)) {
                  updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
              }
              else if (length(update) == 1) {
                  updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
              }
              else {
                  repeated <- rep(update, length.out = prod(target_dim))
                  updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
              }
          }
          else {
              if (length(update) == prod(target_dim)) {
                  updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
              }
              else if (prod(target_dim) == 1) {
                  updated <- matrix(update, nrow = 1, ncol = 1)
              }
              else {
                  repeated <- rep(update, length.out = prod(target_dim))
                  updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
              }
          }
          if (target == "weights") {
              self$weights[[layer]] <- self$weights[[layer]] - updated
              Ppost <- self$weights[[layer]]
          }
          else {
              self$biases[[layer]] <- self$biases[[layer]] - updated
              Ppost <- self$biases[[layer]]
          }
          if (verbose) {
              message("before log")
          }
          optimizers_log_update(optimizer = "adagrad_x", epoch = epoch, layer = layer, target = target, grads_matrix = grads_matrix,
              P_before = Wpre, P_after = Ppost, update_applied = updated, verbose = verbose)
      }
      return(list(updated_weights_or_biases = self[[target]][[layer]], updated_optimizer_params = optimizer_params[[layer]]))
  }

  update_weights_block <- function (self, update_weights, optimizer, optimizer_params_weights, weight_gradients, lr, reg_type,
      beta1, beta2, epsilon, epoch, lookahead_step, verbose = FALSE)
  {
      if (!update_weights) {
          return(list(updated_optimizer_params = optimizer_params_weights))
      }
      if (self$ML_NN) {
          for (layer in 1:self$num_layers) {
              if (!is.null(self$weights[[layer]]) && !is.null(optimizer)) {
                  if (is.null(optimizer_params_weights[[layer]])) {
                    optimizer_params_weights[[layer]] <- initialize_optimizer_params(optimizer, list(dim(self$weights[[layer]])),
                      lookahead_step, layer)
                    if (verbose) {
                      message(">>> After initialize_optimizer_params() for layer ", layer)
                      for (line in utils::capture.output(str(optimizer_params_weights[[layer]]))) message(line)
                      message("Names: ", paste(names(optimizer_params_weights[[layer]]), collapse = ", "))
                    }
                  }
                  grads_matrix <- weight_gradients[[layer]]
                  grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5)
                  if (!is.null(reg_type)) {
                    if (reg_type == "L2") {
                      weight_update <- lr * grads_matrix + self$lambda * self$weights[[layer]]
                    }
                    else if (reg_type == "L1") {
                      l1_grad <- self$lambda * sign(self$weights[[layer]])
                      weight_update <- lr * grads_matrix + l1_grad
                    }
                    else if (reg_type == "L1_L2") {
                      l1_ratio <- 0.5
                      l1_grad <- l1_ratio * sign(self$weights[[layer]])
                      l2_grad <- (1 - l1_ratio) * self$weights[[layer]]
                      weight_update <- lr * grads_matrix + self$lambda * (l1_grad + l2_grad)
                    }
                    else if (reg_type == "Group_Lasso") {
                      if (is.null(self$groups) || is.null(self$groups[[layer]])) {
                        self$groups[[layer]] <- list(1:ncol(self$weights[[layer]]))
                      }
                      group_lasso_grad <- matrix(0, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]))
                      for (group in self$groups[[layer]]) {
                        group_weights <- self$weights[[layer]][, group, drop = FALSE]
                        norm_group <- sqrt(sum(group_weights^2, na.rm = TRUE)) + 1e-08
                        group_lasso_grad[, group] <- group_weights/norm_group
                      }
                      weight_update <- lr * grads_matrix + self$lambda * group_lasso_grad
                    }
                    else if (reg_type == "Max_Norm") {
                      max_norm <- 1
                      weight_norms <- sqrt(colSums(self$weights[[layer]]^2, na.rm = TRUE))
                      clipped_weights <- self$weights[[layer]]
                      for (j in seq_along(weight_norms)) {
                        if (weight_norms[j] > max_norm) {
                          clipped_weights[, j] <- (clipped_weights[, j]/weight_norms[j]) * max_norm
                        }
                      }
                      weight_update <- lr * grads_matrix + self$lambda * (self$weights[[layer]] - clipped_weights)
                    }
                    else {
                      if (verbose)
                        message("Warning: Unknown reg_type provided. No regularization applied.")
                      weight_update <- lr * grads_matrix
                    }
                  }
                  else {
                    weight_update <- lr * grads_matrix
                  }
                  if (all(dim(grads_matrix) == dim(self$weights[[layer]]))) {
                    self$weights[[layer]] <- self$weights[[layer]] - weight_update
                  }
                  else if (prod(dim(self$weights[[layer]])) == 1) {
                    self$weights[[layer]] <- self$weights[[layer]] - sum(weight_update)
                  }
                  else {
                    self$weights[[layer]] <- self$weights[[layer]] - apply(weight_update, 2, mean)
                  }
                  if (!is.null(optimizer_params_weights[[layer]]) && !is.null(optimizer)) {
                    {
                      grads_input <- if (is.list(grads_matrix)) {
                        grads_matrix
                      }
                      else if (is.null(dim(grads_matrix))) {
                        list(matrix(grads_matrix, nrow = 1, ncol = 1))
                      }
                      else if (length(dim(grads_matrix)) == 1) {
                        list(matrix(grads_matrix, nrow = 1))
                      }
                      else {
                        list(grads_matrix)
                      }
                      updated_optimizer <- apply_optimizer_update(optimizer = optimizer, optimizer_params = optimizer_params_weights,
                        grads_matrix = grads_matrix, lr = lr, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
                        epoch = epoch, self = self, layer = layer, target = "weights", verbose = verbose)
                      optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                  }
              }
          }
      }
      else {
          if (!is.null(self$weights) && !is.null(optimizer)) {
              weights_mat <- if (is.list(self$weights)) {
                  as.matrix(self$weights[[1]])
              }
              else {
                  as.matrix(self$weights)
              }
              if (is.null(optimizer_params_weights[[1]])) {
                  optimizer_params_weights[[1]] <- initialize_optimizer_params(optimizer, list(dim(weights_mat)),
                    lookahead_step, 1L)
                  if (verbose) {
                    message(">>> SL initialize_optimizer_params done for layer 1")
                    for (line in utils::capture.output(str(optimizer_params_weights[[1]]))) message(line)
                    message("Names: ", paste(names(optimizer_params_weights[[1]]), collapse = ", "))
                  }
              }
              grads_matrix <- weight_gradients[[1]]
              grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5)
              if (!is.null(reg_type)) {
                  if (reg_type == "L2") {
                    reg_term <- self$lambda * weights_mat
                    weight_update <- lr * grads_matrix + reg_term
                  }
                  else if (reg_type == "L1") {
                    reg_term <- self$lambda * sign(weights_mat)
                    weight_update <- lr * grads_matrix + reg_term
                  }
                  else if (reg_type == "L1_L2") {
                    l1_ratio <- 0.5
                    reg_term <- self$lambda * (l1_ratio * sign(weights_mat) + (1 - l1_ratio) * weights_mat)
                    weight_update <- lr * grads_matrix + reg_term
                  }
                  else if (reg_type == "Group_Lasso") {
                    norm_weights <- sqrt(sum(weights_mat^2, na.rm = TRUE)) + 1e-08
                    reg_term <- self$lambda * (weights_mat/norm_weights)
                    weight_update <- lr * grads_matrix + reg_term
                  }
                  else if (reg_type == "Max_Norm") {
                    max_norm <- 1
                    norm_weights <- sqrt(sum(weights_mat^2, na.rm = TRUE))
                    clipped_weights <- if (norm_weights > max_norm) {
                      (weights_mat/norm_weights) * max_norm
                    }
                    else {
                      weights_mat
                    }
                    reg_term <- self$lambda * (weights_mat - clipped_weights)
                    weight_update <- lr * grads_matrix + reg_term
                  }
                  else {
                    if (verbose)
                      message("Warning: Unknown reg_type in SL. No regularization applied.")
                    weight_update <- lr * grads_matrix
                  }
              }
              else {
                  weight_update <- lr * grads_matrix
              }
              if (verbose) {
                  message(">> SL grads_matrix dim: ", paste(dim(grads_matrix), collapse = " x "))
                  message("SL grads_matrix summary:")
                  for (nm in names(summary(as.vector(grads_matrix)))) {
                    message(nm, ": ", summary(as.vector(grads_matrix))[nm])
                  }
              }
              if (!is.null(optimizer_params_weights[[1]])) {
                  {
                    updated_optimizer <- apply_optimizer_update(optimizer = optimizer, optimizer_params = optimizer_params_weights,
                      grads_matrix = grads_matrix, lr = lr, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
                      epoch = epoch, self = self, layer = 1L, target = "weights", verbose = verbose)
                    weights_mat <- updated_optimizer$updated_weights_or_biases
                    if (is.list(self$weights)) {
                      self$weights[[1]] <- weights_mat
                    }
                    else {
                      self$weights <- weights_mat
                    }
                    optimizer_params_weights[[1]] <- updated_optimizer$updated_optimizer_params
                    if (verbose) {
                      message(">> SL updated weights summary: min = ", min(weights_mat), ", mean = ", mean(weights_mat),
                        ", max = ", max(weights_mat))
                    }
                  }
              }
              else {
                  if (all(dim(grads_matrix) == dim(weights_mat))) {
                    weights_mat <- weights_mat - weight_update
                  }
                  else if (prod(dim(weights_mat)) == 1) {
                    weights_mat <- weights_mat - sum(weight_update)
                  }
                  else {
                    weights_mat <- weights_mat - apply(weight_update, 2, mean)
                  }
                  if (is.list(self$weights)) {
                    self$weights[[1]] <- weights_mat
                  }
                  else {
                    self$weights <- weights_mat
                  }
              }
          }
      }
      return(list(updated_optimizer_params = optimizer_params_weights))
  }

  update_biases_block <- function (self, update_biases, optimizer, optimizer_params_biases, bias_gradients, errors, lr, reg_type,
      beta1, beta2, epsilon, epoch, lookahead_step, verbose = FALSE)
  {
      if (!update_biases) {
          return(list(updated_optimizer_params = optimizer_params_biases))
      }
      if (self$ML_NN) {
          for (layer in 1:self$num_layers) {
              if (!is.null(self$biases[[layer]]) && !is.null(optimizer)) {
                  if (is.null(optimizer_params_biases[[layer]])) {
                    optimizer_params_biases[[layer]] <- initialize_optimizer_params(optimizer, list(dim(as.matrix(self$biases[[layer]]))),
                      lookahead_step, layer)
                  }
                  grads_matrix <- bias_gradients[[layer]]
                  grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5)
                  bias_shape <- dim(as.matrix(self$biases[[layer]]))
                  grad_shape <- dim(grads_matrix)
                  if (!all(bias_shape == grad_shape)) {
                    if (prod(grad_shape) == 1) {
                      grads_matrix <- matrix(grads_matrix, nrow = bias_shape[1], ncol = bias_shape[2])
                    }
                    else if (prod(bias_shape) == 1) {
                      self$biases[[layer]] <- matrix(self$biases[[layer]], nrow = grad_shape[1], ncol = grad_shape[2])
                    }
                    else {
                      grads_matrix <- matrix(rep(grads_matrix, length.out = prod(bias_shape)), nrow = bias_shape[1],
                        ncol = bias_shape[2])
                    }
                  }
                  if (!is.null(reg_type)) {
                    if (reg_type == "L2") {
                      reg_term <- self$lambda * self$biases[[layer]]
                      bias_update <- lr * grads_matrix + reg_term
                    }
                    else if (reg_type == "L1") {
                      reg_term <- self$lambda * sign(self$biases[[layer]])
                      bias_update <- lr * grads_matrix + reg_term
                    }
                    else if (reg_type == "L1_L2") {
                      l1_ratio <- 0.5
                      l1_grad <- l1_ratio * sign(self$biases[[layer]])
                      l2_grad <- (1 - l1_ratio) * self$biases[[layer]]
                      reg_term <- self$lambda * (l1_grad + l2_grad)
                      bias_update <- lr * grads_matrix + reg_term
                    }
                    else if (reg_type == "Group_Lasso") {
                      norm_bias <- sqrt(sum(self$biases[[layer]]^2, na.rm = TRUE)) + 1e-08
                      reg_term <- self$lambda * (self$biases[[layer]]/norm_bias)
                      bias_update <- lr * grads_matrix + reg_term
                    }
                    else if (reg_type == "Max_Norm") {
                      max_norm <- 1
                      norm_bias <- sqrt(sum(self$biases[[layer]]^2, na.rm = TRUE))
                      clipped_bias <- if (norm_bias > max_norm) {
                        (self$biases[[layer]]/norm_bias) * max_norm
                      }
                      else {
                        self$biases[[layer]]
                      }
                      reg_term <- self$lambda * (self$biases[[layer]] - clipped_bias)
                      bias_update <- lr * grads_matrix + reg_term
                    }
                    else {
                      if (verbose)
                        message("Warning: Unknown reg_type in ML bias update. No regularization applied.")
                      bias_update <- lr * grads_matrix
                    }
                  }
                  else {
                    bias_update <- lr * grads_matrix
                  }
                  if (all(dim(grads_matrix) == dim(self$biases[[layer]]))) {
                    self$biases[[layer]] <- self$biases[[layer]] - bias_update
                  }
                  else if (prod(dim(self$biases[[layer]])) == 1) {
                    self$biases[[layer]] <- self$biases[[layer]] - sum(bias_update)
                  }
                  else {
                    self$biases[[layer]] <- self$biases[[layer]] - apply(bias_update, 2, mean)
                  }
                  if (!is.null(optimizer_params_biases[[layer]]) && !is.null(optimizer)) {
                    {
                      grads_matrix2 <- if (is.null(dim(errors[[layer]]))) {
                        matrix(errors[[layer]], nrow = 1)
                      }
                      else {
                        errors[[layer]]
                      }
                      grads_matrix2 <- colSums(grads_matrix2)
                      updated_optimizer <- apply_optimizer_update(optimizer = optimizer, optimizer_params = optimizer_params_biases,
                        grads_matrix = grads_matrix2, lr = lr, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
                        epoch = epoch, self = self, layer = layer, target = "biases", verbose = verbose)
                      optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
                    }
                  }
              }
          }
      }
      else {
          if (verbose) {
              message("Single Layer Bias Update")
          }
          if (is.null(self$biases))
              stop("Biases are NULL in single-layer mode.")
          if (!is.matrix(self$biases))
              self$biases <- matrix(as.numeric(self$biases), nrow = 1)
          if (is.null(optimizer_params_biases))
              optimizer_params_biases <- list()
          if (!is.null(optimizer) && is.null(optimizer_params_biases[[1]])) {
              optimizer_params_biases[[1]] <- initialize_optimizer_params(optimizer, list(dim(self$biases)),
                  lookahead_step, 1L)
              if (verbose) {
                  message(">>> SL initialize_optimizer_params (bias) done for layer 1")
                  for (line in utils::capture.output(str(optimizer_params_biases[[1]]))) message(line)
                  message("Names: ", paste(names(optimizer_params_biases[[1]]), collapse = ", "))
              }
          }
          bias_grad <- colMeans(errors[[1]], na.rm = TRUE)
          bias_grad <- matrix(rep(bias_grad, length.out = ncol(self$biases)), nrow = 1)
          bias_grad <- clip_gradient_norm(bias_grad, max_norm = 5)
          if (verbose) {
              message("SL bias_grad dim: ", paste(dim(bias_grad), collapse = " x "))
              message("SL bias_grad summary:")
              for (nm in names(summary(as.vector(bias_grad)))) {
                  message(nm, ": ", summary(as.vector(bias_grad))[nm])
              }
          }
          if (!is.null(optimizer) && !is.null(optimizer_params_biases[[1]])) {
              updated_optimizer <- apply_optimizer_update(optimizer = optimizer, optimizer_params = optimizer_params_biases,
                  grads_matrix = bias_grad, lr = lr, beta1 = beta1, beta2 = beta2, epsilon = epsilon, epoch = epoch,
                  self = self, layer = 1L, target = "biases", verbose = verbose)
              self$biases <- updated_optimizer$updated_weights_or_biases
              optimizer_params_biases[[1]] <- updated_optimizer$updated_optimizer_params
              if (verbose) {
                  message(">> SL updated biases summary: min = ", min(self$biases), ", mean = ", mean(self$biases),
                    ", max = ", max(self$biases))
              }
          }
          else {
              bias_update <- lr * bias_grad
              if (!is.null(reg_type)) {
                  if (reg_type == "L2") {
                    reg_term <- self$lambda * self$biases
                    bias_update <- bias_update + reg_term
                  }
                  else if (reg_type == "L1") {
                    reg_term <- self$lambda * sign(self$biases)
                    bias_update <- bias_update + reg_term
                  }
                  else if (reg_type == "L1_L2") {
                    l1_ratio <- 0.5
                    l1_grad <- l1_ratio * sign(self$biases)
                    l2_grad <- (1 - l1_ratio) * self$biases
                    bias_update <- bias_update + self$lambda * (l1_grad + l2_grad)
                  }
                  else if (reg_type == "Group_Lasso") {
                    norm_bias <- sqrt(sum(self$biases^2, na.rm = TRUE)) + 1e-08
                    reg_term <- self$lambda * (self$biases/norm_bias)
                    bias_update <- bias_update + reg_term
                  }
                  else if (reg_type == "Max_Norm") {
                    max_norm <- 1
                    norm_bias <- sqrt(sum(self$biases^2, na.rm = TRUE))
                    clipped_bias <- if (norm_bias > max_norm) {
                      (self$biases/norm_bias) * max_norm
                    }
                    else {
                      self$biases
                    }
                    reg_term <- self$lambda * (self$biases - clipped_bias)
                    bias_update <- bias_update + reg_term
                  }
                  else {
                    if (verbose)
                      message("Warning: Unknown reg_type in SL bias update. No regularization applied.")
                  }
              }
              self$biases <- self$biases - bias_update
          }
      }
      return(list(updated_optimizer_params = optimizer_params_biases))
  }

  conform_gradients <- function(self, weight_gradients, bias_gradients) {
          # Conform gradient shapes to weight shapes
          for (lyr in seq_along(weight_gradients)) {
            if (isTRUE(self$ML_NN)) {
              W <- as.matrix(self$weights[[lyr]])
            } else {
              W <- if (is.list(self$weights)) as.matrix(self$weights[[1]]) else as.matrix(self$weights)
            }
            if (is.null(weight_gradients[[lyr]])) {
              weight_gradients[[lyr]] <- matrix(0, nrow = nrow(W), ncol = ncol(W))
            } else {
              G <- as.matrix(weight_gradients[[lyr]])
              if (!all(dim(G) == dim(W))) {
                G2 <- matrix(0, nrow = nrow(W), ncol = ncol(W))
                r <- min(nrow(G), nrow(W)); c <- min(ncol(G), ncol(W))
                G2[seq_len(r), seq_len(c)] <- G[seq_len(r), seq_len(c)]
                G <- G2
              }
              weight_gradients[[lyr]] <- G
            }

            if (is.null(bias_gradients[[lyr]])) {
              bias_gradients[[lyr]] <- matrix(0, nrow = 1, ncol = ncol(W))
            } else {
              b <- as.numeric(bias_gradients[[lyr]])
              if (length(b) != ncol(W)) {
                b2 <- numeric(ncol(W)); len <- min(length(b), ncol(W)); b2[seq_len(len)] <- b[seq_len(len)]
                b <- b2
              }
              bias_gradients[[lyr]] <- matrix(b, nrow = 1L)
            }
          }

    list(weights = weight_gradients, biases = bias_gradients)
  }
  list(weights = update_weights_block, biases = update_biases_block,
       gradients = conform_gradients)
})
