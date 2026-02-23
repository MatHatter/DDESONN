#' Update weights block (ML + SL) for DDESONN
#' 
#' This function is a direct extraction of your inlined `if (update_weights) { ... }`
#' block. It preserves all original logic and debug prints.
#' 
#' @param self R6 model (DDESONN/SONN) – modified in-place
#' @param update_weights logical
#' @param optimizer character (adam, rmsprop, sgd, sgd_momentum, nag, ftrl, lamb, lookahead, adagrad, adadelta)
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
    self,
    update_weights,
    optimizer,
    optimizer_params_weights,
    weight_gradients,
    lr,
    reg_type,
    beta1, beta2, epsilon,
    epoch,
    lookahead_step,
    verbose = FALSE
) {
  if (!update_weights) {
    return(list(updated_optimizer_params = optimizer_params_weights))
  }
  
  # =========================
  # MULTI-LAYER NN BRANCH
  # =========================
  if (self$ML_NN) {
    for (layer in 1:self$num_layers) {
      if (!is.null(self$weights[[layer]]) && !is.null(optimizer)) {
        
        if (is.null(optimizer_params_weights[[layer]])) {
          optimizer_params_weights[[layer]] <- initialize_optimizer_params(
            optimizer,
            list(dim(self$weights[[layer]])),
            lookahead_step,
            layer
          )
          
          if (verbose) {
            message(">>> After initialize_optimizer_params() for layer ", layer)
            for (line in utils::capture.output(str(optimizer_params_weights[[layer]]))) message(line)
            message("Names: ", paste(names(optimizer_params_weights[[layer]]), collapse = ", "))
          }
        }
        
        # Get weight gradients from learn()
        grads_matrix <- weight_gradients[[layer]]
        
        # Clip weight gradient
        grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5.0)
        
        # --------- Apply Regularization to Weight Gradient ---------
        if (!is.null(reg_type)) {
          if (reg_type == "L2") {
            weight_update <- lr * grads_matrix + self$lambda * self$weights[[layer]]
          } else if (reg_type == "L1") {
            l1_grad <- self$lambda * sign(self$weights[[layer]])
            weight_update <- lr * grads_matrix + l1_grad
          } else if (reg_type == "L1_L2") {
            l1_ratio <- 0.5
            l1_grad <- l1_ratio * sign(self$weights[[layer]])
            l2_grad <- (1 - l1_ratio) * self$weights[[layer]]
            weight_update <- lr * grads_matrix + self$lambda * (l1_grad + l2_grad)
          } else if (reg_type == "Group_Lasso") {
            if (is.null(self$groups) || is.null(self$groups[[layer]])) {
              self$groups[[layer]] <- list(1:ncol(self$weights[[layer]]))  # Default: entire layer as one group
            }
            group_lasso_grad <- matrix(0, nrow = nrow(self$weights[[layer]]), ncol = ncol(self$weights[[layer]]))
            for (group in self$groups[[layer]]) {
              group_weights <- self$weights[[layer]][, group, drop = FALSE]
              norm_group <- sqrt(sum(group_weights^2, na.rm = TRUE)) + 1e-8
              group_lasso_grad[, group] <- group_weights / norm_group
            }
            weight_update <- lr * grads_matrix + self$lambda * group_lasso_grad
          } else if (reg_type == "Max_Norm") {
            max_norm <- 1.0
            weight_norms <- sqrt(colSums(self$weights[[layer]]^2, na.rm = TRUE))
            clipped_weights <- self$weights[[layer]]
            for (j in seq_along(weight_norms)) {
              if (weight_norms[j] > max_norm) {
                clipped_weights[, j] <- (clipped_weights[, j] / weight_norms[j]) * max_norm
              }
            }
            weight_update <- lr * grads_matrix + self$lambda * (self$weights[[layer]] - clipped_weights)
          } else {
            if (verbose) message("Warning: Unknown reg_type provided. No regularization applied.")
            weight_update <- lr * grads_matrix
          }
        } else {
          weight_update <- lr * grads_matrix
        }
        
        # Apply weight update safely
        if (all(dim(grads_matrix) == dim(self$weights[[layer]]))) {
          self$weights[[layer]] <- self$weights[[layer]] - weight_update
        } else if (prod(dim(self$weights[[layer]])) == 1) {
          self$weights[[layer]] <- self$weights[[layer]] - sum(weight_update)
        } else {
          self$weights[[layer]] <- self$weights[[layer]] - apply(weight_update, 2, mean)
        }
        
        # --- Optimizer dispatch (ML) ---
        if (!is.null(optimizer_params_weights[[layer]]) && !is.null(optimizer)) {
          if (optimizer == "adam") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "rmsprop") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "sgd") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = NA, beta2 = NA, epsilon = NA,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "sgd_momentum") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1, beta2 = beta2, epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "nag") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1, beta2 = beta2, epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "ftrl") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1, beta2 = beta2,
              alpha = 0.1, lambda1 = 0.01, lambda2 = 0.01,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "lamb") {
            grads_input <- if (is.list(grads_matrix)) { grads_matrix } 
            else if (is.null(dim(grads_matrix))) { list(matrix(grads_matrix, nrow = 1, ncol = 1)) } 
            else if (length(dim(grads_matrix)) == 1) { list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1)) } 
            else { list(grads_matrix) }
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1, beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "lookahead") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = weight_gradients[[layer]],
              lr = lr, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
              epoch = epoch, self = self, layer = layer,
              target = "weights", verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "adagrad") {
            grads_input <- if (is.list(grads_matrix)) { grads_matrix } 
            else if (is.null(dim(grads_matrix))) { list(matrix(grads_matrix, nrow = 1, ncol = 1)) } 
            else if (length(dim(grads_matrix)) == 1) { list(matrix(grads_matrix, nrow = 1)) } 
            else { list(grads_matrix) }
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1, beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "adadelta") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_weights,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1, beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "weights",
              verbose = verbose
            )
            optimizer_params_weights[[layer]] <- updated_optimizer$updated_optimizer_params
          }
        }
      }
    }
    
  } else {
    # =========================
    # SINGLE-LAYER NN BRANCH
    # =========================
    if (!is.null(self$weights) && !is.null(optimizer)) {
      weights_mat <- if (is.list(self$weights)) {
        as.matrix(self$weights[[1]])
      } else {
        as.matrix(self$weights)
      }
      if (is.null(optimizer_params_weights[[1]])) {
        optimizer_params_weights[[1]] <- initialize_optimizer_params(
          optimizer,
          list(dim(weights_mat)),
          lookahead_step,
          1L
        )
        if (verbose) {
          message(">>> SL initialize_optimizer_params done for layer 1")
          for (line in utils::capture.output(str(optimizer_params_weights[[1]]))) message(line)
          message("Names: ", paste(names(optimizer_params_weights[[1]]), collapse = ", "))
        }
      }
      
      # Get weight gradients from learn()
      grads_matrix <- weight_gradients[[1]]
      
      # Clip gradient
      grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5.0)
      
      # --------- Regularization ---------
      if (!is.null(reg_type)) {
        if (reg_type == "L2") {
          reg_term <- self$lambda * weights_mat
          weight_update <- lr * grads_matrix + reg_term
        } else if (reg_type == "L1") {
          reg_term <- self$lambda * sign(weights_mat)
          weight_update <- lr * grads_matrix + reg_term
        } else if (reg_type == "L1_L2") {
          l1_ratio <- 0.5
          reg_term <- self$lambda * (l1_ratio * sign(weights_mat) + (1 - l1_ratio) * weights_mat)
          weight_update <- lr * grads_matrix + reg_term
        } else if (reg_type == "Group_Lasso") {
          norm_weights <- sqrt(sum(weights_mat^2, na.rm = TRUE)) + 1e-8
          reg_term <- self$lambda * (weights_mat / norm_weights)
          weight_update <- lr * grads_matrix + reg_term
        } else if (reg_type == "Max_Norm") {
          max_norm <- 1.0
          norm_weights <- sqrt(sum(weights_mat^2, na.rm = TRUE))
          clipped_weights <- if (norm_weights > max_norm) {
            (weights_mat / norm_weights) * max_norm
          } else {
            weights_mat
          }
          reg_term <- self$lambda * (weights_mat - clipped_weights)
          weight_update <- lr * grads_matrix + reg_term
        } else {
          if (verbose) message("Warning: Unknown reg_type in SL. No regularization applied.")
          weight_update <- lr * grads_matrix
        }
      } else {
        weight_update <- lr * grads_matrix
      }
      
      # ------------------- DEBUG -------------------
      if (verbose) {
        message(">> SL grads_matrix dim: ", paste(dim(grads_matrix), collapse = " x "))
        message("SL grads_matrix summary:")
        for (nm in names(summary(as.vector(grads_matrix)))) {
          message(nm, ": ", summary(as.vector(grads_matrix))[nm])
        }
      }
      
      # ------------------- OPTIMIZER DISPATCH (SL) -------------------
      if (!is.null(optimizer_params_weights[[1]])) {
        if (optimizer %in% c("adam","rmsprop","sgd","sgd_momentum","nag","ftrl","lamb","lookahead","adagrad","adadelta")) {
          updated_optimizer <- apply_optimizer_update(
            optimizer        = optimizer,
            optimizer_params = optimizer_params_weights,
            grads_matrix     = grads_matrix,
            lr               = lr,
            beta1            = beta1,
            beta2            = beta2,
            epsilon          = epsilon,
            epoch            = epoch,
            self             = self,
            layer            = 1L,
            target           = "weights",
            verbose = verbose
          )
          # Most branches in your ML path assign absolute weights; mirror that:
          weights_mat <- updated_optimizer$updated_weights_or_biases
          if (is.list(self$weights)) {
            self$weights[[1]] <- weights_mat
          } else {
            self$weights <- weights_mat
          }
          optimizer_params_weights[[1]] <- updated_optimizer$updated_optimizer_params
          
          if (verbose) {
            message(">> SL updated weights summary: min = ", min(weights_mat),
                    ", mean = ", mean(weights_mat),
                    ", max = ", max(weights_mat))
          }
          
        } else {
          # Unknown optimizer → fall back to manual update
          if (all(dim(grads_matrix) == dim(weights_mat))) {
            weights_mat <- weights_mat - weight_update
          } else if (prod(dim(weights_mat)) == 1) {
            weights_mat <- weights_mat - sum(weight_update)
          } else {
            weights_mat <- weights_mat - apply(weight_update, 2, mean)
          }
          if (is.list(self$weights)) {
            self$weights[[1]] <- weights_mat
          } else {
            self$weights <- weights_mat
          }
        }
      } else {
        # Params not initialized (shouldn’t happen) → safe fallback
        if (all(dim(grads_matrix) == dim(weights_mat))) {
          weights_mat <- weights_mat - weight_update
        } else if (prod(dim(weights_mat)) == 1) {
          weights_mat <- weights_mat - sum(weight_update)
        } else {
          weights_mat <- weights_mat - apply(weight_update, 2, mean)
        }
        if (is.list(self$weights)) {
          self$weights[[1]] <- weights_mat
        } else {
          self$weights <- weights_mat
        }
      }
    }
  }
  
  return(list(updated_optimizer_params = optimizer_params_weights))
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