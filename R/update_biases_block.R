#' Update biases block (ML + SL) for DDESONN
#'
#' Direct extraction of your inlined `if (update_biases) { ... }` block.
#' Preserves all logic, shapes, and debug prints.
#'
#' @param self R6 model (modified in-place)
#' @param update_biases logical
#' @param optimizer character
#' @param optimizer_params_biases list (returned/updated)
#' @param bias_gradients list of per-layer bias grads (from learn())
#' @param errors list (used by SL path for bias grad)
#' @param lr numeric
#' @param reg_type NULL or one of c("L2","L1","L1_L2","Group_Lasso","Max_Norm")
#' @param beta1,beta2,epsilon numerics
#' @param epoch integer
#' @param lookahead_step integer
#' @param verbose logical
#'
#' @return list(updated_optimizer_params = optimizer_params_biases)
#' @keywords internal
#' @noRd
update_biases_block <- function(
    self,
    update_biases,
    optimizer,
    optimizer_params_biases,
    bias_gradients,
    errors,
    lr,
    reg_type,
    beta1, beta2, epsilon,
    epoch,
    lookahead_step,
    verbose = FALSE
) {
  if (!update_biases) {
    return(list(updated_optimizer_params = optimizer_params_biases))
  }

  if (self$ML_NN) {
    # =========================
    # MULTI-LAYER NN BRANCH
    # =========================
    for (layer in 1:self$num_layers) {
      if (!is.null(self$biases[[layer]]) && !is.null(optimizer)) {
        
        # Initialize optimizer parameters only if not already done
        if (is.null(optimizer_params_biases[[layer]])) {
          optimizer_params_biases[[layer]] <- initialize_optimizer_params(
            optimizer,
            list(dim(as.matrix(self$biases[[layer]]))),
            lookahead_step,
            layer
          )
        }
        
        # Get bias gradients from learn()
        grads_matrix <- bias_gradients[[layer]]
        
        # Clip bias gradient
        grads_matrix <- clip_gradient_norm(grads_matrix, max_norm = 5)
        
        # --- Align dimensions if needed ---
        bias_shape <- dim(as.matrix(self$biases[[layer]]))
        grad_shape <- dim(grads_matrix)
        
        if (!all(bias_shape == grad_shape)) {
          if (prod(grad_shape) == 1) {
            grads_matrix <- matrix(grads_matrix, nrow = bias_shape[1], ncol = bias_shape[2])
          } else if (prod(bias_shape) == 1) {
            self$biases[[layer]] <- matrix(self$biases[[layer]], nrow = grad_shape[1], ncol = grad_shape[2])
          } else {
            grads_matrix <- matrix(rep(grads_matrix, length.out = prod(bias_shape)),
                                   nrow = bias_shape[1], ncol = bias_shape[2])
          }
        }
        
        # --------- Regularization ---------
        if (!is.null(reg_type)) {
          if (reg_type == "L2") {
            reg_term <- self$lambda * self$biases[[layer]]
            bias_update <- lr * grads_matrix + reg_term
            
          } else if (reg_type == "L1") {
            reg_term <- self$lambda * sign(self$biases[[layer]])
            bias_update <- lr * grads_matrix + reg_term
            
          } else if (reg_type == "L1_L2") {
            l1_ratio <- 0.5
            l1_grad  <- l1_ratio * sign(self$biases[[layer]])
            l2_grad  <- (1 - l1_ratio) * self$biases[[layer]]
            reg_term <- self$lambda * (l1_grad + l2_grad)
            bias_update <- lr * grads_matrix + reg_term
            
          } else if (reg_type == "Group_Lasso") {
            norm_bias <- sqrt(sum(self$biases[[layer]]^2, na.rm = TRUE)) + 1e-8
            reg_term  <- self$lambda * (self$biases[[layer]] / norm_bias)
            bias_update <- lr * grads_matrix + reg_term
            
          } else if (reg_type == "Max_Norm") {
            max_norm  <- 1.0
            norm_bias <- sqrt(sum(self$biases[[layer]]^2, na.rm = TRUE))
            clipped_bias <- if (norm_bias > max_norm) {
              (self$biases[[layer]] / norm_bias) * max_norm
            } else {
              self$biases[[layer]]
            }
            reg_term <- self$lambda * (self$biases[[layer]] - clipped_bias)
            bias_update <- lr * grads_matrix + reg_term
            
          } else {
            if (verbose) message("Warning: Unknown reg_type in ML bias update. No regularization applied.")
            bias_update <- lr * grads_matrix
          }
        } else {
          # Default: No regularization
          bias_update <- lr * grads_matrix
        }
        
        # Apply bias update safely
        if (all(dim(grads_matrix) == dim(self$biases[[layer]]))) {
          self$biases[[layer]] <- self$biases[[layer]] - bias_update
        } else if (prod(dim(self$biases[[layer]])) == 1) {
          self$biases[[layer]] <- self$biases[[layer]] - sum(bias_update)
        } else {
          self$biases[[layer]] <- self$biases[[layer]] - apply(bias_update, 2, mean)
        }
        
        # ---- Optimizer dispatch (ML) ----
        if (!is.null(optimizer_params_biases[[layer]]) && !is.null(optimizer)) {
          
          if (optimizer == "adam") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "rmsprop") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = bias_gradients[[layer]],
              lr = lr,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "sgd") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,  # kept for signature
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "sgd_momentum") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,  # momentum
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "nag") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "ftrl") {
            updated_optimizer <- apply_optimizer_update(
              optimizer         = optimizer,
              optimizer_params  = optimizer_params_biases,
              grads_matrix      = grads_matrix,
              lr                = lr,
              alpha             = 0.1,
              beta1             = 1.0,
              lambda1           = 0.01,
              lambda2           = 0.01,
              epsilon           = epsilon,
              epoch             = epoch,
              self              = self,
              layer             = layer,
              target            = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "lamb") {
            grads_input <- if (is.list(grads_matrix)) {
              grads_matrix
            } else if (is.null(dim(grads_matrix))) {
              list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
            } else if (length(dim(grads_matrix)) == 1) {
              list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
            } else {
              list(grads_matrix)
            }
            updated_optimizer <- apply_optimizer_update(
              optimizer         = optimizer,
              optimizer_params  = optimizer_params_biases,
              grads_matrix      = grads_matrix,
              lr                = lr,
              beta1             = beta1,
              beta2             = beta2,
              epsilon           = epsilon,
              epoch             = epoch,
              self              = self,
              layer             = layer,
              target            = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "lookahead") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              layer = layer,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "adagrad") {
            grads_matrix2 <- if (is.null(dim(errors[[layer]]))) {
              matrix(errors[[layer]], nrow = 1)
            } else {
              errors[[layer]]
            }
            grads_matrix2 <- colSums(grads_matrix2)
            
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = grads_matrix2,
              lr = lr,
              beta1 = beta1,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
            
          } else if (optimizer == "adadelta") {
            updated_optimizer <- apply_optimizer_update(
              optimizer = optimizer,
              optimizer_params = optimizer_params_biases,
              grads_matrix = grads_matrix,
              lr = lr,
              beta1 = beta1,
              beta2 = beta2,
              epsilon = epsilon,
              epoch = epoch,
              self = self,
              layer = layer,
              target = "biases",
              verbose = verbose
            )
            optimizer_params_biases[[layer]] <- updated_optimizer$updated_optimizer_params
          }
        }
      }
    }
    
  } else {
    # =========================
    # SINGLE-LAYER BIAS UPDATE
    # =========================
    if (verbose) {
      message("Single Layer Bias Update")
    }
    
    # 1) Ensure biases matrix [1 x n_units]
    if (is.null(self$biases)) stop("Biases are NULL in single-layer mode.")
    if (!is.matrix(self$biases)) self$biases <- matrix(as.numeric(self$biases), nrow = 1)
    
    # 2) Ensure optimizer params list + init slot 1
    if (is.null(optimizer_params_biases)) optimizer_params_biases <- list()
    if (!is.null(optimizer) && is.null(optimizer_params_biases[[1]])) {
      optimizer_params_biases[[1]] <- initialize_optimizer_params(
        optimizer,
        list(dim(self$biases)),
        lookahead_step,
        1L
      )
      if (verbose) {
        message(">>> SL initialize_optimizer_params (bias) done for layer 1")
        for (line in utils::capture.output(str(optimizer_params_biases[[1]]))) message(line)
        message("Names: ", paste(names(optimizer_params_biases[[1]]), collapse = ", "))
      }
    }
    
    # 3) Gradient from errors (per-unit mean)
    bias_grad <- colMeans(errors[[1]], na.rm = TRUE)
    
    # shape to [1 x n_units]
    bias_grad <- matrix(rep(bias_grad, length.out = ncol(self$biases)), nrow = 1)
    
    # optional clip
    bias_grad <- clip_gradient_norm(bias_grad, max_norm = 5.0)
    
    # Debug
    if (verbose) {
      message("SL bias_grad dim: ", paste(dim(bias_grad), collapse = " x "))
      message("SL bias_grad summary:")
      for (nm in names(summary(as.vector(bias_grad)))) {
        message(nm, ": ", summary(as.vector(bias_grad))[nm])
      }
    }
    
    # 4) Optimizer dispatch (preferred path)
    if (!is.null(optimizer) && !is.null(optimizer_params_biases[[1]])) {
      updated_optimizer <- apply_optimizer_update(
        optimizer        = optimizer,
        optimizer_params = optimizer_params_biases,
        grads_matrix     = bias_grad,
        lr               = lr,
        beta1            = beta1,
        beta2            = beta2,
        epsilon          = epsilon,
        epoch            = epoch,
        self             = self,
        layer            = 1L,
        target           = "biases",
        verbose = verbose
      )
      
      self$biases <- updated_optimizer$updated_weights_or_biases
      optimizer_params_biases[[1]] <- updated_optimizer$updated_optimizer_params
      
      if (verbose) {
        message(">> SL updated biases summary: min = ", min(self$biases),
                ", mean = ", mean(self$biases),
                ", max = ", max(self$biases))
      }
      
    } else {
      # 5) Manual / fallback with regularization
      bias_update <- lr * bias_grad
      
      if (!is.null(reg_type)) {
        if (reg_type == "L2") {
          reg_term <- self$lambda * self$biases
          bias_update <- bias_update + reg_term
        } else if (reg_type == "L1") {
          reg_term <- self$lambda * sign(self$biases)
          bias_update <- bias_update + reg_term
        } else if (reg_type == "L1_L2") {
          l1_ratio <- 0.5
          l1_grad  <- l1_ratio * sign(self$biases)
          l2_grad  <- (1 - l1_ratio) * self$biases
          bias_update <- bias_update + self$lambda * (l1_grad + l2_grad)
        } else if (reg_type == "Group_Lasso") {
          norm_bias <- sqrt(sum(self$biases^2, na.rm = TRUE)) + 1e-8
          reg_term  <- self$lambda * (self$biases / norm_bias)
          bias_update <- bias_update + reg_term
        } else if (reg_type == "Max_Norm") {
          max_norm  <- 1.0
          norm_bias <- sqrt(sum(self$biases^2, na.rm = TRUE))
          clipped_bias <- if (norm_bias > max_norm) {
            (self$biases / norm_bias) * max_norm
          } else {
            self$biases
          }
          reg_term <- self$lambda * (self$biases - clipped_bias)
          bias_update <- bias_update + reg_term
        } else {
          if (verbose) message("Warning: Unknown reg_type in SL bias update. No regularization applied.")
        }
      }
      
      # Final manual apply
      self$biases <- self$biases - bias_update
    }
  }
  
  return(list(updated_optimizer_params = optimizer_params_biases))
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