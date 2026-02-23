# ===============================================================
# DeepDynamic -- DDESONN
# Deep Dynamic Experimental Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2026 Mathew William Armitage Fok
#
# Released under the MIT License. See the file LICENSE.
# ===============================================================

apply_optimizer_update <- function(optimizer, optimizer_params, grads_matrix, lr, beta1, beta2, epsilon, epoch, self, layer, target,
                                   alpha = NULL, lambda1 = NULL, lambda2 = NULL, verbose = FALSE) {
  
  # -------- SL shim (normalize shapes so ML-style code works) --------
  is_sl <- !self$ML_NN
  
  # In SL, weights/biases are matrices, but branches use [[layer]].
  # Temporarily "box" them into 1-element lists and guarantee shapes.
  if (is_sl) {
    orig_weights <- self$weights
    orig_biases  <- self$biases
    on.exit({
      # Unbox back to matrices when we return
      if (!is.null(orig_weights)) self$weights <- orig_weights
      if (!is.null(orig_biases))  self$biases  <- orig_biases
    }, add = TRUE)
    
    if (target == "weights" && !is.list(self$weights)) self$weights <- list(as.matrix(self$weights))
    if (target == "biases"  && !is.list(self$biases))  self$biases  <- list(as.matrix(self$biases))
  }
  
  # Current param matrix and its dim (works in both ML and SL after boxing)
  param_now <- if (identical(target, "weights")) as.matrix(self$weights[[layer]]) else as.matrix(self$biases[[layer]])
  if (is.null(dim(param_now))) param_now <- matrix(as.numeric(param_now), 1, 1)
  target_dim <- dim(param_now)
  
  # Make grads a matrix with same shape as the target param
  gm <- grads_matrix
  if (is.list(gm)) gm <- gm[[1]]
  gm <- as.matrix(gm)
  if (is.null(dim(gm))) gm <- matrix(as.numeric(gm), nrow = target_dim[1], ncol = target_dim[2])
  if (nrow(gm) != target_dim[1] || ncol(gm) != target_dim[2]) {
    if (length(gm) == prod(target_dim)) {
      gm <- matrix(as.numeric(gm), nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(gm) == 1L) {
      gm <- matrix(rep(as.numeric(gm), prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      gm <- matrix(rep(as.numeric(gm)[1], prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    }
  }
  grads_matrix <- gm
  # -------------------------------------------------------------------
  
  
  if (optimizer == "adam") {
    Wpre <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Boost learning rate for output layer
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # Update optimizer params using Adam
    optimizer_params[[layer]] <- adam_update(
      optimizer_params[[layer]],
      grads   = list(grads_matrix),
      lr      = lr * layer_boost,
      beta1   = beta1,
      beta2   = beta2,
      epsilon = epsilon,
      t       = epoch
    )
    
    # Select correct update matrix based on target
    if (target == "weights") {
      update <- optimizer_params[[layer]]$weights_update
    } else if (target == "biases") {
      update <- optimizer_params[[layer]]$biases_update
    } else {
      stop("Unknown target: must be 'weights' or 'biases'")
    }
    
    # Fix: allow both SL NN (update is list of 1 matrix) and ML NN (direct matrix)
    update_matrix <- if (is.list(update) && length(update) == 1) update[[1]] else update
    
    if (is.null(update_matrix)) {
      stop(paste0("Update matrix for layer ", layer, " is NULL -- check gradients or optimizer output."))
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    update_len <- length(update_matrix)
    
    if (is.null(target_dim)) {
      stop(paste0("Target matrix dimensions for layer ", layer, " are NULL."))
    }
    
    # ------------------------------------------
    #     SHAPE FIXES FOR WEIGHTS VS. BIASES
    # ------------------------------------------
    if (target == "biases") {
      if (length(update_matrix) == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update_matrix) == 1) {
        updated <- matrix(rep(update_matrix, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (update_len == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update_matrix, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      # Clip weights after update if target is "weights"
      clip_threshold <- .5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
      Wpost <- self$weights[[layer]]
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
      Wpost <- self$biases[[layer]]
    }
    
    # Clean compact log (Option A: tagged min/mean/max)
    optimizers_log_update("adam", epoch, layer, target, grads_matrix, Wpre, Wpost, verbose)
  } 
  else if (optimizer == "rmsprop") {
    Wpre <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    if (!exists("target_dim")) {
      target_dim <- if (target == "biases") {
        dim(as.matrix(self$biases[[layer]]))
      } else {
        dim(self$weights[[layer]])
      }
    }
    
    # Boost LR for output layer if needed
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # --- FIX: Make sure grads is a list of 2D matrix ---
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # Update call
    optimizer_params[[layer]] <- rmsprop_update(
      optimizer_params[[layer]],
      grads   = grads_input,
      lr      = lr * layer_boost,
      beta2   = beta2,
      epsilon = epsilon
    )
    
    update <- optimizer_params[[layer]]$updates
    update_matrix <- if (is.list(update) && length(update) == 1) update[[1]] else update
    
    # --- YOUR DIMENSIONAL HANDLING BLOCK (UNCHANGED) ---
    if (target == "biases") {
      if (length(update_matrix) == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update_matrix) == 1) {
        updated <- matrix(rep(update_matrix, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update_matrix) == prod(target_dim)) {
        updated <- matrix(update_matrix, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update_matrix, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update_matrix, length.out = prod(target_dim))
        updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      clip_threshold <- 5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
      Wpost <- self$weights[[layer]]
    } else {
      self$biases[[layer]] <- self$biases[[layer]] - updated
      Wpost <- self$biases[[layer]]
    }
    
    # Clean compact log (Option A)
    optimizers_log_update("rmsprop", epoch, layer, target, grads_matrix, Wpre, Wpost, verbose)
  }
  
  else if (optimizer == "sgd") {
    Wpre <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # Ensure optimizer_params is long enough and has the layer slot
    while (length(optimizer_params) < layer) {
      optimizer_params[[length(optimizer_params) + 1]] <- NULL
    }
    if (is.null(optimizer_params[[layer]])) {
      stop(paste("Missing optimizer_params for layer", layer))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1  # Placeholder
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # SGD update
    sgd_result <- sgd_update(
      params = optimizer_params[[layer]],
      grads  = grads_input,
      lr     = lr * layer_boost
    )
    
    optimizer_params[[layer]] <- sgd_result$params
    
    update <- if (target == "weights") {
      sgd_result$weights_update[[1]]
    } else {
      sgd_result$biases_update[[1]]
    }
    
    if (is.null(update)) {
      stop(paste("SGD update is NULL for layer", layer, "and target", target))
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (length(update) == prod(target_dim)) {
      updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(update) == 1) {
      updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      repeated <- rep(update, length.out = prod(target_dim))
      updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
    }
    
    if (target == "weights") {
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
      Wpost <- self$weights[[layer]]
    } else {
      self$biases[[layer]]  <- self$biases[[layer]] - updated
      Wpost <- self$biases[[layer]]
    }
    
    # Clean compact log (Option A: tagged min/mean/max)
    optimizers_log_update("sgd", epoch, layer, target, grads_matrix, Wpre, Wpost, verbose)
    
    # Build return object
    updated_optimizer <- list(updated_optimizer_params = optimizer_params[[layer]], updated_weights_or_biases = updated)
  }
  
  else if (optimizer == "sgd_momentum") {
    Wpre <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # $$$$$$$$$$$$ Ensure optimizer_params has this layer
    while (length(optimizer_params) < layer) {
      optimizer_params[[length(optimizer_params) + 1]] <- NULL
    }
    if (is.null(optimizer_params[[layer]])) {
      stop(paste("Missing optimizer_params for layer", layer))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # Momentum update
    sgd_result <- sgd_momentum_update(
      params   = optimizer_params[[layer]],
      grads    = grads_input,
      lr       = lr * layer_boost,
      momentum = beta1
    )
    optimizer_params[[layer]] <- sgd_result$params
    
    update <- if (target == "weights") {
      sgd_result$weights_update[[1]]
    } else {
      sgd_result$biases_update[[1]]
    }
    if (is.null(update)) {
      stop(paste("SGD momentum update is NULL for layer", layer, "and target", target))
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim    <- dim(as.matrix(target_matrix))
    
    if (length(update) == prod(target_dim)) {
      updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(update) == 1) {
      updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      repeated <- rep(update, length.out = prod(target_dim))
      updated  <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
    }
    
    if (target == "weights") {
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
      Wpost <- self$weights[[layer]]
    } else {
      self$biases[[layer]]  <- self$biases[[layer]] - updated
      Wpost <- self$biases[[layer]]
    }
    
    # Clean compact log (Option A: tagged min/mean/max)
    optimizers_log_update("sgd_momentum", epoch, layer, target, grads_matrix, Wpre, Wpost, verbose)
    
    # Return object to caller
    updated_optimizer <- list(
      updated_optimizer_params   = optimizer_params[[layer]],
      updated_weights_or_biases  = updated
    )
  }
  
  else if (optimizer == "nag") {
    if (verbose) {
      message(">> Optimizer = nag")
      message("Layer: ", layer)
      message("grads_matrix dim: ", if (is.null(dim(grads_matrix))) "NULL" else paste(dim(grads_matrix), collapse = " x "))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # NAG Update
    nag_result <- nag_update(
      params = optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost,
      beta1 = beta1
    )
    
    optimizer_params[[layer]] <- nag_result$params
    
    # Use correct update from result
    update <- if (target == "weights") nag_result$weights_update[[1]] else nag_result$biases_update[[1]]
    
    # Align shapes
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    if (verbose) {
      message("Updated ", target, " summary (layer ", layer, "): min = ", min(updated),
              ", mean = ", mean(updated), ", max = ", max(updated))
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  
  else if (optimizer == "ftrl") {
    if (verbose) {
      message(">> Optimizer = ftrl")
      message("Layer: ", layer)
      message("grads_matrix dim: ", if (is.null(dim(grads_matrix))) "NULL" else paste(dim(grads_matrix), collapse = " x "))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # FTRL Update
    ftrl_result <- ftrl_update(
      params   = optimizer_params[[layer]],
      grads    = grads_input,
      lr       = lr * layer_boost,
      alpha    = 0.1,
      beta     = 1.0,
      lambda1  = 0.01,
      lambda2  = 0.01
    )
    
    optimizer_params[[layer]] <- ftrl_result$params
    
    # Choose correct update
    update <- if (target == "weights") ftrl_result$weights_update[[1]] else ftrl_result$biases_update[[1]]
    
    # Align shape
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    if (verbose) {
      message("Updated ", target, " summary (layer ", layer, "): min = ", min(updated),
              ", mean = ", mean(updated), ", max = ", max(updated))
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  else if (optimizer == "lamb") {
    if (verbose) {
      message(">> Optimizer = lamb")
      message("Layer: ", layer)
      message("grads_matrix dim: ", if (is.null(dim(grads_matrix))) "NULL" else paste(dim(grads_matrix), collapse = " x "))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = length(grads_matrix), ncol = 1))
    } else {
      list(grads_matrix)
    }
    
    # LAMB Update
    lamb_result <- lamb_update(
      params = optimizer_params[[layer]],
      grads = grads_input[[1]],
      lr = lr * layer_boost,
      beta1 = beta1,
      beta2 = beta2,
      eps = epsilon,
      lambda = 0.01
    )
    
    optimizer_params[[layer]] <- lamb_result$params
    
    # Select correct update
    update <- if (target == "weights") lamb_result$weights_update[[1]] else lamb_result$biases_update[[1]]
    
    # Align shapes
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clipping
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    if (verbose) {
      message("Updated ", target, " summary (layer ", layer, "): min = ", min(updated),
              ", mean = ", mean(updated), ", max = ", max(updated))
    }
    
    # Apply update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  else if (optimizer == "lookahead") {
    if (verbose) {
      message(">> Optimizer = lookahead")
      message("Layer: ", layer)
      message("grads_matrix dim: ", if (is.null(dim(grads_matrix))) "NULL" else paste(dim(grads_matrix), collapse = " x "))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # Call the lookahead optimizer
    lookahead_result <- lookahead_update(
      params = optimizer_params[[layer]],
      grads_list = list(grads_matrix),
      lr = lr * layer_boost,
      beta1 = beta1,
      beta2 = beta2,
      epsilon = epsilon,
      lookahead_step = lookahead_step,
      base_optimizer = "adam_update",
      epoch = epoch,
      lambda = lambda
    )
    
    # Update the state
    optimizer_params[[layer]] <- lookahead_result
    
    # Extract update for weight or bias
    update <- if (target == "weights") lookahead_result$weights_update else {
      if (!is.null(lookahead_result$biases_update)) lookahead_result$biases_update else matrix(0, nrow = 1, ncol = 1)
    }
    
    # Align update shape to target
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (length(update) == prod(target_dim)) {
      updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
    } else if (length(update) == 1) {
      updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
    } else {
      repeated <- rep(update, length.out = prod(target_dim))
      updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
    }
    
    # Optionally clip weights
    if (target == "weights") {
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    if (verbose) {
      message("Updated ", target, " summary (layer ", layer, "): min = ", min(updated),
              ", mean = ", mean(updated), ", max = ", max(updated))
    }
    
    # Apply the update
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else if (target == "biases") {
      self$biases[[layer]] <- self$biases[[layer]] - updated
    }
  }
  
  
  
  else if (optimizer == "adagrad") {
    Wpre <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    
    # --- LR boost placeholder ---
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    # --- Normalize grads input ---
    grads_input <- if (is.list(grads_matrix)) {
      grads_matrix
    } else if (is.null(dim(grads_matrix))) {
      list(matrix(grads_matrix, nrow = 1, ncol = 1))
    } else if (length(dim(grads_matrix)) == 1) {
      list(matrix(grads_matrix, nrow = 1))
    } else {
      list(grads_matrix)
    }
    
    # --- Call Adagrad update ---
    adagrad_result <- adagrad_update(
      params  = optimizer_params[[layer]],
      grads   = grads_input,
      lr      = lr * layer_boost,
      epsilon = epsilon
    )
    optimizer_params[[layer]] <- adagrad_result$params
    
    # --- Extract update ---
    update <- if (target == "weights") {
      adagrad_result$weights_update[[1]]
    } else {
      adagrad_result$biases_update[[1]]
    }
    
    # --- Align shapes ---
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else if (length(update) == 1) {
        updated <- matrix(rep(update, prod(target_dim)), nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else if (prod(target_dim) == 1) {
        updated <- matrix(update, nrow = 1, ncol = 1)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      # Optional clipping
      # clip_threshold <- 0.5
      # updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    # --- Apply update ---
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
      Ppost <- self$weights[[layer]]
    } else {
      self$biases[[layer]]  <- self$biases[[layer]] - updated
      Ppost <- self$biases[[layer]]
    }
    
    # --- Log summary (works for weights or biases) ---
    if (verbose) {
      message("before log")
    }
    optimizers_log_update(
      optimizer       = "adagrad", epoch = epoch,
      layer           = layer,
      target          = target,
      grads_matrix    = grads_matrix,
      P_before        = Wpre,
      P_after         = Ppost,
      update_applied  = updated,
      verbose         = verbose
    )
  }
  
  
  
  
  else if (optimizer == "adadelta") {
    if (verbose) {
      message(">> Optimizer = adadelta (", target, ")")
      message("Layer: ", layer)
    }
    
    err <- errors[[layer]]
    err_dims <- dim(err)
    if (verbose) {
      message("errors dim: ", if (is.null(err_dims)) "NULL" else paste(err_dims, collapse = " x "))
    }
    
    # Normalize error to matrix shape
    if (is.null(err_dims)) {
      err <- matrix(err, nrow = 1)
    } else if (length(err_dims) == 1) {
      err <- matrix(err, nrow = 1)
    }
    
    # Safe to compute gradient matrix now
    grads_matrix <- colSums(err)
    
    if (verbose) {
      message("grads_matrix dim: ", if (is.null(dim(grads_matrix))) "NULL" else paste(dim(grads_matrix), collapse = " x "))
    }
    
    layer_boost <- if (layer == self$num_layers) 1 else 1
    
    grads_input <- list(matrix(grads_matrix, nrow = 1))
    
    adadelta_result <- adadelta_update(
      params = optimizer_params[[layer]],
      grads = grads_input,
      lr = lr * layer_boost,
      epsilon = epsilon
    )
    
    optimizer_params[[layer]] <- adadelta_result$params
    
    update <- if (target == "weights") adadelta_result$weights_update[[1]] else adadelta_result$biases_update[[1]]
    
    if (is.null(update) || !is.numeric(update)) {
      warning("WARNING: update (delta_w) is NULL or non-numeric. Skipping update for layer", layer, " target:", target)
      return(NULL)
    }
    
    target_matrix <- if (target == "weights") self$weights[[layer]] else self$biases[[layer]]
    target_dim <- dim(as.matrix(target_matrix))
    
    if (is.null(target_dim)) {
      warning("WARNING: target matrix has NULL dimension for layer", layer)
      return(NULL)
    }
    
    if (target == "biases") {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2])
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2])
      }
    } else {
      if (length(update) == prod(target_dim)) {
        updated <- matrix(update, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      } else {
        repeated <- rep(update, length.out = prod(target_dim))
        updated <- matrix(repeated, nrow = target_dim[1], ncol = target_dim[2], byrow = TRUE)
      }
      
      # Optional clip for weight updates
      clip_threshold <- 0.5
      updated <- pmin(pmax(updated, -clip_threshold), clip_threshold)
    }
    
    if (verbose) {
      message("Updated ", target, " summary (layer ", layer, "): min = ", min(updated),
              ", mean = ", mean(updated), ", max = ", max(updated))
    }
    
    if (target == "weights") {
      self$weights[[layer]] <- self$weights[[layer]] - updated
    } else {
      self$biases[[layer]] <- self$biases[[layer]] - updated 
    }
  }
  
  return(list(
    updated_weights_or_biases = self[[target]][[layer]],
    updated_optimizer_params = optimizer_params[[layer]])
  )
  
  
}


initialize_optimizer_params <- function(optimizer, dim, lookahead_step, layer, verbose = FALSE) {
  if (length(dim) == 2 && is.null(layer)) {
    dim <- list(dim)
  }
  
  layer_dim <- dim[[1]]  # always using first dim block
  if (length(layer_dim) != 2 || any(is.na(layer_dim)) || any(layer_dim <= 0)) {
    if (verbose) message("Invalid dimensions detected. Setting default dimension [1, 1].")
    layer_dim <- c(1, 1)
  }
  
  nrow_dim <- layer_dim[1]
  ncol_dim <- layer_dim[2]
  
  current_layer <- if (!is.null(layer)) layer else 1
  if (verbose) {
    message("Layer ", current_layer, " dimensions: nrow = ", nrow_dim, ", ncol = ", ncol_dim)
  }
  
  param_init <- matrix(rnorm(nrow_dim * ncol_dim), nrow = nrow_dim, ncol = ncol_dim)
  
  entry <- switch(optimizer,
                  adam = list(param = param_init, m = matrix(0, nrow = nrow_dim, ncol = ncol_dim), v = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  rmsprop = list(param = param_init, m = matrix(0, nrow = nrow_dim, ncol = ncol_dim), v = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  adadelta = list(param = param_init, m = matrix(0, nrow = nrow_dim, ncol = ncol_dim), v = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  adagrad = list(param = param_init, r = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  sgd = list(param = param_init, momentum = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  sgd_momentum = list(param = param_init, momentum = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  nag = list(param = param_init,
                             momentum = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                             fast_weights = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                             fast_biases = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  ftrl = list(param = param_init,
                              z = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                              n = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  lamb = list(param = param_init,
                              m = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                              v = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                              r = matrix(0, nrow = nrow_dim, ncol = ncol_dim)),
                  lookahead = list(param = param_init,
                                   m = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   v = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   r = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   slow_weights = matrix(0, nrow = nrow_dim, ncol = ncol_dim),
                                   lookahead_counter = 0,
                                   lookahead_step = lookahead_step),
                  stop(paste("Optimizer", optimizer, "not supported."))
  )
  
  if (verbose) {
    message("Layer ", current_layer, " optimizer tracking params initialized:")
    for (line in utils::capture.output(str(entry))) message(line)
  }
  
  return(entry)
}


adam_update <- function(params, grads, lr, beta1, beta2, epsilon, t) {
  # Force grads into a list if it's not already
  if (!is.list(grads)) {
    grads <- list(as.matrix(grads))
  }
  
  # Initialize m and v as lists
  if (!is.list(params$m)) {
    params$m <- vector("list", length(grads))
  }
  if (!is.list(params$v)) {
    params$v <- vector("list", length(grads))
  }
  
  # # Learning rate scheduler
  # lr_schedule <- function(t, initial_lr) {
  #   decay_rate <- 0.01
  #   initial_lr * exp(-decay_rate * t)
  # }
  # lr <- lr_schedule(t, lr)
  
  # Update moment estimates
  for (i in seq_along(grads)) {
    grad_matrix <- grads[[i]]
    grad_dims <- dim(grad_matrix)
    if (is.null(grad_dims)) grad_matrix <- matrix(grad_matrix, nrow = 1)
    
    # Initialize if missing or shape mismatch
    if (is.null(params$m[[i]]) || !all(dim(params$m[[i]]) == dim(grad_matrix))) {
      params$m[[i]] <- matrix(0, nrow = nrow(grad_matrix), ncol = ncol(grad_matrix))
    }
    if (is.null(params$v[[i]]) || !all(dim(params$v[[i]]) == dim(grad_matrix))) {
      params$v[[i]] <- matrix(0, nrow = nrow(grad_matrix), ncol = ncol(grad_matrix))
    }
    
    # Update m and v
    params$m[[i]] <- beta1 * params$m[[i]] + (1 - beta1) * grad_matrix
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grad_matrix ^ 2)
  }
  
  # Bias correction
  m_hat <- lapply(params$m, function(m) m / (1 - beta1 ^ t))
  v_hat <- lapply(params$v, function(v) v / (1 - beta2 ^ t))
  
  # Compute updates
  weights_update <- Map(function(m, v) lr * m / (sqrt(v) + epsilon), m_hat, v_hat)
  
  return(list(
    m = params$m,
    v = params$v,
    weights_update = weights_update,
    biases_update = weights_update  # identical in single-layer mode
  ))
}

rmsprop_update <- function(params, grads, lr, beta2 = 0.999, epsilon = 1e-8) {
  # Force grads into a list of matrices
  if (!is.list(grads)) {
    grads <- list(as.matrix(grads))
  }
  
  # Initialize v as list if not already
  if (!is.list(params$v)) {
    params$v <- vector("list", length(grads))
  }
  
  updates <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_matrix <- grads[[i]]
    
    # Ensure it's a matrix (handle scalar/1D vectors)
    grad_matrix <- if (is.null(dim(grad_matrix))) matrix(grad_matrix, nrow = 1) else as.matrix(grad_matrix)
    
    # Initialize v if missing or shape mismatch
    if (is.null(params$v[[i]]) || !all(dim(params$v[[i]]) == dim(grad_matrix))) {
      params$v[[i]] <- matrix(0, nrow = nrow(grad_matrix), ncol = ncol(grad_matrix))
    }
    
    # Update v
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grad_matrix ^ 2)
    
    # Compute update
    updates[[i]] <- lr * grad_matrix / (sqrt(params$v[[i]]) + epsilon)
  }
  
  return(list(
    v = params$v,
    updates = updates
  ))
}


adagrad_update <- function(params, grads, lr, epsilon) {
  # Initialize r as a list if not already
  if (!is.list(params$r)) {
    params$r <- vector("list", length(grads))
  }
  
  # Initialize updates
  updates <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    # Init r if missing or mismatched
    if (is.null(params$r[[i]]) || !all(dim(params$r[[i]]) == grad_dims)) {
      params$r[[i]] <- array(0, dim = grad_dims)
    }
    
    # Update r
    params$r[[i]] <- params$r[[i]] + grads[[i]]^2
    
    # Compute update
    updates[[i]] <- lr * grads[[i]] / (sqrt(params$r[[i]]) + epsilon)
  }
  
  # Return as full structured output
  return(list(
    params = params,
    weights_update = updates,
    biases_update = updates  # If shared logic for both -- differentiate if needed
  ))
}

adadelta_update <- function(params, grads, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, t = 1) {
  if (!is.list(params$m)) params$m <- vector("list", length(grads))
  if (!is.list(params$v)) params$v <- vector("list", length(grads))
  
  updates <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    # Ensure gradient is in array form
    grad_dims <- dim(grads[[i]])
    if (is.null(grad_dims)) {
      grads[[i]] <- array(grads[[i]], dim = c(length(grads[[i]]), 1))
      grad_dims <- dim(grads[[i]])
    }
    
    # Initialize accumulators with same shape
    if (is.null(params$m[[i]]) || !identical(dim(params$m[[i]]), grad_dims)) {
      params$m[[i]] <- array(0, dim = grad_dims)
    }
    if (is.null(params$v[[i]]) || !identical(dim(params$v[[i]]), grad_dims)) {
      params$v[[i]] <- array(0, dim = grad_dims)
    }
    
    # Clean NaNs from gradients
    grads[[i]][is.na(grads[[i]])] <- 0
    params$v[[i]][is.na(params$v[[i]])] <- 0
    
    # Adadelta updates
    params$v[[i]] <- beta2 * params$v[[i]] + (1 - beta2) * (grads[[i]] ^ 2)
    v_hat <- params$v[[i]] / (1 - beta2 ^ t)
    
    delta <- (sqrt(params$m[[i]] + epsilon) / sqrt(v_hat + epsilon)) * grads[[i]]
    delta[is.nan(delta) | is.infinite(delta)] <- 0
    delta <- pmin(pmax(delta, -5), 5)  # optional clip
    
    params$m[[i]] <- beta1 * params$m[[i]] + (1 - beta1) * (delta ^ 2)
    updates[[i]] <- delta
  }
  
  return(list(
    params = list(m = params$m, v = params$v),
    weights_update = updates,
    biases_update = updates
  ))
}

# Stochastic Gradient Descent with Momentum
# Define the sgd_update function with improvements
# Improved SGD Update Function
# Define the sgd_update function
sgd_momentum_update <- function(params, grads, lr, momentum) {
  # Initialize momentum as a list if it is not already
  if (!is.list(params$momentum)) {
    params$momentum <- vector("list", length(grads))
  }
  
  # Initialize weights_update and biases_update as lists
  weights_update <- vector("list", length(grads))
  biases_update  <- vector("list", length(grads))
  
  # Update each element of momentum and calculate weights_update and biases_update
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    
    if (is.null(grad_dims)) {
      grad_dims <- c(1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    if (is.null(params$momentum[[i]]) || !all(dim(params$momentum[[i]]) == grad_dims)) {
      params$momentum[[i]] <- array(0, dim = grad_dims)
    }
    
    # Momentum update
    params$momentum[[i]] <- momentum * params$momentum[[i]] - lr * grads[[i]]
    
    weights_update[[i]] <- params$momentum[[i]]
    biases_update[[i]]  <- lr * grads[[i]]  # This is just a placeholder; bias logic might differ
  }
  
  # Standardize dimensions to match grads
  for (i in seq_along(weights_update)) {
    if (is.null(dim(grads[[i]]))) {
      weights_update[[i]] <- array(weights_update[[i]], dim = c(1))
      biases_update[[i]]  <- array(biases_update[[i]],  dim = c(1))
    } else {
      weights_update[[i]] <- matrix(weights_update[[i]], nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
      biases_update[[i]]  <- matrix(biases_update[[i]],  nrow = dim(grads[[i]])[1], ncol = dim(grads[[i]])[2])
    }
  }
  
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

sgd_update <- function(params, grads, lr) {
  updated_params <- list()
  
  weights_update <- list()
  biases_update  <- list()
  
  for (i in seq_along(grads)) {
    grad_matrix <- as.matrix(grads[[i]])
    param_matrix <- as.matrix(params$param)
    
    # Safe reshape if needed
    if (!all(dim(grad_matrix) == dim(param_matrix))) {
      grad_matrix <- matrix(rep(grad_matrix, length.out = length(param_matrix)), nrow = nrow(param_matrix))
    }
    
    update_matrix <- lr * grad_matrix
    updated_param_matrix <- param_matrix - update_matrix
    
    # Store updates
    updated_params$param <- updated_param_matrix
    updated_params$momentum <- params$momentum  # even if unused
    
    weights_update[[i]] <- update_matrix
    biases_update[[i]]  <- matrix(0, nrow = 1, ncol = 1)  # dummy for compatibility
  }
  
  return(list(
    params = updated_params,
    weights_update = weights_update,
    biases_update = biases_update
  ))
}



nag_update <- function(params, grads, lr, beta1 = 0.9) {
  weights_update <- vector("list", length(grads))
  biases_update <- vector("list", length(grads))
  
  for (i in seq_along(grads)) {
    grad_dims <- dim(grads[[i]])
    if (is.null(grad_dims)) {
      grad_dims <- c(length(grads[[i]]), 1)
      grads[[i]] <- array(grads[[i]], dim = grad_dims)
    }
    
    if (length(params$momentum) < i) {
      params$momentum[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$fast_weights) < i) {
      params$fast_weights[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    if (length(params$fast_biases) < i) {
      params$fast_biases[[i]] <- matrix(0, nrow = grad_dims[1], ncol = grad_dims[2])
    }
    
    # Use beta1
    params$momentum[[i]] <- beta1 * params$momentum[[i]] + grads[[i]]
    weights_update[[i]] <- lr * (beta1 * params$momentum[[i]] + grads[[i]])
    biases_update[[i]] <- lr * grads[[i]]
    
    params$fast_weights[[i]] <- params$fast_weights[[i]] - weights_update[[i]]
    params$fast_biases[[i]] <- params$fast_biases[[i]] - biases_update[[i]]
  }
  
  return(list(params = params, weights_update = weights_update, biases_update = biases_update))
}

ftrl_update <- function(params, grads, lr,
                        alpha   = 0.1,
                        beta    = 1.0,
                        lambda1 = 0.01,
                        lambda2 = 0.01) {
  # 1) Wrap single matrix/vector into a list
  if (!is.list(grads)) {
    grads <- list(grads)
  }
  n_grads <- length(grads)
  
  # 2) Ensure params$z and params$n exist and are lists of correct length
  if (!is.list(params$z) || length(params$z) != n_grads) {
    params$z <- vector("list", n_grads)
  }
  if (!is.list(params$n) || length(params$n) != n_grads) {
    params$n <- vector("list", n_grads)
  }
  
  # Prepare outputs
  weights_update <- vector("list", n_grads)
  biases_update  <- vector("list", n_grads)
  
  for (i in seq_len(n_grads)) {
    # 3) Force grad to matrix
    grad_i <- grads[[i]]
    if (!is.matrix(grad_i)) {
      grad_i <- matrix(grad_i, nrow = length(grad_i), ncol = 1)
    }
    
    # 4) Initialize z[i], n[i] if missing or wrong shape
    if (is.null(params$z[[i]]) ||
        !identical(dim(params$z[[i]]), dim(grad_i))) {
      params$z[[i]] <- matrix(0, nrow = nrow(grad_i), ncol = ncol(grad_i))
    }
    if (is.null(params$n[[i]]) ||
        !identical(dim(params$n[[i]]), dim(grad_i))) {
      params$n[[i]] <- matrix(0, nrow = nrow(grad_i), ncol = ncol(grad_i))
    }
    
    # Shortcut references
    z_i <- params$z[[i]]
    n_i <- params$n[[i]]
    
    # 5) Compute new accumulators
    n_new <- n_i + grad_i^2
    sigma <- (sqrt(n_new) - sqrt(n_i)) / alpha
    z_new <- z_i + grad_i - sigma * 0  # if you had slow weights, you'd subtract them here
    
    # 6) FTRL update step
    denom <- (beta + sqrt(n_new)) / alpha + lambda2
    w_update <- -1/denom * (z_new - lambda1 * sign(z_new))
    
    # 7) Store updates
    weights_update[[i]] <- w_update
    biases_update[[i]]  <- matrix(0,
                                  nrow = nrow(grad_i),
                                  ncol = ncol(grad_i))  # still zero for biases
    
    # 8) Save back updated accumulators
    params$z[[i]] <- z_new
    params$n[[i]] <- n_new
  }
  
  return(list(
    params         = params,
    weights_update = weights_update,
    biases_update  = biases_update
  ))
}


# LAMB Update Function
lamb_update <- function(params, grads, lr, beta1, beta2, eps, lambda) {
  # Ensure param and grads are numeric vectors
  params$param <- as.numeric(params$param)
  grads <- as.numeric(grads)
  
  # Ensure m and v are initialized and numeric
  if (is.null(params$m)) params$m <- rep(0, length(grads))
  if (is.null(params$v)) params$v <- rep(0, length(grads))
  
  m <- beta1 * params$m + (1 - beta1) * grads
  v <- beta2 * params$v + (1 - beta2) * (grads^2)
  
  m_hat <- m / (1 - beta1)
  v_hat <- v / (1 - beta2)
  
  update <- m_hat / (sqrt(v_hat) + eps)
  
  # Trust ratio scaling (LAMB-specific)
  w_norm <- sqrt(sum(params$param^2))
  u_norm <- sqrt(sum(update^2))
  trust_ratio <- ifelse(w_norm > 0 && u_norm > 0, w_norm / u_norm, 1.0)
  
  update <- trust_ratio * update
  
  # Apply weight decay
  update <- update + lambda * params$param
  
  updated_param <- params$param - lr * update
  
  return(list(
    params = list(
      param = updated_param,
      m = m,
      v = v
    ),
    weights_update = list(update),  # Required for downstream [[1]]
    biases_update = list(rep(0, length(update)))  # Placeholder for biases
  ))
}
