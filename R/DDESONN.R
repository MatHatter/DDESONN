# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Experimental Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
#
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#_____/\\\\\\\\\\\__________/\\\\\________/\\\\\_____/\\\___/\\\\\_____/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#___/\\\/////////\\\______/\\\///\\\_____\/\\\\\\___\/\\\__\/\\\\\\___\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#__\//\\\______\///_____/\\\/__\///\\\___\/\\\/\\\__\/\\\__\/\\\/\\\__\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#___\////\\\___________/\\\______\//\\\__\/\\\//\\\_\/\\\__\/\\\//\\\_\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#______\////\\\_______\/\\\_______\/\\\__\/\\\\//\\\\/\\\__\/\\\\//\\\\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#_________\////\\\____\//\\\______/\\\___\/\\\_\//\\\/\\\__\/\\\_\//\\\/\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#__/\\\______\//\\\____\///\\\__/\\\_____\/\\\__\//\\\\\\__\/\\\__\//\\\\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#_\///\\\\\\\\\\\/_______\///\\\\\/______\/\\\___\//\\\\\__\/\\\___\//\\\\\_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#___\///////////___________\/////________\///_____\/////___\///_____\/////_$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Step 1: Define the Self-Organizing Neural Network (SONN) class


SONN <- R6::R6Class( 
  "SONN",
  lock_objects = FALSE,
  public = list(
    input_size = NULL,  # Define input_size as a public property
    hidden_sizes = NULL,
    output_size = NULL,
    num_layers = NULL,
    lambda = NULL,
    weights = NULL,
    biases = NULL,
    ML_NN = NULL,
    N = NULL,
    map = NULL,
    threshold = NULL,
    model_iter_num = NULL,
    activation_functions = NULL,
    activation_functions_predict = NULL,
    dropout_rates = NULL,

    initialize = function(input_size, hidden_sizes = NULL, output_size, Rdata = NULL, N,  lambda, ML_NN, dropout_rates = NULL, activation_functions = NULL, activation_functions_predict = NULL, init_method, custom_scale) {

      # Initialize SONN parameters and architecture
      self$input_size <- input_size
      self$ML_NN <- ML_NN
      if (self$ML_NN) {
        if (is.null(hidden_sizes) || !length(hidden_sizes)) {
          stop("ML_NN=TRUE requires non-empty hidden_sizes.")
        }
        self$hidden_sizes <- as.integer(hidden_sizes)
        self$num_layers   <- length(self$hidden_sizes) + 1L  # including the output layer
      } else {
        self$hidden_sizes <- NULL
        self$num_layers   <- 1L  # single layer: input -> output
      }

      self$output_size <- output_size
      self$lambda <- lambda  # Regularization parameter
      self$ML_NN <- ML_NN
      self$num_layers <- length(hidden_sizes) + 1  # including the output layer

      self$dropout_rates <- dropout_rates

      self$output_size <- output_size
      self$lambda <- lambda  # Regularization parameter



      # Initialize weights and biases using specified initialization method fir MLNN and SLNN
      init <- self$initialize_weights_biases(input_size, hidden_sizes, output_size, init_method, custom_scale)
      self$weights <- init$weights
      self$biases <- init$biases



      # Function to find factors of N that are as close as possible to each other
      find_grid_dimensions <- function(N) {
        factors <- unlist(sapply(1:floor(sqrt(N)), function(x) {
          if (N %% x == 0) return(c(x, N / x))
        }))
        factors <- sort(unique(factors))  # Sort and remove duplicates

        # Find the index of the factor that is closest to the square root of N
        sqrt_N <- sqrt(N)
        idx <- which.min(abs(factors - sqrt_N))

        # Return the pair of factors closest to each other
        c(factors[idx], N / factors[idx])
      }

      # Use the function to dynamically calculate grid_rows and grid_cols
      grid_dimensions <- find_grid_dimensions(N)
      grid_rows <- grid_dimensions[1]
      grid_cols <- grid_dimensions[2]

      self$map <- matrix(1:N, nrow = grid_rows, ncol = grid_cols)

      # Configuration flags for enabling/disabling per-SONN model training plots
      self$PerEpochViewPlotsConfig <- list(
        accuracy_plot = FALSE,  # training accuracy/loss
        saturation_plot = FALSE,  # output saturation
        max_weight_plot = FALSE,  # max weight magnitude
        viewAllPlots = FALSE,
        verbose    = FALSE
      )

    },
    initialize_weights_biases = function(input_size, hidden_sizes, output_size, init_method, custom_scale) {
      # container
      weights <- list()
      biases  <- list()
      
      # --------- light validation (non-intrusive) ---------
      if (!is.finite(input_size) || input_size <= 0) stop("input_size must be a positive number.")
      if (!is.finite(output_size) || output_size <= 0) stop("output_size must be a positive number.")
      if (isTRUE(self$ML_NN)) {
        if (is.null(hidden_sizes) || length(hidden_sizes) == 0L) {
          stop("ML_NN=TRUE requires hidden_sizes with at least one positive integer.")
        }
        if (any(!is.finite(hidden_sizes) | hidden_sizes <= 0)) {
          stop("All hidden_sizes must be positive numbers.")
        }
      }
      
      # local initializer (preserved methods; add sd/fan guards)
      init_weight <- function(fan_in, fan_out, init_method, custom_scale) {
        if (!is.finite(fan_in) || !is.finite(fan_out) || fan_in <= 0 || fan_out <= 0) {
          stop(sprintf("Bad layer dims for init: fan_in=%s fan_out=%s", fan_in, fan_out))
        }
        
        if (init_method == "xavier") {
          scale <- ifelse(is.null(custom_scale), 0.5, custom_scale)
          sd <- sqrt(2 / (fan_in + fan_out)) * scale
          sd <- if (!is.finite(sd) || sd <= 0) .Machine$double.eps else sd
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
          
        } else if (init_method == "he") {
          scale <- ifelse(is.null(custom_scale), 1.0, custom_scale)
          sd <- sqrt(2 / fan_in) * scale
          sd <- if (!is.finite(sd) || sd <= 0) .Machine$double.eps else sd
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
          
        } else if (init_method == "lecun") {
          scale <- ifelse(is.null(custom_scale), 1.0, custom_scale)
          sd <- sqrt(1 / fan_in) * scale
          sd <- if (!is.finite(sd) || sd <= 0) .Machine$double.eps else sd
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
          
        } else if (init_method == "orthogonal") {
          # Preserve your approach; just ensure dimensions match exactly
          A <- matrix(rnorm(fan_in * fan_out), nrow = fan_in, ncol = fan_out)
          QR <- qr(A)
          Q  <- qr.Q(QR)
          if (nrow(Q) != fan_in || ncol(Q) != min(fan_in, fan_out)) {
            # qr.Q can return min(fan_in, fan_out) columns; pad if needed
            if (ncol(Q) < fan_out) {
              Q <- cbind(Q, matrix(0, nrow = fan_in, ncol = fan_out - ncol(Q)))
            } else if (ncol(Q) > fan_out) {
              Q <- Q[, seq_len(fan_out), drop = FALSE]
            }
          }
          W <- Q
          
        } else if (init_method == "variance_scaling") {
          scale <- ifelse(is.null(custom_scale), 0.5, custom_scale)
          sd <- sqrt(1 / (fan_in + fan_out)) * scale
          sd <- if (!is.finite(sd) || sd <= 0) .Machine$double.eps else min(sd, 0.2)
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
          
        } else if (init_method == "glorot_uniform") {
          limit <- sqrt(6 / (fan_in + fan_out))
          limit <- if (!is.finite(limit) || limit <= 0) 1e-3 else limit
          W <- matrix(runif(fan_in * fan_out, min = -limit, max = limit), nrow = fan_in, ncol = fan_out)
          
        } else {
          sd <- 0.01
          W <- matrix(rnorm(fan_in * fan_out, mean = 0, sd = sd), nrow = fan_in, ncol = fan_out)
        }
        return(W)
      }
      
      # ======================
      # Multi-layer vs Single
      # ======================
      if (self$ML_NN) {
        # ---- Multi-layer path ----
        # First hidden layer
        weights[[1]] <- init_weight(as.integer(input_size), as.integer(hidden_sizes[1]), init_method, custom_scale)
        biases[[1]]  <- matrix(0, nrow = as.integer(hidden_sizes[1]), ncol = 1)
        
        # Intermediate hidden layers
        if (length(hidden_sizes) >= 2L) {
          for (layer in 2:length(hidden_sizes)) {
            fan_in  <- as.integer(hidden_sizes[layer - 1])
            fan_out <- as.integer(hidden_sizes[layer])
            weights[[layer]] <- init_weight(fan_in, fan_out, init_method, custom_scale)
            biases[[layer]]  <- matrix(0, nrow = fan_out, ncol = 1)
          }
        }
        
        # Output layer
        last_hidden_size <- as.integer(hidden_sizes[[length(hidden_sizes)]])
        weights[[length(hidden_sizes) + 1]] <- init_weight(last_hidden_size, as.integer(output_size), init_method, custom_scale)
        biases[[length(hidden_sizes) + 1]]  <- matrix(0, nrow = as.integer(output_size), ncol = 1)
        
        # assign into self (lists)
        self$weights <- weights
        self$biases  <- biases
        
      } else {
        # ---- Single-layer path ----
        # keep building in locals for consistency
        W <- init_weight(as.integer(input_size), as.integer(output_size), init_method, custom_scale)
        b <- rep(0, as.integer(output_size))   # <- store as numeric vector for SL
        
        # orientation sanity: rows must match input_size; transpose if needed
        if (nrow(W) != as.integer(input_size) && ncol(W) == as.integer(input_size)) {
          W <- t(W)
        }
        
        # assign into self as plain numeric types (not lists)
        self$weights <- W                       # matrix: input_size x output_size
        self$biases  <- as.numeric(b)           # vector: length = output_size
        
        # keep local lists in sync for return shape below
        weights[[1]] <- self$weights
        biases[[1]]  <- matrix(self$biases, nrow = length(self$biases), ncol = 1)
      }
      
      # return shapes consistent with assignment (SL returns plain; ML returns lists)
      return(list(weights = self$weights, biases = self$biases))
    },
    #Dropout functions with no default rates (training only)
    dropout_forward = function(x, rate) {
      # Keep shapes stable & support NULL / edge rates
      if (is.null(rate) || rate <= 0 || rate >= 1) {
        return(list(out = x, mask = NULL, scale = 1))
      }
      x <- as.matrix(x)
      mask <- matrix(rbinom(length(x), 1, 1 - rate), nrow = nrow(x), ncol = ncol(x))
      scale <- 1 / (1 - rate)              # inverted dropout scaling
      list(out = x * mask * scale, mask = mask, scale = scale)
    },
    dropout_backward = function(grad, mask, rate) {
      # Reuse the SAME mask as forward; apply the same scaling
      if (is.null(mask) || is.null(rate) || rate <= 0 || rate >= 1) {
        return(grad)
      }
      grad <- as.matrix(grad)
      grad * mask * (1 / (1 - rate))
    }
    ,# Method to perform self-organization
    viewPerEpochPlots = function(name) {
      cfg <- self$PerEpochViewPlotsConfig
      on_all <- isTRUE(cfg$viewAllPlots) || isTRUE(cfg$verbose)
      isTRUE(cfg[[name]]) || on_all
    },
    self_organize = function(Rdata, labels, lr, verbose = FALSE) {
      if(verbose){print("----------------------------------------self-organize-begin----------------------------------------")}





      if (self$ML_NN) {
        # Multi-layer mode: First layer
        input_rows <- nrow(Rdata)
        output_cols <- ncol(self$weights[[1]])

        if (length(self$biases[[1]]) == 1) {
          bias_matrix <- matrix(self$biases[[1]], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases[[1]]) == output_cols) {
          bias_matrix <- matrix(self$biases[[1]], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases[[1]]) < output_cols) {
          bias_matrix <- matrix(rep(self$biases[[1]], length.out = output_cols),
                                nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else {
          bias_matrix <- matrix(self$biases[[1]][1:output_cols], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        }

        outputs <- Rdata %*% self$weights[[1]] + bias_matrix

      } else {
        # Single-layer mode
        input_rows <- nrow(Rdata)
        weight_matrix_sl <- if (is.list(self$weights)) as.matrix(self$weights[[1]]) else as.matrix(self$weights)
        output_cols <- ncol(weight_matrix_sl)

        if (length(self$biases) == 1) {
          bias_matrix <- matrix(self$biases, nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases) == output_cols) {
          bias_matrix <- matrix(self$biases, nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else if (length(self$biases) < output_cols) {
          bias_matrix <- matrix(rep(self$biases, length.out = output_cols),
                                nrow = input_rows, ncol = output_cols, byrow = TRUE)
        } else {
          bias_matrix <- matrix(self$biases[1:output_cols], nrow = input_rows, ncol = output_cols, byrow = TRUE)
        }

        outputs <- Rdata %*% weight_matrix_sl + bias_matrix
      }




      if (self$ML_NN) {
        hidden_outputs <- list()
        hidden_outputs[[1]] <- outputs

        outputs <- vector("list", self$num_layers)
        outputs[[1]] <- hidden_outputs[[1]]

        broadcast_bias <- function(bias, nrow_out, ncol_out) {
          if (length(bias) == 1) {
            matrix(bias, nrow_out, ncol_out, byrow = TRUE)
          } else if (length(bias) == ncol_out) {
            matrix(bias, nrow_out, ncol_out, byrow = TRUE)
          } else if (length(bias) < ncol_out) {
            matrix(rep(bias, length.out = ncol_out), nrow_out, ncol_out, byrow = TRUE)
          } else if (length(bias) > ncol_out) {
            matrix(bias[1:ncol_out], nrow_out, ncol_out, byrow = TRUE)
          } else {
            stop("Bias shape mismatch")
          }
        }

        for (layer in 2:self$num_layers) {
          input <- hidden_outputs[[layer - 1]]
          weights <- self$weights[[layer]]
          biases <- broadcast_bias(self$biases[[layer]], nrow(input), ncol(weights))

          if (ncol(input) != nrow(weights)) {
            if (ncol(input) == ncol(weights)) {
              weights <- t(weights)
            } else {
              stop("Dimensions of hidden_outputs and weights are not conformable")
            }
          }

          hidden_outputs[[layer]] <- input %*% weights + biases
          outputs[[layer]] <- hidden_outputs[[layer]]
        }
      }

      print("str(outputs)")
      str(outputs)




      if (self$ML_NN) {
        print(paste("LAYER", self$num_layers))

        expected_shape <- dim(outputs[[self$num_layers]])
        input_shape <- dim(Rdata)

        if (!all(expected_shape == input_shape)) {
          cat("Mismatch between Rdata and outputs[[num_layers]]: correcting...\n")
          # Try to reshape outputs to match Rdata
          output_matrix <- matrix(
            rep(outputs[[self$num_layers]], length.out = nrow(Rdata) * ncol(Rdata)),
            nrow = nrow(Rdata),
            ncol = ncol(Rdata),
            byrow = FALSE
          )
        } else {
          output_matrix <- outputs[[self$num_layers]]
        }

        error_1000x10 <- Rdata - output_matrix

      } else {
        if (!all(dim(outputs) == dim(Rdata))) {
          cat("Mismatch between Rdata and outputs (single-layer): correcting...\n")
          output_matrix <- matrix(
            rep(outputs, length.out = nrow(Rdata) * ncol(Rdata)),
            nrow = nrow(Rdata),
            ncol = ncol(Rdata),
            byrow = FALSE
          )
        } else {
          output_matrix <- outputs
        }

        error_1000x10 <- Rdata - output_matrix
      }

      # Store output error
      errors <- vector("list", self$num_layers)
      errors[[self$num_layers]] <- as.matrix(error_1000x10)
      str(errors[[self$num_layers]])


      # Store output error
      errors <- vector("list", self$num_layers)
      errors[[self$num_layers]] <- as.matrix(error_1000x10)
      str(errors[[self$num_layers]])




      # Propagate the error backwards
      if (self$ML_NN) {
        for (layer in (self$num_layers - 1):1) {
          cat("Layer:", layer, "\n")

          # Load weights and error from next layer
          weights_next <- self$weights[[layer + 1]]
          errors_next  <- errors[[layer + 1]]

          # Check for NULLs
          if (is.null(weights_next) || is.null(errors_next)) {
            cat(paste("Skipping layer", layer, "- weights or errors are NULL\n"))
            next
          }

          # Print actual dimensions
          weight_dims <- dim(weights_next)
          error_dims  <- dim(errors_next)
          cat("Weights dimensions:\n"); print(weight_dims)
          cat("Errors dimensions:\n"); print(error_dims)

          # Sanity checks
          if (is.null(weight_dims) || is.null(error_dims)) {
            cat(paste("Skipping layer", layer, "- dimensions are NULL\n"))
            next
          }

          # Adjust shape dynamically
          expected_error_cols <- ncol(weights_next)
          actual_error_cols   <- ncol(errors_next)
          actual_error_rows   <- nrow(errors_next)

          # Match error columns to weights' output dim
          if (actual_error_cols != expected_error_cols) {
            if (actual_error_cols > expected_error_cols) {
              errors_next <- errors_next[, 1:expected_error_cols, drop = FALSE]
            } else {
              errors_next <- matrix(
                rep(errors_next, length.out = actual_error_rows * expected_error_cols),
                nrow = actual_error_rows,
                ncol = expected_error_cols
              )
            }
          }


          # Propagate error
          cat("Backpropagating errors for layer", layer, "\n")
          errors[[layer]] <- errors_next %*% t(weights_next)
        }
      }

      else {
        cat("Single Layer Backpropagation\n")

        # Check existence
        weights_sl <- if (is.list(self$weights)) self$weights[[1]] else self$weights
        errors_sl  <- errors[[1]]

        if (is.null(weights_sl) || is.null(errors_sl)) {
          stop("Error: Weights or errors for single layer do not exist.")
        }

        # Ensure matrix form
        weights_sl <- as.matrix(weights_sl)
        errors_sl  <- as.matrix(errors_sl)

        # Print current dimensions
        weight_dims <- dim(weights_sl)
        error_dims  <- dim(errors_sl)
        cat("Weights dimensions:\n"); print(weight_dims)
        cat("Errors dimensions:\n"); print(error_dims)

        if (is.null(weight_dims) || is.null(error_dims)) {
          stop("Error: Dimensions for weights or errors are NULL.")
        }

        # Target: errors[[1]] = weights %*% errors
        # Align shapes: [n_input, n_output] %*% [batch_size, n_output]^T

        # Ensure error has matching columns
        expected_cols <- ncol(weights_sl)
        actual_cols   <- ncol(errors_sl)
        actual_rows   <- nrow(errors_sl)

        if (actual_cols != expected_cols) {
          if (actual_cols > expected_cols) {
            errors_sl <- errors_sl[, 1:expected_cols, drop = FALSE]
          } else {
            errors_sl <- matrix(
              rep(errors_sl, length.out = actual_rows * expected_cols),
              nrow = actual_rows,
              ncol = expected_cols
            )
          }
        }

        # If error is still misaligned for matrix multiplication, transpose
        if (ncol(errors_sl) != ncol(weights_sl)) {
          if (ncol(errors_sl) == nrow(weights_sl)) {
            weights_sl <- t(weights_sl)
          } else {
            cat("Warning: shape mismatch persists in single-layer case\n")
          }
        }

        # Perform backpropagation step
        cat("Performing matrix multiplication for single layer\n")
        errors[[1]] <- errors_sl %*% t(weights_sl)
      }


      print("str(errors)")
      str(errors)


      if (self$ML_NN) {

        # Ensure errors[[1]] has same number of rows as Rdata
        if (nrow(errors[[1]]) != nrow(Rdata)) {
          if (nrow(errors[[1]]) > nrow(Rdata)) {
            errors[[1]] <- errors[[1]][1:nrow(Rdata), , drop = FALSE]
          } else {
            errors[[1]] <- errors[[1]][rep(1:nrow(errors[[1]]), length.out = nrow(Rdata)), , drop = FALSE]
          }
        }

        # Update weights for the first layer
        if (ncol(errors[[1]]) == nrow(self$weights[[1]])) {
          self$weights[[1]] <- self$weights[[1]] - (lr * t(Rdata) %*% errors[[1]])
        } else if (nrow(t(errors[[1]])) == nrow(self$weights[[1]]) && ncol(t(errors[[1]])) < ncol(Rdata)) {
          self$weights[[1]] <- self$weights[[1]] - ((lr * t(Rdata) %*% errors[[1]]))[, 1:ncol(self$weights[[1]])]
        } else if (prod(dim(self$weights[[1]])) == 1) {
          update_value <- lr * sum(t(Rdata) %*% errors[[1]])
          self$weights[[1]] <- self$weights[[1]] - update_value
        } else {
          self$weights[[1]] <- self$weights[[1]] - (lr * apply(t(Rdata) %*% errors[[1]], 2, mean))
        }

        # Update biases for the first layer
        if (length(self$biases[[1]]) < length(colMeans(errors[[1]]))) {
          colMeans_shortened <- colMeans(errors[[1]])[1:length(self$biases[[1]])]
          self$biases[[1]] <- self$biases[[1]] - (lr * colMeans_shortened)
        } else if (length(self$biases[[1]]) > length(colMeans(errors[[1]]))) {
          colMeans_extended <- rep(colMeans(errors[[1]]), length.out = length(self$biases[[1]]))
          self$biases[[1]] <- self$biases[[1]] - (lr * colMeans_extended)
        } else {
          self$biases[[1]] <- self$biases[[1]] - (lr * colMeans(errors[[1]]))
        }

      }
      else {
        # Robust single-layer weight update
        weights_mat <- if (is.list(self$weights)) as.matrix(self$weights[[1]]) else as.matrix(self$weights)
        gradient <- tryCatch({
          grad <- t(Rdata) %*% error_1000x10
          if (all(dim(weights_mat) == dim(grad))) {
            grad
          } else if (prod(dim(weights_mat)) == 1) {
            sum(grad)
          } else if (ncol(weights_mat) < ncol(grad)) {
            grad[, 1:ncol(weights_mat), drop = FALSE]
          } else if (ncol(weights_mat) > ncol(grad)) {
            matrix(
              rep(grad, length.out = nrow(weights_mat) * ncol(weights_mat)),
              nrow = nrow(weights_mat),
              ncol = ncol(weights_mat)
            )
          } else {
            apply(grad, 2, mean)
          }
        }, error = function(e) {
          apply(t(Rdata) %*% error_1000x10, 2, mean)
        })

        # Update weights
        if (is.matrix(gradient)) {
          weights_mat <- weights_mat - (lr * gradient)
        } else {
          weights_mat <- weights_mat - (lr * matrix(gradient, nrow = nrow(weights_mat), ncol = ncol(weights_mat)))
        }
        if (is.list(self$weights)) {
          self$weights[[1]] <- weights_mat
        } else {
          self$weights <- weights_mat
        }

        # Robust single-layer bias update
        if (length(self$biases) == ncol(error_1000x10)) {
          self$biases <- self$biases - (lr * colMeans(error_1000x10))
        } else if (length(self$biases) < ncol(error_1000x10)) {
          self$biases <- self$biases - (lr * colMeans(error_1000x10)[1:length(self$biases)])
        } else {
          extended <- rep(colMeans(error_1000x10), length.out = length(self$biases))
          self$biases <- self$biases - (lr * extended)
        }
      }


      if (self$ML_NN) {
        for (layer in 2:self$num_layers) {

          # Ensure the number of columns in errors[[layer]] matches the number of rows in hidden_outputs[[layer - 1]]
          if (ncol(errors[[layer]]) != nrow(hidden_outputs[[layer - 1]])) {
            if (ncol(errors[[layer]]) > nrow(hidden_outputs[[layer - 1]])) {
              # Truncate columns of errors[[layer]] to match the number of rows in hidden_outputs[[layer - 1]]
              errors[[layer]] <- errors[[layer]][, 1:nrow(hidden_outputs[[layer - 1]]), drop = FALSE]
            } else {
              # Replicate columns of errors[[layer]] to match the number of rows in hidden_outputs[[layer - 1]]
              errors[[layer]] <- errors[[layer]][, rep(1:ncol(errors[[layer]]), length.out = nrow(hidden_outputs[[layer - 1]])), drop = FALSE]
            }
          }




          # Calculate dimensions
          result_dim <- dim(errors[[layer]] %*% (hidden_outputs[[layer - 1]]))
          weight_dim <- dim(self$weights[[layer]])

          print("results_dim")
          print(dim(errors[[layer]]))
          print("hidden_outputs[[layer - 1]])")
          print(dim(hidden_outputs[[layer - 1]]))

          # Update weights for the layer
          if (ncol(self$weights[[layer]]) == ncol(hidden_outputs[[layer - 1]])) {
            if (all(weight_dim == result_dim)) {
              grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
              self$weights[[layer]] <- self$weights[[layer]] - lr * grad
            } else {
              cat("Dimensions mismatch, handling default case for weights.\n")
              grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
              grad <- grad[1:nrow(self$weights[[layer]]), 1:ncol(self$weights[[layer]])]
              self$weights[[layer]] <- self$weights[[layer]] - lr * grad
            }

          } else if (nrow(self$weights[[layer]]) == ncol(hidden_outputs[[layer - 1]]) &&
                     ncol(self$weights[[layer]]) < ncol(hidden_outputs[[layer - 1]])) {
            grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
            grad <- grad[1:nrow(self$weights[[layer]]), 1:ncol(self$weights[[layer]])]
            self$weights[[layer]] <- self$weights[[layer]] - lr * grad

          } else if (prod(weight_dim) == 1) {
            grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
            update_value <- lr * sum(grad)
            self$weights[[layer]] <- self$weights[[layer]] - update_value

          } else {
            grad <- t(hidden_outputs[[layer - 1]]) %*% errors[[layer]]
            grad <- grad[1:nrow(self$weights[[layer]]), 1:ncol(self$weights[[layer]])]
            self$weights[[layer]] <- self$weights[[layer]] - lr * grad
          }



          # Update biases for the layer
          if (length(self$biases[[layer]]) < length(colMeans(errors[[layer]]))) {
            colMeans_shortened <- colMeans(errors[[layer]])[1:length(self$biases[[layer]])]
            self$biases[[layer]] <- self$biases[[layer]] - (lr * colMeans_shortened)
          } else if (length(self$biases[[layer]]) > length(colMeans(errors[[layer]]))) {
            colMeans_extended <- rep(colMeans(errors[[layer]]), length.out = length(self$biases[[layer]]))
            self$biases[[layer]] <- self$biases[[layer]] - (lr * colMeans_extended)
          } else {
            self$biases[[layer]] <- self$biases[[layer]] - (lr * colMeans(errors[[layer]]))
          }

        }
      }



      if (is.null(self$map)) {
        cat("[Debug] SOM not yet trained. Training now...\n")
        self$train_map(Rdata)

        # Determine how many SOM neurons to keep based on max allowed
        max_neurons_allowed <- 9  # e.g., for 3x3 SOM map
        map_size <- prod(dim(self$map$codes[[1]]))  # total SOM neurons
        actual_neurons <- min(map_size, max_neurons_allowed)

        input_dim <- ncol(Rdata)

        # Truncate weights to desired number of SOM neurons and match input dimension
        truncated_weights <- self$map$codes[[1]][1:actual_neurons, 1:input_dim, drop = FALSE]
        if (is.list(self$weights)) {
          self$weights[[1]] <- matrix(truncated_weights, nrow = actual_neurons, ncol = input_dim)
          weight_dim <- dim(self$weights[[1]])
        } else {
          self$weights <- matrix(truncated_weights, nrow = actual_neurons, ncol = input_dim)
          weight_dim <- dim(self$weights)
        }

        # Set bias to match output dimension of layer 1
        output_dim_layer1 <- if (self$ML_NN && self$num_layers >= 1) {
          weight_dim[2]
        } else {
          1
        }
        if (is.list(self$biases)) {
          self$biases[[1]] <- rep(0, output_dim_layer1)
        } else {
          self$biases <- rep(0, output_dim_layer1)
        }

        # Debug info
        cat("[Debug] SOM-trained weights dim after truncation:\n")
        print(weight_dim)
      }




      if(verbose){print("----------------------------------------self-organize-end----------------------------------------")}

    },
    #the magical function
    #the magical function
    learn = function(Rdata, labels, lr, CLASSIFICATION_MODE, activation_functions, dropout_rates, sample_weights, verbose = FALSE) {
      if (verbose) { print("----------------------------------------learn-begin----------------------------------------") }
      start_time <- Sys.time()
      
      `%||%` <- function(x, y) if (is.null(x)) y else x
      .safe_get <- function(lst, idx) {
        if (is.list(lst) && length(lst) >= idx && idx >= 1) lst[[idx]] else NULL
      }
      .af_name <- function(f) {
        if (is.function(f)) {
          nm <- attr(f, "name")
          if (is.null(nm)) return("unnamed_function")
          return(nm)
        }
        if (is.character(f)) return(paste(f, collapse = ","))
        if (is.null(f)) return("NULL")
        class(f)[1]
      }
      
      # --- minimal, local resolver (no global normalizer) ---
      .resolve_one <- function(x) {
        if (is.null(x) || is.function(x)) {
          if (is.function(x) && is.null(attr(x, "name"))) attr(x, "name") <- "unnamed_function"
          return(x)
        }
        if (is.character(x) && length(x) >= 1L) {
          key <- tolower(x[1])
          key <- gsub("[- ]", "_", key, perl = TRUE)
          if (key %in% c("linear","none")) key <- "identity"
          if (key == "logistic") key <- "sigmoid"
          fn <- switch(key,
                       "relu"=relu, "tanh"=tanh, "sigmoid"=sigmoid, "softmax"=softmax, "identity"=identity,
                       "leaky_relu"=leaky_relu, "elu"=elu, "swish"=swish, "gelu"=gelu, "selu"=selu, "mish"=mish,
                       # add more of your registry names here if you want them available:
                       "hard_sigmoid"=hard_sigmoid, "softplus"=softplus, "prelu"=prelu,
                       "bent_identity"=bent_identity, "maxout"=maxout,
                       NULL
          )
          if (is.null(fn)) stop(sprintf("Unsupported activation: '%s'", x[1]), call. = FALSE)
          if (is.null(attr(fn, "name"))) attr(fn, "name") <- key
          return(fn)
        }
        stop(sprintf("Unsupported activation spec type: %s", class(x)[1]), call. = FALSE)
      }
      
      ## ---------------------------
      ## Labels & sample_weights prep
      ## ---------------------------
      if (identical(CLASSIFICATION_MODE, "binary")) {
        if (!is.numeric(labels) || !is.matrix(labels) || ncol(labels) != 1) {
          labels <- matrix(as.numeric(labels), ncol = 1)
        }
        pos_weight <- 2
        neg_weight <- 1
        if (is.null(sample_weights)) {
          sample_weights <- ifelse(labels == 1, pos_weight, neg_weight)
        }
        sample_weights <- matrix(sample_weights, nrow = nrow(labels), ncol = 1)
        
        if (!is.matrix(labels)) labels <- as.matrix(labels)
        if (length(dim(labels)) == 2 && nrow(labels) == ncol(labels)) {
          labels <- matrix(diag(labels), ncol = 1)
        }
        labels <- matrix(as.numeric(labels), ncol = 1)
        
      } else if (identical(CLASSIFICATION_MODE, "multiclass")) {
        # one-hot if needed
        if (is.matrix(labels) && ncol(labels) >= 2) {
          labels_mat <- labels
        } else {
          lbl_vec <- as.vector(labels)
          if (!is.null(self$class_levels) && length(self$class_levels) > 0) {
            lbl_fac <- factor(lbl_vec, levels = self$class_levels)
          } else {
            lbl_fac <- factor(lbl_vec)
            self$class_levels <- levels(lbl_fac)
          }
          labels_mat <- model.matrix(~ lbl_fac - 1)
          colnames(labels_mat) <- as.character(self$class_levels)
        }
        labels <- as.matrix(labels_mat)
        
        if (is.null(sample_weights)) {
          sample_weights <- rep(1, nrow(labels))
        }
        sample_weights <- matrix(sample_weights, nrow = nrow(labels), ncol = ncol(labels), byrow = FALSE)
        
      } else if (identical(CLASSIFICATION_MODE, "regression")) {
        if (!is.numeric(labels)) labels <- as.numeric(labels)
        if (!is.matrix(labels) || ncol(labels) != 1L) {
          labels <- matrix(labels, ncol = 1L)
        }
        storage.mode(labels) <- "double"
        
        if (is.null(sample_weights)) {
          sample_weights <- rep(1, nrow(labels))
        }
        sample_weights <- matrix(as.numeric(sample_weights), nrow = nrow(labels), ncol = 1L)
        
      } else {
        stop(sprintf("Unknown CLASSIFICATION_MODE: %s", CLASSIFICATION_MODE))
      }
      
      ## ---------------------------
      ## Normalize dropout list to num_layers
      ## ---------------------------
      self$dropout_rates <- if (is.list(dropout_rates)) dropout_rates else list(dropout_rates)
      if (length(self$dropout_rates) < self$num_layers) {
        self$dropout_rates <- c(self$dropout_rates,
                                rep(list(NULL), self$num_layers - length(self$dropout_rates)))
      } else if (length(self$dropout_rates) > self$num_layers) {
        self$dropout_rates <- self$dropout_rates[1:self$num_layers]
      }
      
      ## ---------------------------
      ## Initialize outputs
      ## ---------------------------
      predicted_output_learn <- NULL
      error_learn <- NULL
      dim_hidden_layers_learn <- list()
      predicted_output_learn_hidden <- NULL
      bias_gradients <- list()
      grads_matrix <- list()
      errors <- list()
      
      ## ---------------------------
      ## Activation functions: show RAW input (exactly as received)
      ## ---------------------------
      cat("=== RAW activation_functions (as passed to learn) ===\n")
      if (is.null(activation_functions)) {
        cat("NULL\n")
      } else if (is.function(activation_functions)) {
        cat("single function: ", .af_name(activation_functions), "\n", sep = "")
      } else if (is.character(activation_functions)) {
        cat("character vector: ", paste(activation_functions, collapse = ", "), "\n", sep = "")
      } else if (is.list(activation_functions)) {
        cat("list len=", length(activation_functions), " (names/types)\n", sep = "")
        for (i in seq_along(activation_functions)) {
          v <- activation_functions[[i]]
          cat(sprintf("  [L%02d] %s\n", i,
                      if (is.function(v)) paste0("fn:", .af_name(v))
                      else if (is.character(v)) paste0("str:", v)
                      else class(v)[1]))
        }
      } else {
        cat("type: ", class(activation_functions)[1], "\n", sep = "")
      }
      cat("=== END RAW ===\n\n")
      
      ## ================================================================
      ## MULTI-LAYER MODE
      ## ================================================================
      if (self$ML_NN) {
        hidden_outputs <- vector("list", self$num_layers)
        activation_derivatives <- vector("list", self$num_layers)
        dropout_masks <- rep(list(NULL), self$num_layers)   # store masks for backprop
        dim_hidden_layers_learn <- vector("list", self$num_layers)
        
        input_matrix <- as.matrix(Rdata)
        
        # Forward pass
        for (layer in 1:self$num_layers) {
          weights_matrix <- as.matrix(self$weights[[layer]])
          bias_vec <- as.numeric(unlist(self$biases[[layer]]))
          input_data <- if (layer == 1) input_matrix else hidden_outputs[[layer - 1]]
          input_data <- as.matrix(input_data)
          
          input_rows <- nrow(input_data)
          weights_rows <- nrow(weights_matrix)
          weights_cols <- ncol(weights_matrix)
          
          cat(sprintf("[Debug] Layer %d : input dim = %d x %d | weights dim = %d x %d\n",
                      layer, input_rows, ncol(input_data), weights_rows, weights_cols))
          
          if (ncol(input_data) != weights_rows) {
            stop(sprintf("Layer %d: input cols (%d) do not match weights rows (%d)",
                         layer, ncol(input_data), weights_rows))
          }
          
          if (length(bias_vec) == 1) {
            bias_matrix <- matrix(bias_vec, nrow = input_rows, ncol = weights_cols)
          } else if (length(bias_vec) == weights_cols) {
            bias_matrix <- matrix(rep(bias_vec, each = input_rows), nrow = input_rows)
          } else if (length(bias_vec) == input_rows * weights_cols) {
            bias_matrix <- matrix(bias_vec, nrow = input_rows)
          } else {
            stop(sprintf("Layer %d: invalid bias shape: length = %d", layer, length(bias_vec)))
          }
          
          Z <- input_data %*% weights_matrix + bias_matrix
          
          # ---- resolve per-layer spec without global normalization ----
          af_spec <- if (is.list(activation_functions)) {
            if (length(activation_functions) >= layer) activation_functions[[layer]] else NULL
          } else if (is.character(activation_functions)) {
            if (length(activation_functions) >= layer) activation_functions[layer] else NULL
          } else if (is.function(activation_functions)) {
            activation_functions
          } else {
            NULL
          }
          activation_function <- .resolve_one(af_spec)
          activation_name <- if (is.function(activation_function)) attr(activation_function, "name") else "none"
          
          # DEBUG: what exactly are we applying?
          if (is.function(activation_function)) {
            cat(sprintf("[Debug] Layer %d : Activation Function = %s (callable)\n", layer, activation_name %||% "unnamed_function"))
          } else if (is.character(activation_function)) { # shouldn't happen after resolve
            cat(sprintf("[Debug] Layer %d : Activation Function (string) = %s\n", layer, paste(activation_function, collapse = ",")))
          } else if (is.null(activation_function)) {
            cat(sprintf("[Debug] Layer %d : Activation Function = NULL (identity)\n", layer))
          } else {
            cat(sprintf("[Debug] Layer %d : Activation placeholder class = %s\n", layer, class(activation_function)[1]))
          }
          
          A <- if (is.function(activation_function)) activation_function(Z) else Z
          
          # Dropout on hidden layers only
          rate <- .safe_get(self$dropout_rates, layer)
          if (layer == self$num_layers) rate <- NULL
          do_out <- self$dropout_forward(A, rate)
          A <- do_out$out
          dropout_masks[[layer]] <- do_out$mask
          
          hidden_outputs[[layer]] <- A
          
          # Derivatives for hidden layers; for output, we’ll handle with CE shortcut
          if (is.function(activation_function)) {
            derivative_name <- paste0(attr(activation_function, "name") %||% "unnamed_function", "_derivative")
            if (!exists(derivative_name, mode = "function")) {
              stop(paste("Layer", layer, ": Activation derivative function", derivative_name, "does not exist."))
            }
            activation_derivatives[[layer]] <- get(derivative_name, mode = "function")(Z)
          } else {
            activation_derivatives[[layer]] <- matrix(1, nrow = nrow(Z), ncol = ncol(Z))
          }
          
          dim_hidden_layers_learn[[layer]] <- dim(A)
        }
        
        predicted_output_learn <- hidden_outputs[[self$num_layers]]
        
        ## ===== MC OUTPUT QUICK ASSERTS =====
        if (identical(CLASSIFICATION_MODE, "multiclass")) {
          P <- predicted_output_learn
          # 1) Activation name on the *output* layer must be softmax for CE shortcut
          af_spec_last <- if (is.list(activation_functions)) activation_functions[[self$num_layers]]
          else if (is.character(activation_functions)) activation_functions[self$num_layers]
          else if (is.function(activation_functions)) activation_functions else NULL
          af_last <- .resolve_one(af_spec_last)
          af_last_name <- if (is.function(af_last)) attr(af_last, "name") else "none"
          cat("[LEARN-MC] output activation:", af_last_name, "\n")
          
          # 2) Non-finite checks and ranges
          cat("[LEARN-MC] P dims:", nrow(P), "x", ncol(P),
              " | finite %:", mean(is.finite(P)), "\n")
          if (any(!is.finite(P))) {
            bad <- which(!is.finite(P), arr.ind = TRUE)
            print(head(bad, 10)); stop("[LEARN-MC] non-finite values in P")
          }
          pr <- range(P[is.finite(P)])
          cat("[LEARN-MC] P range:", paste(pr, collapse=" .. "), "\n")
          
          # 3) If softmax, rows should sum ≈ 1
          if (af_last_name == "softmax") {
            rs <- rowSums(P)
            cat("[LEARN-MC] rowSums(P) min..max:", min(rs), "..", max(rs),
                " | %==1 (±1e-6):", mean(abs(rs - 1) < 1e-6), "\n")
            if (any(!is.finite(rs))) stop("[LEARN-MC] non-finite rowSums(P)")
          }
          
          # 4) Column-order alignment with labels (critical for one-hot)
          if (!is.null(colnames(P)) && !is.null(colnames(labels))) {
            if (!identical(colnames(P), colnames(labels))) {
              cat("[LEARN-MC][WARN] P colnames:", paste(colnames(P), collapse=", "), "\n")
              cat("[LEARN-MC][WARN] Y colnames:", paste(colnames(labels), collapse=", "), "\n")
              # Reorder labels to match P (safer than reordering P)
              common <- intersect(colnames(P), colnames(labels))
              if (length(common) == ncol(P) && length(common) == ncol(labels)) {
                labels <- labels[, colnames(P), drop = FALSE]
                cat("[LEARN-MC] Reordered labels to match P column order.\n")
              } else {
                stop("[LEARN-MC] Incompatible class columns between P and labels")
              }
            }
          } else {
            # If no names, align by index but print class_levels for visibility
            if (!is.null(self$class_levels)) {
              cat("[LEARN-MC] class_levels:", paste(self$class_levels, collapse=", "), "\n")
            }
          }
          
          # 5) Dim checks for sample_weights
          if (!all(dim(labels) == dim(sample_weights))) {
            stop(sprintf("[LEARN-MC] labels dims %dx%d != weights dims %dx%d",
                         nrow(labels), ncol(labels), nrow(sample_weights), ncol(sample_weights)))
          }
        }
        ## ===== END MC OUTPUT QUICK ASSERTS =====
        
        predicted_output_learn_hidden <- hidden_outputs
        
        # Error (kept as (pred - y) * w so we can reuse as CE delta at output)
        error_learn <- (predicted_output_learn - labels) * sample_weights
        
        # Backward pass with CE shortcut at the output
        error_backprop <- error_learn
        for (layer in self$num_layers:1) {
          
          # Use CE shortcut on the output layer IF classification + (sigmoid|softmax)
          use_ce_shortcut <- FALSE
          if (identical(CLASSIFICATION_MODE, "binary") || identical(CLASSIFICATION_MODE, "multiclass")) {
            # resolve again for safety (same logic as forward)
            af_spec <- if (is.list(activation_functions)) {
              if (length(activation_functions) >= layer) activation_functions[[layer]] else NULL
            } else if (is.character(activation_functions)) {
              if (length(activation_functions) >= layer) activation_functions[layer] else NULL
            } else if (is.function(activation_functions)) {
              activation_functions
            } else {
              NULL
            }
            act_fun <- .resolve_one(af_spec)
            act_name <- if (is.function(act_fun)) attr(act_fun, "name") else "none"
            if (layer == self$num_layers && (act_name %in% c("sigmoid", "softmax"))) {
              use_ce_shortcut <- TRUE
            }
          }
          
          if (use_ce_shortcut) {
            delta <- error_learn  # (A_L - Y) * w, no multiply by derivative
            cat(sprintf("[Debug] Layer %d : Using CE shortcut (delta = A - Y)\n", layer))
          } else {
            delta <- error_backprop * activation_derivatives[[layer]]
            cat(sprintf("[Debug] Layer %d : Using derivative-backed delta\n", layer))
          }
          
          # Apply SAME mask/rate as forward for this layer (output layer had rate=NULL)
          rate <- .safe_get(self$dropout_rates, layer)
          mask <- .safe_get(dropout_masks, layer)
          delta <- self$dropout_backward(delta, mask, rate)
          
          # Gradients
          bias_gradients[[layer]] <- matrix(colMeans(delta), nrow = 1)        # average over batch
          input_for_grad <- if (layer == 1) input_matrix else hidden_outputs[[layer - 1]]
          grads_matrix[[layer]] <- t(input_for_grad) %*% delta                # weight grads
          
          errors[[layer]] <- delta
          
          # Propagate to previous layer
          if (layer > 1) {
            weights_t <- t(as.matrix(self$weights[[layer]]))
            error_backprop <- delta %*% weights_t
          }
        }
        
        ## ================================================================
        ## SINGLE-LAYER MODE
        ## ================================================================
      } else {
        cat("Single Layer Learning Phase\n")
        
        X <- as.matrix(Rdata)
        weights_matrix <- if (is.list(self$weights)) {
          as.matrix(self$weights[[1]])
        } else {
          as.matrix(self$weights)
        }
        bias_vec <- if (is.list(self$biases)) {
          as.numeric(unlist(self$biases[[1]]))
        } else {
          as.numeric(self$biases)
        }
        
        if (ncol(X) != nrow(weights_matrix)) {
          stop(sprintf("SL NN: input cols (%d) do not match weights rows (%d)", ncol(X), nrow(weights_matrix)))
        }
        
        if (length(bias_vec) == 1) {
          bias_matrix <- matrix(bias_vec, nrow = nrow(X), ncol = ncol(weights_matrix))
        } else if (length(bias_vec) == ncol(weights_matrix)) {
          bias_matrix <- matrix(rep(bias_vec, each = nrow(X)), nrow = nrow(X))
        } else if (length(bias_vec) == nrow(X) * ncol(weights_matrix)) {
          bias_matrix <- matrix(bias_vec, nrow = nrow(X))
        } else {
          stop(sprintf("SL NN: invalid bias shape: length = %d", length(bias_vec)))
        }
        
        # Optional input dropout (SL)
        if (!is.list(self$dropout_rates)) self$dropout_rates <- list(self$dropout_rates)
        rate <- .safe_get(self$dropout_rates, 1)
        do_x <- self$dropout_forward(X, rate)
        X_dropped <- do_x$out
        mask <- do_x$mask
        
        Z <- X_dropped %*% weights_matrix + bias_matrix
        
        # Resolve SL activation without normalization
        af_spec <- if (is.list(activation_functions)) {
          activation_functions[[1]]
        } else if (is.character(activation_functions)) {
          activation_functions[1]
        } else if (is.function(activation_functions)) {
          activation_functions
        } else {
          NULL
        }
        activation_function <- .resolve_one(af_spec)
        activation_name <- if (is.function(activation_function)) attr(activation_function, "name") else "none"
        
        # DEBUG: what are we using in SL?
        if (is.function(activation_function)) {
          cat(sprintf("[Debug] SL : Activation Function = %s (callable)\n", .af_name(activation_function)))
        } else if (is.character(activation_function)) {
          cat(sprintf("[Debug] SL : Activation Function (string) = %s\n", paste(activation_function, collapse = ",")))
        } else if (is.null(activation_function)) {
          cat("[Debug] SL : Activation Function = NULL (identity)\n")
        } else {
          cat(sprintf("[Debug] SL : Activation placeholder class = %s\n", class(activation_function)[1]))
        }
        
        A <- if (is.function(activation_function)) activation_function(Z) else Z
        predicted_output_learn <- A
        
        ## Mirror ML shapes for downstream consumers
        predicted_output_learn_hidden <- list(A)
        
        if (identical(CLASSIFICATION_MODE, "multiclass") && ncol(predicted_output_learn) != ncol(labels)) {
          stop(sprintf("SL NN (multiclass): output cols (%d) != label cols (%d).",
                       ncol(predicted_output_learn), ncol(labels)))
        }
        
        # Error
        error_learn <- (predicted_output_learn - labels) * sample_weights
        dim_hidden_layers_learn[[1]] <- dim(predicted_output_learn)
        
        # CE shortcut at output if (binary/multiclass) & (sigmoid/softmax)
        use_ce_shortcut <- FALSE
        if (identical(CLASSIFICATION_MODE, "binary") || identical(CLASSIFICATION_MODE, "multiclass")) {
          if (activation_name %in% c("sigmoid", "softmax")) use_ce_shortcut <- TRUE
        }
        
        if (use_ce_shortcut) {
          delta <- error_learn
          cat("[Debug] SL : Using CE shortcut (delta = A - Y)\n")
        } else {
          deriv_fn_name <- if (is.function(activation_function)) paste0(attr(activation_function, "name"), "_derivative") else NULL
          activation_deriv <- if (!is.null(deriv_fn_name) && exists(deriv_fn_name)) {
            get(deriv_fn_name)(Z)
          } else {
            matrix(1, nrow = nrow(Z), ncol = ncol(Z))
          }
          delta <- error_learn * activation_deriv
          cat("[Debug] SL : Using derivative-backed delta\n")
        }
        
        # Backprop through dropout ONLY if mask matches delta's shape
        if (!is.null(rate) && !is.null(mask) && is.matrix(mask) && all(dim(mask) == dim(delta))) {
          delta <- self$dropout_backward(delta, mask, rate)
        }
        
        errors <- vector("list", max(1L, self$num_layers))
        grads_matrix <- vector("list", max(1L, self$num_layers))
        bias_gradients <- vector("list", max(1L, self$num_layers))
        
        bias_gradients[[1]] <- matrix(colMeans(delta), nrow = 1)
        grads_matrix[[1]] <- t(X_dropped) %*% delta
        errors[[1]] <- delta
      }
      
      learn_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      if (verbose) { print("----------------------------------------learn-end----------------------------------------") }
      
      return(list(
        learn_output = predicted_output_learn,
        learn_time = learn_time,
        error = error_learn,
        dim_hidden_layers = dim_hidden_layers_learn,
        hidden_outputs = predicted_output_learn_hidden,
        grads_matrix = grads_matrix,
        bias_gradients = bias_gradients,
        errors = errors
      ))
    },
    # Method to perform prediction
    predict = function(Rdata, weights, biases, activation_functions_predict, verbose = FALSE, debug = FALSE) {
      if (verbose) { print("----------------------------------------predict-begin----------------------------------------") }
      
      # ---- Debug/Verbose toggles ----
      if (is.null(debug)) {
        debug <- isTRUE(get0("DEBUG_PREDICT_FORWARD", inherits = TRUE, ifnotfound = FALSE))
      }
      .dbg <- function(...) if (isTRUE(debug))   cat("[PRED-DBG] ", sprintf(...), "\n", sep = "")
      .vbs <- function(...) if (isTRUE(verbose) && !isTRUE(debug)) cat(sprintf(...), "\n")
      
      # ---------- last-layer variance probe (local helper) ----------
      probe_last_layer <- function(Z_last, A_last, last_af_name = NA_character_, tag = "[PROBE]") {
        vZ  <- as.vector(Z_last); vA <- as.vector(A_last)
        sdZ <- stats::sd(vZ, na.rm = TRUE); sdA <- stats::sd(vA, na.rm = TRUE)
        rngZ <- range(vZ, na.rm = TRUE); rngA <- range(vA, na.rm = TRUE)
        
        if (isTRUE(debug)) {
          cat(sprintf(
            "%s last_af=%s | sd(Z)=%.6g | sd(A)=%.6g | range(Z)=[%.6g, %.6g] | range(A)=[%.6g, %.6g]\n",
            tag, as.character(last_af_name), sdZ, sdA, rngZ[1], rngZ[2], rngA[1], rngA[2]
          ))
          eps_flat <- 1e-6
          if (sdZ < eps_flat) {
            cat(sprintf("%s DIAG: Z_last is ~flat -> possible collapse.\n", tag))
          } else if (sdA < sdZ * 1e-3) {
            cat(sprintf("%s DIAG: Z_last spread but A_last squashed -> activation/head mismatch.\n", tag))
          } else {
            cat(sprintf("%s DIAG: Variance preserved across head.\n", tag))
          }
        } else if (isTRUE(verbose)) {
          cat(sprintf("%s head=%s | sd(Z)=%.6g → sd(A)=%.6g\n", tag, as.character(last_af_name), sdZ, sdA))
        }
        invisible(list(sdZ = sdZ, sdA = sdA, rngZ = rngZ, rngA = rngA))
      }
      # --------------------------------------------------------------
      
      # Fallback to internal state (stateful mode) if weights/biases are missing
      if (is.null(weights)) {
        if (!is.null(self$weights)) weights <- self$weights else stop("predict(): weights not provided and self$weights is NULL.")
      }
      if (is.null(biases)) {
        if (!is.null(self$biases))  biases  <- self$biases  else stop("predict(): biases not provided and self$biases is NULL.")
      }
      
      # Ensure list shapes
      if (!is.list(weights)) weights <- list(weights)
      if (!is.list(biases))  biases  <- list(biases)
      
      # ---------------- Activation resolver (minimal, local) ----------------
      .resolve_one <- function(x) {
        if (is.null(x) || is.function(x)) {
          if (is.function(x) && is.null(attr(x, "name"))) attr(x, "name") <- "unnamed_function"
          return(x)
        }
        if (is.character(x) && length(x) >= 1L) {
          key <- tolower(x[1]); key <- gsub("[- ]", "_", key, perl = TRUE)
          if (key %in% c("linear", "none")) key <- "identity"
          if (key == "logistic") key <- "sigmoid"
          fn <- switch(key,
                       "relu"=relu, "tanh"=tanh, "sigmoid"=sigmoid, "softmax"=softmax, "identity"=identity,
                       "leaky_relu"=leaky_relu, "elu"=elu, "swish"=swish, "gelu"=gelu, "selu"=selu, "mish"=mish,
                       "hard_sigmoid"=hard_sigmoid, "softplus"=softplus, "prelu"=prelu,
                       "bent_identity"=bent_identity, "maxout"=maxout,
                       NULL
          )
          if (is.null(fn)) stop(sprintf("Unsupported activation: '%s'", x[1]), call. = FALSE)
          if (is.null(attr(fn, "name"))) attr(fn, "name") <- key
          return(fn)
        }
        stop(sprintf("Unsupported activation spec type: %s", class(x)[1]), call. = FALSE)
      }
      
      # Choose the prediction-time activation spec
      acts_pred <- if (!missing(activation_functions_predict) && !is.null(activation_functions_predict)) {
        activation_functions_predict
      } else if (!is.null(self$activation_functions_predict)) {
        self$activation_functions_predict
      } else if (!is.null(self$activation_functions)) {
        self$activation_functions
      } else {
        NULL
      }
      
      # Print RAW activations passed to predict (before resolving)
      cat("=== RAW activation_functions_predict (as passed to predict) ===\n")
      if (is.null(acts_pred)) {
        cat("NULL\n")
      } else if (is.function(acts_pred)) {
        nm <- attr(acts_pred, "name"); if (is.null(nm)) nm <- "unnamed_function"
        cat("single function: ", nm, "\n", sep = "")
      } else if (is.character(acts_pred)) {
        cat("character vector: ", paste(acts_pred, collapse = ", "), "\n", sep = "")
      } else if (is.list(acts_pred)) {
        cat("list len=", length(acts_pred), " (names/types)\n", sep = "")
        for (i in seq_along(acts_pred)) {
          v <- acts_pred[[i]]
          cat(sprintf("  [L%02d] %s\n", i,
                      if (is.function(v)) paste0("fn:", (attr(v,"name") %||% "unnamed_function"))
                      else if (is.character(v)) paste0("str:", v)
                      else class(v)[1]))
        }
      } else {
        cat("type: ", class(acts_pred)[1], "\n", sep = "")
      }
      cat("=== END RAW ===\n\n")
      
      # Helper to fetch spec for a given layer and resolve it
      .get_act <- function(layer) {
        spec <- if (is.list(acts_pred)) {
          if (length(acts_pred) >= layer) acts_pred[[layer]] else NULL
        } else if (is.character(acts_pred)) {
          if (length(acts_pred) >= layer) acts_pred[layer] else NULL
        } else if (is.function(acts_pred)) {
          acts_pred
        } else {
          NULL
        }
        .resolve_one(spec)
      }
      # ---------------------------------------------------------------------
      
      start_time  <- Sys.time()
      output      <- as.matrix(Rdata)
      num_layers  <- length(weights)
      
      # Input diagnostics
      .dbg("INPUT dims=%d x %d | mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
           nrow(output), ncol(output),
           mean(output), stats::sd(as.vector(output)), min(output),
           stats::median(as.vector(output)), max(output))
      .vbs("Predict: X dims=%d x %d", nrow(output), ncol(output))
      
      # For final head debug after the loop
      last_w <- NULL; last_b <- NULL; last_Z <- NULL; last_A <- NULL; last_act_name <- "identity"
      
      for (layer in seq_len(num_layers)) {
        w <- as.matrix(weights[[layer]])
        b <- as.numeric(unlist(biases[[layer]]))
        
        # Broadcast bias to match samples × units
        n_samples <- nrow(output)
        n_units   <- ncol(w)
        if (length(b) == 1) {
          bias_mat <- matrix(b, nrow = n_samples, ncol = n_units, byrow = TRUE)
        } else if (length(b) == n_units) {
          bias_mat <- matrix(b, nrow = n_samples, ncol = n_units, byrow = TRUE)
        } else {
          bias_mat <- matrix(rep(b, length.out = n_units), nrow = n_samples, ncol = n_units, byrow = TRUE)
        }
        
        # Weights/bias debug info
        .dbg("L%02d: W dims=%d x %d | W mean=%.6g sd=%.6g min=%.6g max=%.6g",
             layer, nrow(w), ncol(w), mean(w), stats::sd(as.vector(w)), min(w), max(w))
        
        # Concise bias summary
        if (isTRUE(debug) || isTRUE(verbose)) {
          b_sd   <- if (length(b) > 1) stats::sd(b) else 0
          b_min  <- if (length(b) > 0) min(b) else NA_real_
          b_max  <- if (length(b) > 0) max(b) else NA_real_
          if (isTRUE(debug)) {
            b_head <- paste(utils::head(round(b, 6), 6), collapse = ", ")
            cat(sprintf("[BIAS] L%02d: len=%d | mean=%.6g sd=%.6g | range=[%s, %s] | head=[%s]\n",
                        layer, length(b), mean(b), b_sd,
                        format(b_min, digits = 6), format(b_max, digits = 6), b_head))
          } else {
            cat(sprintf("[BIAS] L%02d: len=%d | mean=%.6g sd=%.6g | range=[%s, %s]\n",
                        layer, length(b), mean(b), b_sd,
                        format(b_min, digits = 6), format(b_max, digits = 6)))
          }
        }
        
        # Linear transformation
        output <- output %*% w + bias_mat
        
        # Pre-activation stats
        .dbg("L%02d: Z dims=%d x %d | Z mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
             layer, nrow(output), ncol(output),
             mean(output), stats::sd(as.vector(output)), min(output),
             stats::median(as.vector(output)), max(output))
        
        # Per-layer probe (before activation)
        if (isTRUE(debug) || isTRUE(verbose)) {
          sdZ  <- stats::sd(as.vector(output))
          rngZ <- range(as.vector(output))
          if (isTRUE(debug)) {
            cat(sprintf("[L%02d-PROBE] sd(Z)=%.6g | range(Z)=[%.6g, %.6g]\n", layer, sdZ, rngZ[1], rngZ[2]))
          } else {
            cat(sprintf("[L%02d] sd(Z)=%.6g\n", layer, sdZ))
          }
        }
        
        Z_curr <- output  # cache before activation
        act_fn <- .get_act(layer)
        act_name <- if (is.function(act_fn)) (attr(act_fn, "name") %||% "function") else "identity"
        
        # Apply activation
        if (is.function(act_fn)) {
          cat(sprintf("[PRED] Layer %d activation = %s\n", layer, act_name))
          output <- act_fn(output)
          # After-activation probe
          if (isTRUE(debug) || isTRUE(verbose)) {
            sdA  <- stats::sd(as.vector(output)); rngA <- range(as.vector(output))
            if (isTRUE(debug)) {
              cat(sprintf("[L%02d-PROBE] sd(A)=%.6g | range(A)=[%.6g, %.6g]\n", layer, sdA, rngA[1], rngA[2]))
            } else {
              cat(sprintf("[L%02d] sd(A)=%.6g\n", layer, sdA))
            }
          }
        } else {
          cat(sprintf("[PRED] Layer %d activation = identity (NULL)\n", layer))
          # output already equals Z_curr
          if (isTRUE(debug) || isTRUE(verbose)) {
            sdA  <- stats::sd(as.vector(output)); rngA <- range(as.vector(output))
            if (isTRUE(debug)) {
              cat(sprintf("[L%02d-PROBE] sd(A)=%.6g | range(A)=[%.6g, %.6g]\n", layer, sdA, rngA[1], rngA[2]))
            } else {
              cat(sprintf("[L%02d] sd(A)=%.6g\n", layer, sdA))
            }
          }
        }
        
        # Stash for final head diagnostics
        if (layer == num_layers) {
          last_w <- w; last_b <- b; last_Z <- Z_curr; last_A <- output; last_act_name <- act_name
          probe_last_layer(Z_last = last_Z, A_last = last_A,
                           last_af_name = tolower(last_act_name), tag = "[PROBE]")
        }
      }
      
      # Final head diagnostics
      if (num_layers >= 1) {
        cat("\n[HEAD-DBG] ---- Last layer diagnostic ----\n")
        cat(sprintf("[HEAD-DBG] W_last dims=%d x %d | mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    nrow(last_w), ncol(last_w), mean(last_w), sd(as.vector(last_w)), min(last_w), max(last_w)))
        cat(sprintf("[HEAD-DBG] b_last len=%d | mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    length(last_b), mean(last_b), if (length(last_b)>1) sd(last_b) else 0, min(last_b), max(last_b)))
        cat(sprintf("[HEAD-DBG] Z_last: mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    mean(last_Z), sd(as.vector(last_Z)), min(last_Z), max(last_Z)))
        cat(sprintf("[HEAD-DBG] A_last: mean=%.6f sd=%.6f min=%.6f max=%.6f\n\n",
                    mean(last_A), sd(as.vector(last_A)), min(last_A), max(last_A)))
        probe_last_layer(Z_last = last_Z, A_last = last_A,
                         last_af_name = last_act_name, tag = "[PROBE]")
      }
      
      end_time <- Sys.time()
      prediction_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      .dbg("DONE | total_time=%.6fs | FINAL dims=%d x %d | mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
           prediction_time, nrow(output), ncol(output),
           mean(output), stats::sd(as.vector(output)), min(output),
           stats::median(as.vector(output)), max(output))
      
      if (isTRUE(verbose) && !isTRUE(debug)) {
        cat(sprintf("Predict complete in %.4fs | Output dims=%d x %d\n",
                    prediction_time, nrow(output), ncol(output)))
      }
      
      print("----------------------------------------predict-end----------------------------------------")
      
      # Return both keys for compatibility with existing codepaths
      return(list(
        prediction = output,
        predicted_output = output,
        prediction_time = prediction_time
      ))
    }
    ,# Method for training the SONN with L2 regularization
    train_network = function(Rdata, labels,  X_train = NULL, y_train = NULL, lr, num_networks, CLASSIFICATION_MODE, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, do_ensemble, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type, sample_weights, X_validation, y_validation, validation_metrics, threshold_function, ML_NN, train, verbose, output_root = NULL) {
      
      print("----------------------------------------train_network-begin----------------------------------------")
      start_time <- Sys.time()
      
      # ------------------------------------------------------------------
      # Local copies of the training split
      # ------------------------------------------------------------------
      if (is.null(X_train)) X_train <- Rdata
      if (is.null(y_train)) y_train <- labels
      
      # ----------------------------
      # State/optimizer
      # ----------------------------
      prev_weights <- NULL
      prev_biases  <- NULL
      optimizer_params_weights <- vector("list", self$num_layers)
      optimizer_params_biases  <- vector("list", self$num_layers)
      
      best_train_loss          <- Inf
      best_epoch_train_loss    <- NA_integer_
      best_train_acc           <- -Inf
      best_epoch_train         <- NA_integer_
      best_val_acc             <- -Inf
      best_val_epoch           <- NA_integer_
      best_val_prediction_time <- NA_real_
      
      # === REGRESSION-ONLY tracking (mirrors v2: select best by validation loss) ===
      best_val_loss        <- Inf
      best_val_epoch_loss  <- NA_integer_
      
      # --- persist across epochs (needed so mid-epoch best isn't lost) ---
      best_weights        <- NULL
      best_biases         <- NULL
      best_val_probs      <- NULL
      best_val_labels     <- NULL
      best_val_n_eff      <- NA_integer_
      last_val_predict    <- NULL
      last_train_predict  <- NULL
      
      predicted_output_val <- NULL
      
      val_accuracy_log <- c()
      train_accuracy_log <- c()
      loss_log <- c()
      learning_rate_log <- c()
      val_loss_log <- c()
      train_loss_log <- c()
      mean_output_log <- c()
      sd_output_log <- c()
      max_weight_log <- c()
      
      losses <- numeric(num_epochs)
      
      total_learn_time <- 0
      
      # ======== TRAIN LOOP ========
      if (train) {
        
        # --- LOGS (empty at start, will fill each epoch) ---
        train_accuracy_log <- numeric(0)
        train_loss_log     <- numeric(0)
        mean_output_log    <- numeric(0)
        sd_output_log      <- numeric(0)
        max_weight_log     <- numeric(0)
        val_accuracy_log   <- numeric(0)
        val_loss_log       <- numeric(0)
        
        for (epoch in 1:num_epochs) {
          
          # lr <- lr_scheduler(epoch)
          cat("Epoch:", epoch, "| Learning Rate:", lr, "\n")
          num_epochs_check <<- num_epochs
          
          # 1) Train step
          learn_result <- self$learn(
            Rdata = Rdata,
            labels = labels,
            lr = lr,
            CLASSIFICATION_MODE = CLASSIFICATION_MODE,
            activation_functions = activation_functions,
            dropout_rates = dropout_rates,
            sample_weights = sample_weights
          )
          
          # --- Error debug ---
          if (!is.null(learn_result$error)) {
            cat("[TRAIN-DBG] last-layer error summary ->",
                " min=", min(learn_result$error, na.rm=TRUE),
                " mean=", mean(learn_result$error, na.rm=TRUE),
                " max=", max(learn_result$error, na.rm=TRUE),
                " sd=",  sd(learn_result$error, na.rm=TRUE), "\n")
          }
          
          # =========================
          # TRAINING METRICS (mode-aware)
          # =========================
          
          probs_train <- learn_result$learn_output
          probs_train <- as.matrix(probs_train)
          storage.mode(probs_train) <- "double"
          
          n <- nrow(probs_train)
          K <- max(1L, ncol(probs_train))
          
          # Align labels to n rows (trim only; no recycling)
          labels_epoch <- if (is.matrix(labels)) {
            if (nrow(labels) == n) labels else labels[seq_len(min(nrow(labels), n)), , drop = FALSE]
          } else if (is.data.frame(labels)) {
            v <- labels[[1]]
            v[seq_len(min(length(v), n))]
          } else {
            labels[seq_len(min(length(labels), n))]
          }
          
          # Build targets like in validation
          targs_tr <- .build_targets(labels_epoch, n, K, CLASSIFICATION_MODE)
          
          # Compute metrics by mode
          if (identical(CLASSIFICATION_MODE, "multiclass")) {
            stopifnot(K >= 2)
            pred_idx_tr   <- max.col(probs_train, ties.method = "first")
            train_accuracy <- mean(pred_idx_tr == targs_tr$y_idx, na.rm = TRUE)
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss <- .ce_loss_multiclass(probs_train, targs_tr$Y)
            } else {
              train_loss <- mean((probs_train - targs_tr$Y)^2, na.rm = TRUE)
            }
            
          } else if (identical(CLASSIFICATION_MODE, "binary")) {
            stopifnot(K == 1)
            preds_bin_tr   <- as.integer(probs_train >= 0.5)
            train_accuracy <- mean(preds_bin_tr == targs_tr$y, na.rm = TRUE)
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss <- .bce_loss(probs_train, targs_tr$y)
            } else {
              train_loss <- mean((probs_train - matrix(targs_tr$y, ncol = 1))^2, na.rm = TRUE)
            }
            
          } else if (identical(CLASSIFICATION_MODE, "regression")) {
            # No accuracy for regression; report NA and proper loss instead
            y_reg     <- if (is.matrix(labels_epoch)) as.numeric(labels_epoch[,1]) else as.numeric(labels_epoch)
            preds_reg <- as.numeric(probs_train[,1])
            train_loss <- mean((preds_reg - y_reg)^2, na.rm = TRUE)
            train_accuracy <- NA_real_
            
          } else {
            stop("Unknown CLASSIFICATION_MODE.")
          }
          
          # Track best loss (used for regression; harmless for classification)
          if (is.finite(train_loss) && train_loss < best_train_loss) {
            best_train_loss       <- train_loss
            best_epoch_train_loss <- epoch
          }
          
          # Log metrics (single source of truth)
          train_accuracy_log <- c(train_accuracy_log, train_accuracy)
          train_loss_log     <- c(train_loss_log,     train_loss)
          
          # Track best training accuracy only when defined (classification)
          if (!is.na(train_accuracy) && (is.na(best_train_acc) || train_accuracy > best_train_acc)) {
            best_train_acc   <- train_accuracy
            best_epoch_train <- epoch
          }
          
          cat(sprintf(
            "Epoch %d | Train %s: %s | Loss: %.6f\n",
            epoch,
            if (identical(CLASSIFICATION_MODE, "regression")) "R²/Acc" else "Accuracy",
            if (is.na(train_accuracy)) "NA" else sprintf("%.2f%%", 100 * train_accuracy),
            train_loss
          ))
          
          predicted_output_train_reg <- learn_result
          predicted_output_train_reg_prediction_time <- learn_result$learn_time
          
          # Predicted output (use correct output layer)
          if (self$ML_NN) {
            predicted_output <- predicted_output_train_reg$hidden_outputs[[self$num_layers]]
          } else {
            predicted_output <- predicted_output_train_reg$learn_output
          }
          
          # Output saturation diagnostics
          mean_output <- mean(predicted_output)
          sd_output   <- sd(predicted_output)
          cat("Mean Output:", round(mean_output, 4), "| StdDev:", round(sd_output, 4), "\n")
          mean_output_log <- c(mean_output_log, mean_output)
          sd_output_log   <- c(sd_output_log, sd_output)
          
          # Weight explosion diagnostics
          if (exists("best_weights_record")) {
            max_weight <- max(sapply(best_weights_record, function(w) max(abs(w))))
            cat("Max Weight Abs:", round(max_weight, 4), "\n")
          } else {
            max_weight <- NA
          }
          max_weight_log <- c(max_weight_log, max_weight)
          
          ## =========================
          ## SONN — Top 3 Per-epoch Plots (original scaffolding)
          ## =========================
          if (is.null(self$PerEpochlViewPlotsConfig)) self$PerEpochlViewPlotsConfig <- list()
          .fix_flag <- function(v, default) { if (isTRUE(v)) TRUE else if (isFALSE(v)) FALSE else default }
          defaults <- list(accuracy_plot = FALSE, saturation_plot = FALSE, max_weight_plot = FALSE, viewAllPlots = FALSE, verbose = FALSE)
          for (nm in names(defaults)) {
            self$PerEpochlViewPlotsConfig[[nm]] <- .fix_flag(self$PerEpochlViewPlotsConfig[[nm]], defaults[[nm]])
          }
          pe <- self$PerEpochlViewPlotsConfig
          message(sprintf("SONN per-epoch flags → acc=%s, sat=%s, max=%s, all=%s, verbose=%s",
                          pe$accuracy_plot, pe$saturation_plot, pe$max_weight_plot, pe$viewAllPlots, pe$verbose))
          message(sprintf("SONN gate eval → acc=%s, sat=%s, max=%s",
                          self$viewPerEpochPlots("accuracy_plot"),
                          self$viewPerEpochPlots("saturation_plot"),
                          self$viewPerEpochPlots("max_weight_plot")))
          plots_dir <- ddesonn_plots_dir(output_root)
          ens <- as.integer(if (!is.null(self$ensemble_number)) self$ensemble_number else get0("ensemble_number", 1L))
          mod <- as.integer(if (exists("model_iter_num", inherits = TRUE)) model_iter_num else get0("model_iter_num", 1L))
          
          # =========================
          # MERGED-IN — gradients, normalization, BLOCK A metrics
          # =========================
          
          # ---- Gradients & misc outputs from learn() (normalize shapes early) ----
          weight_gradients_raw <- learn_result$grads_matrix
          bias_gradients_raw   <- learn_result$bias_gradients
          errors               <- learn_result$errors
          error                <- learn_result$error
          dim_hidden_layers    <- learn_result$dim_hidden_layers
          
          .as_list <- function(x, L) {
            if (is.null(x)) {
              vector("list", L)
            } else if (is.list(x)) {
              if (length(x) < L) c(x, vector("list", L - length(x))) else x[seq_len(L)]
            } else {
              lst <- vector("list", L); lst[[L]] <- x; lst
            }
          }
          
          weight_gradients <- .as_list(weight_gradients_raw, if (isTRUE(self$ML_NN)) self$num_layers else 1L)
          bias_gradients   <- .as_list(bias_gradients_raw,   if (isTRUE(self$ML_NN)) self$num_layers else 1L)
          
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
          
          cat(sprintf("Grad norm L1 by layer: %s\n",
                      paste(vapply(weight_gradients, function(G) sum(abs(G), na.rm=TRUE), numeric(1)),
                            collapse=" | ")))
          
          # 2) Final head / predictions (already set as probs_train above)
          storage.mode(probs_train) <- "double"
          n <- nrow(probs_train); K <- ncol(probs_train)
          
          # 3) Align labels for this epoch (trim to n; no padding)
          if (is.matrix(labels)) {
            labels_epoch <- if (nrow(labels) == n) labels else labels[seq_len(n), , drop = FALSE]
          } else {
            labels_epoch <- if (length(labels) == n) labels else labels[seq_len(min(length(labels), n))]
          }
          if (CLASSIFICATION_MODE == "binary") {
            predictions <- probs_train
          } else if (CLASSIFICATION_MODE == "multiclass") {
            predictions <- probs_train
          } else if (CLASSIFICATION_MODE == "regression") {
            predictions <- probs_train
          } else stop("Unknown CLASSIFICATION_MODE")
          
          # =========================
          # BLOCK A — Accuracy & Saturation (aware)
          # =========================
          cat(sprintf("[dbg] BLOCK A: n=%d, K=%d | probs_train range=[%.6f, %.6f]\n",
                      n, K, min(probs_train), max(probs_train)))
          cat("[dbg] BLOCK A: CLASSIFICATION_MODE =", CLASSIFICATION_MODE, "\n")
          
          if (identical(CLASSIFICATION_MODE, "multiclass")) {
            targs <- .build_targets(labels_epoch, n, K, CLASSIFICATION_MODE)
            stopifnot(K >= 2)
            pred_idx <- max.col(probs_train, ties.method = "first")
            cat("[dbg] BLOCK A: pred_idx head =", paste(utils::head(pred_idx, 6), collapse=", "), "\n")
            cat("[dbg] BLOCK A: lbl_idx head  =", paste(utils::head(targs$y_idx, 6), collapse=", "), "\n")
            train_accuracy_blockA <- mean(pred_idx == targs$y_idx, na.rm = TRUE)
            train_accuracy_log    <- c(train_accuracy_log, train_accuracy_blockA)
            cat(sprintf("[dbg] BLOCK A: train_accuracy=%.6f\n", train_accuracy_blockA))
            
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss_blockA <- .ce_loss_multiclass(probs_train, targs$Y)
              cat(sprintf("[dbg] BLOCK A: CE loss=%.6f\n", train_loss_blockA))
            } else {
              train_loss_blockA <- mean((probs_train - targs$Y)^2, na.rm = TRUE)
              cat(sprintf("[dbg] BLOCK A: MSE loss=%.6f\n", train_loss_blockA))
            }
            train_loss_log <- c(train_loss_log, train_loss_blockA)
            
          } else if (identical(CLASSIFICATION_MODE, "binary")) {
            targs <- .build_targets(labels_epoch, n, K, CLASSIFICATION_MODE)
            stopifnot(K == 1)
            preds_bin_blockA <- as.integer(probs_train >= 0.5)
            cat("[dbg] BLOCK A: preds_bin head =", paste(utils::head(preds_bin_blockA, 6), collapse=", "), "\n")
            cat("[dbg] BLOCK A: y head        =", paste(utils::head(targs$y, 6), collapse=", "), "\n")
            train_accuracy_blockA <- mean(preds_bin_blockA == targs$y, na.rm = TRUE)
            train_accuracy_log    <- c(train_accuracy_log, train_accuracy_blockA)
            cat(sprintf("[dbg] BLOCK A: train_accuracy=%.6f\n", train_accuracy_blockA))
            
            if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
              train_loss_blockA <- .bce_loss(probs_train, targs$y)
              cat(sprintf("[dbg] BLOCK A: BCE loss=%.6f\n", train_loss_blockA))
            } else {
              train_loss_blockA <- mean((probs_train - matrix(targs$y, ncol=1))^2, na.rm = TRUE)
              cat(sprintf("[dbg] BLOCK A: MSE loss=%.6f\n", train_loss_blockA))
            }
            train_loss_log <- c(train_loss_log, train_loss_blockA)
            
          } else if (identical(CLASSIFICATION_MODE, "regression")) {
            y_reg    <- if (is.matrix(labels_epoch)) as.numeric(labels_epoch[,1]) else as.numeric(labels_epoch)
            preds_reg <- as.numeric(probs_train[,1])
            cat("[dbg] BLOCK A: y_reg head     =", paste(utils::head(y_reg, 6), collapse=", "), "\n")
            cat("[dbg] BLOCK A: preds_reg head =", paste(utils::head(preds_reg, 6), collapse=", "), "\n")
            
            train_loss_blockA <- mean((preds_reg - y_reg)^2, na.rm = TRUE)
            cat(sprintf("[dbg] BLOCK A: MSE loss=%.6f\n", train_loss_blockA))
            train_loss_log <- c(train_loss_log, train_loss_blockA)
            
            train_accuracy_blockA <- NA_real_
            train_accuracy_log    <- c(train_accuracy_log, NA_real_)
            cat("[dbg] BLOCK A: train_accuracy=NA (regression)\n")
            
            mae  <- mean(abs(preds_reg - y_reg), na.rm = TRUE)
            vary <- stats::var(y_reg, na.rm = TRUE)
            r2   <- if (is.finite(vary) && vary > 0) 1 - train_loss_blockA / vary else NA_real_
            cat(sprintf("[dbg] BLOCK A: MAE=%.6f | R^2=%s\n", mae, ifelse(is.na(r2), "NA", sprintf("%.6f", r2))))
          } else {
            stop("Unknown CLASSIFICATION_MODE. Use 'multiclass', 'binary', or 'regression'.")
          }
          
          # keep your best tracker (already updated earlier by original block); optional reinforce:
          if (is.na(best_train_acc) || (!is.na(train_accuracy) && train_accuracy > best_train_acc)) {
            best_train_acc   <- train_accuracy
            best_epoch_train <- epoch
            cat(sprintf("[dbg] BLOCK A: new best_train_acc=%.6f at epoch=%d\n", best_train_acc, best_epoch_train))
          }
          
          # Saturation stats already computed above from predicted_output
          cat(sprintf("[dbg] BLOCK A: saturation mean=%.6f | sd=%.6f\n", mean_output, sd_output))
          
          # -----------------------------------------
          # (unchanged) plotting & filename handling
          # -----------------------------------------
          fname <- make_fname_prefix(
            do_ensemble     = isTRUE(get0("do_ensemble", ifnotfound = FALSE)),
            num_networks    = get0("num_networks", ifnotfound = NULL),
            total_models    = if (!is.null(self$ensemble)) length(self$ensemble) else get0("num_networks", ifnotfound = NULL),
            ensemble_number = if (exists("self", inherits = TRUE)) self$ensemble_number else get0("ensemble_number", ifnotfound = NULL),
            model_index     = get0("model_iter_num", ifnotfound = NULL),
            who             = "SONN"
          )
          cat("[fname probe] -> ", fname("probe.png"), "\n")
          
          plots_dir <- ddesonn_plots_dir(output_root)
          
          plot_title_prefix <- if (isTRUE(get0("do_ensemble", ifnotfound = FALSE))) {
            sprintf("DDESONN%s SONN%s | lr: %s | lambda: %s",
                    if (!is.null(self$ensemble_number)) paste0(" ", self$ensemble_number) else "",
                    if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                    lr, self$lambda)
          } else {
            sprintf("SONN%s | lr: %s | lambda: %s",
                    if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                    lr, self$lambda)
          }
          
          pad <- function(x, n){ length(x) <- n; x }
          nA  <- max(length(train_accuracy_log), length(train_loss_log),
                     length(mean_output_log), length(sd_output_log))
          df_accsat <- data.frame(
            Epoch      = seq_len(nA),
            Accuracy   = pad(train_accuracy_log, nA),
            Loss       = pad(train_loss_log, nA),
            MeanOutput = pad(mean_output_log, nA),
            StdOutput  = pad(sd_output_log, nA)
          )
          
          if (self$viewPerEpochPlots("accuracy_plot")) {
            tryCatch({
              p <- ggplot2::ggplot(df_accsat, ggplot2::aes(x = Epoch)) +
                ggplot2::geom_line(ggplot2::aes(y = Accuracy), size = 1) +
                ggplot2::geom_line(ggplot2::aes(y = Loss),     size = 1) +
                ggplot2::labs(title = paste(plot_title_prefix, "— Training Accuracy (blue) & Loss (red)"),
                              y = "Value") +
                ggplot2::theme_minimal() +
                ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold", size = 16))
              out <- file.path(plots_dir, fname("training_accuracy_loss_plot.png"))
              message("📸 save: ", out)
              ggplot2::ggsave(filename = out, plot = p, width = 6, height = 4, dpi = 300, device = "png")
            }, error = function(e) message("❌ accuracy_loss_plot: ", e$message))
          }
          
          if (self$viewPerEpochPlots("saturation_plot")) {
            tryCatch({
              p <- ggplot2::ggplot(df_accsat, ggplot2::aes(x = Epoch)) +
                ggplot2::geom_line(ggplot2::aes(y = MeanOutput), size = 1) +
                ggplot2::geom_line(ggplot2::aes(y = StdOutput),  size = 1) +
                ggplot2::labs(title = paste(plot_title_prefix, "— Output Mean & Std Dev"),
                              y = "Output Value") +
                ggplot2::theme_minimal() +
                ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold", size = 16))
              out <- file.path(plots_dir, fname("output_saturation_plot.png"))
              message("📸 save: ", out)
              ggplot2::ggsave(filename = out, plot = p, width = 6, height = 4, dpi = 300, device = "png")
            }, error = function(e) message("❌ output_saturation_plot: ", e$message))
          }
          
          # 5) Regularization (ensure reg_loss_total exists)
          reg_loss_total <- 0
          if (!is.null(reg_type)) {
            if (isTRUE(self$ML_NN)) {
              for (layer in 1:self$num_layers) {
                weights_layer <- self$weights[[layer]]
                
                # === REGRESSION-ONLY: force L2 penalty like v2 ===
                if (identical(CLASSIFICATION_MODE, "regression")) {
                  reg_loss_total <- reg_loss_total + self$lambda * sum(weights_layer^2, na.rm = TRUE)
                } else {
                  # ---- original multi-reg logic preserved for classification ----
                  if (reg_type == "L1") {
                    reg_loss_total <- reg_loss_total + self$lambda * sum(abs(weights_layer), na.rm = TRUE)
                  } else if (reg_type == "L2") {
                    reg_loss_total <- reg_loss_total + self$lambda * sum(weights_layer^2, na.rm = TRUE)
                  } else if (reg_type == "L1_L2") {
                    l1_ratio <- 0.5
                    reg_loss_total <- reg_loss_total + self$lambda * (
                      l1_ratio * sum(abs(weights_layer), na.rm = TRUE) +
                        (1 - l1_ratio) * sum(weights_layer^2, na.rm = TRUE)
                    )
                  } else if (reg_type == "Group_Lasso") {
                    if (is.null(self$groups) || is.null(self$groups[[layer]])) {
                      self$groups <- if (is.null(self$groups)) vector("list", self$num_layers) else self$groups
                      self$groups[[layer]] <- list(1:ncol(weights_layer))
                    }
                    reg_loss_total <- reg_loss_total + self$lambda * sum(sapply(self$groups[[layer]], function(group) {
                      sqrt(sum(weights_layer[, group]^2, na.rm = TRUE))
                    }))
                  } else if (reg_type == "Max_Norm") {
                    max_norm <- 1.0
                    norm_weight <- sqrt(sum(weights_layer^2, na.rm = TRUE))
                    reg_loss_total <- reg_loss_total + self$lambda * ifelse(norm_weight > max_norm, 1, 0)
                  } else if (reg_type == "Orthogonality") {
                    WtW <- t(weights_layer) %*% weights_layer
                    I <- diag(ncol(WtW))
                    reg_loss_total <- reg_loss_total + self$lambda * sum((WtW - I)^2)
                  } else if (reg_type == "Sparse_Bayesian") {
                    stop("Sparse Bayesian Learning is not implemented in this code.")
                  } else {
                    message("Invalid regularization type. Using 0.")
                  }
                }
              } # end for layer
            } else {
              # -------- single-layer reg --------
              weights_list  <- if (is.list(self$weights)) self$weights else list(self$weights)
              weights_layer <- weights_list[[1]]
              
              # === REGRESSION-ONLY: force L2 penalty like v2 ===
              if (identical(CLASSIFICATION_MODE, "regression")) {
                reg_loss_total <- self$lambda * sum(weights_layer^2, na.rm = TRUE)
              } else {
                # ---- original multi-reg logic preserved for classification ----
                if (reg_type == "L1") {
                  reg_loss_total <- self$lambda * sum(abs(weights_layer), na.rm = TRUE)
                } else if (reg_type == "L2") {
                  reg_loss_total <- self$lambda * sum(weights_layer^2, na.rm = TRUE)
                } else if (reg_type == "L1_L2") {
                  l1_ratio <- 0.5
                  reg_loss_total <- self$lambda * (
                    l1_ratio * sum(abs(weights_layer), na.rm = TRUE) +
                      (1 - l1_ratio) * sum(weights_layer^2, na.rm = TRUE)
                  )
                } else if (reg_type == "Group_Lasso") {
                  if (is.null(self$groups)) self$groups <- list(1:ncol(weights_layer))
                  reg_loss_total <- self$lambda * sum(sapply(self$groups, function(group) {
                    sqrt(sum(weights_layer[, group]^2, na.rm = TRUE))
                  }))
                } else if (reg_type == "Max_Norm") {
                  max_norm <- 1.0
                  norm_weight <- sqrt(sum(weights_layer^2, na.rm = TRUE))
                  reg_loss_total <- self$lambda * ifelse(norm_weight > max_norm, 1, 0)
                } else if (reg_type == "Orthogonality") {
                  WtW <- t(weights_layer) %*% weights_layer
                  I <- diag(ncol(WtW))
                  reg_loss_total <- self$lambda * sum((WtW - I)^2)
                } else if (reg_type == "Sparse_Bayesian") {
                  stop("Sparse Bayesian Learning is not implemented in this code.")
                } else {
                  message("Invalid regularization type. Using 0.")
                }
              } # end regression vs classification
            } # end ML_NN else
          } # end if !is.null(reg_type)
          
          # ===== Loss (train) =====
          losses[[epoch]] <- loss_function(
            predictions         = predictions,     # defined right after learn_result
            labels              = labels_epoch,    # aligned to n above
            CLASSIFICATION_MODE = CLASSIFICATION_MODE,
            reg_loss_total      = reg_loss_total,  # always defined
            loss_type           = loss_type,
            verbose             = verbose
          )
          
          # ===== Initialize records and optimizer params (unchanged) =====
          if (self$ML_NN) {
            weights_record <- vector("list", self$num_layers)
            biases_record  <- vector("list", self$num_layers)
          }
          
          if (update_weights) {
            res_upd <- update_weights_block(
              self = self,
              update_weights = update_weights,
              optimizer = optimizer,
              optimizer_params_weights = optimizer_params_weights,
              weight_gradients = weight_gradients,
              lr = lr,
              reg_type = reg_type,
              beta1 = beta1, beta2 = beta2, epsilon = epsilon,
              epoch = epoch,
              lookahead_step = lookahead_step,
              verbose = verbose
            )
            optimizer_params_weights <- res_upd$updated_optimizer_params
          }
          
          # Record the updated weight matrix
          if (self$ML_NN) {
            for (layer in 1:self$num_layers) {
              weights_record[[layer]] <- as.matrix(self$weights[[layer]])
            }
          } else {
            weights_record <- if (is.list(self$weights)) {
              as.matrix(self$weights[[1]])
            } else {
              as.matrix(self$weights)
            }
          }
          
          # =========================
          # BLOCK B — Weights (Max Weight + Plot)
          # =========================
          
          # post-update max|W| and log
          max_weight <- tryCatch({
            if (is.list(self$weights)) {
              max(unlist(lapply(self$weights, function(W) max(abs(as.numeric(W)), na.rm = TRUE))), na.rm = TRUE)
            } else {
              max(abs(as.numeric(self$weights)), na.rm = TRUE)
            }
          }, error = function(e) NA_real_)
          max_weight_log <- c(max_weight_log, max_weight)
          
          # DF for MaxWeight only
          df_maxw <- data.frame(
            Epoch     = seq_len(length(max_weight_log)),
            MaxWeight = max_weight_log
          )
          
          # ensure output dir + title
          plots_dir <- ddesonn_plots_dir(output_root)
          if (!exists("plot_title_prefix", inherits = TRUE)) {
            plot_title_prefix <- if (isTRUE(get0("do_ensemble", ifnotfound = FALSE))) {
              sprintf("DDESONN%s SONN%s | lr: %s | lambda: %s",
                      if (!is.null(self$ensemble_number)) paste0(" ", self$ensemble_number) else "",
                      if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                      lr, lambda)
            } else {
              sprintf("SONN%s | lr: %s | lambda: %s",
                      if (exists("model_iter_num", inherits = TRUE)) paste0(" ", model_iter_num) else "",
                      lr, lambda)
            }
          }
          
          # 3) Max Weight Magnitude
          if (self$viewPerEpochPlots("max_weight_plot")) {
            tryCatch({
              p <- ggplot2::ggplot(df_maxw, ggplot2::aes(x = Epoch, y = MaxWeight)) +
                ggplot2::geom_line(size = 1) +
                ggplot2::labs(title = paste(plot_title_prefix, "— Max Weight Magnitude Over Time"),
                              y = "Max |Weight|") +
                ggplot2::theme_minimal() +
                ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold", size = 16))
              out <- file.path(plots_dir, fname("max_weight_plot.png"))
              message("📸 save: ", out)
              ggplot2::ggsave(filename = out, plot = p, width = 6, height = 4, dpi = 300, device = "png")
            }, error = function(e) message("❌ max_weight_plot: ", e$message))
          }
          
          # Update biases
          if (update_biases) {
            res_bias <- update_biases_block(
              self = self,
              update_biases = update_biases,
              optimizer = optimizer,
              optimizer_params_biases = optimizer_params_biases,
              bias_gradients = bias_gradients,
              errors = errors,  # used by SL path (colMeans(errors[[1]]))
              lr = lr,
              reg_type = reg_type,
              beta1 = beta1, beta2 = beta2, epsilon = epsilon,
              epoch = epoch,
              lookahead_step = lookahead_step,
              verbose = verbose
            )
            optimizer_params_biases <- res_bias$updated_optimizer_params
          }
          
          if (self$ML_NN) {
            for (layer in 1:self$num_layers) {
              biases_record[[layer]] <- as.matrix(self$biases[[layer]])
            }
          } else {
            biases_record <- as.matrix(self$biases)
          }
          
          # ===== Validation-or-Training Metrics Block =====
          last_val_probs    <- NULL
          last_val_labels   <- NULL
          last_train_probs  <- NULL
          last_train_labels <- NULL
          last_val_predict   <- NULL
          last_train_predict <- NULL
          
          if (!is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
            
            to_one_hot_matrix <- function(y_vec, levels_ref = NULL) {
              if (is.matrix(y_vec) && ncol(y_vec) > 1L && all(y_vec %in% c(0, 1, NA))) {
                Y <- matrix(as.numeric(y_vec), nrow = nrow(y_vec), ncol = ncol(y_vec))
                storage.mode(Y) <- "double"
                return(Y)
              }
              if (is.matrix(y_vec) && ncol(y_vec) == 1L) {
                y_vec <- y_vec[, 1]
              }
              if (is.factor(y_vec)) {
                f <- y_vec
              } else {
                levs <- levels_ref
                if (is.null(levs) || !length(levs)) {
                  levs <- unique(as.character(y_vec))
                }
                f <- factor(y_vec, levels = levs)
              }
              mm <- model.matrix(~ f - 1)
              colnames(mm) <- levels(f)
              if (!is.null(levels_ref) && length(levels_ref)) {
                missing_cols <- setdiff(levels_ref, colnames(mm))
                if (length(missing_cols)) {
                  mm <- cbind(mm, matrix(0, nrow(mm), length(missing_cols),
                                         dimnames = list(NULL, missing_cols)))
                }
                mm <- mm[, levels_ref, drop = FALSE]
              }
              storage.mode(mm) <- "double"
              mm[is.na(mm)] <- 0
              mm
            }
            
            # -------- Validation path --------
            cat(sprintf("[PRED-INVOKE] train_network(): per-epoch VALIDATION predict() | epoch=%d | caller=validation_metrics\n", epoch))  # #$$$$$$$$$$$$$
            predicted_output_val <- tryCatch(
              self$predict(
                Rdata                = X_validation,
                weights              = if (isTRUE(self$ML_NN)) weights_record else self$weights,
                biases               = if (isTRUE(self$ML_NN)) biases_record  else self$biases,
                activation_functions = activation_functions,
                verbose = verbose,
                debug = debug
              ),
              error = function(e) {
                message("Validation predict failed: ", e$message)
                NULL
              }
            )
            
            if (!is.null(predicted_output_val)) {
              probs_val <- if (!is.null(predicted_output_val$predicted_output)) {
                predicted_output_val$predicted_output
              } else {
                predicted_output_val
              }
              probs_val <- as.matrix(probs_val)
              
              n_val <- nrow(probs_val)
              K_val <- max(1L, ncol(probs_val))
              
              cat("Debug (val): nrow(X_val)=", nrow(X_validation),
                  " nrow(probs_val)=", n_val,
                  " ncol(probs_val)=", K_val, "\n")
              
              # --- Normalize y_validation to vector or one-hot matrix ---
              if (is.data.frame(y_validation)) {
                y_val_vec <- y_validation[[1]]
                len_y <- length(y_val_vec)
              } else if (is.matrix(y_validation)) {
                if (ncol(y_validation) == 1L) {
                  y_val_vec <- y_validation[, 1]; len_y <- length(y_val_vec)
                } else {
                  y_val_vec <- y_validation;      len_y <- nrow(y_validation)
                }
              } else {
                y_val_vec <- y_validation;        len_y <- length(y_val_vec)
              }
              
              # Align by trimming only (no recycling)
              n_eff <- min(nrow(X_validation), n_val, len_y)
              if (n_eff <= 0) stop("Validation sizes yield zero effective rows.")
              
              probs_val <- probs_val[seq_len(n_eff), , drop = FALSE]
              if (is.matrix(y_val_vec) && ncol(y_val_vec) > 1L) {
                y_val_epoch <- y_val_vec[seq_len(n_eff), , drop = FALSE]
              } else {
                y_val_epoch <- y_val_vec[seq_len(n_eff)]
              }
              
              last_val_probs   <- probs_val
              last_val_labels  <- y_val_epoch
              last_val_predict <- predicted_output_val
              cat("[DBG] Captured LAST-epoch validation probs/labels and predict() list\n")
              
              targs_val <- .build_targets(y_val_epoch, n_eff, K_val, CLASSIFICATION_MODE)
              
              if (identical(CLASSIFICATION_MODE, "multiclass")) {
                stopifnot(K_val >= 2)
                
                ## --- Defensive alignment and conversion ---
                P <- as.matrix(probs_val)
                Y <- as.matrix(targs_val$Y)
                
                if (nrow(P) != nrow(Y) || ncol(P) != ncol(Y)) {
                  stop(sprintf("[VAL-MC] P dims %dx%d != Y dims %dx%d",
                               nrow(P), ncol(P), nrow(Y), ncol(Y)))
                }
                if (any(!is.finite(P))) stop("[VAL-MC] non-finite in probs_val")
                if (any(!is.finite(Y))) stop("[VAL-MC] non-finite in targets")
                
                # Clamp probabilities for stability
                eps <- 1e-12
                P <- pmin(pmax(P, eps), 1 - eps)
                
                ## --- Compute loss manually ---
                if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  row_loss <- -rowSums(Y * log(P), na.rm = TRUE)
                  val_loss <- mean(row_loss, na.rm = TRUE)
                } else {
                  val_loss <- mean((P - Y)^2, na.rm = TRUE)
                }
                
                ## --- Compute accuracy ---
                preds_idx <- max.col(P, ties.method = "first")
                targs_idx <- max.col(Y, ties.method = "first")
                acc_vec   <- as.integer(preds_idx == targs_idx)
                val_acc   <- mean(acc_vec, na.rm = TRUE)
                
                ## --- Append logs ---
                val_accuracy_log <- c(val_accuracy_log, val_acc)
                val_loss_log     <- c(val_loss_log, val_loss)
                
                ## --- Save best model snapshot ---
                if (is.na(best_val_acc) || (!is.na(val_acc) && val_acc > best_val_acc)) {
                  best_val_acc   <- val_acc
                  best_val_epoch <- epoch
                  best_val_n_eff <- n_eff
                  
                  if (isTRUE(self$ML_NN)) {
                    best_weights <- lapply(self$weights, as.matrix)
                    best_biases  <- lapply(self$biases,  as.matrix)
                  } else {
                    best_weights <- as.matrix(self$weights)
                    best_biases  <- as.matrix(self$biases)
                  }
                  
                  if (!is.null(predicted_output_val$prediction_time)) {
                    best_val_prediction_time <- predicted_output_val$prediction_time
                  }
                  
                  best_val_probs  <- as.matrix(P)
                  best_val_labels <- as.matrix(Y)
                  
                  cat("New best model saved at epoch", epoch,
                      "| Val Acc (0.5 thr):", round(100 * val_acc, 2), "%\n")
                }
              } else if (identical(CLASSIFICATION_MODE, "binary")) {
                val_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .bce_loss(probs_val, targs_val$y)
                } else {
                  mean((probs_val - matrix(targs_val$y, ncol = 1))^2, na.rm = TRUE)
                }
                val_acc <- accuracy(
                  SONN                = self,
                  Rdata               = X_validation[seq_len(n_eff), , drop = FALSE],
                  labels              = y_val_epoch,
                  CLASSIFICATION_MODE = CLASSIFICATION_MODE,
                  predicted_output    = probs_val,
                  verbose             = isTRUE(debug)
                )
                val_accuracy_log <- c(val_accuracy_log, val_acc)
                val_loss_log     <- c(val_loss_log,     val_loss)
                
                if (is.na(best_val_acc) || (!is.na(val_acc) && val_acc > best_val_acc)) {
                  best_val_acc       <- val_acc
                  best_val_epoch     <- epoch
                  best_val_n_eff     <- n_eff
                  
                  if (isTRUE(self$ML_NN)) {
                    best_weights <- lapply(self$weights, as.matrix)
                    best_biases  <- lapply(self$biases,  as.matrix)
                  } else {
                    best_weights <- as.matrix(self$weights)
                    best_biases  <- as.matrix(self$biases)
                  }
                  if (!is.null(predicted_output_val$prediction_time)) {
                    best_val_prediction_time <- predicted_output_val$prediction_time
                  }
                  best_val_probs  <- as.matrix(probs_val)
                  best_val_labels <- if (is.matrix(y_val_epoch)) y_val_epoch else matrix(y_val_epoch, ncol = 1L)
                  
                  cat("New best model saved at epoch", epoch,
                      "| Val Acc (0.5 thr):", round(100 * val_acc, 2), "%\n")
                }
                
              } else if (identical(CLASSIFICATION_MODE, "regression")) {
                preds_reg_val <- as.numeric(probs_val[, 1])
                y_reg_val     <- if (is.matrix(y_val_epoch)) as.numeric(y_val_epoch[, 1]) else as.numeric(y_val_epoch)
                val_loss      <- mean((preds_reg_val - y_reg_val)^2, na.rm = TRUE)
                
                mae_val <- mean(abs(preds_reg_val - y_reg_val), na.rm = TRUE)
                vary    <- stats::var(y_reg_val, na.rm = TRUE)
                r2_val  <- if (is.finite(vary) && vary > 0) 1 - val_loss / vary else NA_real_
                cat(sprintf("[dbg] REG VAL: MSE=%.6f | MAE=%.6f | R^2=%s\n",
                            val_loss, mae_val, ifelse(is.na(r2_val), "NA", sprintf("%.6f", r2_val))))
                
                val_accuracy_log <- c(val_accuracy_log, NA_real_)
                val_loss_log     <- c(val_loss_log,     val_loss)
                
                if (is.finite(val_loss) && (val_loss < best_val_loss)) {
                  best_val_loss        <- val_loss
                  best_val_epoch_loss  <- epoch
                  best_val_epoch       <- epoch
                  best_val_n_eff       <- n_eff
                  
                  if (isTRUE(self$ML_NN)) {
                    best_weights <- lapply(self$weights, as.matrix)
                    best_biases  <- lapply(self$biases,  as.matrix)
                  } else {
                    best_weights <- as.matrix(self$weights)
                    best_biases  <- as.matrix(self$biases)
                  }
                  if (!is.null(predicted_output_val$prediction_time)) {
                    best_val_prediction_time <- predicted_output_val$prediction_time
                  }
                  best_val_probs  <- as.matrix(probs_val)
                  best_val_labels <- if (is.matrix(y_val_epoch)) y_val_epoch else matrix(y_val_epoch, ncol = 1L)
                  
                  cat("New best (regression) model saved at epoch", epoch,
                      "| Val MSE:", round(val_loss, 6), "\n")
                }
              } else {
                stop("Unknown CLASSIFICATION_MODE.")
              }
              
            }
            
          } else if (!is.null(X_train) && !is.null(y_train) && isFALSE(validation_metrics)) {
            
            # -------- Training path (when validation metrics are disabled) --------
            cat(sprintf("[PRED-INVOKE] train_network(): per-epoch TRAIN predict() (validation disabled) | epoch=%d | caller=train_no_validation\n", epoch))  # #$$$$$$$$$$$$$
            predicted_output_train <- tryCatch(
              self$predict(
                Rdata                = X_train,
                weights              = if (isTRUE(self$ML_NN)) weights_record else self$weights,
                biases               = if (isTRUE(self$ML_NN)) biases_record  else self$biases,
                activation_functions = activation_functions,
                verbose = verbose,
                debug = debug
              ),
              error = function(e) {
                message("Training predict failed: ", e$message)
                NULL
              }
            )
            
            if (!is.null(predicted_output_train)) {
              
              probs_tr <- if (!is.null(predicted_output_train$predicted_output)) {
                predicted_output_train$predicted_output
              } else {
                predicted_output_train
              }
              probs_tr <- as.matrix(probs_tr)
              
              n_tr <- nrow(probs_tr)
              K_tr <- max(1L, ncol(probs_tr))
              
              cat("Debug (train): nrow(X_train)=", nrow(X_train),
                  " nrow(probs_tr)=", n_tr,
                  " ncol(probs_tr)=", K_tr, "\n")
              
              if (is.data.frame(y_train)) {
                y_tr_vec <- y_train[[1]]; len_y_tr <- length(y_tr_vec)
              } else if (is.matrix(y_train)) {
                if (ncol(y_train) == 1L) {
                  y_tr_vec <- y_train[, 1]; len_y_tr <- length(y_tr_vec)
                } else {
                  y_tr_vec <- y_train;      len_y_tr <- nrow(y_train)
                }
              } else {
                y_tr_vec <- y_train;        len_y_tr <- length(y_tr_vec)
              }
              
              n_eff_tr <- min(nrow(X_train), n_tr, len_y_tr)
              if (n_eff_tr <= 0) stop("Training sizes yield zero effective rows.")
              
              probs_tr <- probs_tr[seq_len(n_eff_tr), , drop = FALSE]
              if (is.matrix(y_tr_vec) && ncol(y_tr_vec) > 1L) {
                y_tr_epoch <- y_tr_vec[seq_len(n_eff_tr), , drop = FALSE]
              } else {
                y_tr_epoch <- y_tr_vec[seq_len(n_eff_tr)]
              }
              
              last_train_probs    <- probs_tr
              last_train_labels   <- y_tr_epoch
              last_train_predict  <- predicted_output_train
              cat("[DBG] Captured LAST-epoch training probs/labels and predict() list\n")
              
              targs_tr <- .build_targets(y_tr_epoch, n_eff_tr, K_tr, CLASSIFICATION_MODE)
              
              if (identical(CLASSIFICATION_MODE, "multiclass")) {
                stopifnot(K_tr >= 2)
                tr_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .ce_loss_multiclass(probs_tr, targs_tr$Y)
                } else {
                  mean((probs_tr - targs_tr$Y)^2, na.rm = TRUE)
                }
                tr_acc <- accuracy(
                  SONN                = self,
                  Rdata               = X_train[seq_len(n_eff_tr), , drop = FALSE],
                  labels              = y_tr_epoch,
                  CLASSIFICATION_MODE = CLASSIFICATION_MODE,
                  predicted_output    = probs_tr,
                  verbose             = isTRUE(debug)
                )
                
              } else if (identical(CLASSIFICATION_MODE, "binary")) {
                tr_loss <- if (!is.null(loss_type) && identical(loss_type, "cross_entropy")) {
                  .bce_loss(probs_tr, targs_tr$y)
                } else {
                  mean((probs_tr - matrix(targs_tr$y, ncol = 1))^2, na.rm = TRUE)
                }
                tr_acc <- accuracy(
                  SONN                = self,
                  Rdata               = X_train[seq_len(n_eff_tr), , drop = FALSE],
                  labels              = y_tr_epoch,
                  CLASSIFICATION_MODE = CLASSIFICATION_MODE,
                  predicted_output    = probs_tr,
                  verbose             = isTRUE(debug)
                )
                
              } else if (identical(CLASSIFICATION_MODE, "regression")) {
                preds_reg_tr <- as.numeric(probs_tr[, 1])
                y_reg_tr     <- if (is.matrix(y_tr_epoch)) as.numeric(y_tr_epoch[, 1]) else as.numeric(y_tr_epoch)
                tr_loss      <- mean((preds_reg_tr - y_reg_tr)^2, na.rm = TRUE)
                tr_acc       <- NA_real_
              } else {
                stop("Unknown CLASSIFICATION_MODE.")
              }
              
              # Log metrics (training)
              train_accuracy_log <- c(train_accuracy_log, tr_acc)
              train_loss_log     <- c(train_loss_log,     tr_loss)
              
              # Track "best" by training accuracy when validation is disabled (classification only)
              if (!identical(CLASSIFICATION_MODE, "regression")) {
                if (is.na(best_train_acc) || (!is.na(tr_acc) && tr_acc > best_train_acc)) {
                  best_train_acc   <- tr_acc
                  best_epoch_train <- epoch
                  
                  if (isTRUE(self$ML_NN)) {
                    best_weights <- lapply(self$weights, as.matrix)
                    best_biases  <- lapply(self$biases,  as.matrix)
                  } else {
                    best_weights <- as.matrix(self$weights)
                    best_biases  <- as.matrix(self$biases)
                  }
                  
                  assign("best_train_probs",  probs_tr,   envir = .GlobalEnv)
                  assign("best_train_labels", y_tr_epoch, envir = .GlobalEnv)
                  
                  cat("New best (train) model saved at epoch", epoch,
                      "| Train Acc (0.5 thr):", round(100 * tr_acc, 2), "%\n")
                }
              } else {
                # For regression + no validation: track best by training loss (closest to v2 simplicity)
                if (is.finite(tr_loss) && (tr_loss < best_train_loss)) {
                  best_train_loss       <- tr_loss
                  best_epoch_train_loss <- epoch
                  
                  if (isTRUE(self$ML_NN)) {
                    best_weights <- lapply(self$weights, as.matrix)
                    best_biases  <- lapply(self$biases,  as.matrix)
                  } else {
                    best_weights <- as.matrix(self$weights)
                    best_biases  <- as.matrix(self$biases)
                  }
                  
                  assign("best_train_probs",  probs_tr,   envir = .GlobalEnv)
                  assign("best_train_labels", y_tr_epoch, envir = .GlobalEnv)
                  
                  cat("New best (regression, train) model saved at epoch", epoch,
                      "| Train MSE:", round(tr_loss, 6), "\n")
                }
              }
            }
          }
          
          # -------- FINALIZE WITH BEST SNAPSHOT --------
          use_best <- !is.null(best_weights) && !is.null(best_biases)
          
          if (isTRUE(use_best)) {
            
            # recompute TRAIN preds with best snapshot
            if (!is.null(X_train)) {
              cat(sprintf("[PRED-INVOKE] train_network(): BEST-SNAPSHOT recompute TRAIN predict() | epoch=%d | caller=best_snapshot_train\n", epoch))  # #$$$$$$$$$$$$$
              pred_train_best <- tryCatch(
                self$predict(
                  Rdata                = X_train,
                  weights              = best_weights,
                  biases               = best_biases,
                  activation_functions = activation_functions,
                  verbose = FALSE, debug = FALSE
                ),
                error = function(e) NULL
              )
              if (!is.null(pred_train_best)) {
                predicted_output_train_reg <- pred_train_best
              }
            }
            
            # recompute VAL preds with best snapshot when applicable
            if (!is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
              cat(sprintf("[PRED-INVOKE] train_network(): BEST-SNAPSHOT recompute VALIDATION predict() | epoch=%d | caller=best_snapshot_validation\n", epoch))  # #$$$$$$$$$$$$$
              pred_val_best <- tryCatch(
                self$predict(
                  Rdata                = X_validation,
                  weights              = best_weights,
                  biases               = best_biases,
                  activation_functions = activation_functions,
                  verbose = FALSE, debug = FALSE
                ),
                error = function(e) NULL
              )
              
              if (!is.null(pred_val_best)) {
                predicted_output_val <- pred_val_best
                if (!is.null(pred_val_best$prediction_time)) {
                  best_val_prediction_time <- pred_val_best$prediction_time
                }
                
                probs_val_best <- if (!is.null(pred_val_best$predicted_output)) {
                  pred_val_best$predicted_output
                } else {
                  pred_val_best
                }
                probs_val_best <- as.matrix(probs_val_best)
                
                if (is.data.frame(y_validation)) {
                  y_val_vec2 <- y_validation[[1]]
                } else if (is.matrix(y_validation)) {
                  y_val_vec2 <- if (ncol(y_validation) == 1L) y_validation[, 1] else y_validation
                } else {
                  y_val_vec2 <- y_validation
                }
                
                n_eff2 <- min(nrow(X_validation), nrow(probs_val_best), if (is.matrix(y_val_vec2)) nrow(y_val_vec2) else length(y_val_vec2))
                if (is.finite(best_val_n_eff) && !is.na(best_val_n_eff)) {
                  n_eff2 <- min(n_eff2, best_val_n_eff)
                }
                if (n_eff2 <= 0) stop("Validation sizes yield zero effective rows (best snapshot).")
                
                probs_val_best <- probs_val_best[seq_len(n_eff2), , drop = FALSE]
                if (is.matrix(y_val_vec2) && !is.null(dim(y_val_vec2)) && ncol(y_val_vec2) > 1L) {
                  y_val_epoch2 <- y_val_vec2[seq_len(n_eff2), , drop = FALSE]
                } else {
                  y_val_epoch2 <- y_val_vec2[seq_len(n_eff2)]
                }
                
                best_val_probs  <- as.matrix(probs_val_best)
                if (identical(CLASSIFICATION_MODE, "multiclass")) {
                  base_levels <- self$class_levels
                  best_val_labels <- to_one_hot_matrix(y_val_epoch2, levels_ref = base_levels)
                  if (!is.null(base_levels) && length(base_levels) && ncol(best_val_probs) == length(base_levels)) {
                    colnames(best_val_probs) <- base_levels
                  }
                } else if (identical(CLASSIFICATION_MODE, "binary")) {
                  if (is.matrix(y_val_epoch2) && ncol(y_val_epoch2) >= 1L) {
                    best_val_labels <- matrix(as.numeric(y_val_epoch2[, 1]), ncol = 1L)
                  } else {
                    best_val_labels <- matrix(as.numeric(y_val_epoch2), ncol = 1L)
                  }
                } else {
                  best_val_labels <- as.matrix(y_val_epoch2)
                }
              }
            }
            
            if (identical(CLASSIFICATION_MODE, "regression")) {
              cat(sprintf("[BEST-SNAPSHOT] using %s epoch=%s | best_val_loss=%.7f | n_eff=%s\n",
                          if (isTRUE(validation_metrics)) "validation-best" else "train-best",
                          if (isTRUE(validation_metrics)) as.character(best_val_epoch_loss) else as.character(best_epoch_train_loss),
                          if (isTRUE(validation_metrics)) best_val_loss else best_train_loss,
                          as.character(best_val_n_eff)))
            } else {
              cat(sprintf("[BEST-SNAPSHOT] using %s epoch=%s | best_val_acc=%.7f | thr=0.5 | n_eff=%s\n",
                          if (isTRUE(validation_metrics)) "validation-best" else "train-best",
                          if (isTRUE(validation_metrics)) as.character(best_val_epoch) else as.character(best_epoch_train),
                          if (is.na(best_val_acc)) NA_real_ else best_val_acc,
                          as.character(best_val_n_eff)))
            }
          } else {
            cat("[BEST-SNAPSHOT] no best snapshot captured; returning last evaluated predictions.\n")
          }
          
          # --- STRICT CHECK: only when regression + validation enabled ---
          if (identical(CLASSIFICATION_MODE, "regression")) {
            if (is.null(best_val_probs) || is.null(best_val_labels)) {
              warning("[WARN] Regression best snapshot missing explicit val_probs/labels — using last validation predictions instead.")
              best_val_probs  <- as.matrix(probs_val)
              best_val_labels <- if (is.matrix(y_val_epoch)) y_val_epoch else matrix(y_val_epoch, ncol = 1L)
            }
          }
          
          # === Keep predict()-style list selection logic as in your original ===
          if (isTRUE(train) && isTRUE(validation_metrics)) {
            cat(sprintf("[PRED-INVOKE] train_network(): implicit last_val_predict selection (NO new predict) | epoch=%d | caller=implicit_last_val_predict\n", epoch))  # #$$$$$$$$$$$$$
            predicted_output_train_reg <- last_val_predict
          } else if (!isTRUE(train) && !isTRUE(validation_metrics)) {
            cat(sprintf("[PRED-INVOKE] train_network(): implicit last_train_predict selection (NO new predict) | epoch=%d | caller=implicit_last_train_predict\n", epoch))  # #$$$$$$$$$$$$$
            predicted_output_train_reg <- last_train_predict
          } else {
            predicted_output_train_reg <- predicted_output_train_reg
          }
          
        } # end for epoch
        
        total_learn_time <- total_learn_time + learn_result$learn_time
        
        # mirror the best snapshot onto self for downstream readers
        self$best_weights <- best_weights
        self$best_biases  <- best_biases
        
        cat(sprintf("\nBest Training Accuracy: %.2f%% at Epoch %d\n", 100 * best_train_acc, best_epoch_train))
        cat("Best Epoch (validation accuracy):", best_val_epoch, "\n")
        cat("Best Validation Accuracy:", round(100 * best_val_acc, 2), "%\n")
        if (identical(CLASSIFICATION_MODE, "regression")) {
          cat(sprintf("Best Training MSE: %.6f at Epoch %s\n", best_train_loss, as.character(best_epoch_train_loss)))
          if (isTRUE(validation_metrics)) {
            cat(sprintf("Best Validation MSE: %.6f at Epoch %s\n", best_val_loss, as.character(best_val_epoch_loss)))
          }
        }
        
      } else {
        predicted_output_train_reg_prediction_time <- NULL
        weights_record <- NULL
        biases_record <- NULL
        dim_hidden_layers <- NULL
      }
      
      if (train) {
        if (self$ML_NN) {
          for (layer in 1:self$num_layers) {
            cat(sprintf("Layer %d weights summary:\n", layer))
            print(summary(as.vector(weights_record[[layer]])))
            
            cat(sprintf("Layer %d biases summary:\n", layer))
            print(summary(as.vector(biases_record[[layer]])))
          }
        } else {
          cat("Single-layer weights summary:\n")
          print(summary(as.vector(weights_record)))
          
          cat("Single-layer biases summary:\n")
          print(summary(as.vector(biases_record)))
        }
      }
      
      optimal_epoch <- which(diff(losses) > 0)[1]
      loss_increase_flag <- any(losses > losses[1])
      lossesatoptimalepoch <- if (is.na(optimal_epoch)) tail(losses, 1) else losses[optimal_epoch]
      
      # --- Robust loss plot saver (base R) ---
      if (all(is.finite(losses))) {
        plots_dir <- ddesonn_plots_dir(output_root)
        
        fname_prefixer <- make_fname_prefix(
          do_ensemble     = do_ensemble,
          num_networks    = num_networks,
          ensemble_number = ensemble_number,
          model_index     = model_iter_num,
          who             = "SONN"
        )
        
        output_file <- file.path(plots_dir, paste0(fname_prefixer("loss_plot"), ".png"))
        cat("Saving to:", normalizePath(output_file, mustWork = FALSE), "\n")
        
        if (capabilities("cairo")) {
          png(output_file, width = 900, height = 650, res = 96, type = "cairo-png")
        } else {
          png(output_file, width = 900, height = 650, res = 96)
        }
        cat("Device before:", dev.cur(), "\n")
        
        plot(losses, type = 'l',
             main = paste('Loss Over Epochs for DDESONN', ensemble_number,
                          'SONN', model_iter_num, 'lr:', lr, 'lambda:', self$lambda),
             xlab = 'Epoch', ylab = 'Loss', col = 'turquoise', lwd = 2.0)
        
        points(optimal_epoch, losses[optimal_epoch], col = 'limegreen', pch = 16)
        offset <- max(losses) * 0.06
        eq <- paste("Optimal Epoch:", optimal_epoch,
                    "\nLoss:", round(losses[optimal_epoch], 4))
        text(optimal_epoch + 1.65,
             losses[optimal_epoch] + offset,
             eq, pos = 4, col = "limegreen", adj = 0)
        
        dev.off()
        cat("Device after:", dev.cur(), "\n")
        
        fi <- file.info(output_file)
        cat("Saved OK. Size:", fi$size, "bytes\n")
      } else {
        cat("Skipping plot: non-finite losses.\n")
      }
      
      end_time <- Sys.time()
training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
if(verbose){print("----------------------------------------train_network-end----------------------------------------")}

return(list(
  predicted_output_l2      = predicted_output_train_reg,
  training_time            = training_time,
  best_train_acc           = best_train_acc,
  best_epoch_train         = best_epoch_train,
  best_train_loss          = best_train_loss,
  best_epoch_train_loss    = best_epoch_train_loss,
  best_val_acc             = best_val_acc,
  best_val_epoch           = best_val_epoch,
  best_val_prediction_time = best_val_prediction_time,
  learn_output             = learn_result$learn_output,
  learn_time               = total_learn_time,
  learn_dim_hidden_layers  = learn_result$dim_hidden_layers,
  learn_hidden_outputs     = learn_result$hidden_outputs,
  learn_grads_matrix       = learn_result$grads_matrix,
  learn_bias_gradients     = learn_result$bias_gradients,
  learn_errors             = learn_result$errors,
  optimal_epoch            = optimal_epoch,
  weights_record           = weights_record,
  biases_record            = biases_record,
  best_weights_record      = best_weights,
  best_biases_record       = best_biases,
  lossesatoptimalepoch     = NULL,
  loss_increase_flag       = NULL,
  loss_status              = NULL,
  dim_hidden_layers        = dim_hidden_layers,
  predicted_output_val     = predicted_output_val,
  best_val_probs           = best_val_probs,
  best_val_labels          = best_val_labels,
  # --- regression additions (harmless for cls) ---
  best_val_loss            = best_val_loss,
  best_val_epoch_loss      = best_val_epoch_loss
))
} #end of train_network()
) #end off SONN class
)
#
#    ________  ________  ___________ _________________    _______
#    \______ \ \______ \ \_   _____//   _____/\_____  \   \      \
#   |     |  \ |    |  \ |    __)_ \_____  \  /   |   \  /   |   \
#  |     `   \|    `   \|        \/        \/    |    \/    |    \
# /_______  /_______  /_______  /_______  /\_______  /\____|__  /
#         \/        \/        \/        \/         \/         \/
#

# Step 2: Define the Deep Dynamic Experimental of Self-Organizing Neural Networks (DDESONN) class
DDESONN <- R6::R6Class( 
  "DDESONN",
  lock_objects = FALSE,
  public = list(
    ensemble = NULL,  # Define ensemble as a public property
    results_list = NULL,
    predicted_outputAndTime = NULL,
    numeric_columns = NULL,
    #ensemble_number = NULL,
    # Constructor
    initialize = function(num_networks, input_size, hidden_sizes, output_size, N, lambda, ensemble_number, ensembles, ML_NN, activation_functions, activation_functions_predict, init_method = init_method, custom_scale = custom_scale) {


      # Initialize an ensemble of SONN networks
      self$ensemble <- lapply(1:num_networks, function(i) {

        # Determine ensemble and model names

        ensemble_name <- ensemble_number
        model_name <- i

        # Initialize variables
        run_result <- NULL
        weights_record_extract <- NULL
        biases_record_extract <- NULL


        # Instantiate SONN network based on conditions
        if (ML_NN) {
          new_network <- SONN$new(input_size = input_size, hidden_sizes = hidden_sizes, output_size = output_size, N = N, lambda = lambda, ML_NN = ML_NN, activation_functions = activation_functions, activation_functions_predict = activation_functions_predict, init_method = init_method, custom_scale = custom_scale)
        } else {
          new_network <- SONN$new(input_size = input_size, output_size = output_size, N = N, lambda = lambda, ML_NN = ML_NN, activation_functions = activation_functions, activation_functions_predict = activation_functions_predict, init_method = init_method, custom_scale = custom_scale)
        }

        # Set names for the model
        attr(new_network, "ensemble_name") <- ensemble_name
        attr(new_network, "model_name") <- model_name



        # Check if ML_NN is TRUE before loading weights and biases for multiple layers
        if (ML_NN == TRUE) {
          if (!is.null(weights_record_extract[[1]]) && !is.null(biases_record_extract[[1]])) {
            # print((as.matrix(new_network$weights)))
            # print(weights_record_extract)
            # print(biases_record_extract)
            # for (k in 1:(length(hidden_sizes)+1)) {
            new_network$load_all_weights(weights_list = weights_record_extract)
            new_network$load_all_biases(biases_list = biases_record_extract[[k]])
            # }
          }
        }else{
          # Load weights and biases for the first layer
          if (!is.null(weights_record_extract[[1]]) && !is.null(biases_record_extract[[1]])) {
            # print((as.matrix(new_network$weights)))
            new_network$load_weights(new_weights = weights_record_extract)
            new_network$load_biases(new_biases = biases_record_extract)
          }
        }

        return(new_network)
      })


      self$predicted_outputAndTime <- list() #vector("list", length(self$ensemble) * 2) #* nrow(hyperparameter_grid))
      results_list <- list() #vector("list", length(self$ensemble) * 2) #* nrow(hyperparameter_grid))
      self$numeric_columns <- NULL

      self$ensembles <- ensembles

      # Configuration flags for enabling/disabling per-DDESONN model performance/relevance plots
      self$FinalUpdatePerformanceandRelevanceViewPlotsConfig  <- list(
        performance_high_mean_plots = FALSE,  # high mean performance plots
        performance_low_mean_plots  = FALSE,   # low mean performance plots
        relevance_high_mean_plots   = FALSE,    # high mean relevance plots
        relevance_low_mean_plots    = FALSE,     # low mean relevance plots
        viewAllPlots = FALSE,
        verbose      = FALSE
      )


    ## ============================================================
    ## SECTION: EvaluatePredictionsReportPlotsConfig (module)       
    ## ============================================================
      self$EvaluatePredictionsReportPlotsConfig <- list(
        pred_vs_error_scatter = FALSE,  # pred_vs_error_scatter.png
        roc_curve             = FALSE,  # roc_curve.png
        pr_curve              = FALSE,  # pr_curve.png
        legacy_conf_heatmap   = FALSE,  # confusion_heatmap_legacy.png
        
        # Accuracy plot family (single toggle + selector)               
        accuracy_plots         = FALSE,                                
        accuracy_plot_mode     = "both",  # "accuracy"|"accuracy_tuned"|"both"  
        
        multiclass_heatmap    = FALSE,  # confusion_matrix_multiclass_heatmap.png
        
        viewAllPlots          = FALSE,  # overrides everything above
        verbose               = FALSE
      )
      
    
    },
    # Function to normalize specific columns in the data
    normalize_data = function(Rdata, numeric_columns) {
      # Calculate mean and standard deviation for each numeric feature
      means <- colMeans(Rdata[, numeric_columns])
      std_devs <- apply(Rdata[, numeric_columns], 2, sd)

      # Print mean and standard deviation before normalization
      print(paste("Before normalization - Mean: ", means))
      print(paste("Before normalization - Standard Deviation: ", std_devs))

      # Normalize the numeric data
      normalized_Rdata <- Rdata
      normalized_Rdata[, numeric_columns] <- scale(Rdata[, numeric_columns], center = means, scale = std_devs)

      # Calculate mean and standard deviation after normalization
      normalized_means <- colMeans(normalized_Rdata[, numeric_columns])
      normalized_std_devs <- apply(normalized_Rdata[, numeric_columns], 2, sd)

      # Print mean and standard deviation after normalization
      print(paste("After normalization - Mean: ", normalized_means))
      print(paste("After normalization - Standard Deviation: ", normalized_std_devs))

      return(normalized_Rdata)
    },
    # Function to perform batch normalization on specific columns in the data
    batch_normalize_data = function(Rdata, numeric_columns, gamma_bn, beta_bn, epsilon_bn = epsilon_bn, momentum_bn = momentum_bn, is_training_bn = is_training_bn) {

      if (is_training_bn) {
        # Training mode: Compute mean and variance from the current batch
        batch_mean_bn <- colMeans(Rdata[, numeric_columns])
        batch_var_bn <- apply(Rdata[, numeric_columns], 2, var)

        # Update running statistics
        if (is.null(running_mean_bn)) {
          running_mean_bn <- batch_mean_bn
          running_var_bn <- batch_var_bn
        } else {
          running_mean_bn <- momentum_bn * running_mean_bn + (1 - momentum_bn) * batch_mean_bn
          running_var_bn <- momentum_bn * running_var_bn + (1 - momentum_bn) * batch_var_bn
        }

        # Normalize using batch statistics
        normalized_data_bn <- Rdata
        normalized_data_bn[, numeric_columns] <- (Rdata[, numeric_columns] - batch_mean_bn) / sqrt(batch_var_bn + epsilon_bn)

        # Apply gamma and beta
        normalized_data_bn[, numeric_columns] <- (normalized_data_bn[, numeric_columns] * gamma_bn) + beta_bn

        # Print diagnostic information
        print(paste("Batch Mean: ", batch_mean_bn))
        print(paste("Batch Variance: ", batch_var_bn))

      } else {
        # Inference mode: Use running statistics computed during training
        if (is.null(running_mean_bn) || is.null(running_var_bn)) {
          stop("Running mean and variance must be provided for inference.")
        }

        # Normalize using running statistics
        normalized_data_bn <- Rdata
        normalized_data_bn[, numeric_columns] <- (Rdata[, numeric_columns] - running_mean_bn) / sqrt(running_var_bn + epsilon_bn)

        # Apply gamma and beta
        normalized_data_bn[, numeric_columns] <- (normalized_data_bn[, numeric_columns] * gamma_bn) + beta_bn
      }

      # Print diagnostic information
      post_norm_mean_bn <- colMeans(normalized_data_bn[, numeric_columns])
      post_norm_var_bn <- apply(normalized_data_bn[, numeric_columns], 2, var)

      print(paste("After normalization - Mean: ", post_norm_mean_bn))
      print(paste("After normalization - Variance: ", post_norm_var_bn))

      # Return the normalized data and the updated running mean/variance
      return(list(normalized_data = normalized_data_bn, running_mean = running_mean_bn, running_var = running_var_bn))
    },
    # Function to calculate a reasonable batch size
    calculate_batch_size = function(data_size, max_batch_size = 512, min_batch_size = 16) {
      # Get the number of rows from the dataset
      n <- nrow(data_size)

      # Default to the minimum batch size
      batch_size <- min_batch_size

      # Adjust batch size if the dataset is sufficiently large
      if (n >= 1000) {
        # Set batch size to a fraction of dataset size, constrained by max_batch_size
        batch_size <- min(max_batch_size, n / 10)
      }

      # Return the computed batch size
      return(batch_size)
    },
    viewFinalUpdatePerformanceandRelevancePlots = function(name) {
      cfg <- self$FinalUpdatePerformanceandRelevanceViewPlotsConfig
      on_all <- isTRUE(cfg$viewAllPlots) || isTRUE(cfg$verbose)
      val <- cfg[[name]]
      flag <- isTRUE(val) || (is.logical(val) && length(val) == 1 && !is.na(val) && val)
      on_all || flag
    },
    ## ============================================================
    ## SECTION: Plot gate — EvaluatePredictionsReport               
    ## ============================================================
    viewEvaluatePredictionsReportPlots=function(name){
      cfg<-self$EvaluatePredictionsReportPlotsConfig
      if(!is.list(cfg))return(FALSE)
      on_all<-isTRUE(cfg$viewAllPlots)||isTRUE(cfg$verbose)
      val<-cfg[[name]]
      flag<-isTRUE(val)||(is.logical(val)&&length(val)==1L&&!is.na(val)&&val)
      on_all||flag
    },
    
    train = function(Rdata, labels, X_train, y_train, lr, lr_decay_rate, lr_decay_epoch, lr_min, num_networks, ensemble_number, do_ensemble, num_epochs, self_org, threshold, reg_type, numeric_columns, CLASSIFICATION_MODE, activation_functions, activation_functions_predict, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, batch_normalize_data, gamma_bn = NULL, beta_bn = NULL, epsilon_bn = 1e-5, momentum_bn = 0.9, is_training_bn = TRUE, shuffle_bn = FALSE, loss_type, update_weights, update_biases, sample_weights, preprocessScaledData, X_validation, y_validation, validation_metrics, threshold_function, best_weights_on_latest_weights_off, ML_NN, train, grouped_metrics, viewTables, verbose, output_root = NULL) {
      if(verbose){print("----------------------------------------train-begin----------------------------------------")}
      # Normalize the input data
      if (!is.null(numeric_columns) && !batch_normalize_data) {
        Rdata <- self$normalize_data(Rdata, numeric_columns)
      }
      
      # Initialize batch normalization parameters if not set
      if (batch_normalize_data) {
        if (is.null(gamma_bn)) gamma_bn <- rep(1, length(numeric_columns))
        if (is.null(beta_bn)) beta_bn <- rep(0, length(numeric_columns))
        if (is.null(self$mean_bn)) self$mean_bn <- rep(0, length(numeric_columns))
        if (is.null(self$var_bn)) self$var_bn <- rep(1, length(numeric_columns))
      }
      
      for (epoch in 1:num_epochs) {
        
        # Create mini-batches
        n <- nrow(Rdata)
        indices <- 1:n
        
        # Shuffle data if prompted
        if (shuffle_bn) {
          indices <- sample(indices)
        }
        
        batch_size <- self$calculate_batch_size(Rdata)
        mini_batches <- split(indices, ceiling(seq_along(indices) / batch_size))
        
        if (batch_normalize_data) {
          if (train) {
            for (batch in mini_batches) {
              batch_data <- Rdata[batch, , drop = FALSE]
              batch_labels <- labels[batch]
              batch_mean_bn <- colMeans(batch_data[, numeric_columns, drop = FALSE])
              batch_var_bn <- apply(batch_data[, numeric_columns, drop = FALSE], 2, var)
              batch_var_bn[!is.finite(batch_var_bn) | batch_var_bn < 0] <- 0
              self$mean_bn <- momentum_bn * self$mean_bn + (1 - momentum_bn) * batch_mean_bn
              self$var_bn <- momentum_bn * self$var_bn + (1 - momentum_bn) * batch_var_bn
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, batch_mean_bn, `-`)
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, sqrt(batch_var_bn + epsilon_bn), `/`)
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, gamma_bn, `*`)
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, beta_bn, `+`)
              if (verbose) {
                print(paste("Batch Mean: ", toString(round(batch_mean_bn, 6))))
                print(paste("Batch Variance: ", toString(round(batch_var_bn, 6))))
              }
            }
          } else {
            if (is.null(self$mean_bn) || is.null(self$var_bn)) stop("Running mean and variance must be provided for inference.")
            for (batch in mini_batches) {
              batch_data <- Rdata[batch, , drop = FALSE]
              batch_labels <- labels[batch]
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, self$mean_bn, `-`)
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, sqrt(self$var_bn + epsilon_bn), `/`)
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, gamma_bn, `*`)
              batch_data[, numeric_columns] <- sweep(batch_data[, numeric_columns, drop = FALSE], 2, beta_bn, `+`)
              if (verbose) {
                post_norm_mean_bn <- colMeans(batch_data[, numeric_columns, drop = FALSE])
                post_norm_var_bn <- apply(batch_data[, numeric_columns, drop = FALSE], 2, var)
                print(paste("After normalization - Mean: ", toString(round(post_norm_mean_bn, 6))))
                print(paste("After normalization - Variance: ", toString(round(post_norm_var_bn, 6))))
              }
            }
          }
        } else {
          for (batch in mini_batches) {
            batch_data <- Rdata[batch, , drop = FALSE]
            batch_labels <- labels[batch]
          }
        }#end of batch normalize data
      }
      
      # Initialize lists to store results
      all_predicted_outputAndTime      <- vector("list", length(self$ensemble))
      all_predicted_outputs_learn      <- vector("list", length(self$ensemble))
      all_predicted_outputs            <- vector("list", length(self$ensemble))
      all_prediction_times             <- vector("list", length(self$ensemble))
      all_training_times               <- vector("list", length(self$ensemble))
      all_best_val_prediction_time     <- vector("list", length(self$ensemble))
      all_learn_times                  <- vector("list", length(self$ensemble))
      all_ensemble_name_model_name     <- vector("list", length(self$ensemble))
      all_model_iter_num               <- vector("list", length(self$ensemble))
      all_best_train_acc               <- vector("list", length(self$ensemble))
      all_best_epoch_train             <- vector("list", length(self$ensemble))
      all_best_train_loss              <- vector("list", length(self$ensemble))
      all_best_epoch_train_loss        <- vector("list", length(self$ensemble))
      all_best_val_acc                 <- vector("list", length(self$ensemble))
      all_best_val_epoch               <- vector("list", length(self$ensemble))
      all_errors                       <- vector("list", length(self$ensemble))
      all_hidden_outputs               <- vector("list", length(self$ensemble))
      all_layer_dims                   <- vector("list", length(self$ensemble))
      all_best_val_probs               <- vector("list", length(self$ensemble))
      all_best_val_labels              <- vector("list", length(self$ensemble))
      all_weights                      <- vector("list", length(self$ensemble))
      all_biases                       <- vector("list", length(self$ensemble))
      all_activation_functions         <- vector("list", length(self$ensemble))
      all_activation_functions_predict <- vector("list", length(self$ensemble))
      
      for (i in 1:length(self$ensemble)) {
        # Add Ensemble and Model names to performance_list
        ensemble_name <- attr(self$ensemble[[i]], "ensemble_name")
        model_name <- attr(self$ensemble[[i]], "model_name")
        
        ensemble_name_model_name <- paste("Ensemble:", ensemble_name, "Model:", model_name)
        
        model_iter_num <- i
        
        if(self_org){
          self$ensemble[[i]]$self_organize(Rdata, labels, lr)
        }
        
        # might remove if, but keep contents
        if (train) {
          
          predicted_outputAndTime <- suppressMessages(
            self$ensemble[[i]]$train_network(
              Rdata, labels, X_train, y_train, lr, num_networks, CLASSIFICATION_MODE, num_epochs, model_iter_num, update_weights, update_biases, ensemble_number, do_ensemble, reg_type, activation_functions, dropout_rates, optimizer, beta1, beta2, epsilon, lookahead_step, loss_type, sample_weights, X_validation, y_validation, validation_metrics, threshold_function, ML_NN, train = TRUE, verbose = FALSE, output_root = output_root
            ))
          
          # Store core model info
          all_ensemble_name_model_name[[i]] <- ensemble_name_model_name
          all_model_iter_num[[i]] <- model_iter_num
          
          all_predicted_outputAndTime[[i]] <- list(
            predicted_output         = predicted_outputAndTime$predicted_output_l2$predicted_output, #this is last_val_predict or last_train_predict based on what is toggled upstream (isTrue(validation_metrics))
            prediction_time          = predicted_outputAndTime$predicted_output_l2$prediction_time,
            learn_time               = predicted_outputAndTime$learn_time,
            training_time            = predicted_outputAndTime$training_time,
            best_val_prediction_time = predicted_outputAndTime$best_val_prediction_time,
            optimal_epoch            = predicted_outputAndTime$optimal_epoch,
            weights_record           = predicted_outputAndTime$best_weights_record,
            biases_record            = predicted_outputAndTime$best_biases_record,
            losses_at_optimal_epoch  = predicted_outputAndTime$lossesatoptimalepoch,
            best_train_acc           = predicted_outputAndTime$best_train_acc,
            best_epoch_train         = predicted_outputAndTime$best_epoch_train,
            best_train_loss          = predicted_outputAndTime$best_train_loss,
            best_epoch_train_loss    = predicted_outputAndTime$best_epoch_train_loss,
            best_val_acc             = predicted_outputAndTime$best_val_acc,
            best_val_epoch           = predicted_outputAndTime$best_val_epoch
          )
          
          y <- labels
          # Continue if predictions are available
          if (!is.null(predicted_outputAndTime$predicted_output_l2)) {
            
            all_predicted_outputs[[i]]              <- predicted_outputAndTime$predicted_output_l2$predicted_output
            all_learn_times[[i]]                    <- predicted_outputAndTime$learn_time
            all_training_times[[i]]                 <- predicted_outputAndTime$training_time
            all_prediction_times[[i]]               <- predicted_outputAndTime$predicted_output_l2$prediction_time
            all_best_val_prediction_time[[i]]       <- predicted_outputAndTime$best_val_prediction_time
            all_errors[[i]]                         <- compute_error(predicted_outputAndTime$predicted_output_l2$predicted_output, y, CLASSIFICATION_MODE)
            all_hidden_outputs[[i]]                 <- predicted_outputAndTime$learn_hidden_outputs
            all_layer_dims[[i]]                     <- predicted_outputAndTime$learn_dim_hidden_layers
            all_best_val_probs[[i]]                 <- predicted_outputAndTime$best_val_probs
            all_best_val_labels[[i]]                <- predicted_outputAndTime$best_val_labels
            all_weights[[i]]                        <- predicted_outputAndTime$best_weights_record
            all_biases[[i]]                         <- predicted_outputAndTime$best_biases_record
            all_activation_functions[[i]]           <- activation_functions
            all_activation_functions_predict[[i]]   <- activation_functions_predict
            all_best_train_acc[[i]]                 <- predicted_outputAndTime$best_train_acc
            all_best_epoch_train[[i]]               <- predicted_outputAndTime$best_epoch_train
            all_best_train_loss[[i]]                <- predicted_outputAndTime$best_train_loss
            all_best_epoch_train_loss[[i]]          <- predicted_outputAndTime$best_epoch_train_loss
            all_best_val_acc[[i]]                   <- predicted_outputAndTime$best_val_acc
            all_best_val_epoch[[i]]                 <- predicted_outputAndTime$best_val_epoch
            
            if(verbose){
              # --- Debug prints ---
              cat(">> Ensemble Index:", i, "\n")
              cat("Predicted Output (first 5):\n"); print(head(all_predicted_outputs[[i]], 5))
              cat("Prediction Time:\n"); print(all_prediction_times[[i]])
              cat("Shape of Predicted Output:\n"); print(dim(all_predicted_outputs[[i]]))
              
              cat("Error Preview (first 5):\n"); print(head(all_errors[[i]], 5))
              if(ML_NN){
                cat("Hidden Output Layer Count:\n"); print(length(all_hidden_outputs[[i]]))
                cat("Hidden Layer Dims:\n"); print(all_layer_dims[[i]])
              }
              cat("Best Validation Probabilities (first 5):\n"); print(head(all_best_val_probs[[i]], 5))
              cat("Best Validation Labels (first 5):\n"); print(head(all_best_val_labels[[i]], 5))
              
              # Debug weights and biases
              cat("Weights Record (layer 1 preview):\n"); str(all_weights[[i]][[1]])
              cat("Biases Record (layer 1 preview):\n"); str(all_biases[[i]][[1]])
              
              # Debug activation functions
              cat("Activation Functions Used:\n"); print(all_activation_functions[[i]])
              cat("--------------------------------------------------------\n")
            } #end of verbose
          } # end of if (!is.null(predicted_outputAndTime$predicted_output_l2)){}
        }
      } # end of for (i in 1:length(self$ensemble))
      
      if(verbose){
        print(all_ensemble_name_model_name)
      }
      
      for (i in seq_along(all_predicted_outputAndTime)) {
        if (isTRUE(verbose)) {
          message("\n── Model ", i, " ──")
        }
        
        model_result <- all_predicted_outputAndTime[[i]]
        
        if (is.null(model_result)) {
          if (isTRUE(verbose)) message("Empty slot.")
          next
        }
        
        if (isTRUE(verbose)) {
          message("Prediction length: ", length(model_result$predicted_output))
          message("Prediction time: ", model_result$prediction_time)
          message("Training time: ", model_result$training_time)
          message("Optimal epoch: ", model_result$optimal_epoch)
          message("Loss at optimal: ", model_result$losses_at_optimal_epoch)
        }
        
        # ---- Weights ----
        if (is.list(model_result$weights_record)) {
          if (isTRUE(verbose)) message("Weights record dims by layer:")
          for (L in seq_along(model_result$weights_record)) {
            W <- model_result$weights_record[[L]]
            if (!is.null(W)) {
              if (isTRUE(verbose)) message(sprintf("  Layer %d: %s", L, paste(dim(W), collapse = " x ")))
            } else {
              if (isTRUE(verbose)) message(sprintf("  Layer %d: NULL", L))
            }
          }
        } else {
          if (isTRUE(verbose)) message("Weights record dims (SL): ", paste(dim(model_result$weights_record), collapse = " x "))
        }
        
        # ---- Biases ----
        if (is.list(model_result$biases_record)) {
          if (isTRUE(verbose)) message("Biases record length by layer:")
          for (L in seq_along(model_result$biases_record)) {
            b <- model_result$biases_record[[L]]
            if (!is.null(b)) {
              # bias could be vector or 1-col matrix
              blen <- if (is.matrix(b)) nrow(b) * ncol(b) else length(b)
              if (isTRUE(verbose)) message(sprintf("  Layer %d: %d", L, blen))
            } else {
              if (isTRUE(verbose)) message(sprintf("  Layer %d: NULL", L))
            }
          }
        } else {
          blen <- if (is.matrix(model_result$biases_record)) length(model_result$biases_record) else length(model_result$biases_record)
          if (isTRUE(verbose)) message("Biases record length (SL): ", blen)
        }
        
        # if (isTRUE(verbose)) Sys.sleep(0.25)  # pause slightly for readability
      }
      
      .first_nonnull <- function(...) { for (x in list(...)) if (!is.null(x)) return(x); NULL }
      
      labels_arg <- .first_nonnull(
        get0("labels", ifnotfound = NULL, inherits = FALSE),  # only if bound here
        y_validation,                                         # since you already have it here
        get0("y", ifnotfound = NULL, inherits = TRUE)         # legacy fallback; no X_* fallbacks
      )
      

      
      cat(sprintf("[TRACE] BEFORE update_performance_and_relevance @ %s\n", #$$$$$$$$$$$$$
                  format(Sys.time(), "%H:%M:%OS3"))) #$$$$$$$$$$$$$
      
      performance_relevance_data <- self$update_performance_and_relevance(
        Rdata                            = Rdata,
        labels                           = labels_arg,
        num_networks                     = num_networks,
        update_weights                   = update_weights,
        update_biases                    = update_biases,
        preprocessScaledData             = preprocessScaledData,
        X_validation                     = X_validation,
        y_validation                     = y_validation,
        validation_metrics               = validation_metrics,
        lr                               = lr,
        CLASSIFICATION_MODE              = CLASSIFICATION_MODE,
        ensemble_number                  = ensemble_number,
        model_iter_num                   = model_iter_num,
        num_epochs                       = num_epochs,
        threshold                        = threshold,
        threshold_function               = threshold_function,
        learn_results                    = learn_results,
        predicted_output_list            = all_predicted_outputs,
        all_best_val_probs               = all_best_val_probs,
        all_best_val_labels              = all_best_val_labels,
        all_best_val_prediction_time     = all_best_val_prediction_time,
        learn_time                       = all_learn_times,
        prediction_time_list             = all_prediction_times,
        run_id                           = all_ensemble_name_model_name,
        all_predicted_outputAndTime      = all_predicted_outputAndTime,
        all_weights                      = all_weights,
        all_biases                       = all_biases,
        all_activation_functions         = all_activation_functions,
        all_activation_functions_predict = all_activation_functions_predict,
        all_best_train_acc               = all_best_train_acc,
        all_best_epoch_train             = all_best_epoch_train,
        all_best_train_loss              = all_best_train_loss,
        all_best_epoch_train_loss        = all_best_epoch_train_loss,
        all_best_val_acc                 = all_best_val_acc,
        all_best_val_epoch               = all_best_val_epoch,
        best_weights_on_latest_weights_off = best_weights_on_latest_weights_off,
        ML_NN = ML_NN,
        train = train,
        grouped_metrics = grouped_metrics,
        viewTables = viewTables,
        verbose = verbose
      )
      
      cat(sprintf("[TRACE] AFTER update_performance_and_relevance @ %s\n", #$$$$$$$$$$$$$
                  format(Sys.time(), "%H:%M:%OS3"))) #$$$$$$$$$$$$$
      
      `%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a
      
      # Prints to RStudio Plots pane ONLY. Never saves. Handles ggplot, list, and nested list.
      print_plotlist_verbose <- function(x, label = NULL, print_plots = TRUE) {
        lab <- label %||% "Plot"
        if (inherits(x, c("gg","ggplot"))) {
          if (print_plots) print(x)
          return(invisible(NULL))
        }
        if (is.list(x)) {
          for (i in seq_along(x)) {
            item <- x[[i]]
            if (inherits(item, c("gg","ggplot")) && print_plots) print(item)
            if (is.list(item)) {
              for (j in seq_along(item)) {
                p <- item[[j]]
                if (inherits(p, c("gg","ggplot")) && print_plots) print(p)
              }
            }
          }
        }
        invisible(NULL)
      }
      
      # =========================
      # DDESONN — Final perf/relevance lists (uses self$viewFinalUpdatePerformanceandRelevancePlots)
      # =========================
      
      `%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a
      
      # Helper: ask your R6 toggle
      .allow <- function(name) {
        # name should match a field in self$FinalUpdatePerformanceandRelevanceViewPlotsConfig
        # on_all = viewAllPlots || verbose
        self$viewFinalUpdatePerformanceandRelevancePlots(name)
      }
      
      # Optional: allow disabling saving too (defaults TRUE if unset)
      .save_enabled <- isTRUE(self$FinalUpdatePerformanceandRelevanceViewPlotsConfig$saveAlso %||% TRUE)
      
      # Prepare output dir only if we might save
      if (.save_enabled) plots_dir <- ddesonn_plots_dir(output_root)
      
      ens <- as.integer(ensemble_number)
      tot <- if (!is.null(self$ensemble)) length(self$ensemble) else as.integer(get0("num_networks", ifnotfound = 1L))
      mod <- if (exists("model_iter_num", inherits = TRUE) && length(model_iter_num)) as.integer(model_iter_num) else 1L
      
      # Ensure we have a filename namer
      if (!exists("fname", inherits = TRUE) || !is.function(fname)) {
        fname <- make_fname_prefix(
          isTRUE(get0("do_ensemble", ifnotfound = FALSE)),
          num_networks    = tot,
          total_models    = tot,
          ensemble_number = ens,
          model_index     = mod,
          who             = "DDESONN"
        )
      }
      
      .slug <- function(s) {
        s <- trimws(as.character(s))
        s <- gsub("\\s+", "_", s)
        s <- gsub("[^A-Za-z0-9_]+", "_", s)
        tolower(gsub("_+", "_", s))
      }
      
      .plot_label_slug <- function(p) {
        .first_label <- function(obj) {
          if (is.null(obj)) return(NULL)
          s <- tryCatch(as.character(obj), error = function(e) NULL)
          if (is.null(s) || !length(s)) return(NULL)
          s1 <- s[[1]]; if (is.null(s1) || !nzchar(s1)) return(NULL)
          s1
        }
        t1 <- .first_label(tryCatch(p$labels$title, error = function(e) NULL))
        if (!is.null(t1)) return(.slug(t1))
        y1 <- .first_label(tryCatch(p$labels$y,     error = function(e) NULL))
        if (!is.null(y1)) return(.slug(y1))
        NULL
      }
      
      # Walk any mixture of ggplot / list / nested list; save/print only if allowed(name)
      .walk_save_view <- function(x, base, idx_env, name_flag) {
        if (is.null(x) || !length(x)) return(invisible(NULL))
        
        # Respect your config per group
        do_action <- .allow(name_flag)
        
        save_one <- function(p, nm_fallback) {
          if (!do_action) return(invisible(NULL))   # neither save nor print if flag is off
          
          nm <- .plot_label_slug(p) %||% .slug(nm_fallback)
          idx_env[[nm]] <- (idx_env[[nm]] %||% 0L) + 1L
          file_base <- sprintf("%s_%03d", nm, idx_env[[nm]])
          out <- file.path(ddesonn_plots_dir(output_root), fname(sprintf("%s.png", file_base)))
          
          # Save only if global save is enabled
          if (.save_enabled) {
            try(suppressWarnings(suppressMessages(
              ggplot2::ggsave(out, p, width = 6, height = 4, dpi = 300)
            )), silent = TRUE)
          }
          
          # Print (view) — gated by same per-group flag
          try(print(p), silent = TRUE)
        }
        
        if (inherits(x, c("gg","ggplot"))) {
          save_one(x, base)
        } else if (is.list(x)) {
          for (k in seq_along(x)) {
            elem <- x[[k]]
            if (inherits(elem, c("gg","ggplot"))) {
              save_one(elem, sprintf("%s_%02d", base, k))
            } else if (is.list(elem)) {
              .walk_save_view(elem, sprintf("%s_%02d", base, k), idx_env, name_flag)
            }
          }
        }
        invisible(NULL)
      }
      
      # Independent counters per group to keep filenames stable
      .idx <- new.env(parent = emptyenv())
      
      # Map each holder to its config flag name (keys must match your config fields)
      .walk_save_view(performance_relevance_data$performance_high_mean_plots, "performance_high_mean",
                      .idx, "performance_high_mean_plots")
      .walk_save_view(performance_relevance_data$performance_low_mean_plots,  "performance_low_mean",
                      .idx, "performance_low_mean_plots")
      .walk_save_view(performance_relevance_data$relevance_high_mean_plots,   "relevance_high_mean",
                      .idx, "relevance_high_mean_plots")
      .walk_save_view(performance_relevance_data$relevance_low_mean_plots,    "relevance_low_mean",
                      .idx, "relevance_low_mean_plots")
      
      invisible(NULL)
      
      
      # ============================================================
      # SECTION: EVOKE — post-train metadata enrichment (NOT per-epoch)
      # ============================================================
      cat(sprintf(
        "[EVOKE-ENRICH] where=%s | why=%s | note=%s | epochs_configured=%s\n",
        "DDESONN$train::post_train_metadata",
        "About to attach per-slot metadata; this may call ddesonn_predict() again (aggregate='none')",
        "This EVOKE is NOT per-epoch; epochs run inside model$train() (not exposed here)",
        as.character(num_epochs %||% NA_integer_)
      ))
      
      # ============================================================
      # SECTION: EVOKE — starting enrichment predicts (NOT per-epoch)
      # ============================================================
      cat(sprintf(
        "[EVOKE-ENRICH-BEGIN] where=%s | why=%s\n",
        "DDESONN$train::post_train_metadata",
        "Computing per-member TRAIN/VALID predictions for metadata (calls ddesonn_predict -> net$predict)"
      ))
      
      # =========================
      # Attach per-slot metrics
      # =========================
      model <- self
      mode <- tolower(CLASSIFICATION_MODE %||% "binary")
      
      if (mode %in% c("binary", "multiclass")) {
        thr_used <- threshold %||% .ddesonn_threshold_default(mode)
        
        # TRAIN predictions (per-member)
        pr_train <- try(ddesonn_predict(model, Rdata, aggregate = "none", type = "response"), silent = TRUE)
        per_model_train <- if (!inherits(pr_train, "try-error")) pr_train$per_model else NULL
        
        # VALID predictions (per-member) if present
        per_model_valid <- NULL
        if (!is.null(X_validation)) {
          pr_valid <- try(ddesonn_predict(model, X_validation, aggregate = "none", type = "response"), silent = TRUE)
          per_model_valid <- if (!inherits(pr_valid, "try-error")) pr_valid$per_model else NULL
        }
        
        # prepare true labels as vectors for metric calc
        y_train_vec <- NULL
        y_valid_vec <- NULL
        if (mode == "binary") {
          y_train_vec <- as.numeric(labels[, 1])
          if (!is.null(y_validation)) y_valid_vec <- as.numeric(y_validation[, 1])
        } else { # multiclass
          y_train_vec <- max.col(labels, ties.method = "first")
          if (!is.null(y_validation)) y_valid_vec <- max.col(y_validation, ties.method = "first")
        }
        
        best_train_acc             <- tryCatch(predicted_outputAndTime$best_train_acc,           error = function(e) NA_real_)
        best_epoch_train           <- tryCatch(predicted_outputAndTime$best_epoch_train,         error = function(e) NA_integer_)
        best_train_loss            <- tryCatch(predicted_outputAndTime$best_train_loss,          error = function(e) NA_real_) 
        best_epoch_train_loss      <- tryCatch(predicted_outputAndTime$best_epoch_train_loss,    error = function(e) NA_integer_)
        best_val_acc               <- tryCatch(predicted_outputAndTime$best_val_acc,             error = function(e) NA_real_)
        best_val_epoch             <- tryCatch(predicted_outputAndTime$best_val_epoch,           error = function(e) NA_integer_)
        best_val_prediction_time   <- tryCatch(predicted_outputAndTime$best_val_prediction_time, error = function(e) NA_real_)
        
        Kslots <- try(length(model$ensemble), silent = TRUE)
        if (!inherits(Kslots, "try-error") && is.finite(Kslots) && Kslots >= 1L) {
          for (k in seq_len(Kslots)) {
            slot_obj <- try(model$ensemble[[k]], silent = TRUE)
            if (inherits(slot_obj, "try-error") || is.null(slot_obj)) next
            if (is.null(slot_obj$metadata)) slot_obj$metadata <- list()
            
            # Always stamp mode so downstream writers can respect it
            slot_obj$metadata$classification_mode <- mode
            
            # TRAIN metrics per slot
            if (!is.null(per_model_train) && length(per_model_train) >= k) {
              Pt <- as.matrix(per_model_train[[k]])
              if (mode == "binary") {
                cm_tr <- confusion_matrix(
                  SONN = slot_obj,
                  labels = matrix(y_train_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "binary",
                  predicted_output = matrix(as.numeric(Pt[, 1]), ncol = 1L),
                  threshold = thr_used,
                  verbose = FALSE
                )
                TP <- cm_tr$TP; FP <- cm_tr$FP; TN <- cm_tr$TN; FN <- cm_tr$FN
                total <- TP + FP + TN + FN
                acc <- if (total > 0) (TP + TN) / total else NA_real_
                prec <- if ((TP + FP) > 0) TP / (TP + FP) else NA_real_
                rec  <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
                f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) 2 * prec * rec / (prec + rec) else NA_real_
                slot_obj$metadata$performance_metric <- list(
                  accuracy = acc,
                  precision = prec,
                  recall = rec,
                  f1 = f1,
                  f1_score = f1
                )
                slot_obj$metadata$confusion_matrix <- cm_tr
              } else {
                acc_val <- accuracy(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_train_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pt,
                  verbose = FALSE
                )
                prec_val <- precision(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_train_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pt,
                  verbose = FALSE
                )
                rec_val <- recall(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_train_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pt,
                  verbose = FALSE
                )
                f1_val <- f1(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_train_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pt,
                  verbose = FALSE
                )
                slot_obj$metadata$performance_metric <- list(
                  accuracy = acc_val,
                  precision = prec_val,
                  recall = rec_val,
                  f1 = f1_val,
                  f1_score = f1_val
                )
              }
            }
            
            # VALID metrics per slot (optional)
            if (!is.null(per_model_valid) && length(per_model_valid) >= k && !is.null(y_valid_vec)) {
              Pv <- as.matrix(per_model_valid[[k]])
              if (mode == "binary") {
                cm_va <- confusion_matrix(
                  SONN = slot_obj,
                  labels = matrix(y_valid_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "binary",
                  predicted_output = matrix(as.numeric(Pv[, 1]), ncol = 1L),
                  threshold = thr_used,
                  verbose = FALSE
                )
                TP <- cm_va$TP; FP <- cm_va$FP; TN <- cm_va$TN; FN <- cm_va$FN
                total <- TP + FP + TN + FN
                acc <- if (total > 0) (TP + TN) / total else NA_real_
                prec <- if ((TP + FP) > 0) TP / (TP + FP) else NA_real_
                rec  <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
                f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) 2 * prec * rec / (prec + rec) else NA_real_
                # Only in BINARY: expose the tuned bundle (prevents utils from “thinking binary” otherwise)
                slot_obj$metadata$accuracy_precision_recall_f1_tuned <- list(
                  accuracy = acc,
                  precision = prec,
                  recall = rec,
                  f1 = f1,
                  confusion_matrix = cm_va,
                  chosen_threshold = thr_used
                )
              } else {
                # Multiclass: NO tuned bundle, NO confusion_matrix (keeps downstream from mapping binary fields)
                acc_val <- accuracy(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_valid_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pv,
                  verbose = FALSE
                )
                prec_val <- precision(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_valid_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pv,
                  verbose = FALSE
                )
                rec_val <- recall(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_valid_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pv,
                  verbose = FALSE
                )
                f1_val <- f1(
                  SONN = slot_obj,
                  Rdata = Rdata,
                  labels = matrix(y_valid_vec, ncol = 1L),
                  CLASSIFICATION_MODE = "multiclass",
                  predicted_output = Pv,
                  verbose = FALSE
                )
                # keep validation macro metrics merged into performance_metric if you want:
                # (optional) slot_obj$metadata$valid_performance_metric <- list(
                #   accuracy = acc_val,
                #   precision = prec_val,
                #   recall = rec_val,
                #   f1 = f1_val,
                #   f1_score = f1_val
                # )
              }
            }
            
            # Best fields
            slot_obj$metadata$best_train_acc           <- .take1num(best_train_acc)
            slot_obj$metadata$best_epoch_train         <- .int(best_epoch_train %||% NA_integer_)
            slot_obj$metadata$best_train_loss          <- .take1num(best_train_loss)
            slot_obj$metadata$best_epoch_train_loss    <- .int(best_epoch_train_loss %||% NA_integer_)
            slot_obj$metadata$best_val_acc             <- .take1num(best_val_acc)
            slot_obj$metadata$best_val_epoch           <- .int(best_val_epoch %||% NA_integer_)
            slot_obj$metadata$best_val_prediction_time <- .take1num(best_val_prediction_time %||% NA_real_)
          }
        }
      } else if (mode == "regression") {
        # TRAIN predictions (per-member)
        pr_train <- try(ddesonn_predict(model, Rdata, aggregate = "none", type = "response"), silent = TRUE)
        per_model_train <- if (!inherits(pr_train, "try-error")) pr_train$per_model else NULL
        
        # VALID predictions (per-member) if present
        per_model_valid <- NULL
        if (!is.null(X_validation)) {
          pr_valid <- try(ddesonn_predict(model, X_validation, aggregate = "none", type = "response"), silent = TRUE)
          per_model_valid <- if (!inherits(pr_valid, "try-error")) pr_valid$per_model else NULL
        }
        
        # true labels as numeric vectors
        y_train_vec <- as.numeric(labels[, 1])
        y_valid_vec <- if (!is.null(y_validation)) as.numeric(y_validation[, 1]) else NULL
        
        Kslots <- try(length(model$ensemble), silent = TRUE)
        if (!inherits(Kslots, "try-error") && is.finite(Kslots) && Kslots >= 1L) {
          for (k in seq_len(Kslots)) {
            slot_obj <- try(model$ensemble[[k]], silent = TRUE)
            if (inherits(slot_obj, "try-error") || is.null(slot_obj)) next
            if (is.null(slot_obj$metadata)) slot_obj$metadata <- list()
            
            # Always stamp mode so downstream writers can respect it
            slot_obj$metadata$classification_mode <- mode
            
            # TRAIN metrics per-slot
            if (!is.null(per_model_train) && length(per_model_train) >= k) {
              pt <- as.numeric(per_model_train[[k]][, 1])
              slot_obj$metadata$performance_metric <- list(
                MSE = MSE(slot_obj, Rdata, y_train_vec, "regression", pt, verbose = FALSE),
                RMSE = RMSE(slot_obj, Rdata, y_train_vec, "regression", pt, verbose = FALSE),
                MAE = MAE(slot_obj, Rdata, y_train_vec, "regression", pt, verbose = FALSE),
                R2 = R2(slot_obj, Rdata, y_train_vec, "regression", pt, verbose = FALSE)
              )
            }
            
            # VALID metrics per-slot (optional)
            if (!is.null(per_model_valid) && length(per_model_valid) >= k && !is.null(y_valid_vec)) {
              pv <- as.numeric(per_model_valid[[k]][, 1])
              slot_obj$metadata$validation_metrics <- list(
                MSE = MSE(slot_obj, X_validation, y_valid_vec, "regression", pv, verbose = FALSE),
                RMSE = RMSE(slot_obj, X_validation, y_valid_vec, "regression", pv, verbose = FALSE),
                MAE = MAE(slot_obj, X_validation, y_valid_vec, "regression", pv, verbose = FALSE),
                R2 = R2(slot_obj, X_validation, y_valid_vec, "regression", pv, verbose = FALSE)
              )
            }
            
            # Carry best_* fields if present from training result (harmless if NA)
            slot_obj$metadata$best_train_acc           <- .take1num(tryCatch(predicted_outputAndTime$best_train_acc,           error=function(e) NA_real_))
            slot_obj$metadata$best_epoch_train         <- .int(     tryCatch(predicted_outputAndTime$best_epoch_train,         error=function(e) NA_integer_))
            slot_obj$metadata$best_train_loss          <- .take1num(tryCatch(predicted_outputAndTime$best_train_loss,          error=function(e) NA_real_))
            slot_obj$metadata$best_epoch_train_loss    <- .int(     tryCatch(predicted_outputAndTime$best_epoch_train_loss,    error=function(e) NA_integer_))
            slot_obj$metadata$best_val_acc             <- .take1num(tryCatch(predicted_outputAndTime$best_val_acc,             error=function(e) NA_real_))
            slot_obj$metadata$best_val_epoch           <- .int(     tryCatch(predicted_outputAndTime$best_val_epoch,           error=function(e) NA_integer_))
            slot_obj$metadata$best_val_prediction_time <- .take1num(tryCatch(predicted_outputAndTime$best_val_prediction_time, error=function(e) NA_real_))
          }
        }
      }
      
      # ============================================================
      # SECTION: EVOKE — enrichment finished (NOT per-epoch)
      # ============================================================
      cat(sprintf(
        "[EVOKE-ENRICH-END] where=%s | why=%s | note=%s\n",
        "DDESONN$train::post_train_metadata",
        "Finished attaching per-slot metadata; predict() prints after FINAL SUMMARY usually come from this block",
        "Not per-epoch: epoch-level prints must live inside model$train()/train_network() predict call sites"
      ))
      
      
      
      if(verbose){print("----------------------------------------train-end----------------------------------------")}
      
      return(list(predicted_outputAndTime = predicted_outputAndTime, performance_relevance_data = performance_relevance_data))
    }
    , # Method for updating performance and relevance metrics

    update_performance_and_relevance = function(Rdata, labels, num_networks, update_weights, update_biases, preprocessScaledData, X_validation, y_validation, validation_metrics, lr, CLASSIFICATION_MODE, ensemble_number, model_iter_num, num_epochs, threshold, threshold_function, learn_results, predicted_output_list, all_best_val_probs, all_best_val_labels, all_best_val_prediction_time, learn_time, prediction_time_list, run_id, all_predicted_outputAndTime, all_weights, all_biases, all_activation_functions, all_activation_functions_predict, all_best_train_acc, all_best_epoch_train, all_best_train_loss, all_best_epoch_train_loss, all_best_val_acc, all_best_val_epoch, best_weights_on_latest_weights_off, ML_NN, train, grouped_metrics, viewTables, verbose) {
      if(verbose){print("----------------------------------------update_performance_and_relevance-begin----------------------------------------")}

      # Initialize lists to store performance and relevance metrics for each SONN
      performance_list <- list()
      relevance_list <- list()
      model_name_list <-  list()
      #████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████

      # Calculate performance and relevance for each SONN in the ensemble

      for (i in 1:length(self$ensemble)) {


        best_val_probs <- all_best_val_probs[[i]]
        best_val_labels <- all_best_val_labels[[i]]
        best_val_prediction_time <- all_best_val_prediction_time[[i]]

        best_train_acc <- all_best_train_acc[[i]]
        best_epoch_train <- all_best_epoch_train[[i]]
        best_train_loss <- all_best_train_loss[[i]]
        best_epoch_train_loss <- all_best_epoch_train_loss[[i]]
        best_val_acc <- all_best_val_acc[[i]]
        best_val_epoch <- all_best_val_epoch[[i]]

        single_predicted_outputAndTime <- all_predicted_outputAndTime[[i]]  # metadata
        single_predicted_output <- predicted_output_list[[i]]
        single_ensemble_name_model_name <- run_id[[i]]
        
        single_activation_functions <- all_activation_functions[[i]]
        single_activation_functions_predict <- all_activation_functions_predict[[i]]
        
        # might remove if, but keep contents
        if (train) {


          cat("___________________________________________________________________________\n")
          cat("______________________________DESONN_", ensemble_number, "_SONN_", i, "______________________________\n", sep = "")

          single_prediction_time <- prediction_time_list[[i]]

          #brought X_validation and y_validation as close as possible to metrics without "doubling-up" vars per se
          if (validation_metrics){
            Rdata = X_validation
            labels = y_validation
          }

          # ---- PRE-REPORT TUNING (binary only) ----
          tuned <- NULL
          best_threshold <- NA_real_
          # need to figure out if I need to add is.null(predicted_output_val) && to the ifs below
          if ( !is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
            # choose snapshot to tune: prefer best_val_*; else last-epoch predictions
            if (!is.null(best_val_probs) && !is.null(best_val_labels)) {
              probs_for_tuning  <- as.matrix(best_val_probs)
              labels_for_tuning <- if (is.matrix(best_val_labels)) best_val_labels[, 1] else best_val_labels
              labels_for_tuning <- as.integer(labels_for_tuning)
            } else {
              fallback_probs <- single_predicted_output
              if (is.list(fallback_probs) && !is.null(fallback_probs$predicted_output)) {
                fallback_probs <- fallback_probs$predicted_output
              }
              probs_for_tuning <- as.matrix(fallback_probs)
              n_eff <- min(NROW(probs_for_tuning), if (is.matrix(y_validation)) NROW(y_validation) else length(y_validation))
              y_val_vec <- if (is.matrix(y_validation)) {
                y_validation[seq_len(n_eff), 1]
              } else {
                y_validation[seq_len(n_eff)]
              }
              labels_for_tuning <- as.integer(y_val_vec)
            }

            tuned <- accuracy_precision_recall_f1_tuned(
              SONN                = self$ensemble[[i]],
              Rdata               = tryCatch(X_validation, error = function(e) NULL),
              labels              = matrix(labels_for_tuning, ncol = 1L),
              CLASSIFICATION_MODE = "binary",
              predicted_output    = probs_for_tuning,
              metric_for_tuning   = "accuracy",
              threshold_grid      = seq(0.05, 0.95, by = 0.01),
              verbose             = isTRUE(verbose)
            )

            chosen_threshold <- suppressWarnings(as.numeric(tuned$details$best_threshold))
            if (!is.finite(chosen_threshold)) chosen_threshold <- 0.5

            best_threshold <- chosen_threshold
            self$ensemble[[i]]$chosen_threshold <- chosen_threshold
          }
# print(head(single_predicted_output))
# print(head(y_validation))
# stop()
          # === Evaluate Prediction Diagnostics ===
          if (!is.null(X_validation) && !is.null(y_validation) && isTRUE(validation_metrics)) {
            eval_result <- EvaluatePredictionsReport(
              X_validation = X_validation,
              y_validation = y_validation,
              CLASSIFICATION_MODE = CLASSIFICATION_MODE,
              probs = single_predicted_output,
              predicted_outputAndTime = single_predicted_outputAndTime,
              threshold_function = threshold_function,    # still accepted, no-op
              all_best_val_probs = best_val_probs,
              all_best_val_labels = best_val_labels,
              verbose = verbose,
              tuned_threshold_override = best_threshold,
              SONN = self$ensemble[[i]]
            )
            # Mirror back for downstream readers that expect the report field
            if (is.finite(best_threshold)) {
              eval_result$best_threshold <- best_threshold
            }
          } else {
            # Ensure eval_result exists for later fields even if no validation path
            eval_result <- list(best_threshold = NA_real_, best_thresholds = NULL, accuracy = NA_real_, accuracy_percent = NA_real_, metrics = NULL, misclassified = NULL)
          }

          # Safely get number of columns from many shapes
          safe_ncol <- function(x) {
            if (is.null(x)) return(0L)
            # If it's a list from self$predict, try common fields first
            if (is.list(x)) {
              if (!is.null(x$predicted_output)) return(safe_ncol(x$predicted_output))
              if (!is.null(x$preds))            return(safe_ncol(x$preds))
              if (!is.null(x$output))           return(safe_ncol(x$output))
              # last resort: if it still has a dim attribute
              if (!is.null(dim(x))) return(ncol(x))
              return(0L)
            }
            if (is.matrix(x))      return(ncol(x))
            if (is.data.frame(x))  return(ncol(x))
            if (!is.null(dim(x)))  return(dim(x)[2])
            # vectors/scalars count as 1 col (binary probs)
            if (is.atomic(x))      return(1L)
            0L
          }

          k_labels <- safe_ncol(y_validation)
          k_probs  <- safe_ncol(single_predicted_output)

          # Prefer label-driven K when available; otherwise use predictions
          K <- if (k_labels > 0L) max(1L, k_labels) else max(1L, k_probs)


          # Pull out both fields (back-compat + multiclass)
          best_threshold_scalar <- eval_result$best_threshold          # numeric (binary) or NA (multiclass)
          best_thresholds_vec   <- eval_result$best_thresholds         # vector: length 1 (binary) or K (multiclass)

          # Decide what to store/use
          if (K == 1L) {
            # Binary: prefer tuned scalar; fallback to 0.5 if NA
            threshold_used   <- if (is.finite(best_threshold_scalar)) best_threshold_scalar else 0.5
            thresholds_used  <- best_thresholds_vec  # length-1 vector (kept for consistency)
          } else {
            # Multiclass: no single scalar; keep the whole vector
            threshold_used   <- NA_real_
            # if somehow missing, fallback to 0.5 per class
            thresholds_used  <- if (!is.null(best_thresholds_vec) && length(best_thresholds_vec) == K) {
              best_thresholds_vec
            } else {
              rep(0.5, K)
            }
          }

          # Optional: logs
          if (isTRUE(verbose)) {
            if (K == 1L) {
              message(sprintf("[train] Using tuned binary threshold: %.3f", threshold_used))
            } else {
              message(sprintf("[train] Using tuned per-class thresholds: %s",
                              paste0(sprintf("%.3f", thresholds_used), collapse = ", ")))
            }
          }


          if (best_weights_on_latest_weights_off && !is.null(best_val_probs) && !is.null(best_val_labels)) {
            probs   <- best_val_probs
            targets <- best_val_labels
            prediction_time <- best_val_prediction_time
            cat("[calculate_performance] Using best validation snapshot (@ best epoch)\n")
          } else {
            probs   <- single_predicted_output
            targets <- labels
            prediction_time <- single_prediction_time
            cat("[calculate_performance] Using last-epoch predictions\n")
          }


          performance_list[[i]] <- calculate_performance(
            SONN = self$ensemble[[i]],
            Rdata = Rdata,
            labels = targets,
            lr = lr,
            CLASSIFICATION_MODE = CLASSIFICATION_MODE,
            model_iter_num = i,
            num_epochs = num_epochs,
            threshold = if (K == 1L) threshold_used else threshold,
            learn_time = learn_time,
            predicted_output = probs,
            prediction_time = prediction_time,
            ensemble_number = ensemble_number,
            run_id = run_id,
            weights = all_weights[[i]],
            biases = all_biases[[i]],
            activation_functions = all_activation_functions[[i]],
            ML_NN = ML_NN,
            verbose = verbose
          )

          relevance_list[[i]] <- calculate_relevance(
            self$ensemble[[i]],
            Rdata = Rdata,
            labels = targets,
            CLASSIFICATION_MODE = CLASSIFICATION_MODE,
            model_iter_num = i,
            predicted_output = probs,
            ensemble_number = ensemble_number,
            weights = self$ensemble[[i]]$weights,
            biases = self$ensemble[[i]]$biases,
            activation_functions = self$ensemble[[i]]$activation_functions,
            ML_NN = ML_NN,
            verbose = verbose
          )

          performance_metric <- performance_list[[i]]$metrics

          # Fill tuned columns directly from the pre-report tuning pass
          if (!is.null(tuned) && identical(CLASSIFICATION_MODE, "binary")) {
            performance_metric$accuracy_tuned  <- tuned$accuracy
            performance_metric$precision_tuned <- tuned$precision
            performance_metric$recall_tuned    <- tuned$recall
            performance_metric$f1_tuned        <- tuned$f1
          }

          relevance_metric <- relevance_list[[i]]$metrics

          if (ensemble_number < 1 && length(self$ensemble) >= 1 || (verbose && (ensemble_number < 1 && length(self$ensemble) >= 1))) {
            if (verbose || viewTables) {
              message(sprintf(">> METRICS FOR ENSEMBLE: %s MODEL: %s", ensemble_number, i))
              emit_table(
                performance_metric,
                title = "[PERFORMANCE metrics]",
                verbose = verbose,
                viewTables = viewTables
              )
              emit_table(
                relevance_metric,
                title = "[RELEVANCE metrics]",
                verbose = verbose,
                viewTables = viewTables
              )
            }

          }

        }


        if (verbose) {
          message("\n=============================================")
          message("DEBUG: Preparing to store metadata (Network Container)")
          message(sprintf("Ensemble number: %s", ensemble_number))
          message(sprintf("Model iteration: %s", i))
          message(sprintf("Run ID: %s", single_ensemble_name_model_name))
          pred_shape <- tryCatch(paste(dim(single_predicted_output), collapse = " × "), error = function(...) "<unknown>")
          message(sprintf("Predicted output shape: %s", pred_shape))
          message(sprintf("Checking self$ensemble[[%s]]", i))
          captured <- utils::capture.output(str(self$ensemble[[i]]))
          for (line in captured) message(line)
          message("=============================================\n")
        }


        self$store_metadata(
          single_predicted_outputAndTime, actual_values = NULL, do_ensemble = NULL, self$input_size, self$output_size, self$N,
          total_num_samples = NULL, num_test_samples = NULL, num_training_samples = NULL, num_validation_samples = NULL,
          num_networks, update_weights, update_biases, lr, self$lambda, num_epochs,
          run_id = single_ensemble_name_model_name, ensemble_number, model_iter_num = i,
          model_serial_num = sprintf("%d.0.%d", ensemble_number, i),
          threshold = if (exists("threshold_used") && isTRUE(is.finite(threshold_used))) threshold_used else NULL,
          CLASSIFICATION_MODE, predicted_output = single_predicted_output, preprocessScaledData,
          X = NULL, y = NULL, X_test_scaled = NULL, y_test = NULL, all_weights, all_biases, artifact_names, artifact_paths,
          validation_metrics = validation_metrics, single_activation_functions, single_activation_functions_predict,
          self$dropout_rates, self$hidden_sizes, self$ML_NN, best_val_prediction_time, best_train_acc,
          best_epoch_train, best_train_loss, best_epoch_train_loss, best_val_acc, best_val_epoch, performance_metric, relevance_metric, NULL
        )

      }


      #████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████

      # Extract names and metrics for performance and relevance
      performance_metrics <- lapply(seq_along(performance_list), function(i) performance_list[[i]]$metrics) #<<-
      performance_names <- lapply(seq_along(performance_list), function(i) performance_list[[i]]$names) #<<-

      relevance_metrics <- lapply(seq_along(relevance_list), function(i) relevance_list[[i]]$metrics) #<<-
      relevance_names <- lapply(seq_along(relevance_list), function(i) relevance_list[[i]]$names) #<<-

      # Check for NULL values in performance_metrics and relevance_metrics
      check_null <- function(metrics_list) {
        unlist(lapply(seq_along(metrics_list), function(i) {
          if (is.null(metrics_list[[i]])) {
            return(paste0("NULL at index: ", i))
          } else {
            return(paste0("Not NULL at index: ", i))
          }
        }))
      }
      check_null(performance_metrics)
      null_check_performance <- check_null(performance_metrics) #<<-
      check_null(relevance_metrics)
      null_check_relevance <- check_null(relevance_metrics) #<<-




      # Convert the matrix to a data frame without modifying row names
      # Initialize empty vectors to store values and row names

      # Function to process performance metrics
      process_performance <- function(metrics_data, model_names, high_threshold = 10, verbose = FALSE) {
        EXCLUDE_METRICS_REGEX <- paste(
          c(
            "^accuracy_precision_recall_f1_tuned_details_accuracy_percent$",
            "^accuracy_percent$",
            "^accuracy_precision_recall_f1_tuned_details_y_pred_class",
            "^y_pred_class",
            "^accuracy_precision_recall_f1_tuned_details_grid_used",
            "^grid_used"
          ),
          collapse = "|"
        )

        # ---- model names handling ----
        if (length(model_names) == 1L && length(metrics_data) > 1L) {
          model_names <- rep(model_names, length(metrics_data))
        }
        if (is.null(model_names) || length(model_names) != length(metrics_data)) {
          model_names <- paste0("Model_", seq_along(metrics_data))
        }

        # ---- helpers ----
        to_numeric_safely <- function(v) {
          v <- as.character(v)
          cleaned <- gsub("[^0-9eE+\\-\\.]", "", v)
          suppressWarnings(as.numeric(cleaned))
        }
        norm_atom <- function(x) {
          if (inherits(x, "Duration"))  return(as.numeric(x))                 # seconds
          if (inherits(x, "difftime"))  return(as.numeric(x, units = "secs")) # seconds
          if (inherits(x, "POSIXct") || inherits(x, "POSIXt")) return(as.numeric(x))
          if (inherits(x, "Date"))      return(as.numeric(x))
          if (is.logical(x))            return(as.numeric(x))
          if (is.factor(x))             return(as.character(x))
          x
        }
        # Flatten into named atomic elements (as list of scalars/vectors)
        flatten_metrics <- function(x, prefix = NULL) {
          out <- list()
          nm_prefix <- function(base, name) if (is.null(base) || base == "") name else paste0(base, "_", name)

          if (is.null(x)) return(out)

          if (is.atomic(x) && length(x) >= 1L) {
            # keep vectors; caller will split to rows
            nm <- if (is.null(prefix)) "value" else prefix
            out[[nm]] <- x
            return(out)
          }

          if (is.data.frame(x)) {
            for (nm in names(x)) {
              out <- c(out, flatten_metrics(x[[nm]], nm_prefix(prefix, nm)))
            }
            return(out)
          }

          if (is.list(x)) {
            nms <- names(x)
            for (i in seq_along(x)) {
              nm <- if (!is.null(nms) && nzchar(nms[i])) nms[i] else as.character(i)
              out <- c(out, flatten_metrics(x[[i]], nm_prefix(prefix, nm)))
            }
            return(out)
          }

          out  # unknown type -> ignore
        }

        build_long_df <- function(lst, model_name) {
          if (is.null(lst) || length(lst) == 0L) {
            if (verbose) message("[process_performance] empty metrics for ", model_name)
            return(data.frame(Model_Name = character(0), Metric = character(0), Value = numeric(0)))
          }
          rows <- list()
          idx <- 1L
          for (nm in names(lst)) {
            val <- lst[[nm]]
            if (length(val) == 0L) next
            val <- norm_atom(val)
            if (length(val) == 0L) next

            # split vectors to multiple rows with indexed metric names (stable & unique)
            if (length(val) > 1L) {
              for (k in seq_along(val)) {
                rows[[idx]] <- data.frame(
                  Model_Name = model_name,
                  Metric     = paste0(nm, "_", k),
                  Value      = as.character(val[[k]]),
                  stringsAsFactors = FALSE, check.names = FALSE
                )
                idx <- idx + 1L
              }
            } else {
              rows[[idx]] <- data.frame(
                Model_Name = model_name,
                Metric     = nm,
                Value      = as.character(val),
                stringsAsFactors = FALSE, check.names = FALSE
              )
              idx <- idx + 1L
            }
          }
          if (length(rows) == 0L) {
            data.frame(Model_Name = character(0), Metric = character(0), Value = numeric(0))
          } else {
            do.call(rbind, rows)
          }
        }

        high_mean_df <- NULL
        low_mean_df  <- NULL

        for (i in seq_along(metrics_data)) {
          mdl_name <- model_names[[i]]
          met_raw  <- metrics_data[[i]]

          flat <- flatten_metrics(met_raw, NULL)
          long <- build_long_df(flat, mdl_name)

          if (!nrow(long)) next

          # drop unwanted metrics
          long <- long[!grepl(EXCLUDE_METRICS_REGEX, long$Metric), , drop = FALSE]
          if (!nrow(long)) next

          # numeric coercion
          long$Value <- to_numeric_safely(long$Value)
          long <- long[is.finite(long$Value), , drop = FALSE]
          if (!nrow(long)) next

          mean_metrics <- long |>
            dplyr::group_by(Metric) |>
            dplyr::summarise(mean_value = mean(Value, na.rm = TRUE), .groups = "drop")

          high_metrics <- mean_metrics |>
            dplyr::filter(mean_value > high_threshold) |>
            dplyr::pull(Metric)

          high_mean_df <- dplyr::bind_rows(high_mean_df, long[long$Metric %in% high_metrics, , drop = FALSE])
          low_mean_df  <- dplyr::bind_rows(low_mean_df,  long[!long$Metric %in% high_metrics, , drop = FALSE])
        }

        list(high_mean_df = high_mean_df, low_mean_df = low_mean_df)
      }

      # Assuming performance_metrics and relevance_metrics are already defined
      performance_results <- process_performance(performance_metrics, run_id) #<<-
      relevance_results <- process_performance(relevance_metrics, run_id) #<<-

      performance_high_mean_df <- performance_results$high_mean_df #<<-
      performance_low_mean_df <- performance_results$low_mean_df #<<-

      relevance_high_mean_df <- relevance_results$high_mean_df #<<-
      relevance_low_mean_df <- relevance_results$low_mean_df #<<-

      # Function to check and print if a dataframe is NULL
      check_and_print_null <- function(df, df_name) {
        if (is.null(df)) {
          print(paste(df_name, "is NULL"))
          return(TRUE)
        } else {
          print(paste(df_name, "is not NULL"))
          return(FALSE)
        }
      }

      # Check and print if any of the dataframes are NULL
      performance_high_mean_is_null <- check_and_print_null(performance_high_mean_df, "performance_high_mean_df")
      performance_low_mean_is_null <- check_and_print_null(performance_low_mean_df, "performance_low_mean_df")
      relevance_high_mean_is_null <- check_and_print_null(relevance_high_mean_df, "relevance_high_mean_df")
      relevance_low_mean_is_null <- check_and_print_null(relevance_low_mean_df, "relevance_low_mean_df")

      # Call the functions and get the plots only if the dataframes are not NULL
      # print("Calling Performance update_performance_and_relevance_high")
      performance_high_mean_plots <- if (!performance_high_mean_is_null) {
        self$update_performance_and_relevance_high(performance_high_mean_df)
      } else {
        NULL
      }
      performance_low_mean_plots <- if (!performance_low_mean_is_null) {
        self$update_performance_and_relevance_low(performance_low_mean_df)
      } else {
        NULL
      }
      # print("Finished Performance update_performance_and_relevance_low")
      # print("Calling Relevance update_performance_and_relevance_high")
      relevance_high_mean_plots <- if (!relevance_high_mean_is_null) {
        self$update_performance_and_relevance_high(relevance_high_mean_df)
      } else {
        NULL
      }
      # print("Finished Relevance update_performance_and_relevance_high")
      # print("Calling Relevance update_performance_and_relevance_low")
      relevance_low_mean_plots <- if (!relevance_low_mean_is_null) {
        self$update_performance_and_relevance_low(relevance_low_mean_df)
      } else {
        NULL
      }


      # =====================================================================
      # GROUPED METRICS  ≠  FUSED ENSEMBLE
      #
      # Audience (plain English):
      # - "Fused" means we COMBINE outputs from multiple models into ONE
      #   consensus prediction stream (used for final decisions/evaluation).
      #   Examples: average, weighted-average, majority vote.
      #
      # What THIS block does:
      # - Provides convenient, experimental insight / reporting across multiple models.
      # - Two modes:
      #     (A) "aggregate_predictions+rep_sonn":
      #         We first call aggregate_predictions() to create a temporary
      #         aggregated prediction vector (mean/median/vote), then score it
      #         with ONE representative SONN to reuse metric code.
      #         • More appropriate when you want a single *proxy* view of the
      #           ensemble’s output (close to how fusion works, but reporting-only).
      #     (B) "average_per_model":
      #         Compute metrics for each model separately, then average the
      #         metric values numerically.
      #         • More of an experimental “sampling pulse” — tells you the
      #           typical performance level across models, not a consensus output.
      #
      # What this is NOT:
      # - NOT the fused-ensemble decision path (avg / wavg / vote_soft /
      #   vote_hard) implemented in DDESONN_fuse_from_agg(), which creates the
      #   SINGLE prediction stream used downstream.
      #
      # Side note:
      # - aggregate_predictions() supports mean/median/vote for reporting use.
      #   The true decision fusion (with weights and soft/hard voting) lives in
      #   DDESONN_fuse_from_agg().
      # =====================================================================

      # Predeclare so they're always in scope
      perf_df <- relev_df <- NULL
      perf_group_summary <- relev_group_summary <- NULL
      group_perf <- group_relev <- NULL

      if(grouped_metrics) {

        # Build per-model long DFs (works for 1+ models)
        perf_df  <- flatten_metrics_to_df(performance_list, run_id)
        relev_df <- flatten_metrics_to_df(relevance_list,     run_id)

        # # --- Vanilla group summaries (across models) ---
        perf_group_summary  <- summarize_grouped(perf_df)
        relev_group_summary <- summarize_grouped(relev_df)

        # --- Optional notify user ---
        if (!verbose && !viewTables) {
          message("[ℹ] Group summaries computed silently. Set `verbose = verbose` to print data frames, or `viewTables = TRUE` to see tables.")
        }

        # Grouped metrics (run whenever you have ≥1 model)
        if (ensemble_number >= 1 && length(self$ensemble) > 1) {
          group_perf <- calculate_performance_grouped(
            SONN_list             = self$ensemble,
            Rdata                 = Rdata,
            labels                = labels,
            lr                    = lr,
            CLASSIFICATION_MODE   = CLASSIFICATION_MODE,
            num_epochs            = num_epochs,
            threshold             = threshold,
            predicted_output_list = predicted_output_list,
            prediction_time_list  = prediction_time_list,
            ensemble_number       = ensemble_number,
            run_id                = run_id,
            ML_NN                 = ML_NN,
            verbose               = verbose,
            agg_method            = "mean",
            metric_mode           = "aggregate_predictions+rep_sonn",
            weights_list          = NULL,
            biases_list           = NULL,
            act_list              = NULL
          )

          group_relev <- calculate_relevance_grouped(
            SONN_list             = self$ensemble,
            Rdata                 = Rdata,
            labels                = labels,
            CLASSIFICATION_MODE   = CLASSIFICATION_MODE,
            predicted_output_list = predicted_output_list,
            ensemble_number       = ensemble_number,
            run_id                = run_id,
            ML_NN                 = ML_NN,
            verbose               = verbose,
            agg_method            = "mean",
            metric_mode           = "aggregate_predictions+rep_sonn"
          )

          # ---------- Printing policy ----------
          # Tables (DF heads) print if EITHER verbose OR viewTables
          if (verbose || viewTables) {
            if (!is.null(perf_df)) {
              emit_table(
                utils::head(perf_df, 12),
                title = "--- performance_long_df (head) ---",
                verbose = verbose,
                viewTables = viewTables
              )
            }
            if (!is.null(relev_df)) {
              emit_table(
                utils::head(relev_df, 12),
                title = "--- relevance_long_df (head) ---",
                verbose = verbose,
                viewTables = viewTables
              )
            }
          }

          # Summaries + grouped metrics print ONLY when verbose = verbose
          if (verbose) {
            if (!is.null(perf_group_summary)) {
              emit_table(
                perf_group_summary,
                title = "=== PERFORMANCE group summary ===",
                verbose = verbose,
                viewTables = viewTables
              )
            }
            if (!is.null(relev_group_summary)) {
              emit_table(
                relev_group_summary,
                title = "=== RELEVANCE group summary ===",
                verbose = verbose,
                viewTables = viewTables
              )
            }
            if (!is.null(group_perf)) {
              emit_table(
                group_perf$metrics,
                title = "=== GROUPED PERFORMANCE metrics ===",
                verbose = verbose,
                viewTables = viewTables
              )
            }
            if (!is.null(group_relev)) {
              emit_table(
                group_relev$metrics,
                title = "=== GROUPED RELEVANCE metrics ===",
                verbose = verbose,
                viewTables = viewTables
              )
            }
          }

        }

      } #end of if(grouped_metrics)

      if(verbose){print("----------------------------------------update_performance_and_relevance-end----------------------------------------")}
      # Return the lists of plots
      return(list(performance_metric = performance_metric, relevance_metric = relevance_metric, performance_high_mean_plots = performance_high_mean_plots, performance_low_mean_plots = performance_low_mean_plots, relevance_high_mean_plots = relevance_high_mean_plots, relevance_low_mean_plots = relevance_low_mean_plots, performance_group_summary = perf_group_summary, relevance_group_summary = relev_group_summary, performance_long_df = perf_df, relevance_long_df = relev_df, performance_grouped = if (exists("group_perf")  && !is.null(group_perf))  group_perf$metrics  else NULL, relevance_grouped   = if (exists("group_relev") && !is.null(group_relev)) group_relev$metrics else NULL, threshold = threshold_used, thresholds = thresholds_used, accuracy = eval_result$accuracy, accuracy_percent = eval_result$accuracy_percent, metrics = if (!is.null(eval_result$metrics)) eval_result$metrics else NULL, misclassified = if (!is.null(eval_result$misclassified)) eval_result$misclassified else NULL))


    },
    # Function to identify outliers
    identify_outliers = function(y) {
      o <- boxplot.stats(y)$out
      return(if(length(o) == 0) NA else o)
    },

    # Function to create bin labels
    create_bin_labels = function(x) {
      breaks <- c(0, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
      labels <- cut(x, breaks = breaks, include.lowest = TRUE, right = FALSE, labels = FALSE)
      return(sapply(labels, function(l) {
        switch(as.character(l),
               "1" = "0%-0.05%",
               "2" = "0.05%-0.1%",
               "3" = "0.1%-0.5%",
               "4" = "0.5%-1%",
               "5" = "1%-2%",
               "6" = "2%-5%",
               "7" = "5%-10%",
               "8" = "10%-20%",
               "9" = "20%-30%",
               "10" = "30%-40%",
               "11" = "40%-50%",
               "12" = "50%-60%",
               "13" = "60%-70%",
               "14" = "70%-80%",
               "15" = "80%-90%",
               "16" = "90%-100%",
               "100%+")  # Add a catch-all label for unexpected values
      }))
    },
    update_performance_and_relevance_high = function(high_mean_df) {

      # ============================================================
      # Init
      # ============================================================
      high_mean_plots <- list()

      # ============================================================
      # Per-metric loop
      # ============================================================
      for (metric in unique(high_mean_df$Metric)) {

        # Filter out rows where the Value is 0 for metrics containing "precision" or "mean_precision"
        filtered_high_mean_df <- high_mean_df[
          !(grepl("precision", high_mean_df$Metric, ignore.case = TRUE) & high_mean_df$Value == 0),
        ]

        # Filter out rows where the Value is NA or infinite
        filtered_high_mean_df <- filtered_high_mean_df[
          !is.na(filtered_high_mean_df$Value) & !is.infinite(filtered_high_mean_df$Value),
        ]

        # Subset the data for the current metric
        plot_data_high <- filtered_high_mean_df[filtered_high_mean_df$Metric == metric, ]

        # Check if plot_data is not empty
        if (nrow(plot_data_high) > 0) {

          # ============================================================
          # Build plot_data + outlier labels
          # ============================================================
          plot_data_high$Outlier <- ifelse(
            !is.na(plot_data_high$Value) &
              plot_data_high$Value %in% self$identify_outliers(plot_data_high$Value),
            plot_data_high$Value,
            NA
          )
          
          # Add columns for outliers
          plot_data_high$Model_Name_Outlier <- plot_data_high$Model_Name

          # Set the RowName to NA where there are no outliers
          plot_data_high$Model_Name_Outlier[is.na(plot_data_high$Outlier)] <- NA

          # Create bin labels for "precisions" or "mean_precisions"
          if (grepl("precision", metric, ignore.case = TRUE)) {
            plot_data_high$Title <- paste0(
              "Boxplot for ",
              metric,
              " (",
              self$create_bin_labels(plot_data_high$Value),
              ")"
            )
          } else {
            plot_data_high$Title <- paste("Boxplot for", metric)
          }

          # ============================================================
          # Plot
          # ============================================================
          high_mean_plot <- ggplot2::ggplot(plot_data_high, ggplot2::aes(x = Metric, y = Value)) +
            ggplot2::geom_boxplot() +
            ggplot2::labs(
              title = unique(plot_data_high$Title),
              x = "Metric",
              y = "Value"
            ) +
            ggplot2::theme_minimal()

          # Add text labels for the outliers
          high_mean_plot <- high_mean_plot +
            ggplot2::geom_text(
              ggplot2::aes(label = Model_Name_Outlier),
              na.rm = TRUE,
              hjust = -0.3
            )

          # Store the plot in the list
          high_mean_plots[[metric]] <- high_mean_plot

        }
      }

      return(high_mean_plots)
    },
    update_performance_and_relevance_low = function(low_mean_df) {
      low_mean_plots <- list()
  
      # Loop over each unique metric
      for (metric in unique(low_mean_df$Metric)) {
    
        # Filter out rows where the Value is 0 for metrics containing "precision" or "mean_precision"
        filtered_low_mean_df <- low_mean_df[
          !(grepl("precision", low_mean_df$Metric, ignore.case = TRUE) & low_mean_df$Value == 0),
        ]
    
        # Filter out rows where the Value is NA or infinite
        filtered_low_mean_df <- filtered_low_mean_df[
          !is.na(filtered_low_mean_df$Value) & !is.infinite(filtered_low_mean_df$Value),
        ]
    
        # Subset the data for the current metric
        plot_data_low <- filtered_low_mean_df[filtered_low_mean_df$Metric == metric, ]
    
        # Check if plot_data is not empty
        if (nrow(plot_data_low) > 0) {
      
          # Add a column to identify outliers
          plot_data_low$Outlier <- ifelse(
            !is.na(plot_data_low$Value) &
              plot_data_low$Value %in% self$identify_outliers(plot_data_low$Value),
            plot_data_low$Value,
            NA
          )
          
          # Add columns for outliers
          plot_data_low$Model_Name_Outlier <- plot_data_low$Model_Name
      
          # Set the RowName to NA where there are no outliers
          plot_data_low$Model_Name_Outlier[is.na(plot_data_low$Outlier)] <- NA
      
          # Create bin labels for "precisions" or "mean_precisions"
          if (grepl("precision", metric, ignore.case = TRUE)) {
            plot_data_low$Title <- paste0(
              "Boxplot for ",
              metric,
              " (",
              self$create_bin_labels(plot_data_low$Value),
              ")"
            )
          } else {
            plot_data_low$Title <- paste("Boxplot for", metric)
          }
      
          # Create box plot
          low_mean_plot <- ggplot2::ggplot(plot_data_low, ggplot2::aes(x = Metric, y = Value)) +
            ggplot2::geom_boxplot() +
            ggplot2::labs(
              title = unique(plot_data_low$Title),
              x = "Metric",
              y = "Value"
            ) +
            ggplot2::theme_minimal()
      
          # Add text labels for the outliers
          low_mean_plot <- low_mean_plot +
            ggplot2::geom_text(
              ggplot2::aes(label = Model_Name_Outlier),
              na.rm = TRUE,
              hjust = -0.3
            )
      
          # Store the plot in the list
          low_mean_plots[[metric]] <- low_mean_plot
      
        }
      }
  
      return(low_mean_plots)
    },
    store_metadata = function(predicted_outputAndTime, actual_values, do_ensemble, input_size, output_size, N, total_num_samples, num_test_samples, num_training_samples, num_validation_samples, num_networks, update_weights, update_biases, lr, lambda, num_epochs, run_id, ensemble_number, model_iter_num, model_serial_num, threshold, CLASSIFICATION_MODE, predicted_output, preprocessScaledData, X, y, X_test_scaled, y_test, all_weights, all_biases, artifact_names, artifact_paths, validation_metrics, activation_functions, activation_functions_predict, dropout_rates, hidden_sizes, ML_NN, best_val_prediction_time, best_train_acc, best_epoch_train, best_train_loss, best_epoch_train_loss, best_val_acc, best_val_epoch, performance_metric, relevance_metric, plot_epochs) {


      # ---------------- helpers (lightweight; keep most original structure) ----------------
      to_num_mat <- function(x) {
        # Guard: never allow functions/closures through
        if (is.function(x)) {
          stop("[store_metadata] Received a function where labels/predictions were expected. Check your `labels`/`actual_values`/`y_validation` sources.")
        }
        
        # Coerce common inputs to a numeric matrix (no factors/characters left)
        if (is.data.frame(x)) {
          if (ncol(x) == 1L && (is.factor(x[[1]]) || is.character(x[[1]]))) {
            x <- matrix(as.numeric(as.factor(x[[1]])), ncol = 1L)
          } else {
            x[] <- lapply(x, function(col) if (is.numeric(col)) col else as.numeric(as.factor(col)))
            x <- as.matrix(x)
          }
        } else if (is.matrix(x)) {
          if (!is.numeric(x)) {
            x <- apply(x, 2, function(col) if (is.numeric(col)) col else as.numeric(as.factor(col)))
          }
          x <- as.matrix(x)
        } else if (is.factor(x) || is.character(x)) {
          x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
        } else if (is.atomic(x) && !is.matrix(x)) {
          x <- matrix(x, ncol = 1L)
        } else {
          x <- as.matrix(x)
          if (!is.numeric(x)) {
            x <- matrix(as.numeric(as.factor(x)), nrow = nrow(x), ncol = ncol(x))
          }
        }
        storage.mode(x) <- "double"
        x
      }
      clamp <- function(v, lo, hi) pmax(lo, pmin(hi, v))
      
      # ---- helper: first non-NULL (skips functions) -----------------------------------
      .first_nonnull <- function(...) {
        for (x in list(...)) {
          if (!is.null(x) && !is.function(x)) return(x)
        }
        NULL
      }
      
      # ---------------- pick target set (labels) with graceful NULL support ----------------
      # Order of preference (NO 'y', NO X_* fallbacks):
      #   validation mode: y_validation -> actual_values -> labels
      #   training/test  : actual_values -> labels
      target_raw <- if (isTRUE(validation_metrics)) {
        .first_nonnull(
          get0("y_validation", ifnotfound = NULL, inherits = TRUE),
          actual_values,
          get0("labels",       ifnotfound = NULL, inherits = TRUE)
        )
      } else {
        .first_nonnull(
          actual_values,
          get0("labels", ifnotfound = NULL, inherits = TRUE)
        )
      }
      
      labels_missing <- is.null(target_raw)
      
      # ---------------- Conform/compute by CLASSIFICATION_MODE (prevents non-numeric - operator) ----------------
      # Guard predicted_output too (avoid closures sneaking in)
      if (is.function(predicted_output)) {
        stop("[store_metadata] `predicted_output` evaluated to a function/closure; check upstream.")
      }
      
      Pm0 <- to_num_mat(predicted_output)
      
      if (!labels_missing) {
        Lm0 <- to_num_mat(target_raw)
        
        # Ensure same number of rows (trim to min; preserves original spirit without rowname alignment)
        n_common <- min(nrow(Lm0), nrow(Pm0))
        if (n_common == 0L) stop("[store_metadata] Empty labels/predictions after trim.")
        if (nrow(Lm0) != nrow(Pm0)) {
          Lm0 <- Lm0[seq_len(n_common), , drop = FALSE]
          Pm0 <- Pm0[seq_len(n_common), , drop = FALSE]
        }
        
        # ---- STRICT: require explicit CLASSIFICATION_MODE; no auto-infer ----
        mode <- tolower(as.character(CLASSIFICATION_MODE))
        if (!mode %in% c("binary","multiclass","regression")) {
          stop("Invalid CLASSIFICATION_MODE. Must be one of: 'binary', 'multiclass', or 'regression'.")
        }
        
        if (identical(mode, "binary")) {
          if (ncol(Lm0) == 2L) {
            y_true <- as.integer(Lm0[, 2] >= Lm0[, 1])
          } else {
            v <- as.numeric(Lm0[, 1])
            u <- sort(unique(as.integer(round(v))))
            if (length(u) == 2L) {
              y_true <- as.integer(v == max(u))
            } else if (all(v %in% c(0, 1))) {
              y_true <- as.integer(v)
            } else {
              y_true <- as.integer(v >= 0.5)
            }
          }
          p_pos <- if (ncol(Pm0) >= 2L) as.numeric(Pm0[, 2]) else as.numeric(Pm0[, 1])
          p_pos[!is.finite(p_pos)] <- NA_real_
          
          target_matrix           <- matrix(as.numeric(y_true), ncol = 1L)
          predicted_output_matrix <- matrix(p_pos, ncol = 1L)
          
          error_prediction <- target_matrix - predicted_output_matrix
          differences      <- error_prediction
          
        } else if (identical(mode, "multiclass")) {
          Kp <- ncol(Pm0); Kl <- ncol(Lm0)
          K  <- if (Kp > 1L) Kp else if (Kl > 1L) Kl else max(2L, Kp, Kl)
          
          if (Kl > 1L) {
            true_ids <- max.col(Lm0, ties.method = "first")
          } else {
            if (is.data.frame(target_raw) && ncol(target_raw) == 1L &&
                (is.factor(target_raw[[1]]) || is.character(target_raw[[1]]))) {
              true_ids <- as.integer(factor(target_raw[[1]]))
            } else {
              vv <- as.integer(round(Lm0[, 1]))
              if (length(vv) && min(vv, na.rm = TRUE) == 0L) vv <- vv + 1L
              true_ids <- vv
            }
            true_ids <- clamp(true_ids, 1L, K)
          }
          target_matrix <- one_hot_from_ids(true_ids, K)
          
          if (ncol(Pm0) < K) {
            total_needed <- nrow(Pm0) * K
            rep_vec <- rep(as.vector(Pm0), length.out = total_needed)
            predicted_output_matrix <- matrix(rep_vec, nrow = nrow(Pm0), ncol = K, byrow = FALSE)
          } else {
            predicted_output_matrix <- Pm0[, seq_len(K), drop = FALSE]
          }
          
          error_prediction <- target_matrix - predicted_output_matrix
          differences      <- error_prediction
          
        } else {  # regression
          if (ncol(Lm0) != ncol(Pm0)) {
            total_elements_needed <- nrow(Lm0) * ncol(Lm0)
            if (ncol(Pm0) < ncol(Lm0)) {
              rep_factor <- ceiling(total_elements_needed / length(Pm0))
              replicated_predicted_output <- rep(Pm0, rep_factor)[1:total_elements_needed]
              predicted_output_matrix <- matrix(
                replicated_predicted_output,
                nrow = nrow(Lm0), ncol = ncol(Lm0), byrow = FALSE
              )
            } else {
              truncated_predicted_output <- Pm0[, 1:ncol(Lm0), drop = FALSE]
              predicted_output_matrix <- matrix(
                truncated_predicted_output,
                nrow = nrow(Lm0), ncol = ncol(Lm0), byrow = FALSE
              )
            }
          } else {
            predicted_output_matrix <- Pm0
          }
          target_matrix    <- Lm0
          error_prediction <- target_matrix - predicted_output_matrix
          differences      <- error_prediction
        }
        
        # Label-present stats
        summary_stats <- summary(differences)
        boxplot_stats <- boxplot.stats(as.numeric(differences))
        
      } else {
        # ---------- Labels are missing: compute what we can without error ----------
        target_matrix           <- NULL
        predicted_output_matrix <- Pm0
        error_prediction        <- NULL
        differences             <- NULL
        
        # NA/empty stats placeholders (CRAN-safe)
        summary_stats <- setNames(
          rep(NA_real_, 6),
          c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")
        )
        boxplot_stats <- list(
          stats = rep(NA_real_, 5),
          n     = nrow(Pm0),
          conf  = c(NA_real_, NA_real_),
          out   = numeric()
        )
      }
      
      # --- Load plot_epochs from file (placeholder kept) ---
      # plot_epochs <- readRDS(paste0("plot_epochs_DESONN", ensemble_number, "SONN", model_iter_num, ".rds"))
      plot_epochs <- NULL
      
      # --- Generate model_serial_num (preserved) ---
      model_serial_num <- sprintf("%d.0.%d", as.integer(ensemble_number), as.integer(model_iter_num))
      
      # --- Build filename prefix / artifact names (preserved) ---
      fname <- make_fname_prefix(
        do_ensemble     = isTRUE(get0("do_ensemble", ifnotfound = FALSE, inherits = TRUE)),
        num_networks    = get0("num_networks", ifnotfound = NULL, inherits = TRUE),
        total_models    = get0("num_networks", ifnotfound = NULL, inherits = TRUE),
        ensemble_number = ensemble_number,
        model_index     = model_iter_num,
        who             = "SONN"
      )
      
      artifact_names <- list(
        training_accuracy_loss_plot = fname("training_accuracy_loss_plot.png"),
        output_saturation_plot      = fname("output_saturation_plot.png"),
        max_weight_plot             = fname("max_weight_plot.png")
      )
      
      # Ensure output dir exists (CRAN-safe: no warnings, recursive ok)
      plots_dir <- file.path(ddesonn_plots_dir(get0("output_root", inherits = TRUE, ifnotfound = NULL)), "training")
      if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
      
      # Drop any NULL names before building paths (prevents file.path() errors)
      artifact_names <- Filter(Negate(is.null), artifact_names)
      
      artifact_paths <- lapply(artifact_names, function(nm) file.path(plots_dir, nm))
      # keep alias if referenced elsewhere
      t_paths <- artifact_paths
      
      # --- Create metadata list (preserved, using explicit CLASSIFICATION_MODE) ---
      metadata <- list(
        input_size = input_size,
        output_size = output_size,
        N = N,
        num_samples = total_num_samples,
        num_test_samples = num_test_samples,
        num_training_samples = num_training_samples,
        num_validation_samples = num_validation_samples,
        num_networks = num_networks,
        update_weights = update_weights,
        update_biases = update_biases,
        lr = lr,
        lambda = lambda,
        num_epochs = num_epochs,
        optimal_epoch = predicted_outputAndTime$optimal_epoch,
        run_id = run_id,
        ensemble_number = ensemble_number,
        model_iter_num = model_iter_num,
        model_serial_num = model_serial_num,
        threshold = threshold,
        CLASSIFICATION_MODE = mode,

        # --- Predictions / errors
        predicted_output = predicted_output,
        predicted_output_tail = tail(predicted_output),
        actual_values_tail = tail(actual_values),
        differences = tail(differences),
        summary_stats = summary_stats,
        boxplot_stats = boxplot_stats,

        # --- Preprocessing
        preprocessScaledData = preprocessScaledData,
        target_transform = if (CLASSIFICATION_MODE == "regression") {preprocessScaledData$target_transform} else NULL,
        reg_target_mode = if (CLASSIFICATION_MODE == "regression") {{
          mode <- preprocessScaledData$reg_target_mode %||%
            get0("REG_TARGET_MODE", inherits = TRUE, ifnotfound =
                   get0("reg_target_mode", inherits = TRUE, ifnotfound = "price"))
          if (is.null(mode) || !nzchar(as.character(mode))) mode <- "price"
          tolower(as.character(mode))
        }} else NULL,
        reg_target_mode_applied = if (CLASSIFICATION_MODE == "regression") {isTRUE(preprocessScaledData$reg_target_mode_applied)} else NULL,

        # --- Data
        X = X,
        y = y,
        X_test = X_test_scaled,
        y_test = y_test,

        # --- Training state
        lossesatoptimalepoch = predicted_outputAndTime$lossesatoptimalepoch,
        loss_increase_flag = predicted_outputAndTime$loss_increase_flag,
        performance_metric = performance_metric,
        relevance_metric = relevance_metric,
        plot_epochs = plot_epochs,
        best_weights_record = all_weights,
        best_biases_record = all_biases,

        # --- Artifacts
        fname_artifact_names = artifact_names,
        fname_artifact_paths = artifact_paths,
        validation_metrics = validation_metrics,

        # Model-critical configs
        activation_functions = activation_functions,
        activation_functions_predict = activation_functions_predict,
        dropout_rates        = dropout_rates,
        hidden_sizes         = self$hidden_sizes %||% hidden_sizes,
        ML_NN                = self$ML_NN %||% ML_NN,

        best_val_prediction_time = best_val_prediction_time,
        best_train_acc = best_train_acc,
        best_epoch_train = best_epoch_train,
        best_train_loss = best_train_loss,
        best_epoch_train_loss = best_epoch_train_loss,
        best_val_acc = best_val_acc,
        best_val_epoch = best_val_epoch
      )


      metadata_main_ensemble <- list()
      metadata_temp_ensemble <- list()

      # --- Store metadata by ensemble type (preserved) ---
      if (ensemble_number <= 1) {
        print(paste("Storing metadata for main ensemble model", model_iter_num, "as", model_serial_num))
        assign(paste0("Ensemble_Main_", ensemble_number, "_model_", model_iter_num, "_metadata"),
               metadata, envir = .GlobalEnv)
      } else {
        print(paste("Storing metadata for temp ensemble model", model_iter_num, "as", model_serial_num))
        assign(paste0("Ensemble_Temp_", ensemble_number, "_model_", model_iter_num, "_metadata"),
               metadata, envir = .GlobalEnv)
      }
    }



  )
)

is_binary <- function(column) {
  unique_values <- unique(column)
  return(length(unique_values) == 2)
}







lr_scheduler <- function(epoch, initial_lr = lr, decay_rate = 0.5, decay_epoch = 20, min_lr = 1e-6) {
  decayed_lr <- initial_lr * decay_rate ^ floor(epoch / decay_epoch)
  return(max(min_lr, decayed_lr))
}

calculate_performance <- function(SONN, Rdata, labels, lr, CLASSIFICATION_MODE, model_iter_num, num_epochs, threshold, learn_time, predicted_output, prediction_time, ensemble_number, run_id, weights, biases, activation_functions, ML_NN, verbose) {

  # --- Elbow method for clustering (robust) ---
  calculate_wss <- function(X, max_k = 15L) {
    max_k <- min(max_k, max(2L, nrow(X) - 1L))
    wss <- numeric(max_k)
    for (k in 1:max_k) {
      wss[k] <- kmeans(X, centers = k, iter.max = 20)$tot.withinss
    }
    wss
  }
  wss <- calculate_wss(Rdata)
  dd  <- diff(diff(wss))
  if (length(dd)) {
    optimal_k <- max(2L, which.max(dd) + 1L)
  } else {
    optimal_k <- min(3L, max(2L, nrow(Rdata) - 1L))
  }
  cluster_assignments <- kmeans(Rdata, centers = optimal_k, iter.max = 50)$cluster


  cat("Length of SONN$weights: ", length(SONN$weights), "\n")
  cat("Length of SONN$map: ", if (is.null(SONN$map)) "NULL" else length(SONN$map), "\n")


  # --- Metrics (all take SONN) ---
  perf_metrics <- list(
    quantization_error                     = quantization_error(SONN, Rdata, run_id, verbose),
    topographic_error                      = topographic_error(SONN, Rdata, threshold, verbose),
    clustering_quality_db                  = clustering_quality_db(SONN, Rdata, cluster_assignments, verbose),
    MSE                                    = MSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    MAE                                    = MAE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    RMSE                                   = RMSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    R2                                     = R2(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    MAPE                                   = MAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    SMAPE                                  = SMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    WMAPE                                  = WMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    MASE                                   = MASE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    accuracy                               = accuracy(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    precision                              = precision(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    recall                                 = recall(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    f1_score                               = f1_score(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose),
    confusion_matrix                       = confusion_matrix(SONN, labels, CLASSIFICATION_MODE, predicted_output, threshold, verbose),
    accuracy_precision_recall_f1_tuned     = accuracy_precision_recall_f1_tuned(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, metric_for_tuning = "accuracy", grid, verbose),
    generalization_ability                 = generalization_ability(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE),
    speed                                  = speed(SONN, prediction_time, verbose),
    speed_learn                            = speed_learn(SONN, learn_time, verbose),
    memory_usage                           = memory_usage(SONN, Rdata, verbose),
    robustness                             = robustness(SONN, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose),
    custom_relative_error_binned           = custom_relative_error_binned(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
  )


  for (name in names(perf_metrics)) {
    val <- perf_metrics[[name]]
    if (is.null(val) || any(is.na(val)) || isTRUE(val)) {
      perf_metrics[[name]] <- NA_real_
    }
  }



  return(list(metrics = perf_metrics, names = names(perf_metrics)))
}

calculate_relevance <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, ML_NN, verbose) {

  # --- Standardize single-layer to list format ---
  if (!is.list(SONN$weights)) {
    SONN$weights <- list(SONN$weights)
  }

  # --- Active relevance metrics ---
  rel_metrics <- list(
    hit_rate     = tryCatch(hit_rate(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose), error = function(e) NULL),
    ndcg         = tryCatch(ndcg(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose), error = function(e) NULL),
    diversity    = tryCatch(diversity(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose), error = function(e) NULL),
    serendipity  = tryCatch(serendipity(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose), error = function(e) NULL)
  )


  # --- Inactive for future implementation ---
  # precision_boolean = precision_boolean(...)
  # recall            = recall(...)
  # f1_score          = f1_score(...)
  # mean_precision    = mean_precision(...)
  # novelty           = novelty(...)

  # --- Validate and clean ---
  for (name in names(rel_metrics)) {
    val <- rel_metrics[[name]]
    if (is.null(val) || any(is.na(val)) || isTRUE(val)) {
      rel_metrics[[name]] <- NA_real_
    }
  }

  return(list(metrics = rel_metrics, names = names(rel_metrics)))
}


# =====================================================================
# PURPOSE: Convenient, experimental insight / reporting across models
#
# NOT "fused" in the deployment sense:
# - "Fused" = combine multiple model outputs into ONE final prediction
#   stream for decisions/evaluation (see DDESONN_fuse_from_agg()).
#
# Modes here:
#   1) metric_mode = "aggregate_predictions+rep_sonn"
#      - Use aggregate_predictions() (mean/median/vote) to create a
#        temporary aggregated prediction vector across models, then score
#        it with a representative SONN.
#      - Best when you want a single proxy view that resembles how fusion
#        behaves, but is still reporting-only (not the true ensemble output).
#
#   2) metric_mode = "average_per_model"
#      - Compute each model’s metrics independently, then average the
#        numeric metric values across models.
#      - More of an experimental “sampling pulse” — shows the typical
#        performance level across models, not a consensus output.
#
# TL;DR:
# - These helpers provide quick experimental insight and reporting across
#   multiple models.
# - Final fused ensemble decisions (avg / wavg / vote_soft / vote_hard)
#   are produced in DDESONN_fuse_from_agg().
# =====================================================================

calculate_performance_grouped <- function(SONN_list, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, threshold, predicted_output_list, prediction_time_list, ensemble_number, run_id, ML_NN, verbose,
                                          agg_method = c("mean","median","vote"),
                                          metric_mode = c("aggregate_predictions+rep_sonn", "average_per_model"),
                                          weights_list = NULL, biases_list = NULL, act_list = NULL
) {
  agg_method <- match.arg(agg_method)
  metric_mode <- match.arg(metric_mode)

  # 1) Aggregate predictions once
  p_agg <- aggregate_predictions(predicted_output_list, method = agg_method)
  pred_time_agg <- mean(unlist(prediction_time_list), na.rm = TRUE)

  if (metric_mode == "aggregate_predictions+rep_sonn") {
    # 2a) Use ONE representative SONN (best F1) so we can reuse your metric code as-is
    rep_sonn <- pick_representative_sonn(SONN_list, predicted_output_list, labels)
    rep_w <- rep_sonn$weights; rep_b <- rep_sonn$biases; rep_af <- rep_sonn$activation_functions

    calculate_performance(
      SONN             = rep_sonn,
      Rdata            = Rdata,
      labels           = labels,
      lr               = lr,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      model_iter_num   = NA_integer_,
      num_epochs       = num_epochs,
      threshold        = threshold,
      learn_time       = NA_real_,
      predicted_output = p_agg,
      prediction_time  = pred_time_agg,
      ensemble_number  = ensemble_number,
      run_id           = paste0(run_id[[1]], "::GROUP"),
      weights          = rep_w,
      biases           = rep_b,
      activation_functions = rep_af,
      ML_NN            = ML_NN,
      verbose          = verbose
    )
  } else {
    # 2b) Average per-model metrics (no re-implementation): compute each, then average numerics
    perfs <- lapply(seq_along(SONN_list), function(i) {
      calculate_performance(
        SONN             = SONN_list[[i]],
        Rdata            = Rdata,
        labels           = labels,
        lr               = lr,
        CLASSIFICATION_MODE = CLASSIFICATION_MODE,
        model_iter_num   = i,
        num_epochs       = num_epochs,
        threshold        = threshold,
        learn_time       = NA_real_,
        predicted_output = predicted_output_list[[i]],
        prediction_time  = prediction_time_list[[i]],
        ensemble_number  = ensemble_number,
        run_id           = run_id[[i]],
        weights          = if (is.null(weights_list)) SONN_list[[i]]$weights else weights_list[[i]],
        biases           = if (is.null(biases_list))  SONN_list[[i]]$biases  else biases_list[[i]],
        activation_functions = if (is.null(act_list)) SONN_list[[i]]$activation_functions else act_list[[i]],
        ML_NN            = ML_NN,
        verbose          = FALSE
      )
    })
    # fold to a single metrics list by averaging numeric leafs
    keys <- Reduce(union, lapply(perfs, `[[`, "names"))
    avg_metrics <- lapply(keys, function(k) {
      vals <- lapply(perfs, function(p) p$metrics[[k]])
      nums <- suppressWarnings(as.numeric(unlist(vals)))
      if (all(is.na(nums))) NULL else mean(nums, na.rm = TRUE)
    })
    names(avg_metrics) <- keys
    list(metrics = avg_metrics, names = names(avg_metrics))
  }
}

calculate_relevance_grouped <- function(SONN_list, Rdata, labels, CLASSIFICATION_MODE, predicted_output_list, ensemble_number, run_id, ML_NN, verbose,
                                        agg_method = c("mean","median","vote"),
                                        metric_mode = c("aggregate_predictions+rep_sonn", "average_per_model")
) {
  agg_method <- match.arg(agg_method)
  metric_mode <- match.arg(metric_mode)
  p_agg <- aggregate_predictions(predicted_output_list, method = agg_method)

  if (metric_mode == "aggregate_predictions+rep_sonn") {
    rep_sonn <- pick_representative_sonn(SONN_list, predicted_output_list, labels)
    calculate_relevance(
      SONN                 = rep_sonn,
      Rdata                = Rdata,
      labels               = labels,
      CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
      model_iter_num       = NA_integer_,
      predicted_output     = p_agg,
      ensemble_number      = ensemble_number,
      weights              = rep_sonn$weights,
      biases               = rep_sonn$biases,
      activation_functions = rep_sonn$activation_functions,
      ML_NN                = ML_NN,
      verbose              = verbose
    )
  } else {
    rels <- lapply(seq_along(SONN_list), function(i) {
      calculate_relevance(
        SONN                 = SONN_list[[i]],
        Rdata                = Rdata,
        labels               = labels,
        CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
        model_iter_num       = i,
        predicted_output     = predicted_output_list[[i]],
        ensemble_number      = ensemble_number,
        weights              = SONN_list[[i]]$weights,
        biases               = SONN_list[[i]]$biases,
        activation_functions = SONN_list[[i]]$activation_functions,
        ML_NN                = ML_NN,
        verbose              = FALSE
      )
    })
    keys <- Reduce(union, lapply(rels, `[[`, "names"))
    avg_metrics <- lapply(keys, function(k) {
      vals <- lapply(rels, function(r) r$metrics[[k]])
      nums <- suppressWarnings(as.numeric(unlist(vals)))
      if (all(is.na(nums))) NULL else mean(nums, na.rm = TRUE)
    })
    names(avg_metrics) <- keys
    list(metrics = avg_metrics, names = names(avg_metrics))
  }
}



# Unified Loss Function (with optional verbose printing)
# Supports binary, multiclass, and regression.
# Strictly enforces valid (mode, loss_type) combos with clear error messages.
loss_function <- function(predictions, labels, CLASSIFICATION_MODE, reg_loss_total, loss_type, verbose) {
  # Default reg_loss_total to 0 if NULL
  if (is.null(reg_loss_total)) reg_loss_total <- 0
  
  if (verbose) {
    print(dim(predictions))
    print(dim(labels))
  }
  
  # Handle missing or NULL loss_type gracefully
  if (is.null(loss_type)) {
    if (verbose) print("Loss type is NULL. Please specify 'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'.")
    return(NA)
  }
  
  mode <- tolower(as.character(CLASSIFICATION_MODE))
  lt   <- tolower(as.character(loss_type))
  
  P <- as.matrix(predictions)
  storage.mode(P) <- "double"
  Y_raw <- as.matrix(labels)
  Y_num <- suppressWarnings(matrix(as.numeric(Y_raw), nrow = nrow(Y_raw), ncol = ncol(Y_raw)))
  
  # ---- FIX: Sanitize labels/preds before checks ----
  if (any(!is.finite(P))) stop("[LOSS] non-finite predictions")
  
  # Replace non-finite labels with 0 (only for numeric modes)
  if (any(!is.finite(Y_num))) {
    bad_rows <- which(!is.finite(rowSums(Y_num)))
    if (length(bad_rows) > 0L) {
      if (verbose) message("[LOSS-FIX] Replacing ", length(bad_rows), " non-finite label rows with 0.")
      Y_num[bad_rows, ] <- 0
    }
  }
  
  if (nrow(P) != nrow(Y_num)) {
    stop(sprintf("[LOSS] shape mismatch rows P %d vs Y %d", nrow(P), nrow(Y_num)))
  }
  if (identical(mode, "regression")) {
    if (ncol(P) != ncol(Y_num)) {
      stop(sprintf("[LOSS] shape mismatch P %dx%d vs Y %dx%d", nrow(P), ncol(P), nrow(Y_num), ncol(Y_num)))
    }
  } else {
    if (!(ncol(Y_num) %in% c(1L, ncol(P)))) {
      stop(sprintf("[LOSS] unexpected label shape P %dx%d vs Y %dx%d", nrow(P), ncol(P), nrow(Y_num), ncol(Y_num)))
    }
  }
  
  sample_weights_local <- get0("sample_weights", inherits = TRUE, ifnotfound = NULL)
  if (!is.null(sample_weights_local)) {
    W <- as.matrix(sample_weights_local)
    W_num <- suppressWarnings(matrix(as.numeric(W), nrow = nrow(W), ncol = ncol(W)))
    if (any(!is.finite(W_num))) stop("[LOSS] non-finite weights")
    if (nrow(W_num) != nrow(Y_num)) {
      stop(sprintf("[LOSS] weights shape mismatch W %dx%d vs Y %dx%d", nrow(W_num), ncol(W_num), nrow(Y_num), ncol(Y_num)))
    }
    if (!(ncol(W_num) %in% c(1L, ncol(Y_num)))) {
      stop(sprintf("[LOSS] weights shape mismatch W %dx%d vs Y %dx%d", nrow(W_num), ncol(W_num), nrow(Y_num), ncol(Y_num)))
    }
  }
  
  # ---- Compatibility checks (clear + strict) ----
  if (mode == "regression") {
    if (lt %in% c("crossentropy", "categoricalcrossentropy")) {
      stop("Invalid loss_type for regression: '", loss_type,
           "'. Use 'MSE' or 'MAE' for CLASSIFICATION_MODE = 'regression'.\n",
           "# Tip: Cross-entropy losses require class probabilities and one-hot/0-1 labels; ",
           "regression targets are continuous.")
    }
  } else if (mode == "binary") {
    if (lt == "categoricalcrossentropy") {
      stop("Invalid loss_type for binary: 'CategoricalCrossEntropy'. ",
           "Use 'CrossEntropy' (binary) or 'MSE'/'MAE'.")
    }
  } else if (mode == "multiclass") {
    # all four allowed; CE names both valid here
    ;
  } else {
    stop("Unknown CLASSIFICATION_MODE: must be 'binary', 'multiclass', or 'regression'")
  }
  
  n <- nrow(P); K <- ncol(P)
  
  # small helpers (kept)
  one_hot <- function(idx, n, K) {
    Y <- matrix(0, n, K)
    ok <- !is.na(idx) & idx >= 1 & idx <= K
    if (any(ok)) Y[cbind(which(ok), idx[ok])] <- 1
    Y
  }
  row_softmax <- function(X) {
    X <- as.matrix(X)
    m <- apply(X, 1L, max)
    ex <- exp(sweep(X, 1L, m, "-"))
    ex / rowSums(ex)
  }
  
  # ---- LOSS COMPUTATION ----
  if (lt == "mse") {
    
    loss <- mean((P - Y_num)^2, na.rm = TRUE)
    
  } else if (lt == "mae") {
    
    loss <- mean(abs(P - Y_num), na.rm = TRUE)
    
  } else if (lt %in% c("crossentropy", "categoricalcrossentropy")) {
    
    eps <- 1e-12
    
    if (mode == "binary") {
      # Binary Cross-Entropy
      y <- if (ncol(Y_num) >= 1L) Y_num[, 1] else Y_num
      if (all(is.na(y))) y <- as.integer(factor(labels)) - 1L
      y[is.na(y)] <- 0
      y <- pmin(pmax(y, 0), 1)
      
      # assume P is sigmoid probs; clamp
      P <- pmin(pmax(P, eps), 1 - eps)
      
      # numeric guards
      if (is.matrix(P) && ncol(P) == 1L) P <- as.vector(P)
      P <- pmin(pmax(P, eps), 1 - eps)
      y <- as.numeric(y)
      loss <- -mean(y * log(P) + (1 - y) * log1p(-P))
      
    } else if (mode == "multiclass") {
      stopifnot(K >= 2)
      # FIX: Proper multiclass cross-entropy
      if (is.matrix(labels) && nrow(labels) == n && ncol(labels) == K &&
          all(labels %in% c(0, 1))) {
        Y <- matrix(as.numeric(labels), n, K)
      } else {
        f  <- if (is.factor(labels)) labels else factor(labels)
        ix <- as.integer(f)
        L  <- nlevels(f)
        if (L > K) ix[ix > K] <- K
        Y <- one_hot(ix, n, K)
      }
      
      # Ensure valid probability matrix
      if (any(P < 0) || any(P > 1) || any(abs(rowSums(P) - 1) > 1e-6)) {
        P <- row_softmax(P)
      }
      
      P <- pmin(pmax(P, eps), 1 - eps)
      loss <- -mean(rowSums(Y * log(P)), na.rm = TRUE)
      
    } else if (mode == "regression") {
      stop("Cross-entropy not valid for regression.")
    }
    
  } else {
    if (verbose) print("Invalid loss type. Choose from 'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'.")
    return(NA)
  }
  
  total_loss <- loss + reg_loss_total
  if (verbose) print(paste("Final loss (with regularization):", total_loss))
  
  return(total_loss)
}
