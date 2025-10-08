source("utils/utils.R")

#' Internal package environment used to lazily load the legacy DDESONN stack.
.ddesonn_env <- new.env(parent = .GlobalEnv)

#' Null-coalescing helper used across the high-level API.
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

## ============================================================
## Robust root resolver + legacy source (drop-in replacement)
## ============================================================

normalize_architecture <- function(architecture = c("auto", "single", "multi"), hidden_sizes) {
  arch <- match.arg(architecture)
  
  hs <- hidden_sizes
  if (is.null(hs) || length(hs) == 0L) {
    hs <- integer(0)
  } else {
    if (!is.numeric(hs)) {
      stop("hidden_sizes must be numeric/integer.", call. = FALSE)
    }
    hs <- hs[!is.na(hs)]
    if (!length(hs)) {
      hs <- integer(0)
    } else {
      if (any(!is.finite(hs))) stop("hidden_sizes must be non-negative finite integers.", call. = FALSE)
      if (any(hs < 0))        stop("hidden_sizes must be non-negative finite integers.", call. = FALSE)
      if (any(hs != as.integer(hs))) stop("hidden_sizes must be non-negative finite integers.", call. = FALSE)
      hs <- as.integer(hs)
      hs <- hs[hs > 0L]
    }
  }
  
  if (arch == "single") {
    if (length(hs) > 0L) {
      warning("architecture='single' but hidden_sizes provided; ignoring hidden_sizes.", call. = FALSE)
      hs <- integer(0)
    }
    return(list(arch = "single", hidden_sizes = hs))
  }
  
  if (arch == "multi") {
    if (length(hs) == 0L) {
      stop("architecture='multi' requires hidden_sizes with one or more positive integers (e.g., c(8) or c(16,8)).", call. = FALSE)
    }
    return(list(arch = "multi", hidden_sizes = hs))
  }
  
  if (length(hs) == 0L) list(arch = "single", hidden_sizes = hs) else list(arch = "multi", hidden_sizes = hs)
}

.ddesonn_find_root <- function() {
  opt <- getOption("DDESONN_ROOT")
  root <- if (!is.null(opt) && nzchar(opt)) opt else {
    pkg_root <- tryCatch(system.file(package = "DDESONN"), error = function(e) "")
    if (nzchar(pkg_root)) pkg_root else getwd()
  }
  root <- normalizePath(root, winslash = "/", mustWork = FALSE)
  if (basename(root) == "inst") root <- dirname(root)  # <-- key line
  root
}


.ddesonn_source_legacy <- function(force = FALSE) {
  # Fast-return if we already sourced successfully
  if (!force && exists(".__LEGACY_OK__", envir = .ddesonn_env, inherits = FALSE)) {
    return(invisible(.ddesonn_env))
  }
  
  root <- .ddesonn_find_root()
  
  # Look in several common spots for the entry file
  candidates <- file.path(root, c(
    "DDESONN.R",
    "legacy/DDESONN.R",
    "R/DDESONN.R",
    "src/DDESONN.R"
  ))
  hit <- candidates[file.exists(candidates)]
  
  if (!length(hit)) {
    stop(
      "Unable to locate DDESONN.R under root: ", root, "\n",
      "Searched: ", paste(candidates, collapse = ", "), "\n",
      "Tip: set options(DDESONN_ROOT = normalizePath('<repo_root>')) before calling."
    )
  }
  
  target <- hit[[1L]]
  
  # Clean env before re-source (prevents stale symbols)
  if (length(ls(.ddesonn_env))) rm(list = ls(.ddesonn_env), envir = .ddesonn_env)
  
  # chdir=TRUE so relative sources inside DDESONN.R resolve from its folder
  sys.source(target, envir = .ddesonn_env, keep.source = TRUE, chdir = TRUE)
  
  # mark as OK to avoid re-work
  assign(".__LEGACY_OK__", TRUE, envir = .ddesonn_env)
  invisible(.ddesonn_env)
}

.ddesonn_get <- function(name) {
  .ddesonn_source_legacy()
  obj <- get0(name, envir = .ddesonn_env, inherits = FALSE)
  if (is.null(obj)) {
    stop(sprintf("Object '%s' was not initialised from the legacy stack.", name), call. = FALSE)
  }
  obj
}

#' Default activation sequences used by the high-level helpers.
#'
#' @param mode Problem mode. One of `"binary"`, `"multiclass"`, or `"regression"`.
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#' @param stage Stage for which activations are required. Either `"train"` or `"predict"`.
#'
#' @return A list of activation functions suitable for passing into the
#'   underlying R6 classes.
#' @export
#'
#' @examples
#' ddesonn_activation_defaults("binary", hidden_sizes = c(32, 16))
#' ddesonn_activation_defaults("regression", hidden_sizes = 64, stage = "predict")
ddesonn_activation_defaults <- function(mode = c("binary", "multiclass", "regression"),
                                        hidden_sizes = NULL,
                                        stage = c("train", "predict")) {
  mode <- match.arg(mode)
  stage <- match.arg(stage)
  
  .ddesonn_source_legacy()
  
  fetch_activation <- function(name) {
    fn <- get0(name, envir = .ddesonn_env, inherits = TRUE)
    if (!is.function(fn)) {
      stop(sprintf("Activation '%s' is not available in the legacy stack.", name), call. = FALSE)
    }
    fn
  }
  
  hidden_len <- length(hidden_sizes %||% integer())
  
  defaults <- switch(
    mode,
    binary = list(hidden = "relu", output = if (stage == "predict") "sigmoid" else "sigmoid"),
    multiclass = list(hidden = "relu", output = "softmax"),
    regression = list(hidden = "relu", output = "identity")
  )
  
  hidden_fns <- rep(list(fetch_activation(defaults$hidden)), length.out = hidden_len)
  c(hidden_fns, list(fetch_activation(defaults$output)))
}

#' Default dropout configuration.
#'
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#'
#' @return A list of dropout rates for each hidden layer.
#' @export
#'
#' @examples
#' ddesonn_dropout_defaults(c(64, 32))
ddesonn_dropout_defaults <- function(hidden_sizes) {
  hidden_sizes <- hidden_sizes %||% integer()
  if (!length(hidden_sizes)) {
    return(list())
  }
  as.list(rep(0.1, length(hidden_sizes)))
}

#' Optimiser options understood by the legacy training loop.
#'
#' @return A character vector of supported optimiser identifiers.
#' @export
#'
#' @examples
#' ddesonn_optimizer_options()
ddesonn_optimizer_options <- function() {
  c("adagrad", "adam", "lamb", "sgd", "sgd_momentum", "nag", "rmsprop", "ftrl", "lookahead")
}

.ddesonn_threshold_default <- function(mode) {
  if (mode %in% c("binary", "multiclass")) 0.5 else NA_real_
}

#' Construct the default training control list.
#'
#' @param mode Problem mode used to determine sensible defaults.
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#'
#' @return A named list that can be modified and supplied to [ddesonn_fit()].
#' @export
#'
#' @examples
#' ddesonn_training_defaults("binary", hidden_sizes = c(32, 16))
ddesonn_training_defaults <- function(mode = c("binary", "multiclass", "regression"),
                                      hidden_sizes = NULL) {
  mode <- match.arg(mode)
  .ddesonn_source_legacy()
  
  list(
    lr = 0.125,
    lr_decay_rate = 0.5,
    lr_decay_epoch = 20L,
    lr_min = 1e-5,
    num_epochs = 3L,
    self_org = FALSE,
    threshold = .ddesonn_threshold_default(mode),
    reg_type = "L1",
    numeric_columns = NULL,
    activation_functions = NULL,
    activation_functions_predict = NULL,
    dropout_rates = ddesonn_dropout_defaults(hidden_sizes),
    optimizer = "adagrad",
    beta1 = 0.9,
    beta2 = 0.8,
    epsilon = 1e-7,
    lookahead_step = 5L,
    batch_normalize_data = TRUE,
    gamma_bn = 0.6,
    beta_bn = 0.6,
    epsilon_bn = 1e-6,
    momentum_bn = 0.9,
    is_training_bn = TRUE,
    shuffle_bn = FALSE,
    loss_type = if (mode %in% c("binary", "multiclass")) "CrossEntropy" else "MSE",
    update_weights = TRUE,
    update_biases  = TRUE,
    sample_weights = NULL,
    preprocessScaledData = NULL,
    X_validation = NULL,
    y_validation = NULL,
    validation_metrics = TRUE,
    threshold_function = NULL,  # keep NULL-safe
    ML_NN = TRUE,
    train = TRUE,
    grouped_metrics = FALSE,
    viewTables = FALSE,
    verbose = FALSE,
    # passthrough controls
    num_networks = NULL,   # if NULL, use model$num_networks
    do_ensemble = NULL,    # legacy expects logical; NULL -> handled by train()
    X_train = NULL,        # optional explicit train features
    y_train = NULL         # optional explicit train labels
  )
}

.as_matrix <- function(x) {
  if (is.matrix(x)) return(x)
  if (is.data.frame(x)) return(as.matrix(x))
  if (is.vector(x)) return(matrix(x, ncol = 1L))
  stop("Input must be a matrix, data.frame, or vector.", call. = FALSE)
}

.as_numeric_matrix <- function(x) {
  m <- .as_matrix(x)
  storage.mode(m) <- "double"
  m
}

#' Create a high-level DDESONN model wrapper.
#'
#' @param input_size Number of input features.
#' @param output_size Number of outputs.
#' @param hidden_sizes Integer vector describing hidden layer widths.
#' @param architecture Neural network architecture strategy. One of
#'   `"auto"`, `"single"`, or `"multi"`. See Details.
#' @param num_networks Number of SONN members to initialise within the ensemble.
#' @param lambda Regularisation strength.
#' @param classification_mode Problem mode: `"binary"`, `"multiclass"`, or `"regression"`.
#' @param activation_functions Optional list of activation functions for training.
#' @param activation_functions_predict Optional list of activation functions used during prediction.
#' @param method Weight initialisation scheme passed to the legacy constructor.
#' @param custom_scale Optional scaling factor for the initialiser.
#' @param N Optional total node count. If omitted it is inferred from the architecture.
#' @param ensembles Optional pre-existing ensemble container.
#' @param ensemble_number Identifier used when combining multiple ensembles.
#'
#' @details
#' The `architecture` argument provides guardrails around the hidden layer
#' configuration. When set to `"auto"`, non-empty positive `hidden_sizes`
#' imply a multi-layer network, while empty, zero, or `NA` values imply a
#' single-layer network. Forcing `architecture = "single"` drops any provided
#' hidden sizes (with a warning). Forcing `architecture = "multi"` requires at
#' least one positive hidden size and raises an error otherwise.
#' 
#' @return A `ddesonn_model` (R6) instance ready for training.
#' @export
#'
#' @examples
#' model <- ddesonn_model(
#'   input_size = 5,
#'   output_size = 1,
#'   hidden_sizes = c(32, 16),
#'   classification_mode = "binary"
#' )
ddesonn_model <- function(input_size,
                          output_size,
                          hidden_sizes = c(64, 32),
                          architecture = c("auto", "single", "multi"),
                          num_networks = 1L,
                          lambda = 2.8e-4,
                          classification_mode = c("binary", "multiclass", "regression"),
                          activation_functions = NULL,
                          activation_functions_predict = NULL,
                          init_method = "he",
                          custom_scale = 1,
                          N = NULL,
                          ensembles = NULL,
                          ensemble_number = 0L) {
  classification_mode <- match.arg(classification_mode)
  architecture <- match.arg(architecture)

  arch_norm <- normalize_architecture(architecture, hidden_sizes)
  arch <- arch_norm$arch
  hidden_sizes <- arch_norm$hidden_sizes
  ML_NN <- (arch == "multi")

  if (arch == "multi" && length(hidden_sizes) == 1L) {
    message(sprintf("[DDESONN] Architecture='multi' with a single hidden layer (%d neurons). This is still MULTI-LAYER (input -> hidden -> output).", hidden_sizes[1]))
  }

  num_layers <- if (isTRUE(ML_NN)) length(hidden_sizes) + 1L else 1L

  activation_functions <- activation_functions %||%
    ddesonn_activation_defaults(classification_mode, hidden_sizes, stage = "train")
  activation_functions <- learn_predict_normalize_activations(
    activation_functions,
    num_layers
  )

  activation_functions_predict <- activation_functions_predict %||%
    ddesonn_activation_defaults(classification_mode, hidden_sizes, stage = "predict")
  activation_functions_predict <- learn_predict_normalize_activations(
    activation_functions_predict,
    num_layers
  )

  if (is.null(N)) {
    N <- if (isTRUE(ML_NN)) {
      input_size + sum(hidden_sizes %||% 0) + output_size
    } else {
      input_size + output_size
    }
  }
  
  DDESONN_class <- .ddesonn_get("DDESONN")
  model <- DDESONN_class$new(
    num_networks = num_networks,
    input_size = input_size,
    hidden_sizes = hidden_sizes,
    output_size = output_size,
    N = N,
    lambda = lambda,
    ensemble_number = ensemble_number,
    ensembles = ensembles,
    ML_NN = ML_NN,
    activation_functions = activation_functions,
    activation_functions_predict = activation_functions_predict,
    init_method = init_method,
    custom_scale = custom_scale
  )

  attr(model, "classification_mode") <- classification_mode
  attr(model, "activation_functions") <- activation_functions
  attr(model, "activation_functions_predict") <- activation_functions_predict
  attr(model, "hidden_sizes") <- hidden_sizes
  attr(model, "lambda") <- lambda
  attr(model, "architecture") <- arch
  attr(model, "ML_NN") <- ML_NN
  model$architecture <- arch
  class(model) <- unique(c("ddesonn_model", class(model)))
  model
}

.prepare_training_data <- function(x) {
  data <- as.data.frame(x)
  numeric_cols <- names(Filter(is.numeric, data))
  list(
    data = .as_numeric_matrix(data),
    numeric_columns = numeric_cols
  )
}

#' Fit a `ddesonn_model` using data frames or matrices.
#'
#' @param model A model created by [ddesonn_model()].
#' @param x Training features.
#' @param y Training targets/labels.
#' @param validation Optional list containing `x` and `y` elements for validation.
#' @param architecture Neural network architecture strategy. One of `"auto"`, `"single"`, or `"multi"`.
#' @param ... Named overrides (kept for forward-compat; values here are ignored to preserve model’s architecture).
#' @param num_epochs,lr,lr_decay_rate,lr_decay_epoch,lr_min Optimizer schedule.
#' @param self_org Logical; legacy self-organization flag.
#' @param threshold Optional initial threshold.
#' @param reg_type Regularization type.
#' @param numeric_columns Optional numeric column names.
#' @param CLASSIFICATION_MODE "binary","multiclass","regression".
#' @param activation_functions,activation_functions_predict Activation lists.
#' @param dropout_rates Dropout per hidden layer.
#' @param optimizer,beta1,beta2,epsilon,lookahead_step Optimizer options.
#' @param batch_normalize_data,gamma_bn,beta_bn,epsilon_bn,momentum_bn,is_training_bn,shuffle_bn BN options.
#' @param loss_type "CrossEntropy" or "MSE" (legacy names supported).
#' @param update_weights,update_biases Whether to update params.
#' @param sample_weights Optional vector/matrix of sample weights.
#' @param preprocessScaledData Optional preprocessing toggle.
#' @param validation_metrics Logical; compute/track validation metrics.
#' @param threshold_function Optional custom threshold fn (or NULL).
#' @param ML_NN Logical; multi-layer SONN.
#' @param train Logical; train loop on/off.
#' @param grouped_metrics,viewTables,verbose Legacy diagnostics flags.
#' @param ensemble_number Integer ensemble id.
#' @param best_weights_on_lastest_weights_off Logical; restore best-on-val weights if TRUE.
#' @param num_networks,do_ensemble,X_train,y_train Optional passthroughs to legacy `train()`.
#'
#' @return The trained model (invisibly).
#' @export
ddesonn_fit <- function(
    model,
    x,
    y,
    validation = NULL,
    architecture = c("auto","single","multi"),
    ...,
    num_epochs = 100,
    lr = 0.01,
    lr_decay_rate = 0.9,
    lr_decay_epoch = 50,
    lr_min = 1e-5,
    self_org = TRUE,
    threshold = NULL,
    reg_type = NULL,
    numeric_columns = NULL,
    CLASSIFICATION_MODE = "binary",
    activation_functions = NULL,
    activation_functions_predict = NULL,
    dropout_rates = NULL,
    optimizer = "adam",
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
    lookahead_step = 5,
    batch_normalize_data = FALSE,
    gamma_bn = 1,
    beta_bn = 0,
    epsilon_bn = 1e-5,
    momentum_bn = 0.9,
    is_training_bn = TRUE,
    shuffle_bn = TRUE,
    loss_type = "mse",
    update_weights = TRUE,
    update_biases  = TRUE,
    sample_weights = NULL,
    preprocessScaledData = NULL,
    validation_metrics = TRUE,
    threshold_function = NULL,
    ML_NN = TRUE,
    train = TRUE,
    grouped_metrics = FALSE,
    viewTables = FALSE,
    verbose = FALSE,
    ensemble_number = 0L,
    best_weights_on_lastest_weights_off = TRUE,
    # optional passthroughs
    num_networks = NULL,
    do_ensemble = NULL,
    X_train = NULL,
    y_train = NULL
) {
  if (!inherits(model, "ddesonn_model")) stop("'model' must be created with ddesonn_model().", call. = FALSE)
  
  # Soft-check architecture mismatch if user passed it here
  architecture <- match.arg(architecture)
  model_arch <- attr(model, "architecture") %||% if (length(attr(model, "hidden_sizes") %||% integer())) "multi" else "single"
  if (!identical(architecture, "auto") && !identical(architecture, model_arch)) {
    warning(sprintf("architecture='%s' ignored at fit-time; model was created as '%s'.", architecture, model_arch), call. = FALSE)
  }
  # Ignore `...` on purpose (kept for forward-compat)
  
  # Prepare data
  data_prep <- .prepare_training_data(x)
  labels <- .as_numeric_matrix(y)
  
  # Resolve mode/architecture
  mode <- attr(model, "classification_mode") %||% CLASSIFICATION_MODE
  hidden_sizes <- attr(model, "hidden_sizes") %||% NULL
  ML_NN_flag <- attr(model, "ML_NN")
  num_layers <- if (isTRUE(ML_NN_flag)) length(hidden_sizes %||% integer()) + 1L else 1L

  activation_train <- activation_functions %||%
    attr(model, "activation_functions") %||% model$activation_functions
  activation_train <- learn_predict_normalize_activations(activation_train, num_layers)

  activation_predict <- activation_functions_predict %||%
    attr(model, "activation_functions_predict") %||% model$activation_functions_predict
  activation_predict <- learn_predict_normalize_activations(activation_predict, num_layers)
  
  # Start from defaults, then overwrite with explicit arguments
  defaults <- ddesonn_training_defaults(mode, hidden_sizes)
  cfg <- utils::modifyList(defaults, list(
    lr = lr,
    lr_decay_rate = lr_decay_rate,
    lr_decay_epoch = lr_decay_epoch,
    lr_min = lr_min,
    num_epochs = num_epochs,
    self_org = self_org,
    threshold = threshold,
    reg_type = reg_type,
    numeric_columns = numeric_columns %||% data_prep$numeric_columns,
    CLASSIFICATION_MODE = mode,
    activation_functions = activation_train,
    activation_functions_predict = activation_predict,
    dropout_rates = dropout_rates %||% ddesonn_dropout_defaults(hidden_sizes),
    optimizer = optimizer,
    beta1 = beta1,
    beta2 = beta2,
    epsilon = epsilon,
    lookahead_step = lookahead_step,
    batch_normalize_data = batch_normalize_data,
    gamma_bn = gamma_bn,
    beta_bn = beta_bn,
    epsilon_bn = epsilon_bn,
    momentum_bn = momentum_bn,
    is_training_bn = is_training_bn,
    shuffle_bn = shuffle_bn,
    loss_type = loss_type,
    update_weights = update_weights,
    update_biases = update_biases,
    sample_weights = sample_weights,
    preprocessScaledData = preprocessScaledData,
    validation_metrics = isTRUE(validation_metrics),
    threshold_function = if (!is.null(threshold_function)) threshold_function else NULL,
    best_weights_on_lastest_weights_off = best_weights_on_lastest_weights_off,
    ML_NN = isTRUE(ML_NN %||% ML_NN_flag),
    train = train,
    grouped_metrics = grouped_metrics,
    viewTables = viewTables,
    verbose = verbose,
    ensemble_number = ensemble_number,
    # passthroughs
    num_networks = num_networks %||% model$num_networks %||% NULL,
    do_ensemble = do_ensemble,
    X_train = if (!is.null(X_train)) .as_numeric_matrix(X_train) else NULL,
    y_train = if (!is.null(y_train)) .as_numeric_matrix(y_train) else NULL
  ), keep.null = TRUE)
  
  # Validation wiring
  if (!is.null(validation)) {
    if (!is.list(validation) || !all(c("x", "y") %in% names(validation))) {
      stop("'validation' must be a list with elements 'x' and 'y'.", call. = FALSE)
    }
    cfg$X_validation <- .as_numeric_matrix(validation$x)
    cfg$y_validation <- .as_numeric_matrix(validation$y)
  }
  if (is.null(cfg$X_validation)) cfg$X_validation <- cfg$X_train
  if (is.null(cfg$y_validation)) cfg$y_validation <- cfg$y_train
  
  # Build training args to match legacy `train()`
  train_args <- list(
    Rdata = data_prep$data,
    labels = labels,
    X_train = cfg$X_train,
    y_train = cfg$y_train,
    lr = cfg$lr,
    lr_decay_rate = cfg$lr_decay_rate,
    lr_decay_epoch = cfg$lr_decay_epoch,
    lr_min = cfg$lr_min,
    num_networks = cfg$num_networks,
    ensemble_number = cfg$ensemble_number,
    do_ensemble = cfg$do_ensemble,
    num_epochs = cfg$num_epochs,
    self_org = cfg$self_org,
    threshold = cfg$threshold,
    reg_type = cfg$reg_type,
    numeric_columns = cfg$numeric_columns,
    CLASSIFICATION_MODE = mode,
    activation_functions = cfg$activation_functions,
    activation_functions_predict = cfg$activation_functions_predict,
    dropout_rates = cfg$dropout_rates,
    optimizer = cfg$optimizer,
    beta1 = cfg$beta1,
    beta2 = cfg$beta2,
    epsilon = cfg$epsilon,
    lookahead_step = cfg$lookahead_step,
    batch_normalize_data = cfg$batch_normalize_data,
    gamma_bn = cfg$gamma_bn,
    beta_bn = cfg$beta_bn,
    epsilon_bn = cfg$epsilon_bn,
    momentum_bn = cfg$momentum_bn,
    is_training_bn = cfg$is_training_bn,
    shuffle_bn = cfg$shuffle_bn,
    loss_type = cfg$loss_type,
    update_weights = cfg$update_weights,
    update_biases  = cfg$update_biases,
    sample_weights = cfg$sample_weights,
    preprocessScaledData = cfg$preprocessScaledData,
    X_validation = cfg$X_validation,
    y_validation = cfg$y_validation,
    validation_metrics = cfg$validation_metrics,
    threshold_function = cfg$threshold_function,
    best_weights_on_lastest_weights_off = cfg$best_weights_on_lastest_weights_off,
    ML_NN = cfg$ML_NN,
    train = cfg$train,
    grouped_metrics = cfg$grouped_metrics,
    viewTables = cfg$viewTables,
    verbose = cfg$verbose,
    model_iter_num = 1L
  )
  formals_train <- names(formals(model$train))
  train_args <- train_args[intersect(names(train_args), formals_train)]
  
  result <- do.call(model$train, train_args)
  model$last_training <- result
  attr(model, "threshold") <- cfg$threshold
  
  # Restore best-on-validation weights if available
  if (isTRUE(best_weights_on_lastest_weights_off)) {
    restored <- FALSE
    if (!is.null(model$best_weights) && !is.null(model$best_biases)) {
      message("[DDESONN] Restoring best validation weights from model...")
      model$weights <- model$best_weights; model$biases <- model$best_biases; restored <- TRUE
    } else if (!is.null(result$best_weights) && !is.null(result$best_biases)) {
      message("[DDESONN] Restoring best validation weights from training result (model-level)...")
      model$weights <- result$best_weights; model$biases <- result$best_biases; restored <- TRUE
    }
    if (!restored && length(model$ensemble)) {
      per_ok <- FALSE
      if (!is.null(result$best_weights_per_member)) {
        for (i in seq_along(model$ensemble)) {
          src <- result$best_weights_per_member[[i]]
          if (!is.null(src$best_weights) && !is.null(src$best_biases)) {
            model$ensemble[[i]]$weights <- src$best_weights
            model$ensemble[[i]]$biases  <- src$best_biases
            per_ok <- TRUE
          }
        }
      } else {
        for (i in seq_along(model$ensemble)) {
          net <- model$ensemble[[i]]
          if (!is.null(net$best_weights) && !is.null(net$best_biases)) {
            net$weights <- net$best_weights; net$biases <- net$best_biases; per_ok <- TRUE
          }
        }
      }
      if (per_ok) message("[DDESONN] Restored best validation weights per ensemble member.")
    }
  } else {
    message("[DDESONN] Using final-epoch weights (best restore disabled).")
  }
  
  invisible(model)
}


.aggregate_predictions <- function(preds, aggregate) {
  if (identical(aggregate, "none")) {
    return(preds)
  }
  arr <- simplify2array(preds)
  if (length(dim(arr)) == 2L) {
    arr <- array(arr, dim = c(dim(arr), 1L))
  }
  if (aggregate == "mean") {
    apply(arr, c(1, 2), mean)
  } else if (aggregate == "median") {
    apply(arr, c(1, 2), stats::median)
  } else {
    stop("Unsupported aggregation method.", call. = FALSE)
  }
}

#' Generate predictions from a fitted `ddesonn_model`.
#'
#' @param model A trained model produced by [ddesonn_model()].
#' @param new_data New feature matrix or data frame.
#' @param aggregate Aggregation strategy across ensemble members. One of
#'   `"mean"`, `"median"`, or `"none"`.
#' @param type Prediction type. `"response"` returns numeric predictions,
#'   while `"class"` applies thresholding for classification problems.
#' @param threshold Optional threshold override when `type = "class"`.
#' @param architecture Neural network architecture strategy. One of
#'   `"auto"`, `"single"`, or `"multi"`.
#'
#' @return A list containing the aggregated prediction matrix and the
#'   per-model outputs when `aggregate = "none"`.
#' @export
ddesonn_predict <- function(model, new_data, aggregate = c("mean", "median", "none"),
                            type = c("response", "class"), threshold = NULL,
                            architecture = c("auto", "single", "multi")) {
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  aggregate <- match.arg(aggregate)
  type <- match.arg(type)
  architecture <- match.arg(architecture)
  
  X <- .as_numeric_matrix(new_data)
  mode <- attr(model, "classification_mode") %||% "binary"
  
  # Add harmless compatibility check (from 'theirs')
  hidden_sizes <- attr(model, "hidden_sizes") %||% integer()
  model_arch <- attr(model, "architecture") %||% { if (length(hidden_sizes)) "multi" else "single" }
  if (!identical(architecture, "auto") && !identical(architecture, model_arch)) {
    warning(sprintf("architecture='%s' is incompatible with the model architecture ('%s'); using '%s'.",
                    architecture, model_arch, model_arch), call. = FALSE)
  }
  
  # Per-member predictions (always collect raw member outputs first)
  preds <- lapply(seq_along(model$ensemble), function(i) {
    net <- model$ensemble[[i]]
    acts <- net$activation_functions_predict %||% net$activation_functions
    res <- net$predict(
      Rdata = X,
      weights = net$weights,
      biases  = net$biases,
      activation_functions_predict = acts,
      verbose = FALSE,
      debug   = FALSE
    )
    out <- res$predicted_output %||% res
    .as_numeric_matrix(out)
  })
  
  # Aggregate across ensemble members if requested
  aggregated <- .aggregate_predictions(preds, aggregate)
  
  # === Enforce probabilities + class handling for classification ===
  output <- list()
  
  if (mode %in% c("binary", "multiclass")) {
    if (mode == "binary") {
      # Keep your safeguard: squash to [0,1] if legacy path emitted logits.
      probs <- .as_numeric_matrix(aggregated)
      rng <- range(probs, finite = TRUE)
      if ((is.finite(rng[1]) && rng[1] < 0) || (is.finite(rng[2]) && rng[2] > 1)) {
        probs <- 1 / (1 + exp(-probs))
      }
      output$prediction <- probs
      
      if (type == "class") {
        thr <- threshold %||% attr(model, "threshold") %||% 0.5
        output$class <- ifelse(probs >= thr, 1L, 0L)
      }
    } else {
      output$prediction <- aggregated
      if (type == "class") {
        output$class <- max.col(aggregated, ties.method = "first")
      }
    }
    
    if (identical(aggregate, "none")) output$per_model <- preds
    return(output)
  }
  
  # Regression (unchanged)
  output <- list(prediction = aggregated)
  if (identical(aggregate, "none")) output$per_model <- preds
  output
}



# ========================================================================
# Legacy artifact helpers used by ddesonn_run() persistence
# ========================================================================

.num <- function(x) suppressWarnings(as.numeric(x))
.int <- function(x) suppressWarnings(as.integer(x))
.chr <- function(x) suppressWarnings(as.character(x))
.take1num <- function(x) {
  v <- suppressWarnings(as.numeric(x))
  if (length(v) && is.finite(v[1])) v[1] else NA_real_
}

.stamp_ddesonn_rundir <- function(do_ensemble, num_networks, seeds) {
  ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  if (!isTRUE(do_ensemble)) {
    seed_tag <- if (length(seeds) > 1L) "wSeed" else "wNoSeed"
    list(
      root = file.path("artifacts", "SingleRuns"),
      run = sprintf("%s__m%d__%s", ts_stamp, as.integer(num_networks), seed_tag),
      ts = ts_stamp
    )
  } else {
    list(
      root = file.path("artifacts", "EnsembleRuns"),
      run = ts_stamp,
      ts = ts_stamp
    )
  }
}

.make_dirs_legacy <- function(base) {
  dirs <- c(
    file.path(base, "models", "main"),
    file.path(base, "fused"),
    file.path(base, "logs")
  )
  for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  invisible(NULL)
}

.flatten_metric_list <- function(x) {
  if (is.null(x)) return(list())
  flat <- tryCatch(
    rapply(x, f = function(z) z, how = "unlist"),
    error = function(e) setNames(vector("list", 0L), character(0))
  )
  if (!length(flat)) return(list())
  L <- as.list(flat)
  keep <- vapply(L, function(z) is.atomic(z) && length(z) == 1, logical(1))
  as.list(flat[keep])
}

.compute_f1 <- function(precision, recall) {
  p <- .num(precision)
  r <- .num(recall)
  ok <- (p + r) > 0
  out <- rep(NA_real_, length(p))
  out[ok] <- 2 * p[ok] * r[ok] / (p[ok] + r[ok])
  out
}

.build_slot_metadata <- function(slot_obj, fallback_serial, k) {
  md <- try(slot_obj$metadata, silent = TRUE)
  if (inherits(md, "try-error") || is.null(md)) md <- list()
  md$model_serial_num <- as.character(md$model_serial_num %||% fallback_serial)
  md$model_name <- md$model_name %||% paste0("model_", k)
  
  pm <- try(slot_obj$performance_metric, silent = TRUE)
  if (!inherits(pm, "try-error")) md$performance_metric <- md$performance_metric %||% pm
  rm <- try(slot_obj$relevance_metric, silent = TRUE)
  if (!inherits(rm, "try-error")) md$relevance_metric <- md$relevance_metric %||% rm
  
  md$best_train_acc <- .take1num(md$best_train_acc)
  md$best_epoch_train <- .int(md$best_epoch_train %||% NA_integer_)
  md$best_val_acc <- .take1num(md$best_val_acc)
  md$best_val_epoch <- .int(md$best_val_epoch %||% NA_integer_)
  md$best_val_prediction_time <- .take1num(md$best_val_prediction_time)
  
  md$predictor <- slot_obj
  md$predictor_fn <- function(X, ...) slot_obj$predict(X, ...)
  md
}

.build_metrics_row <- function(md, run_index, seed, slot, split = "test") {
  pm <- .flatten_metric_list(md$performance_metric)
  rm <- .flatten_metric_list(md$relevance_metric)
  
  row <- list(
    run_index = as.integer(run_index),
    seed = as.integer(seed),
    MODEL_SLOT = as.integer(slot),
    model_slot = as.integer(slot),
    split = .chr(split),
    serial = .chr(md$model_serial_num %||% NA_character_),
    model_name = .chr(md$model_name %||% paste0("model_", slot)),
    best_train_acc = .take1num(md$best_train_acc),
    best_epoch_train = .int(md$best_epoch_train),
    best_val_acc = .take1num(md$best_val_acc),
    best_val_epoch = .int(md$best_val_epoch),
    best_val_prediction_time = .take1num(md$best_val_prediction_time)
  )
  
  add_flat <- function(src, prefix = NULL) {
    if (!length(src)) return(NULL)
    for (nm in names(src)) {
      key <- if (!is.null(prefix)) paste0(prefix, ".", nm) else nm
      val <- src[[nm]]
      vn <- suppressWarnings(as.numeric(val))
      row[[key]] <- if (!is.na(vn)) vn else as.character(val)
    }
    NULL
  }
  add_flat(pm, "performance_metric")
  add_flat(rm, "relevance_metric")
  
  pref_num <- function(...) {
    keys <- c(...)
    for (k in keys) {
      v <- row[[k]]
      if (!is.null(v)) {
        vn <- suppressWarnings(as.numeric(v))
        if (!is.na(vn)) return(vn)
      }
    }
    NA_real_
  }
  
  row$accuracy <- pref_num(
    "performance_metric.accuracy",
    "relevance_metric.accuracy",
    "accuracy_precision_recall_f1_tuned.accuracy"
  )
  row$precision <- pref_num(
    "performance_metric.precision",
    "relevance_metric.precision",
    "accuracy_precision_recall_f1_tuned.precision"
  )
  row$recall <- pref_num(
    "performance_metric.recall",
    "relevance_metric.recall",
    "accuracy_precision_recall_f1_tuned.recall"
  )
  row$f1 <- pref_num(
    "performance_metric.f1",
    "relevance_metric.f1",
    "accuracy_precision_recall_f1_tuned.f1"
  )
  if (is.na(row$f1)) row$f1 <- .compute_f1(row$precision, row$recall)
  row$f1_score <- pref_num("performance_metric.f1_score", "relevance_metric.f1_score")
  if (is.na(row$f1_score)) row$f1_score <- row$f1
  
  for (c0 in c("TP", "FP", "TN", "FN")) {
    base <- paste0("confusion_matrix.", c0)
    tuned <- paste0("accuracy_precision_recall_f1_tuned.confusion_matrix.", c0)
    if (is.null(row[[base]]) && !is.null(row[[tuned]])) row[[base]] <- .num(row[[tuned]])[1]
  }
  
  as.data.frame(row, check.names = TRUE, stringsAsFactors = FALSE)
}

.write_single_runs_metrics <- function(result, run_dir, ts, seeds_vec) {
  s_chr <- as.character(length(seeds_vec))
  pretty_test_path <- file.path(run_dir, sprintf("SingleRun_Pretty_Test_Metrics_%s_seeds_%s.rds", s_chr, ts))
  test_path <- file.path(run_dir, sprintf("SingleRun_Test_Metrics_%s_seeds_%s.rds", s_chr, ts))
  train_path <- file.path(run_dir, sprintf("SingleRun_Train_Acc_Val_Metrics_%s_seeds_%s.rds", s_chr, ts))
  
  rows_train <- list()
  rows_test <- list()
  ptr_tr <- 0L
  ptr_te <- 0L
  
  for (i in seq_along(result$runs)) {
    seed_i <- result$runs[[i]]$seed %||% i
    main_model <- result$runs[[i]]$main$model
    if (is.null(main_model)) next
    K <- length(main_model$ensemble) %||% 0L
    if (K < 1L) next
    
    for (k in seq_len(K)) {
      slot_obj <- try(main_model$ensemble[[k]], silent = TRUE)
      if (inherits(slot_obj, "try-error") || is.null(slot_obj)) next
      md <- try(slot_obj$metadata, silent = TRUE)
      if (inherits(md, "try-error") || is.null(md)) md <- list()
      md$model_serial_num <- md$model_serial_num %||% sprintf("0.main.%d", k)
      md$model_name <- md$model_name %||% paste0("model_", k)
      
      ptr_tr <- ptr_tr + 1L
      rows_train[[ptr_tr]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "train")
      
      ptr_te <- ptr_te + 1L
      rows_test[[ptr_te]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "test")
    }
  }
  
  bind <- function(lst) if (!length(lst)) data.frame() else do.call(rbind, lst)
  df_train <- bind(rows_train)
  df_test <- bind(rows_test)
  
  id_order <- c("run_index", "seed", "model_slot", "MODEL_SLOT", "split", "serial", "model_name")
  metric_pref <- c(
    "accuracy", "precision", "recall", "f1", "f1_score",
    "confusion_matrix.TP", "confusion_matrix.FP", "confusion_matrix.TN", "confusion_matrix.FN",
    "best_train_acc", "best_epoch_train", "best_val_acc", "best_val_epoch", "best_val_prediction_time"
  )
  ord <- function(df) c(
    intersect(id_order, names(df)),
    intersect(metric_pref, names(df)),
    setdiff(names(df), c(id_order, metric_pref))
  )
  if (ncol(df_train)) df_train <- df_train[, ord(df_train), drop = FALSE]
  if (ncol(df_test)) df_test <- df_test[, ord(df_test), drop = FALSE]
  
  saveRDS(df_test, pretty_test_path)
  saveRDS(df_test, test_path)
  saveRDS(df_train, train_path)
}

.write_ensemble_runs_metrics <- function(result, run_dir, ts, seeds_vec) {
  s_chr <- as.character(length(seeds_vec))
  agg_metrics_path <- file.path(run_dir, sprintf("agg_metrics_test__%s_seeds_%s.rds", s_chr, ts))
  
  rows <- list()
  ptr <- 0L
  for (i in seq_along(result$runs)) {
    seed_i <- result$runs[[i]]$seed %||% i
    main_model <- result$runs[[i]]$main$model
    if (is.null(main_model)) next
    K <- length(main_model$ensemble) %||% 0L
    if (K < 1L) next
    for (k in seq_len(K)) {
      slot_obj <- try(main_model$ensemble[[k]], silent = TRUE)
      if (inherits(slot_obj, "try-error") || is.null(slot_obj)) next
      md <- try(slot_obj$metadata, silent = TRUE)
      if (inherits(md, "try-error") || is.null(md)) md <- list()
      md$model_serial_num <- md$model_serial_num %||% sprintf("1.main.%d", k)
      md$model_name <- md$model_name %||% paste0("model_", k)
      
      ptr <- ptr + 1L
      rows[[ptr]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "test")
    }
  }
  
  df <- if (!length(rows)) data.frame() else do.call(rbind, rows)
  
  id_order <- c("run_index", "seed", "model_slot", "MODEL_SLOT", "split", "serial", "model_name")
  metric_pref <- c(
    "accuracy", "precision", "recall", "f1", "f1_score", "auc", "balanced_accuracy",
    "specificity", "sensitivity", "logloss", "brier",
    "MSE", "MAE", "RMSE", "R2", "MAPE", "SMAPE", "WMAPE", "MASE",
    "confusion_matrix.TP", "confusion_matrix.FP", "confusion_matrix.TN", "confusion_matrix.FN",
    "generalization_ability", "speed", "speed_learn1", "speed_learn2",
    "memory_usage", "robustness", "hit_rate", "ndcg", "diversity", "serendipity",
    "best_train_acc", "best_epoch_train", "best_val_acc", "best_val_epoch", "best_val_prediction_time"
  )
  ord <- c(
    intersect(id_order, names(df)),
    intersect(metric_pref, names(df)),
    setdiff(names(df), c(id_order, metric_pref))
  )
  if (ncol(df)) df <- df[, ord, drop = FALSE]
  
  saveRDS(df, agg_metrics_path)
}

.write_agg_predictions <- function(result, run_dir, ts, seeds_vec) {
  s_chr <- as.character(length(seeds_vec))
  out_path <- file.path(run_dir, sprintf("agg_predictions_test__%s_seeds_%s.rds", s_chr, ts))
  
  X <- result$`.__prediction_matrix`
  if (is.null(X)) {
    saveRDS(data.frame(), out_path)
    return(invisible(NULL))
  }
  
  rows <- list()
  ptr <- 0L
  
  for (i in seq_along(result$runs)) {
    seed_i <- result$runs[[i]]$seed %||% i
    main_model <- result$runs[[i]]$main$model
    if (is.null(main_model)) next
    
    pr <- try(ddesonn_predict(
      model = main_model,
      new_data = X,
      aggregate = "none",
      type = "response"
    ), silent = TRUE)
    if (inherits(pr, "try-error") || is.null(pr$per_model)) next
    
    K <- length(pr$per_model)
    for (k in seq_len(K)) {
      yk <- pr$per_model[[k]]
      y_prob <- as.numeric(yk)
      n <- length(y_prob)
      if (!n) next
      rows[[ptr <- ptr + 1L]] <- data.frame(
        run_index = rep.int(i, n),
        seed = rep.int(seed_i, n),
        MODEL_SLOT = rep.int(k, n),
        model_slot = rep.int(k, n),
        obs = seq_len(n),
        split = "test",
        y_pred = y_prob,
        y_true = NA_real_,
        stringsAsFactors = FALSE
      )
    }
  }
  
  out <- if (!length(rows)) data.frame() else do.call(rbind, rows)
  
  id_order <- c("run_index", "seed", "MODEL_SLOT", "model_slot", "obs", "split")
  rest <- setdiff(names(out), c(id_order, "y_pred", "y_true"))
  if (ncol(out)) {
    out <- out[, c(id_order, "y_pred", "y_true", rest), drop = FALSE]
  }
  
  saveRDS(out, out_path)
  invisible(NULL)
}

.write_temp_agg_predictions <- function(result, run_dir, ts, seeds_vec) {
  X <- result$`.__prediction_matrix`
  if (is.null(X)) return(invisible(NULL))
  
  max_temp <- 0L
  for (i in seq_along(result$runs)) {
    ti <- result$runs[[i]]$temp_iterations
    if (!is.null(ti)) max_temp <- max(max_temp, length(ti))
  }
  if (max_temp == 0L) return(invisible(NULL))
  
  s_chr <- as.character(length(seeds_vec))
  
  for (e in seq_len(max_temp)) {
    temp_dir_e <- file.path(run_dir, sprintf("models/temp_e%02d", e))
    dir.create(temp_dir_e, recursive = TRUE, showWarnings = FALSE)
    out_path <- file.path(
      temp_dir_e,
      sprintf("agg_predictions_temp_e%02d__%s_seeds_%s.rds", e, s_chr, ts)
    )
    
    rows <- list()
    ptr <- 0L
    
    for (i in seq_along(result$runs)) {
      seed_i <- result$runs[[i]]$seed %||% i
      ti <- result$runs[[i]]$temp_iterations
      if (is.null(ti)) next
      entry <- NULL
      for (t in ti) if (identical(as.integer(t$iteration), as.integer(e))) { entry <- t; break }
      if (is.null(entry) || is.null(entry$model)) next
      tmodel <- entry$model
      
      pr <- try(ddesonn_predict(
        model = tmodel,
        new_data = X,
        aggregate = "none",
        type = "response"
      ), silent = TRUE)
      if (inherits(pr, "try-error") || is.null(pr$per_model)) next
      
      K <- length(pr$per_model)
      for (k in seq_len(K)) {
        yk <- pr$per_model[[k]]
        y_prob <- as.numeric(yk)
        n <- length(y_prob)
        if (!n) next
        rows[[ptr <- ptr + 1L]] <- data.frame(
          run_index = rep.int(i, n),
          seed = rep.int(seed_i, n),
          MODEL_SLOT = rep.int(k, n),
          model_slot = rep.int(k, n),
          obs = seq_len(n),
          split = "test",
          y_pred = y_prob,
          y_true = NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
    
    out <- if (!length(rows)) data.frame() else do.call(rbind, rows)
    
    id_order <- c("run_index", "seed", "MODEL_SLOT", "model_slot", "obs", "split")
    rest <- setdiff(names(out), c(id_order, "y_pred", "y_true"))
    if (ncol(out)) {
      out <- out[, c(id_order, "y_pred", "y_true", rest), drop = FALSE]
    }
    
    saveRDS(out, out_path)
  }
  
  invisible(NULL)
}

.persist_ddesonn_run <- function(result, output_root, save_models = TRUE) {
  if (is.null(output_root) || !nzchar(output_root)) return(invisible(NULL))
  
  cfg <- result$configuration %||% list()
  stamp <- .stamp_ddesonn_rundir(
    do_ensemble = isTRUE(cfg$do_ensemble),
    num_networks = as.integer(cfg$num_networks %||% 1L),
    seeds = cfg$seeds %||% 1L
  )
  
  run_dir <- file.path(output_root, stamp$root, stamp$run)
  .make_dirs_legacy(run_dir)
  
  ts <- stamp$ts
  models_main_dir <- file.path(run_dir, "models", "main")
  dir.create(models_main_dir, recursive = TRUE, showWarnings = FALSE)
  
  saveRDS(result, file.path(run_dir, "run_result.rds"))
  
  if (isTRUE(save_models) && length(result$runs)) {
    for (i in seq_along(result$runs)) {
      seed_i <- result$runs[[i]]$seed %||% i
      mdl <- result$runs[[i]]$main$model
      if (!is.null(mdl)) {
        saveRDS(mdl, file.path(models_main_dir, sprintf("main_model_seed_%s.rds", seed_i)))
      }
    }
  }
  
  for (i in seq_along(result$runs)) {
    seed_i <- result$runs[[i]]$seed %||% i
    main_model <- result$runs[[i]]$main$model
    if (is.null(main_model)) next
    K <- try(length(main_model$ensemble), silent = TRUE)
    if (inherits(K, "try-error") || !is.finite(K) || K < 1L) next
    for (k in seq_len(K)) {
      slot_obj <- try(main_model$ensemble[[k]], silent = TRUE)
      if (inherits(slot_obj, "try-error") || is.null(slot_obj)) next
      if (!isTRUE(cfg$do_ensemble)) {
        serial <- sprintf("0.main.%d", k)
        md <- .build_slot_metadata(slot_obj, serial, k)
        out_file <- file.path(models_main_dir, sprintf("Ensemble_Main_0_model_%d_metadata_%s_seed%s.rds", k, ts, seed_i))
        saveRDS(md, out_file)
      } else {
        serial <- sprintf("1.main.%d", k)
        md <- .build_slot_metadata(slot_obj, serial, k)
        out_file <- file.path(models_main_dir, sprintf("Ensemble_Main_1_model_%d_metadata_%s_seed%s.rds", k, ts, seed_i))
        saveRDS(md, out_file)
      }
    }
  }
  
  max_temp <- 0L
  for (i in seq_along(result$runs)) {
    ti <- result$runs[[i]]$temp_iterations
    if (!is.null(ti)) max_temp <- max(max_temp, length(ti))
  }
  if (max_temp > 0L) {
    for (e in seq_len(max_temp)) {
      temp_dir_e <- file.path(run_dir, "models", sprintf("temp_e%02d", e))
      dir.create(temp_dir_e, recursive = TRUE, showWarnings = FALSE)
    }
    for (i in seq_along(result$runs)) {
      seed_i <- result$runs[[i]]$seed %||% i
      ti <- result$runs[[i]]$temp_iterations
      if (is.null(ti)) next
      for (entry in ti) {
        e <- as.integer(entry$iteration)
        tmd <- entry$model
        if (!is.null(tmd) && isTRUE(save_models)) {
          saveRDS(tmd, file.path(run_dir, "models", sprintf("temp_e%02d", e), sprintf("temp_model_e%02d_seed_%s.rds", e, seed_i)))
        }
        if (is.null(tmd)) next
        Kt <- try(length(tmd$ensemble), silent = TRUE)
        if (inherits(Kt, "try-error") || !is.finite(Kt) || Kt < 1L) next
        temp_dir_e <- file.path(run_dir, "models", sprintf("temp_e%02d", e))
        for (k in seq_len(Kt)) {
          slot_obj <- try(tmd$ensemble[[k]], silent = TRUE)
          if (inherits(slot_obj, "try-error") || is.null(slot_obj)) next
          serial <- sprintf("%d.temp.%d", e, k)
          md <- .build_slot_metadata(slot_obj, serial, k)
          out_file <- file.path(temp_dir_e, sprintf("Ensemble_Temp_%d_model_%d_metadata_%s_seed%s.rds", e, k, ts, seed_i))
          saveRDS(md, out_file)
        }
      }
    }
  }
  
  if (!is.null(result$predictions)) {
    saveRDS(result$predictions, file.path(run_dir, "predictions_main.rds"))
  }
  if (!is.null(result$temp_predictions) && length(result$temp_predictions)) {
    saveRDS(result$temp_predictions, file.path(run_dir, "predictions_temp.rds"))
  }
  
  seeds_vec <- cfg$seeds %||% 1L
  if (isTRUE(cfg$do_ensemble)) {
    .write_ensemble_runs_metrics(result, run_dir, ts, seeds_vec)
  } else {
    .write_single_runs_metrics(result, run_dir, ts, seeds_vec)
  }
  
  .write_agg_predictions(result, run_dir, ts, seeds_vec)
  .write_temp_agg_predictions(result, run_dir, ts, seeds_vec)
  
  logs_dir <- file.path(run_dir, "logs")
  dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_along(seeds_vec)) {
    seed_i <- seeds_vec[[i]]
    saveRDS(data.frame(), file.path(logs_dir, sprintf("movement_log_run%03d_seed%s_%s.rds", i, seed_i, ts)))
    saveRDS(data.frame(), file.path(logs_dir, sprintf("change_log_run%03d_seed%s_%s.rds", i, seed_i, ts)))
    saveRDS(data.frame(), file.path(logs_dir, sprintf("main_log_run%03d_seed%s_%s.rds", i, seed_i, ts)))
  }
  
  saveRDS(list(
    timestamp = ts,
    seeds = cfg$seeds,
    num_networks = cfg$num_networks,
    do_ensemble = cfg$do_ensemble,
    num_temp_iterations = cfg$num_temp_iterations,
    classification_mode = cfg$classification_mode %||% NA_character_,
    hidden_sizes = cfg$hidden_sizes %||% NA
  ), file.path(run_dir, "run_lineage_metadata.rds"))
  
  invisible(run_dir)
}

#' Run DDESONN across common ensemble scenarios.
#'
#' This helper re-creates the four orchestration modes that previously lived in
#' `TestDDESONN.R`:
#'
#' * Scenario A – single model (`do_ensemble = FALSE`, `num_networks = 1`).
#' * Scenario B – single run with multiple members inside a single model
#'   (`do_ensemble = FALSE`, `num_networks > 1`).
#' * Scenario C – main ensemble container (`do_ensemble = TRUE`,
#'   `num_temp_iterations = 0`).
#' * Scenario D – main ensemble plus TEMP iterations (`do_ensemble = TRUE`,
#'   `num_temp_iterations > 0`).
#'
#' The function accepts a training set, optional validation data, and optional
#' prediction features. It repeatedly instantiates [ddesonn_model()] objects,
#' fits them with [ddesonn_fit()], and (when requested) calls
#' [ddesonn_predict()] to surface aggregated predictions.
#'
#' @param x Training features as a data frame, matrix, or tibble.
#' @param y Training labels/targets.
#' @param classification_mode Overall problem mode. One of `"binary"`,
#'   `"multiclass"`, or `"regression"`.
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#' @param seeds Integer vector of seeds. A separate model (or ensemble stack) is
#'   trained for each seed.
#' @param do_ensemble Logical flag selecting the ensemble container modes
#'   (scenarios C/D). When `FALSE`, scenarios A/B are executed.
#' @param num_networks Number of ensemble members inside each
#'   [ddesonn_model()] instance.
#' @param num_temp_iterations Number of TEMP iterations to run when
#'   `do_ensemble = TRUE` (scenario D). Ignored otherwise.
#' @param validation Optional validation list with elements `x` and `y`.
#' @param model_overrides Named list forwarded to [ddesonn_model()] allowing
#'   custom architectures.
#' @param training_overrides Named list forwarded to [ddesonn_fit()] for the
#'   main run(s).
#' @param temp_overrides Optional named list forwarded to [ddesonn_fit()] for
#'   TEMP iterations. Defaults to `training_overrides`.
#' @param prediction_data Optional features for prediction. When supplied,
#'   predictions are computed for each seed/iteration.
#' @param prediction_type Passed to [ddesonn_predict()].
#' @param aggregate Aggregation strategy within a single model (across ensemble
#'   members).
#' @param seed_aggregate Aggregation strategy across seeds. Set to `"none"` to
#'   keep per-seed prediction matrices.
#' @param threshold Optional threshold override for classification prediction.
#' @param output_root Optional directory where legacy-style artifacts are
#'   written. When `NULL` (default) no files are created.
#' @param save_models Logical; if `TRUE` (default) individual models are
#'   persisted when `output_root` is supplied.
#'
#' @return A list (classed as `"ddesonn_run_result"`) containing the
#'   configuration, per-seed models, and optional prediction summaries.
#' @export
#'
#' @examples
#' \donttest{
#' data <- mtcars
#' x <- data[, c("disp", "hp", "wt", "qsec", "drat")]
#' y <- data$am
#' res <- ddesonn_run(
#'   x,
#'   y,
#'   classification_mode = "binary",
#'   training_overrides = list(num_epochs = 1, lr = 0.05, validation_metrics = FALSE)
#' )
#' length(res$runs)
#' }
ddesonn_run <- function(x,
                        y,
                        classification_mode = c("binary", "multiclass", "regression"),
                        hidden_sizes = c(64, 32),
                        seeds = 1L,
                        do_ensemble = FALSE,
                        num_networks = if (isTRUE(do_ensemble)) 3L else 1L,
                        num_temp_iterations = 0L,
                        validation = NULL,
                        model_overrides = list(),
                        training_overrides = list(),
                        temp_overrides = NULL,
                        prediction_data = NULL,
                        prediction_type = c("response", "class"),
                        aggregate = c("mean", "median", "none"),
                        seed_aggregate = c("mean", "median", "none"),
                        threshold = NULL,
                        output_root = NULL,
                        save_models = TRUE) {
  classification_mode <- match.arg(classification_mode)
  aggregate <- match.arg(aggregate)
  seed_aggregate <- match.arg(seed_aggregate)
  prediction_type <- match.arg(prediction_type)
  
  seeds <- as.integer(seeds)
  seeds <- seeds[is.finite(seeds)]
  if (!length(seeds)) {
    seeds <- 1L
  }
  
  x_matrix <- .as_numeric_matrix(x)
  y_matrix <- .as_numeric_matrix(y)
  
  input_size <- ncol(x_matrix)
  output_size <- ncol(y_matrix)
  
  base_model_args <- list(
    input_size = input_size,
    output_size = output_size,
    hidden_sizes = hidden_sizes,
    num_networks = max(1L, as.integer(num_networks)),
    classification_mode = classification_mode
  )
  base_model_args <- utils::modifyList(base_model_args, model_overrides, keep.null = TRUE)
  
  # Prepare training params
  base_train_overrides <- ddesonn_training_defaults(classification_mode, hidden_sizes)
  base_train_overrides <- utils::modifyList(base_train_overrides, training_overrides, keep.null = TRUE)
  
  prediction_matrix <- NULL
  if (!is.null(prediction_data)) {
    prediction_matrix <- .as_numeric_matrix(prediction_data)
  }
  
  # Per-seed main runs
  runs <- lapply(seq_along(seeds), function(i) {
    set.seed(seeds[[i]])
    mdl <- do.call(ddesonn_model, base_model_args)
    val <- NULL
    if (!is.null(validation)) {
      val <- list(x = validation$x, y = validation$y)
    }
    do.call(ddesonn_fit, c(list(model = mdl, x = x_matrix, y = y_matrix, validation = val), base_train_overrides))
    
    main_pred <- NULL
    aggregate_pred <- NULL
    if (!is.null(prediction_matrix)) {
      preds <- ddesonn_predict(mdl, prediction_matrix, aggregate = aggregate, type = prediction_type, threshold = threshold)
      main_pred <- preds
    }
    
    temp_summary <- list()
    if (isTRUE(do_ensemble) && num_temp_iterations > 0L) {
      tmp_overrides <- temp_overrides %||% base_train_overrides
      temp_list <- vector("list", length = num_temp_iterations)
      for (iter in seq_len(num_temp_iterations)) {
        # TEMP iteration: clone model or reuse with potential tweaks
        tmp_model <- do.call(ddesonn_model, base_model_args)
        # training for TEMP
        do.call(ddesonn_fit, c(list(model = tmp_model, x = x_matrix, y = y_matrix, validation = val), tmp_overrides))
        
        per_seed <- NULL
        aggregate_tmp <- NULL
        if (!is.null(prediction_matrix)) {
          tpred <- ddesonn_predict(tmp_model, prediction_matrix, aggregate = "none", type = "response")
          per_seed <- tpred$per_model
          aggregate_tmp <- .aggregate_predictions(per_seed, aggregate)
        }
        
        temp_list[[iter]] <- list(iteration = iter, model = tmp_model, per_seed = per_seed, aggregate = aggregate_tmp)
      }
      temp_summary <- temp_list
    }
    
    list(
      seed = seeds[[i]],
      main = list(model = mdl, predictions = main_pred),
      temp_iterations = temp_summary
    )
  })
  
  # Aggregate across seeds (if prediction_matrix provided)
  main_seed_predictions <- NULL
  main_aggregate <- NULL
  temp_summary <- list()
  if (!is.null(prediction_matrix)) {
    # collect per-seed per-model predictions (aggregate across ensemble members inside each seed)
    per_seed_preds <- lapply(runs, function(run) {
      if (is.null(run$main$predictions)) return(NULL)
      if (identical(aggregate, "none")) {
        run$main$predictions$per_model
      } else {
        list(run$main$predictions$prediction)
      }
    })
    # remove NULLs
    per_seed_preds <- Filter(Negate(is.null), per_seed_preds)
    main_seed_predictions <- per_seed_preds
    
    # aggregate across seeds if requested
    if (!identical(seed_aggregate, "none") && length(per_seed_preds)) {
      # Convert to common array and aggregate
      # If "aggregate=none", per_seed elements are lists of per-model matrices → stack inside seed then aggregate
      # If "aggregate!=none", each per_seed element is a length-1 list with the already-aggregated matrix
      mats <- lapply(per_seed_preds, function(pe) {
        if (is.list(pe) && length(pe) > 1L) {
          .aggregate_predictions(pe, aggregate = aggregate) # collapse inside seed
        } else if (is.list(pe) && length(pe) == 1L) {
          pe[[1]]
        } else {
          pe
        }
      })
      arr <- simplify2array(mats)
      if (length(dim(arr)) == 2L) arr <- array(arr, dim = c(dim(arr), 1L))
      if (seed_aggregate == "mean") {
        main_aggregate <- apply(arr, c(1, 2), mean)
      } else if (seed_aggregate == "median") {
        main_aggregate <- apply(arr, c(1, 2), stats::median)
      }
    }
  }
  
  # TEMP summaries (if present) — keep structure per iteration across seeds
  if (isTRUE(do_ensemble) && num_temp_iterations > 0L) {
    temp_summary <- lapply(seq_len(num_temp_iterations), function(iter) {
      preds <- lapply(runs, function(run) {
        ent <- NULL
        for (e in run$temp_iterations) if (identical(e$iteration, iter)) { ent <- e; break }
        ent
      })
      # Aggregate per iteration across seeds if needed
      aggregate_pred <- NULL
      if (!identical(seed_aggregate, "none")) {
        mats <- lapply(preds, function(e) e$aggregate)
        mats <- Filter(Negate(is.null), mats)
        if (length(mats)) {
          arr <- simplify2array(mats)
          if (length(dim(arr)) == 2L) arr <- array(arr, dim = c(dim(arr), 1L))
          if (seed_aggregate == "mean") {
            aggregate_pred <- apply(arr, c(1, 2), mean)
          } else if (seed_aggregate == "median") {
            aggregate_pred <- apply(arr, c(1, 2), stats::median)
          }
        }
      }
      list(iteration = iter, per_seed = preds, aggregate = aggregate_pred)
    })
  }
  
  result <- list(
    configuration = list(
      classification_mode = classification_mode,
      hidden_sizes = hidden_sizes,
      seeds = seeds,
      do_ensemble = isTRUE(do_ensemble),
      num_networks = as.integer(base_model_args$num_networks),
      num_temp_iterations = as.integer(num_temp_iterations),
      aggregate = aggregate,
      seed_aggregate = seed_aggregate,
      prediction_type = prediction_type
    ),
    runs = runs
  )
  
  if (!is.null(prediction_matrix)) {
    result$predictions <- list(
      per_seed = main_seed_predictions,
      aggregate = main_aggregate
    )
    result$`.__prediction_matrix` <- prediction_matrix
  }
  
  if (length(temp_summary)) {
    result$temp_predictions <- temp_summary
  }
  
  class(result) <- unique(c("ddesonn_run_result", class(result)))
  
  # Optional legacy-style persistence
  if (!is.null(output_root)) {
    .persist_ddesonn_run(result, output_root = output_root, save_models = save_models)
  }
  # Clean result before returning (drop large matrix used only for persistence)
  if (!is.null(result$`.__prediction_matrix`)) {
    result$`.__prediction_matrix` <- NULL
  }
  
  result
}
#' @export
print.ddesonn_run_result <- function(x, ...) {
  cfg <- x$configuration %||% list()
  cat("DDESONN run result\n")
  
  if (length(cfg)) {
    cat(sprintf("  Mode: %s\n", cfg$classification_mode %||% "unknown"))
    
    if (!is.null(cfg$hidden_sizes)) {
      cat(sprintf("  Hidden sizes: %s\n", paste(cfg$hidden_sizes, collapse = ", ")), "\n")
    }
    
    cat(sprintf("  Seeds: %s\n", paste(cfg$seeds, collapse = ", ")))
    cat(sprintf(
      "  Ensemble: %s (members = %s, TEMP iterations = %s)\n",
      if (isTRUE(cfg$do_ensemble)) "enabled" else "disabled",
      cfg$num_networks %||% "n/a",
      cfg$num_temp_iterations %||% 0L
    ))
  }
  
  cat(sprintf("  Runs captured: %d\n", length(x$runs %||% list())))
  invisible(x)
}
