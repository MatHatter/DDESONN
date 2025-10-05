#' Internal package environment used to lazily load the legacy DDESONN stack.
.ddesonn_env <- new.env(parent = emptyenv())

#' Null-coalescing helper used across the high-level API.
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

.ddesonn_find_root <- function() {
  pkg_root <- system.file(package = "DDESONN")
  if (nzchar(pkg_root)) {
    return(pkg_root)
  }
  getOption("DDESONN_ROOT", default = getwd())
}

.ddesonn_source_legacy <- function() {
  if (exists("DDESONN", envir = .ddesonn_env, inherits = FALSE)) {
    return(invisible(.ddesonn_env))
  }
  
  root <- .ddesonn_find_root()
  target <- file.path(root, "DDESONN.R")
  if (!file.exists(target)) {
    stop("Unable to locate 'DDESONN.R'. Set options(DDESONN_ROOT=...) to the repository root before calling the API.")
  }
  
  base_source <- base::source
  assign(
    "source",
    function(file, ...) {
      resolved <- file.path(root, file)
      if (!file.exists(resolved)) {
        stop(sprintf("Unable to locate dependency file '%s' relative to '%s'", file, root), call. = FALSE)
      }
      base_source(resolved, local = .ddesonn_env, ...)
    },
    envir = .ddesonn_env
  )
  
  sys.source(target, envir = .ddesonn_env, chdir = FALSE)
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
    sample_weights = NULL,
    preprocessScaledData = NULL,
    X_validation = NULL,
    y_validation = NULL,
    validation_metrics = TRUE,
    threshold_function = .ddesonn_get("tune_threshold_accuracy"),
    ML_NN = TRUE,
    train_flag = TRUE,
    grouped_metrics = FALSE,
    viewTables = FALSE,
    verbose = FALSE
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
#' @param num_networks Number of SONN members to initialise within the ensemble.
#' @param lambda Regularisation strength.
#' @param classification_mode Problem mode: `"binary"`, `"multiclass"`, or `"regression"`.
#' @param ML_NN Logical; whether to initialise a multi-layer SONN.
#' @param activation_functions Optional list of activation functions for training.
#' @param activation_functions_predict Optional list of activation functions used during prediction.
#' @param init_method Weight initialisation scheme passed to the legacy constructor.
#' @param custom_scale Optional scaling factor for the initialiser.
#' @param N Optional total node count. If omitted it is inferred from the architecture.
#' @param ensembles Optional pre-existing ensemble container.
#' @param ensemble_number Identifier used when combining multiple ensembles.
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
                          num_networks = 1L,
                          lambda = 2.8e-4,
                          classification_mode = c("binary", "multiclass", "regression"),
                          ML_NN = TRUE,
                          activation_functions = NULL,
                          activation_functions_predict = NULL,
                          init_method = "he",
                          custom_scale = 1,
                          N = NULL,
                          ensembles = NULL,
                          ensemble_number = 0L) {
  classification_mode <- match.arg(classification_mode)
  
  activation_functions <- activation_functions %||%
    ddesonn_activation_defaults(classification_mode, hidden_sizes, stage = "train")
  activation_functions_predict <- activation_functions_predict %||%
    ddesonn_activation_defaults(classification_mode, hidden_sizes, stage = "predict")
  
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
    method = init_method,
    custom_scale = custom_scale
  )
  
  attr(model, "classification_mode") <- classification_mode
  attr(model, "activation_functions") <- activation_functions
  attr(model, "activation_functions_predict") <- activation_functions_predict
  attr(model, "hidden_sizes") <- hidden_sizes
  attr(model, "lambda") <- lambda
  attr(model, "ML_NN") <- ML_NN
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
#' @param ... Named overrides for entries in [ddesonn_training_defaults()].
#'
#' @return The trained model (invisibly). The underlying R6 object is modified
#'   in-place and the last training result is stored under `model$last_training`.
#' @export
#'
#' @examples
#' data <- mtcars
#' x <- data[, c("disp", "hp", "wt", "qsec", "drat")]
#' y <- data$am
#' model <- ddesonn_model(input_size = ncol(x), output_size = 1, hidden_sizes = 8)
#' ddesonn_fit(model, x, y, num_epochs = 1, lr = 0.05, validation_metrics = FALSE)
ddesonn_fit <- function(model, x, y, validation = NULL, ...) {
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  
  data_prep <- .prepare_training_data(x)
  labels <- .as_numeric_matrix(y)
  
  mode <- attr(model, "classification_mode") %||% "binary"
  hidden_sizes <- attr(model, "hidden_sizes") %||% NULL
  
  defaults <- ddesonn_training_defaults(mode, hidden_sizes)
  overrides <- list(...)
  cfg <- utils::modifyList(defaults, overrides, keep.null = TRUE)
  
  cfg$activation_functions <- cfg$activation_functions %||%
    attr(model, "activation_functions")
  cfg$activation_functions_predict <- cfg$activation_functions_predict %||%
    attr(model, "activation_functions_predict")
  cfg$dropout_rates <- cfg$dropout_rates %||% ddesonn_dropout_defaults(hidden_sizes)
  cfg$numeric_columns <- cfg$numeric_columns %||% data_prep$numeric_columns
  cfg$threshold_function <- cfg$threshold_function %||% .ddesonn_get("tune_threshold_accuracy")
  cfg$ML_NN <- isTRUE(cfg$ML_NN %||% attr(model, "ML_NN"))
  cfg$ensemble_number <- overrides$ensemble_number %||% cfg$ensemble_number %||% 0L
  
  if (!is.null(validation)) {
    if (!is.list(validation) || !all(c("x", "y") %in% names(validation))) {
      stop("'validation' must be a list with elements 'x' and 'y'.", call. = FALSE)
    }
    cfg$X_validation <- .as_numeric_matrix(validation$x)
    cfg$y_validation <- .as_numeric_matrix(validation$y)
  }
  
  train_args <- list(
    Rdata = data_prep$data,
    labels = labels,
    lr = cfg$lr,
    lr_decay_rate = cfg$lr_decay_rate,
    lr_decay_epoch = cfg$lr_decay_epoch,
    lr_min = cfg$lr_min,
    ensemble_number = cfg$ensemble_number,
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
    sample_weights = cfg$sample_weights,
    preprocessScaledData = cfg$preprocessScaledData,
    X_validation = cfg$X_validation,
    y_validation = cfg$y_validation,
    validation_metrics = cfg$validation_metrics,
    threshold_function = cfg$threshold_function,
    ML_NN = cfg$ML_NN,
    train = cfg$train_flag,
    grouped_metrics = cfg$grouped_metrics,
    viewTables = cfg$viewTables,
    verbose = cfg$verbose
  )
  
  result <- do.call(model$train, train_args)
  model$last_training <- result
  attr(model, "threshold") <- cfg$threshold
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
#'
#' @return A list containing the aggregated prediction matrix and the
#'   per-model outputs when `aggregate = "none"`.
#' @export
#'
#' @examples
#' data <- mtcars
#' x <- data[, c("disp", "hp", "wt", "qsec", "drat")]
#' y <- data$am
#' model <- ddesonn_model(input_size = ncol(x), output_size = 1, hidden_sizes = 8)
#' ddesonn_fit(model, x, y, num_epochs = 1, lr = 0.05, validation_metrics = FALSE)
#' preds <- ddesonn_predict(model, x)
#' head(preds$prediction)
ddesonn_predict <- function(model, new_data, aggregate = c("mean", "median", "none"),
                            type = c("response", "class"), threshold = NULL) {
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  aggregate <- match.arg(aggregate)
  type <- match.arg(type)
  
  X <- .as_numeric_matrix(new_data)
  mode <- attr(model, "classification_mode") %||% "binary"
  
  preds <- lapply(seq_along(model$ensemble), function(i) {
    net <- model$ensemble[[i]]
    acts <- net$activation_functions_predict %||% net$activation_functions
    res <- net$predict(
      Rdata = X,
      weights = net$weights,
      biases = net$biases,
      activation_functions_predict = acts,
      verbose = FALSE,
      debug = FALSE
    )
    out <- res$predicted_output %||% res
    .as_numeric_matrix(out)
  })
  
  aggregated <- .aggregate_predictions(preds, aggregate)
  output <- list(prediction = aggregated, per_model = preds)
  
  if (type == "class") {
    if (!mode %in% c("binary", "multiclass")) {
      stop("Class predictions are only available for classification modes.", call. = FALSE)
    }
    thr <- threshold %||% attr(model, "threshold") %||% .ddesonn_threshold_default(mode)
    if (mode == "binary") {
      output$class <- ifelse(aggregated >= thr, 1L, 0L)
    } else {
      output$class <- max.col(aggregated, ties.method = "first")
    }
  }
  
  output
}
