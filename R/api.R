#' Internal package environment used to lazily load the legacy DDESONN stack.
#'
#' @keywords internal
#' @noRd
.ddesonn_env <- new.env(parent = emptyenv())

#' Null-coalescing helper used across the high-level API.
#'
#' @keywords internal
#' @noRd
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

.ddesonn_find_root <- function() {
  pkg_root <- system.file(package = "DDESONN")
  if (nzchar(pkg_root)) return(pkg_root)
  getOption("DDESONN_ROOT", default = getwd())  # should be the *repo root*
}

.ddesonn_source_legacy <- function() {
  if (isTRUE(get0(".ddesonn_initialized", envir = .ddesonn_env, inherits = FALSE))) {
    return(invisible(.ddesonn_env))
  }

  rm(list = ls(envir = .ddesonn_env, all.names = TRUE), envir = .ddesonn_env)
  ns <- getNamespace("DDESONN")
  objs <- setdiff(ls(ns, all.names = TRUE), ".ddesonn_env")
  for (nm in objs) {
    assign(nm, get(nm, envir = ns, inherits = FALSE), envir = .ddesonn_env)
  }
  assign(".ddesonn_initialized", TRUE, envir = .ddesonn_env)
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

# special helper for picking up on the user's input in the model's set-up.

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
      if (any(!is.finite(hs))) {
        stop("hidden_sizes must be non-negative finite integers.", call. = FALSE)
      }
      if (any(hs < 0)) {
        stop("hidden_sizes must be non-negative finite integers.", call. = FALSE)
      }
      if (any(hs != as.integer(hs))) {
        stop("hidden_sizes must be non-negative finite integers.", call. = FALSE)
      }
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
  
  if (length(hs) == 0L) {
    list(arch = "single", hidden_sizes = hs)
  } else {
    list(arch = "multi", hidden_sizes = hs)
  }
}

#' @title Default activation sequences for DDESONN helpers
#' @description Compute sensible activation functions for hidden and output
#'   layers based on the modelling mode and stage (training or prediction).
#'
#' @param mode Problem mode. One of `"binary"`, `"multiclass"`, or `"regression"`.
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#' @param stage Stage for which activations are required. Either `"train"` or `"predict"`.
#'
#' @return A list of activation functions suitable for passing into the
#'   underlying R6 classes.
#'
#' @examples
#' ddesonn_activation_defaults("binary", hidden_sizes = c(32, 16))
#' ddesonn_activation_defaults("regression", hidden_sizes = 64, stage = "predict")
#'
#' @export
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

#' @title Default dropout configuration
#' @description Produce a simple dropout configuration matching the supplied
#'   hidden layer sizes.
#'
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#'
#' @return A list of dropout rates for each hidden layer.
#'
#' @examples
#' ddesonn_dropout_defaults(c(64, 32))
#'
#' @export
ddesonn_dropout_defaults <- function(hidden_sizes) {
  hidden_sizes <- hidden_sizes %||% integer()
  if (!length(hidden_sizes)) {
    return(list())
  }
  as.list(rep(0.1, length(hidden_sizes)))
}

#' @title Supported optimizer identifiers
#' @description List the optimizer strings understood by the legacy DDESONN
#'   training loop.
#'
#' @return A character vector of supported optimiser identifiers.
#'
#' @examples
#' ddesonn_optimizer_options()
#'
#' @export
ddesonn_optimizer_options <- function() {
  c("adagrad", "adam", "lamb", "sgd", "sgd_momentum", "nag", "rmsprop", "ftrl", "lookahead")
}

.ddesonn_threshold_default <- function(mode) {
  if (mode %in% c("binary", "multiclass")) 0.5 else NA_real_
}

#' @title Construct default training controls
#' @description Build a list of training hyperparameters that mirror the
#'   expectations of the legacy DDESONN training loop.
#'
#' @param mode Problem mode used to determine sensible defaults.
#' @param hidden_sizes Integer vector describing the hidden layer widths.
#'
#' @return A named list that can be modified and supplied to [ddesonn_fit()].
#'
#' @examples
#' ddesonn_training_defaults("binary", hidden_sizes = c(32, 16))
#'
#' @export
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
    verbose = FALSE,
    output_root = NULL
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

#' @title Create a high-level DDESONN model wrapper
#' @description Initialise a `ddesonn_model` (R6) instance backed by the legacy
#'   `DDESONN` class, while handling sensible defaults for activations and node
#'   counts.
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
#'
#' @examples
#' model <- ddesonn_model(
#'   input_size = 5,
#'   output_size = 1,
#'   hidden_sizes = c(32, 16),
#'   classification_mode = "binary"
#' )
#'
#' @seealso [DDESONN]
#' @export
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
    init_method = init_method,
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

#' @title Fit a `ddesonn_model` with tidy inputs
#' @description Train a `ddesonn_model` (backed by `DDESONN`) using matrices or
#'   data frames, handling label coercion, validation data, and training control
#'   defaults.
#'
#' @param model A model created by [ddesonn_model()].
#' @param x Training features.
#' @param y Training targets/labels.
#' @param validation Optional list containing `x` and `y` elements for validation.
#' @param ... Named overrides for entries in [ddesonn_training_defaults()].
#'
#' @return The trained model (invisibly). The underlying R6 object is modified
#'   in-place and the last training result is stored under `model$last_training`.
#'
#' @examples
#' data <- mtcars
#' x <- data[, c("disp", "hp", "wt", "qsec", "drat")]
#' y <- data$am
#' model <- ddesonn_model(input_size = ncol(x), output_size = 1, hidden_sizes = 8)
#' ddesonn_fit(model, x, y, num_epochs = 1, lr = 0.05, validation_metrics = FALSE)
#'
#' @seealso [DDESONN]
#' @export
ddesonn_fit <- function(model, x, y, validation = NULL, ...) {
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  
  data_prep <- .prepare_training_data(x)
  
  overrides <- list(...)  # <-- move earlier so we can use it for mode
  # 1) Resolve mode with explicit override first, then model attr, then default
  mode <- tolower(overrides$classification_mode %||% attr(model, "classification_mode") %||% "binary")
  hidden_sizes <- attr(model, "hidden_sizes") %||% NULL
  
  # --- Coerce TRAIN labels by mode (no external helpers) ---
  y_in <- if (is.list(y) && !is.data.frame(y)) unlist(y, use.names = FALSE) else y
  labels <- NULL
  if (mode == "regression") {
    if (is.factor(y_in)) y_in <- as.numeric(as.character(y_in))
    labels <- .as_numeric_matrix(y_in)
  } else if (mode == "binary") {
    if (is.matrix(y_in) || is.data.frame(y_in)) {
      yy <- as.matrix(y_in); storage.mode(yy) <- "double"
      if (ncol(yy) == 2L && all(yy %in% c(0,1), na.rm = TRUE)) {
        labels <- yy[, 2, drop = FALSE]
      } else if (ncol(yy) == 1L) {
        v <- yy[,1]
        u <- sort(unique(as.numeric(v)))
        if (length(u) == 2L && !all(u %in% c(0,1))) {
          map <- setNames(c(0,1), u)
          v <- as.numeric(map[as.character(as.numeric(v))])
        }
        labels <- matrix(as.numeric(v), ncol = 1L)
      } else {
        stop("Binary labels must be a single column (or 2-col one-hot).", call. = FALSE)
      }
    } else {
      if (is.logical(y_in)) {
        v <- ifelse(y_in, 1, 0)
      } else if (is.factor(y_in) || is.character(y_in)) {
        lvls <- if (is.factor(y_in)) levels(y_in) else sort(unique(y_in))
        if (length(lvls) != 2L) stop("Binary labels must have exactly 2 levels.", call. = FALSE)
        map <- setNames(c(0,1), lvls)
        v <- as.numeric(map[as.character(if (is.factor(y_in)) as.character(y_in) else y_in)])
      } else {
        v0 <- as.numeric(y_in); u <- sort(unique(v0))
        if (length(u) != 2L) stop("Binary numeric labels must have exactly two unique values.", call. = FALSE)
        map <- setNames(c(0,1), u)
        v <- as.numeric(map[as.character(v0)])
      }
      labels <- matrix(v, ncol = 1L)
    }
  } else if (mode == "multiclass") {
    if (is.matrix(y_in) || is.data.frame(y_in)) {
      yy <- as.matrix(y_in); storage.mode(yy) <- "double"
      vals_ok <- all(yy %in% c(0,1), na.rm = TRUE)
      row_ok  <- all(rowSums(yy, na.rm = TRUE) >= 0.99 & rowSums(yy, na.rm = TRUE) <= 1.01)
      if (ncol(yy) >= 2L && vals_ok && row_ok) {
        labels <- yy
      } else if (ncol(yy) == 1L) {
        cls <- as.vector(yy[,1])
        u <- sort(unique(as.numeric(cls)))
        K <- length(u)
        idx <- match(as.numeric(cls), u)
        if (any(is.na(idx))) stop("Multiclass labels contain NA/unknown.", call. = FALSE)
        M <- matrix(0, nrow = length(idx), ncol = K); M[cbind(seq_along(idx), idx)] <- 1
        labels <- M
      } else {
        stop("Multiclass labels must be one-hot or a single class column.", call. = FALSE)
      }
    } else {
      if (is.factor(y_in)) {
        lvls <- levels(y_in); idx <- as.integer(y_in); K <- length(lvls)
      } else if (is.character(y_in)) {
        lvls <- sort(unique(y_in)); idx <- match(y_in, lvls); K <- length(lvls)
      } else {
        v0 <- as.numeric(y_in); u <- sort(unique(v0)); idx <- match(v0, u); K <- length(u)
      }
      if (any(is.na(idx))) stop("Multiclass labels contain NA/unknown.", call. = FALSE)
      M <- matrix(0, nrow = length(idx), ncol = K); M[cbind(seq_along(idx), idx)] <- 1
      labels <- M
    }
  } else {
    stop("Unknown classification_mode.", call. = FALSE)
  }
  
  defaults <- ddesonn_training_defaults(mode, hidden_sizes)
  cfg <- utils::modifyList(defaults, overrides, keep.null = TRUE)
  
  cfg$activation_functions <- cfg$activation_functions %||% attr(model, "activation_functions")
  cfg$activation_functions_predict <- cfg$activation_functions_predict %||% attr(model, "activation_functions_predict")
  cfg$dropout_rates <- cfg$dropout_rates %||% ddesonn_dropout_defaults(hidden_sizes)
  cfg$numeric_columns <- cfg$numeric_columns %||% data_prep$numeric_columns
  
  # ============================================================
  # SECTION: Final summary formatting control (presentation-only)  #$$$$$$$$$$$$$
  # - User override name: final_summary_decimals
  # - Applies ONLY to values we attach for reporting / metadata display
  # ============================================================
  cfg$final_summary_decimals <- overrides$final_summary_decimals %||% NULL  #$$$$$$$$$$$$$
  
  plot_cfg_override <- overrides$EvaluatePredictionsReportPlotsConfig %||%
    overrides$evaluate_predictions_report_plots %||%
    overrides$eval_report_plots
  if (is.list(plot_cfg_override)) {
    current_cfg <- tryCatch(model$EvaluatePredictionsReportPlotsConfig, error = function(e) list())
    model$EvaluatePredictionsReportPlotsConfig <- utils::modifyList(current_cfg %||% list(), plot_cfg_override, keep.null = TRUE)
  }
  
  # ============================================================
  # SECTION: Plot controls wiring (required arg support)  #$$$$$$$$$$$$$
  # - train_network()/model$train may require plot_controls with NO default
  # - accept both plot_controls and PlotControls keys from ...
  # ============================================================
  cfg$plot_controls <- overrides$plot_controls %||% overrides$PlotControls %||% cfg$plot_controls %||% NULL  #$$$$$$$$$$$$$
  
  # 2) Threshold tuner only for binary; NULL otherwise (prevents downstream b