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
#' # Inspect regression defaults (includes LR decay by default).
#' cfg_reg <- ddesonn_training_defaults("regression", hidden_sizes = c(16, 8))
#' cfg_reg$lr
#' cfg_reg$lr_decay_rate
#' cfg_reg$lr_decay_epoch
#' cfg_reg$lr_min
#'
#' # If you prefer a fixed LR in regression, disable decay explicitly.
#' cfg_reg$lr_decay_rate <- 1.0
#'
#' @export
ddesonn_training_defaults <- function(mode = c("binary", "multiclass", "regression"),
                                      hidden_sizes = NULL) {
  mode <- match.arg(mode)
  .ddesonn_source_legacy()

  # Empirically conservative defaults:
  # - classification modes keep a fixed learning rate unless user overrides,
  # - regression keeps decay enabled.
  lr_decay_rate_default <- if (mode == "regression") 0.5 else 1.0
  lr_decay_epoch_default <- if (mode == "regression") 20L else 1L
  
  list(
    lr = 0.125,
    lr_decay_rate = lr_decay_rate_default,
    lr_decay_epoch = lr_decay_epoch_default,
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
#' @param verbose Logical; emit detailed progress output when TRUE.
#' @param verboseLow Logical; emit important progress output when TRUE.
#' @param debug Logical; emit debug diagnostics when TRUE.
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
#' @seealso [DDESONN-package]
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
                          ensemble_number = 0L,  
                          verbose = FALSE,  
                          verboseLow = FALSE,  
                          debug = FALSE) {  
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
#' @param self_org Optional logical override for the legacy self-organization
#'   phase. `TRUE` enables `self_organize()` during training and `FALSE`
#'   disables it. `NULL` keeps the configured default (`self_org = FALSE` in
#'   [ddesonn_training_defaults()]). Self-organization acts on input-space
#'   topology error (how well neighborhood structure is organized), not on the
#'   supervised prediction-loss term.
#' @param ... Named overrides for entries in [ddesonn_training_defaults()].
#' @param verbose Logical; emit detailed progress output when TRUE.
#' @param verboseLow Logical; emit important progress output when TRUE.
#' @param debug Logical; emit debug diagnostics when TRUE.
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
#' # Regression example (mtcars) with explicit scheduler controls.
#' # If you do NOT want LR decay, set lr_decay_rate = 1.0.
#' reg_x <- mtcars[, c("disp", "hp", "wt", "qsec", "drat")]
#' reg_y <- mtcars$mpg
#' reg_model <- ddesonn_model(
#'   input_size = ncol(reg_x),          # number of input features
#'   output_size = 1,                   # one numeric target
#'   hidden_sizes = c(16, 8),           # hidden-layer widths
#'   classification_mode = "regression" # problem type
#' )
#' ddesonn_fit(
#'   model = reg_model,                 # model object from ddesonn_model()
#'   x = reg_x,                         # training predictors
#'   y = reg_y,                         # training target
#'   num_epochs = 10,                   # training epochs
#'   lr = 0.05,                         # initial learning rate
#'   lr_decay_rate = 0.5,               # decay multiplier (use 1.0 to disable)
#'   lr_decay_epoch = 20L,              # decay step interval in epochs
#'   lr_min = 1e-5,                     # lower bound for learning rate
#'   validation_metrics = FALSE         # disable validation metric pass in this example
#' )
#'
#' @seealso [DDESONN-package]
#' @export
ddesonn_fit <- function(model, x, y, validation = NULL, self_org = NULL, ..., verbose = FALSE, verboseLow = FALSE, debug = FALSE) {  
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  
  debug <- isTRUE(debug %||% getOption("DDESONN.debug", FALSE))  
  debug <- isTRUE(debug) && identical(Sys.getenv("DDESONN_DEBUG"), "1")  
  
  data_prep <- .prepare_training_data(x)
  
  overrides <- list(...)  # <-- move earlier so we can use it for mode  
  if (is.null(overrides$self_org) && !is.null(self_org)) overrides$self_org <- self_org
  if (is.null(overrides$verbose)) overrides$verbose <- verbose  
  if (is.null(overrides$verboseLow)) overrides$verboseLow <- verboseLow  
  if (is.null(overrides$debug)) overrides$debug <- debug  
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
  # SECTION: Final summary formatting control (presentation-only)  
  # - User override name: final_summary_decimals
  # - Applies ONLY to values we attach for reporting / metadata display
  # ============================================================  
  cfg$final_summary_decimals <- overrides$final_summary_decimals %||% NULL  
  
  plot_cfg_override <- overrides$EvaluatePredictionsReportPlotsConfig %||%
    overrides$evaluate_predictions_report_plots %||%
    overrides$eval_report_plots
  if (is.list(plot_cfg_override)) {
    current_cfg <- tryCatch(model$EvaluatePredictionsReportPlotsConfig, error = function(e) list())
    model$EvaluatePredictionsReportPlotsConfig <- utils::modifyList(current_cfg %||% list(), plot_cfg_override, keep.null = TRUE)
  }
  
  # ============================================================  
  # SECTION: Plot controls wiring (required arg support)  
  # - train_network()/model$train may require plot_controls with NO default
  # - accept both plot_controls and PlotControls keys from ...
  # ============================================================  
  cfg$plot_controls <- overrides$plot_controls %||% overrides$PlotControls %||% cfg$plot_controls %||% NULL  
  
  # 2) Threshold tuner only for binary; NULL otherwise (prevents downstream “tuned” bundles)
  if (identical(mode, "binary")) {
    cfg$threshold_function <- cfg$threshold_function %||% .ddesonn_get("tune_threshold_accuracy")
  } else {
    cfg$threshold_function <- NULL
  }
  
  cfg$ML_NN <- isTRUE(cfg$ML_NN %||% attr(model, "ML_NN"))
  cfg$ensemble_number <- overrides$ensemble_number %||% cfg$ensemble_number %||% 0L
  
  # VALID labels (if present) — coerce by mode
  if (!is.null(validation)) {
    cfg$X_validation <- .as_numeric_matrix(validation$x)
    yv_in <- validation$y
    if (is.list(yv_in) && !is.data.frame(yv_in)) yv_in <- unlist(yv_in, use.names = FALSE)
    
    if (mode == "regression") {
      if (is.factor(yv_in)) yv_in <- as.numeric(as.character(yv_in))
      cfg$y_validation <- .as_numeric_matrix(yv_in)
    } else if (mode == "binary") {
      if (is.matrix(yv_in) || is.data.frame(yv_in)) {
        yy <- as.matrix(yv_in); storage.mode(yy) <- "double"
        if (ncol(yy) == 2L && all(yy %in% c(0,1), na.rm = TRUE)) {
          cfg$y_validation <- yy[, 2, drop = FALSE]
        } else if (ncol(yy) == 1L) {
          v <- yy[,1]
          u <- sort(unique(as.numeric(v)))
          if (length(u) == 2L && !all(u %in% c(0,1))) {
            map <- setNames(c(0,1), u); v <- as.numeric(map[as.character(as.numeric(v))])
          }
          cfg$y_validation <- matrix(as.numeric(v), ncol = 1L)
        } else stop("Binary validation labels must be 1 col (or 2-col one-hot).", call. = FALSE)
      } else {
        if (is.logical(yv_in)) {
          v <- ifelse(yv_in, 1, 0)
        } else if (is.factor(yv_in) || is.character(yv_in)) {
          lvls <- if (is.factor(yv_in)) levels(yv_in) else sort(unique(yv_in))
          if (length(lvls) != 2L) stop("Binary validation labels must have exactly 2 levels.", call. = FALSE)
          map <- setNames(c(0,1), lvls)
          v <- as.numeric(map[as.character(if (is.factor(yv_in)) as.character(yv_in) else yv_in)])
        } else {
          v0 <- as.numeric(yv_in); u <- sort(unique(v0))
          if (length(u) != 2L) stop("Binary numeric validation labels must have exactly two unique values.", call. = FALSE)
          map <- setNames(c(0,1), u); v <- as.numeric(map[as.character(v0)])
        }
        cfg$y_validation <- matrix(v, ncol = 1L)
      }
    } else if (mode == "multiclass") {
      if (is.matrix(yv_in) || is.data.frame(yv_in)) {
        yy <- as.matrix(yv_in); storage.mode(yy) <- "double"
        vals_ok <- all(yy %in% c(0,1), na.rm = TRUE)
        row_ok  <- all(rowSums(yy, na.rm = TRUE) >= 0.99 & rowSums(yy, na.rm = TRUE) <= 1.01)
        if (ncol(yy) >= 2L && vals_ok && row_ok) {
          cfg$y_validation <- yy
        } else if (ncol(yy) == 1L) {
          cls <- as.vector(yy[,1]); u <- sort(unique(as.numeric(cls))); K <- length(u)
          idx <- match(as.numeric(cls), u)
          if (any(is.na(idx))) stop("Multiclass validation labels contain NA/unknown.", call. = FALSE)
          M <- matrix(0, nrow = length(idx), ncol = K); M[cbind(seq_along(idx), idx)] <- 1
          cfg$y_validation <- M
        } else stop("Multiclass validation labels must be one-hot or a single class column.", call. = FALSE)
      } else {
        if (is.factor(yv_in))        { lvls <- levels(yv_in); idx <- as.integer(yv_in); K <- length(lvls) }
        else if (is.character(yv_in)){ lvls <- sort(unique(yv_in)); idx <- match(yv_in, lvls); K <- length(lvls) }
        else                         { v0 <- as.numeric(yv_in); u <- sort(unique(v0)); idx <- match(v0, u); K <- length(u) }
        if (any(is.na(idx))) stop("Multiclass validation labels contain NA/unknown.", call. = FALSE)
        M <- matrix(0, nrow = length(idx), ncol = K); M[cbind(seq_along(idx), idx)] <- 1
        cfg$y_validation <- M
      }
    }
  }
  
  # derive from model unless overridden
  model_num_networks <- tryCatch(model$num_networks, error = function(e) NULL)
  cfg$num_networks   <- overrides$num_networks %||% model_num_networks %||% 1L
  cfg$do_ensemble    <- overrides$do_ensemble %||% isTRUE(cfg$num_networks > 1L)
  cfg$best_weights_on_latest_weights_off <- overrides$best_weights_on_latest_weights_off %||% FALSE
  
  # --- add defaults (overridable via ...) ---
  cfg$update_weights <- overrides$update_weights %||% TRUE
  cfg$update_biases  <- overrides$update_biases  %||% TRUE
  
  train_args <- list(
    Rdata = data_prep$data,
    labels = labels,
    X_train = data_prep$data,
    y_train = labels,
    lr = cfg$lr,
    lr_decay_rate = cfg$lr_decay_rate,
    lr_decay_epoch = cfg$lr_decay_epoch,
    lr_min = cfg$lr_min,
    num_networks = cfg$num_networks,
    ensemble_number = cfg$ensemble_number,
    do_ensemble  = cfg$do_ensemble,
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
    threshold_function = cfg$threshold_function,  # will be NULL unless binary
    best_weights_on_latest_weights_off = cfg$best_weights_on_latest_weights_off,
    ML_NN = cfg$ML_NN,
    train = cfg$train_flag,
    grouped_metrics = cfg$grouped_metrics,
    viewTables = cfg$viewTables,
    verbose = cfg$verbose,
    verboseLow = cfg$verboseLow,
    output_root = cfg$output_root,
    plot_controls = cfg$plot_controls                               # FIX: always pass required arg
  )
  
  # ============================================================  
  # SECTION: EVOKE — post-train enrichment (NOT per-epoch)  
  # ============================================================  
  ddesonn_console_log(                                                                               
    sprintf(                                                                                          
      "[EVOKE-ENRICH] where=%s | why=%s | note=%s | epochs_configured=%s\n",                           
      "ddesonn_fit::post_train_metadata",                                                             
      "About to attach per-slot metadata; this may call ddesonn_predict() again (aggregate='none')",  
      "This EVOKE is NOT per-epoch; epochs run inside model$train() (not exposed here)",              
      as.character(cfg$num_epochs %||% NA_integer_)                                                   
    ),                                                                                                
    level = "info",                                                                                   
    verbose = verbose,                                                                                
    verboseLow = verboseLow                                                                           
  )                                                                                                   
  
  result <- do.call(model$train, train_args)
  model$last_training <- result
  attr(model, "threshold") <- cfg$threshold
  
  # ============================================================  
  # SECTION: EVOKE — starting enrichment predicts (NOT per-epoch)  
  # ============================================================  
  ddesonn_console_log(                                                                               
    sprintf(                                                                                          
      "[EVOKE-ENRICH-BEGIN] where=%s | why=%s\n",                                                      
      "ddesonn_fit::post_train_metadata",                                                             
      "Computing per-member TRAIN/VALID predictions for metadata (calls ddesonn_predict -> net$predict)"  
    ),                                                                                                
    level = "info",                                                                                   
    verbose = verbose,                                                                                
    verboseLow = verboseLow                                                                           
  )                                                                                                   
  
  # =========================
  # Attach per-slot metrics
  # =========================
  if (mode %in% c("binary", "multiclass")) {
    thr_used <- cfg$threshold %||% .ddesonn_threshold_default(mode)
    
    # TRAIN predictions (per-member)
    pr_train <- try(ddesonn_predict(model, x, aggregate = "none", type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
    per_model_train <- if (!inherits(pr_train, "try-error")) pr_train$per_model else NULL
    
    # VALID predictions (per-member) if present
    per_model_valid <- NULL
    if (!is.null(validation)) {
      pr_valid <- try(ddesonn_predict(model, validation$x, aggregate = "none", type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
      per_model_valid <- if (!inherits(pr_valid, "try-error")) pr_valid$per_model else NULL
    }
    
    # prepare true labels as vectors for metric calc
    y_train_vec <- NULL
    y_valid_vec <- NULL
    if (mode == "binary") {
      y_train_vec <- as.numeric(labels[,1])
      if (!is.null(cfg$y_validation)) y_valid_vec <- as.numeric(cfg$y_validation[,1])
    } else { # multiclass
      y_train_vec <- max.col(labels, ties.method = "first")
      if (!is.null(cfg$y_validation)) y_valid_vec <- max.col(cfg$y_validation, ties.method = "first")
    }
    
    compute_binary_metrics <- function(y_true, p_hat, thr) {
      y_pred <- as.integer(p_hat >= thr)
      TP <- sum(y_pred == 1L & y_true == 1L, na.rm = TRUE)
      FP <- sum(y_pred == 1L & y_true == 0L, na.rm = TRUE)
      TN <- sum(y_pred == 0L & y_true == 0L, na.rm = TRUE)
      FN <- sum(y_pred == 0L & y_true == 1L, na.rm = TRUE)
      N  <- TP + FP + TN + FN
      acc  <- if (N > 0) (TP + TN)/N else NA_real_
      prec <- if ((TP + FP) > 0) TP/(TP + FP) else NA_real_
      rec  <- if ((TP + FN) > 0) TP/(TP + FN) else NA_real_
      f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) 2*prec*rec/(prec + rec) else NA_real_
      list(
        performance_metric = list(accuracy = acc, precision = prec, recall = rec, f1 = f1, f1_score = f1),
        confusion_matrix   = list(TP = TP, FP = FP, TN = TN, FN = FN)
      )
    }
    compute_multiclass_metrics <- function(y_true_cls, prob_mat) {
      if (is.null(prob_mat) || !length(prob_mat)) {
        return(list(performance_metric = list(accuracy = NA_real_, precision = NA_real_, recall = NA_real_, f1 = NA_real_, f1_score = NA_real_)))
      }
      y_pred_cls <- max.col(prob_mat, ties.method = "first")
      acc <- mean(y_pred_cls == y_true_cls)
      K <- ncol(prob_mat)
      precs <- recs <- f1s <- rep(NA_real_, K)
      for (c in seq_len(K)) {
        TP <- sum(y_pred_cls == c & y_true_cls == c)
        FP <- sum(y_pred_cls == c & y_true_cls != c)
        FN <- sum(y_pred_cls != c & y_true_cls == c)
        prec <- if ((TP + FP) > 0) TP/(TP + FP) else NA_real_
        rec  <- if ((TP + FN) > 0) TP/(TP + FN) else NA_real_
        f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) 2*prec*rec/(prec + rec) else NA_real_
        precs[c] <- prec; recs[c] <- rec; f1s[c] <- f1
      }
      list(
        performance_metric = list(
          accuracy  = acc,
          precision = mean(precs, na.rm = TRUE),
          recall    = mean(recs,  na.rm = TRUE),
          f1        = mean(f1s,   na.rm = TRUE),
          f1_score  = mean(f1s,   na.rm = TRUE)
        )
      )
    }
    
    best_train_acc             <- tryCatch(result$predicted_outputAndTime$best_train_acc,           error = function(e) NA_real_)
    best_epoch_train           <- tryCatch(result$predicted_outputAndTime$best_epoch_train,         error = function(e) NA_integer_)
    best_train_loss            <- tryCatch(result$predicted_outputAndTime$best_train_loss,          error = function(e) NA_real_)
    best_epoch_train_loss      <- tryCatch(result$predicted_outputAndTime$best_epoch_train_loss,    error = function(e) NA_integer_)
    best_val_acc               <- tryCatch(result$predicted_outputAndTime$best_val_acc,             error = function(e) NA_real_)
    best_val_epoch             <- tryCatch(result$predicted_outputAndTime$best_val_epoch,           error = function(e) NA_integer_)
    best_val_prediction_time   <- tryCatch(result$predicted_outputAndTime$best_val_prediction_time, error = function(e) NA_real_)
    
    # ============================================================
    # SECTION: Presentation rounding (metadata only)  
    # ============================================================
    dec <- cfg$final_summary_decimals %||% NULL  
    best_train_acc           <- .ddesonn_format_final_summary_decimals(best_train_acc, dec)           
    best_train_loss          <- .ddesonn_format_final_summary_decimals(best_train_loss, dec)          
    best_val_acc             <- .ddesonn_format_final_summary_decimals(best_val_acc, dec)             
    best_val_prediction_time <- .ddesonn_format_final_summary_decimals(best_val_prediction_time, dec) 
    
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
            m_tr <- compute_binary_metrics(y_train_vec, as.numeric(Pt[,1]), thr_used)
          } else {
            m_tr <- compute_multiclass_metrics(y_train_vec, Pt)
          }
          # format only for display (metadata)                              
          m_tr <- .ddesonn_format_final_summary_decimals(m_tr, dec)         
          slot_obj$metadata$performance_metric <- m_tr$performance_metric
          if (mode == "binary" && !is.null(m_tr$confusion_matrix)) {
            slot_obj$metadata$confusion_matrix <- m_tr$confusion_matrix
          }
        }
        
        # VALID metrics per slot (optional)
        if (!is.null(per_model_valid) && length(per_model_valid) >= k && !is.null(y_valid_vec)) {
          Pv <- as.matrix(per_model_valid[[k]])
          if (mode == "binary") {
            m_va <- compute_binary_metrics(y_valid_vec, as.numeric(Pv[,1]), thr_used)
            m_va <- .ddesonn_format_final_summary_decimals(m_va, dec)       
            # Only in BINARY: expose the tuned bundle (prevents utils from “thinking binary” otherwise)
            slot_obj$metadata$accuracy_precision_recall_f1_tuned <- list(
              accuracy = m_va$performance_metric$accuracy,
              precision = m_va$performance_metric$precision,
              recall = m_va$performance_metric$recall,
              f1 = m_va$performance_metric$f1,
              confusion_matrix = m_va$confusion_matrix,
              chosen_threshold = thr_used
            )
          } else {
            # Multiclass: NO tuned bundle, NO confusion_matrix (keeps downstream from mapping binary fields)
            m_va <- compute_multiclass_metrics(y_valid_vec, Pv)
            m_va <- .ddesonn_format_final_summary_decimals(m_va, dec)       
            # (optional) slot_obj$metadata$valid_performance_metric <- m_va$performance_metric
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
    pr_train <- try(ddesonn_predict(model, x, aggregate = "none", type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
    per_model_train <- if (!inherits(pr_train, "try-error")) pr_train$per_model else NULL
    
    # VALID predictions (per-member) if present
    per_model_valid <- NULL
    if (!is.null(validation)) {
      pr_valid <- try(ddesonn_predict(model, validation$x, aggregate = "none", type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
      per_model_valid <- if (!inherits(pr_valid, "try-error")) pr_valid$per_model else NULL
    }
    
    # true labels as numeric vectors
    y_train_vec <- as.numeric(labels[, 1])
    y_valid_vec <- if (!is.null(cfg$y_validation)) as.numeric(cfg$y_validation[, 1]) else NULL
    
    compute_regression_metrics <- function(y_true, y_hat) {
      y_true <- as.numeric(y_true); y_hat <- as.numeric(y_hat)
      ok <- is.finite(y_true) & is.finite(y_hat)
      y_true <- y_true[ok]; y_hat <- y_hat[ok]
      if (!length(y_true)) {
        return(list(MSE=NA_real_, RMSE=NA_real_, MAE=NA_real_, R2=NA_real_))
      }
      err  <- y_hat - y_true
      mse  <- mean(err^2)
      rmse <- sqrt(mse)
      mae  <- mean(abs(err))
      sst  <- sum((y_true - mean(y_true))^2)
      ssr  <- sum(err^2)
      r2   <- if (sst > 0) 1 - (ssr / sst) else NA_real_
      list(MSE=mse, RMSE=rmse, MAE=mae, R2=r2)
    }
    
    dec <- cfg$final_summary_decimals %||% NULL  
    
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
          m_tr <- compute_regression_metrics(y_train_vec, pt)                                  
          m_tr <- .ddesonn_format_final_summary_decimals(m_tr, dec)                            
          slot_obj$metadata$performance_metric <- m_tr
        }
        
        # VALID metrics per-slot (optional)
        if (!is.null(per_model_valid) && length(per_model_valid) >= k && !is.null(y_valid_vec)) {
          pv <- as.numeric(per_model_valid[[k]][, 1])
          m_va <- compute_regression_metrics(y_valid_vec, pv)                                  
          m_va <- .ddesonn_format_final_summary_decimals(m_va, dec)                            
          slot_obj$metadata$validation_metrics <- m_va
        }
        
        # Carry best_* fields if present from training result (harmless if NA)
        slot_obj$metadata$best_train_acc           <- .take1num(tryCatch(result$predicted_outputAndTime$best_train_acc,           error=function(e) NA_real_))
        slot_obj$metadata$best_epoch_train         <- .int(     tryCatch(result$predicted_outputAndTime$best_epoch_train,         error=function(e) NA_integer_))
        slot_obj$metadata$best_train_loss          <- .take1num(tryCatch(result$predicted_outputAndTime$best_train_loss,          error=function(e) NA_real_))
        slot_obj$metadata$best_epoch_train_loss    <- .int(     tryCatch(result$predicted_outputAndTime$best_epoch_train_loss,    error=function(e) NA_integer_))
        slot_obj$metadata$best_val_acc             <- .take1num(tryCatch(result$predicted_outputAndTime$best_val_acc,             error=function(e) NA_real_))
        slot_obj$metadata$best_val_epoch           <- .int(     tryCatch(result$predicted_outputAndTime$best_val_epoch,           error=function(e) NA_integer_))
        slot_obj$metadata$best_val_prediction_time <- .take1num(tryCatch(result$predicted_outputAndTime$best_val_prediction_time, error=function(e) NA_real_))
      }
    }
  }
  
  # ============================================================  
  # SECTION: EVOKE — enrichment finished (NOT per-epoch)  
  # ============================================================  
  ddesonn_console_log(                                                                               
    sprintf(                                                                                          
      "[EVOKE-ENRICH-END] where=%s | why=%s | note=%s\n",                                              
      "ddesonn_fit::post_train_metadata",                                                             
      "Finished attaching per-slot metadata; predict() prints after FINAL SUMMARY usually come from this block",  
      "Not per-epoch: epoch-level prints must live inside model$train()/train_network() predict call sites"  
    ),                                                                                                 
    level = "info",                                                                                    
    verbose = verbose,                                                                                 
    verboseLow = verboseLow                                                                            
  )                                                                                                    
  
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

.emit_final_run_summary <- function(pred_summary_final,
                                    performance_relevance_data,
                                    cfg,
                                    mode,
                                    test_metrics = NULL) {
  if (is.null(pred_summary_final) || !length(pred_summary_final)) {
    return(invisible(NULL))
  }
  
  .first_scalar <- function(x) {
    if (is.null(x) || !length(x)) return(NULL)
    if (is.list(x)) {
      for (val in x) {
        if (!is.null(val) && length(val)) return(val[[1]])
      }
      return(NULL)
    }
    x[[1]]
  }
  
  best_epoch_train <- suppressWarnings(as.integer(.first_scalar(pred_summary_final$best_epoch_train)))
  best_epoch_train_loss <- suppressWarnings(as.integer(.first_scalar(pred_summary_final$best_epoch_train_loss)))
  best_val_epoch <- suppressWarnings(as.integer(.first_scalar(pred_summary_final$best_val_epoch)))
  best_val_epoch_loss <- suppressWarnings(as.integer(.first_scalar(pred_summary_final$best_val_epoch_loss)))
  best_train_acc <- suppressWarnings(as.numeric(.first_scalar(pred_summary_final$best_train_acc)))
  best_val_acc <- suppressWarnings(as.numeric(.first_scalar(pred_summary_final$best_val_acc)))
  best_train_loss <- suppressWarnings(as.numeric(.first_scalar(pred_summary_final$best_train_loss)))
  best_val_loss <- suppressWarnings(as.numeric(.first_scalar(pred_summary_final$best_val_loss)))
  
  summary_best_epoch <- if (isTRUE(cfg$validation_metrics)) {
    if (identical(mode, "regression")) best_val_epoch_loss else best_val_epoch
  } else {
    if (identical(mode, "regression")) best_epoch_train_loss else best_epoch_train
  }
  
  # ============================================================  
  # SECTION: Presentation decimals (FINAL SUMMARY)  
  # - User control: cfg$final_summary_decimals
  # - Applies ONLY to printed scalar metrics (epochs remain ints)
  # ============================================================  
  dec <- cfg$final_summary_decimals %||% NULL  
  
  best_train_acc_disp  <- .ddesonn_format_final_summary_decimals(best_train_acc, dec)   
  best_val_acc_disp    <- .ddesonn_format_final_summary_decimals(best_val_acc, dec)     
  best_train_loss_disp <- .ddesonn_format_final_summary_decimals(best_train_loss, dec)  
  best_val_loss_disp   <- .ddesonn_format_final_summary_decimals(best_val_loss, dec)    
  
  threshold_disp <- NULL  
  if (!is.null(performance_relevance_data$threshold) &&
      is.finite(performance_relevance_data$threshold)) {
    threshold_disp <- .ddesonn_format_final_summary_decimals(performance_relevance_data$threshold, dec)  
  }
  
  # ============================================================  
  # SECTION: FINAL SUMMARY line alignment (pad labels to colons)  
  # ============================================================  
  .summary_line <- function(label, value, label_width) {          
    sprintf(paste0("%-", label_width, "s: %s"), label, value)      
  }                                                               
  label_width <- 20L                                              
  
  summary_lines <- c("===== FINAL SUMMARY =====")
  summary_lines <- c(summary_lines, .summary_line("Best epoch", as.character(summary_best_epoch), label_width))                  
  summary_lines <- c(summary_lines, .summary_line("Train accuracy", as.character(best_train_acc_disp), label_width))            
  if (isTRUE(cfg$validation_metrics)) {
    summary_lines <- c(summary_lines, .summary_line("Val accuracy", as.character(best_val_acc_disp), label_width))              
  }
  if (!is.null(best_train_loss) && is.finite(best_train_loss)) {
    summary_lines <- c(summary_lines, .summary_line("Train loss", as.character(best_train_loss_disp), label_width))             
  }
  if (!is.null(best_val_loss) && is.finite(best_val_loss)) {
    summary_lines <- c(summary_lines, .summary_line("Val loss", as.character(best_val_loss_disp), label_width))                 
  }
  if (!is.null(threshold_disp)) {  
    summary_lines <- c(summary_lines, .summary_line("Threshold", as.character(threshold_disp), label_width))                    
  }
  
  roc_plot <- performance_relevance_data$eval_report_plots$roc_png %||% NULL
  if (!is.null(roc_plot) && is.character(roc_plot)) {
    roc_plot <- roc_plot[1]
    if (isTRUE(nzchar(roc_plot)) && !is.na(roc_plot)) {
      summary_lines <- c(summary_lines, .summary_line("ROC plot", roc_plot, label_width))                                      
    }
  }
  pr_plot <- performance_relevance_data$eval_report_plots$pr_png %||% NULL
  if (!is.null(pr_plot) && is.character(pr_plot)) {
    pr_plot <- pr_plot[1]
    if (isTRUE(nzchar(pr_plot)) && !is.na(pr_plot)) {
      summary_lines <- c(summary_lines, .summary_line("PR plot", pr_plot, label_width))                                        
    }
  }
  
  if (is.list(test_metrics) && length(test_metrics)) {
    test_acc <- suppressWarnings(as.numeric(test_metrics$accuracy))
    test_loss <- suppressWarnings(as.numeric(test_metrics$loss))
    
    test_acc_disp  <- .ddesonn_format_final_summary_decimals(test_acc, dec)   
    test_loss_disp <- .ddesonn_format_final_summary_decimals(test_loss, dec)  
    
    if (is.finite(test_acc)) {
      summary_lines <- c(summary_lines, .summary_line("Test accuracy", as.character(test_acc_disp), label_width))               
    }
    if (is.finite(test_loss)) {
      summary_lines <- c(summary_lines, .summary_line("Test loss", as.character(test_loss_disp), label_width))                 
    }
  }
  
  # ============================================================  
  # SECTION: knitr HTML detection (for newline preservation)      
  # ============================================================  
  .is_knitr_html <- function() {                                                       
    if (!requireNamespace("knitr", quietly = TRUE)) return(FALSE)                      
    if (!isTRUE(getOption("knitr.in.progress"))) return(FALSE)                         
    out <- try(knitr::is_html_output(), silent = TRUE)                                 
    isTRUE(out)                                                                        
  }                                                                                    
  
  if (isTRUE(cfg$viewTables)) {
    
    summary_tbl <- do.call(  
      rbind,                 
      lapply(summary_lines[-1], function(line) {                                   
        key <- sub(":.*$", "", line)                                               
        val <- sub("^[^:]*:\\s*", "", line)                                        
        data.frame(Metric = key, Value = val, stringsAsFactors = FALSE)            
      })                                                                           
    )                                                                              
    
    cat("\n**CORE METRICS**\n\n")                                                   
    cat("**Final Summary**\n\n")                                                   
    
    md_lines <- paste0("**", summary_tbl$Metric, ":** ", summary_tbl$Value)         
    cat(paste0(md_lines, "  \n"), "\n")                                             
    
    # Optional: keep the metric table renderer too (leave commented unless desired)
    # ddesonn_viewTables(summary_tbl)                                               
    
  } else {
    
    # ============================================================  
    # SECTION: CORE METRICS (viewTables = FALSE)                    
    # FIX: In HTML vignettes, wrap in ```text so newlines DON'T     
    # collapse and ASCII doesn't become giant markdown headings.    
    # ============================================================  
    if (isTRUE(.is_knitr_html())) {                                                  
      cat("\n```text\n")                                                             
    }                                                                                
    
    cat("\n# ================================================================================\n")
    cat("# ================================= CORE METRICS =================================\n")
    cat("# ================================================================================\n\n")
    
    # FIX: print stacked lines (do NOT rely on HTML respecting \n unless fenced)     
    cat(paste(summary_lines, collapse = "\n"), "\n")                                 
    
    if (isTRUE(.is_knitr_html())) {                                                  
      cat("```\n")                                                                   
    }                                                                                
  }
  
  options(DDESONN_LAST_SUMMARY_TS = Sys.time())
  invisible(NULL)
}


.compute_test_metrics <- function(model,
                                  x_test,
                                  y_test,
                                  classification_mode,
                                  cfg,
                                  threshold) {
  if (is.null(model) || is.null(x_test) || is.null(y_test)) {
    return(list(ok = FALSE, reason = "missing test data"))
  }
  pr <- try(predict(model, x_test, type = "response"), silent = TRUE)
  if (inherits(pr, "try-error") || is.null(pr$predicted_output)) {
    return(list(ok = FALSE, reason = "response prediction failed"))
  }
  preds <- .as_numeric_matrix(pr$predicted_output)
  n <- nrow(preds)
  if (!is.finite(n) || n < 1L) {
    return(list(ok = FALSE, reason = "no test predictions returned"))
  }
  mode <- tolower(classification_mode)
  targs <- .build_targets(y_test, n, ncol(preds), mode, debug = FALSE)
  labels_for_loss <- if (identical(mode, "binary")) {
    matrix(as.numeric(targs$y), ncol = 1L)
  } else {
    targs$Y
  }
  loss <- try(loss_function(
    predictions = preds,
    labels = labels_for_loss,
    CLASSIFICATION_MODE = mode,
    reg_loss_total = 0,
    loss_type = cfg$loss_type,
    verbose = FALSE
  ), silent = TRUE)
  loss <- if (inherits(loss, "try-error")) NA_real_ else suppressWarnings(as.numeric(loss))
  accuracy <- NA_real_
  if (mode %in% c("binary", "multiclass")) {
    class_pred <- try(predict(model, x_test, type = "class", threshold = threshold), silent = TRUE)
    if (inherits(class_pred, "try-error") || is.null(class_pred)) {
      return(list(ok = FALSE, reason = "class prediction failed"))
    }
    y_true <- if (identical(mode, "multiclass")) {
      if (is.matrix(labels_for_loss) && ncol(labels_for_loss) > 1L) {
        max.col(labels_for_loss, ties.method = "first")
      } else {
        suppressWarnings(as.integer(.extract_vec(y_test)))
      }
    } else {
      suppressWarnings(as.integer(.extract_vec(y_test)))
    }
    y_true <- .align_len(y_true, n)
    y_pred <- .align_len(as.integer(class_pred), n)
    accuracy <- mean(y_pred == y_true, na.rm = TRUE)
  }
  list(
    ok = TRUE,
    loss = loss,
    accuracy = accuracy,
    loss_type = cfg$loss_type %||% NA_character_,
    n = n
  )
}

# Helper rationale: keep the Keras-style classification report formatting in one
# place (AUC/AUPRC + confusion matrix + report table) so Train/Validation/Test
# can print consistently without duplicating logic, even though core metrics
# already exist elsewhere in the codebase.
.coerce_binary_labels <- function(y_source) {
  if (is.null(y_source)) return(NULL)
  y_mat <- try(.as_numeric_matrix(y_source), silent = TRUE)
  if (inherits(y_mat, "try-error") || !is.matrix(y_mat) || !nrow(y_mat)) {
    return(NULL)
  }
  if (ncol(y_mat) == 2L && all(y_mat %in% c(0, 1, NA), na.rm = TRUE)) {
    return(as.integer(y_mat[, 2]))
  }
  as.integer(y_mat[, 1])
}

.build_binary_report <- function(y_true, p_pos, threshold) {
  if (is.null(y_true) || is.null(p_pos)) return(NULL)
  n <- min(length(y_true), length(p_pos))
  if (!is.finite(n) || n < 1L) return(NULL)
  y_true <- as.integer(y_true[seq_len(n)])
  p_pos <- as.numeric(p_pos[seq_len(n)])
  if (all(y_true %in% c(1, 2), na.rm = TRUE) && !any(y_true %in% c(0), na.rm = TRUE)) {
    y_true <- y_true - 1L
  }
  keep <- is.finite(y_true) & is.finite(p_pos)
  y_true <- y_true[keep]
  p_pos <- p_pos[keep]
  if (!length(y_true) || !length(p_pos)) return(NULL)
  if (any(p_pos < 0 | p_pos > 1, na.rm = TRUE)) {
    p_pos <- 1 / (1 + exp(-p_pos))
  }
  thr <- threshold %||% 0.5
  y_pred <- as.integer(p_pos >= thr)
  TP <- sum(y_pred == 1L & y_true == 1L, na.rm = TRUE)
  TN <- sum(y_pred == 0L & y_true == 0L, na.rm = TRUE)
  FP <- sum(y_pred == 1L & y_true == 0L, na.rm = TRUE)
  FN <- sum(y_pred == 0L & y_true == 1L, na.rm = TRUE)
  support0 <- sum(y_true == 0L, na.rm = TRUE)
  support1 <- sum(y_true == 1L, na.rm = TRUE)
  total_support <- support0 + support1
  prec0 <- if ((TN + FN) > 0) TN / (TN + FN) else 0
  rec0 <- if ((TN + FP) > 0) TN / (TN + FP) else 0
  f1_0 <- if ((prec0 + rec0) > 0) 2 * prec0 * rec0 / (prec0 + rec0) else 0
  prec1 <- if ((TP + FP) > 0) TP / (TP + FP) else 0
  rec1 <- if ((TP + FN) > 0) TP / (TP + FN) else 0
  f1_1 <- if ((prec1 + rec1) > 0) 2 * prec1 * rec1 / (prec1 + rec1) else 0
  accuracy <- if (total_support > 0) (TP + TN) / total_support else NA_real_
  macro_precision <- mean(c(prec0, prec1), na.rm = TRUE)
  macro_recall <- mean(c(rec0, rec1), na.rm = TRUE)
  macro_f1 <- mean(c(f1_0, f1_1), na.rm = TRUE)
  weighted_precision <- if (total_support > 0) {
    (prec0 * support0 + prec1 * support1) / total_support
  } else {
    NA_real_
  }
  weighted_recall <- if (total_support > 0) {
    (rec0 * support0 + rec1 * support1) / total_support
  } else {
    NA_real_
  }
  weighted_f1 <- if (total_support > 0) {
    (f1_0 * support0 + f1_1 * support1) / total_support
  } else {
    NA_real_
  }
  report <- data.frame(
    precision = c(prec0, prec1, accuracy, macro_precision, weighted_precision),
    recall = c(rec0, rec1, accuracy, macro_recall, weighted_recall),
    `f1-score` = c(f1_0, f1_1, accuracy, macro_f1, weighted_f1),
    support = c(support0, support1, total_support, total_support, total_support),
    check.names = FALSE,
    row.names = c("0", "1", "accuracy", "macro avg", "weighted avg")
  )
  auc_val <- NA_real_
  auprc_val <- NA_real_
  if (length(unique(y_true)) == 2L && requireNamespace("pROC", quietly = TRUE)) {
    roc_obj <- try(pROC::roc(response = y_true, predictor = p_pos, levels = c(0, 1), direction = "<", quiet = TRUE), silent = TRUE)
    if (!inherits(roc_obj, "try-error")) {
      auc_val <- tryCatch(as.numeric(pROC::auc(roc_obj)), error = function(e) NA_real_)
    }
  }
  if (length(unique(y_true)) == 2L && requireNamespace("PRROC", quietly = TRUE)) {
    pr_obj <- try(
      PRROC::pr.curve(scores.class0 = p_pos[y_true == 1L], scores.class1 = p_pos[y_true == 0L], curve = FALSE),
      silent = TRUE
    )
    if (!inherits(pr_obj, "try-error")) {
      auprc_val <- tryCatch(as.numeric(pr_obj$auc.integral), error = function(e) NA_real_)
    }
  }
  confusion <- matrix(c(TP, FP, FN, TN), nrow = 2, byrow = TRUE,
                      dimnames = list("Actual" = c("Positive (1)", "Negative (0)"),
                                      "Predicted" = c("Positive (1)", "Negative (0)")))
  list(report = report, confusion = confusion, auc = auc_val, auprc = auprc_val)
}

# final summary Classification Report formatting (decimals)
.format_report_value <- function(x, digits = 3L) {
  if (length(x) == 0L) return(x)
  ifelse(
    is.na(x),
    "NA",
    sprintf(paste0("%.", as.integer(digits), "f"), as.numeric(x))
  )
}

# final summary AUC (ROC)/AUPRC formatting (decimals)
.format_report_table <- function(df, digits = 3L) {
  formatted <- df
  for (col in names(formatted)) {
    if (is.numeric(formatted[[col]]) ||
        (is.character(formatted[[col]]) && suppressWarnings(all(is.na(as.numeric(formatted[[col]])) == is.na(formatted[[col]]))))) {
      formatted[[col]] <- .format_report_value(formatted[[col]], digits = digits)
    }
  }
  formatted
}


.emit_binary_classification_report <- function(split_label,
                                               y_true,
                                               prob_mat,
                                               threshold,
                                               final_summary_decimals = NULL,
                                               viewTables = FALSE) {
  if (is.null(y_true) || is.null(prob_mat)) return(invisible(NULL))
  
  probs <- try(.as_numeric_matrix(prob_mat), silent = TRUE)
  if (inherits(probs, "try-error") || !is.matrix(probs) || !nrow(probs)) {
    return(invisible(NULL))
  }
  
  p_pos <- as.numeric(probs[, 1])
  report <- .build_binary_report(y_true, p_pos, threshold)
  if (is.null(report)) return(invisible(NULL))
  
  # ============================================================  
  # SECTION: Presentation decimals (Classification Report + AUC)  
  # ============================================================  
  dec <- suppressWarnings(as.integer(final_summary_decimals))                              
  if (length(dec) != 1L || !is.finite(dec) || dec < 0L) dec <- 3L                          
  split_up <- toupper(as.character(split_label))                                            
  
  # ============================================================  
  # SECTION: Console headers (viewTables = FALSE)                 
  # - FIX: REMOVE "CLR" noise entirely                            
  # - FIX: "===== TRAIN =====" on its own line                    
  # - FIX: "Classification Report" on its own line                
  # ============================================================  
  .cat_console_split_header <- function() {                                                 
    cat("\n")                                                                               
    cat(sprintf("===== %s =====\n\n", split_up))                                              
  }                                                                                         
  .cat_console_section <- function(section_label) {                                         
    cat(sprintf("%s\n", section_label))                                                     
  }                                                                                         
  
  # ============================================================  
  # SECTION: Classification Report header                         
  # ============================================================  
  if (isTRUE(viewTables)) {
    cat(sprintf("\n**%s**\n\n", split_label))                                                
    cat("**Classification Report**\n")                                                       
  } else {
    .cat_console_split_header()                                                              
    .cat_console_section("Classification Report")                                            
  }
  
  ddesonn_viewTables(
    .format_report_table(report$report, digits = dec),
    na.print = ""
  )
  
  # ============================================================  
  # SECTION: Confusion Matrix header                              
  # ============================================================  
  if (isTRUE(viewTables)) {
    cat("\n**Confusion Matrix**\n")                                                          
  } else {
    cat("\n")                                                                               
    .cat_console_section("Confusion Matrix")                                                
  }
  
  # ============================================================
  # SECTION: Confusion Matrix (INTEGER-ONLY)  
  # ============================================================
  confusion_int <- report$confusion                                                         
  confusion_int <- unclass(confusion_int)                                                    
  storage.mode(confusion_int) <- "integer"                                                   
  dim(confusion_int) <- dim(report$confusion)                                                
  dimnames(confusion_int) <- dimnames(report$confusion)                                      
  ddesonn_viewTables(confusion_int, quote = FALSE)                                           
  
  # ============================================================  
  # SECTION: AUC/AUPRC header                                     
  # ============================================================  
  if (isTRUE(viewTables)) {
    cat(sprintf("\n**AUC (ROC):** %s\n", .format_report_value(report$auc, digits = dec)))
    cat(sprintf("**AUPRC:** %s\n", .format_report_value(report$auprc, digits = dec)))
  } else {
    cat("\n")                                                                               
    .cat_console_section("AUC/AUPRC")                                                       
    cat(sprintf("AUC (ROC): %s\n", .format_report_value(report$auc, digits = dec)))         
    cat(sprintf("AUPRC: %s\n", .format_report_value(report$auprc, digits = dec)))           
  }
  
  invisible(report)
}





#' @title Generate predictions from a fitted `ddesonn_model`
#' @description Internal prediction engine / forward-pass primitive that
#'   produces ensemble or per-model predictions from a trained `ddesonn_model`.
#'   For user-facing inference, prefer [predict.ddesonn_model()], which wraps
#'   this helper to provide a stable API for `type`/`aggregate`/`threshold`
#'   handling and return shapes.
#'   Multiclass note: For multiclass classification, y should be encoded as integer class indices 1..K (or a one-hot matrix whose columns follow the model’s class order), otherwise accuracy comparisons may be incorrect.
#'
#' @param model A trained model produced by [ddesonn_model()].
#' @param new_data New feature matrix or data frame.
#' @param aggregate Aggregation strategy across ensemble members. One of
#'   `"mean"`, `"median"`, or `"none"`.
#' @param type Prediction type. `"response"` returns numeric predictions,
#'   while `"class"` applies thresholding for classification problems.
#' @param threshold Optional threshold override when `type = "class"`.
#' @param verbose Logical; emit detailed progress output when TRUE.
#' @param verboseLow Logical; emit important progress output when TRUE.
#' @param debug Logical; emit debug diagnostics when TRUE.
#'
#' @return A list containing the aggregated prediction matrix and the
#'   per-model outputs when `aggregate = "none"`.
#'
#' @examples
#' # ============================================================
#' # Example 1 — Manual API (minimal, CRAN-safe)
#' # ============================================================
#' # This is the base mtcars binary classification example.
#' # The exact same setup is used again below in the full workflow script.
#'
#' data <- mtcars
#' target <- "am"
#' features <- setdiff(colnames(data), target)
#'
#' x <- data[, features]
#' y <- data[[target]]
#'
#' model <- ddesonn_model(
#'   input_size = ncol(x),
#'   output_size = 1,
#'   hidden_sizes = c(32, 16),
#'   classification_mode = "binary",
#'   activation_functions = c("relu", "relu", "sigmoid"),
#'   activation_functions_predict = c("relu", "relu", "sigmoid"),
#'   num_networks = 1
#' )
#'
#' ddesonn_fit(
#'   model,
#'   x,
#'   y,
#'   num_epochs = 3,
#'   lr = 0.02,
#'   validation_metrics = FALSE
#' )
#'
#' preds <- ddesonn_predict(model, x)
#' head(preds$prediction)
#'
#' # ============================================================
#' # Example 2 — Same example, extended (A–D scenarios)
#' # ============================================================
#' # This is the SAME mtcars example shown above.
#' # The only difference is that the full script adds:
#' # - train/validation/test splitting
#' # - scaling (fit on training data)
#' # - ensemble configurations
#' # - scenario orchestration (A–D)
#'
#' # View the full version of this same example:
#' system.file("scripts", "DDESONN_mtcars_A-D_examples.R", package = "DDESONN")
#'
#' # Repository path:
#' # /DDESONN/inst/scripts/DDESONN_mtcars_A-D_examples.R
#' 
#' @seealso [DDESONN-package]
#' @export
ddesonn_predict <- function(model, new_data,  
                            aggregate = c("mean", "median", "none"),  
                            type = c("response", "class"),  
                            threshold = NULL,  
                            verbose = FALSE,  
                            verboseLow = FALSE,  
                            debug = FALSE) {  
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  debug <- isTRUE(debug %||% getOption("DDESONN.debug", FALSE))  
  debug <- isTRUE(debug) && identical(Sys.getenv("DDESONN_DEBUG"), "1")  
  aggregate <- match.arg(aggregate)
  type <- match.arg(type)
  
  X <- .as_numeric_matrix(new_data)
  mode <- attr(model, "classification_mode") %||% "binary"
  
  preds <- lapply(seq_along(model$ensemble), function(i) {
    net <- model$ensemble[[i]]
    
    # ---- choose predict activations (no extra helpers) ----
    acts_raw <- net$activation_functions_predict %||%
      attr(model, "activation_functions_predict") %||%
      net$activation_functions %||%
      attr(model, "activation_functions")
    
    # infer L from weights
    L <- length(net$weights %||% list())
    if (!is.numeric(L) || L < 1L) stop("ddesonn_predict: cannot infer number of layers.", call. = FALSE)
    
    # normalize to a length-L list of callables (or explicit NULLs)
    acts_norm <- .ddesonn_normalize_activations(acts_raw, L)
    
    res <- net$predict(  
      Rdata = X,
      weights = net$weights,
      biases  = net$biases,
      activation_functions_predict = acts_norm,
      verbose = verbose,  
      debug   = debug  
    )
    
    out <- res$predicted_output %||% res$prediction %||% res
    .as_numeric_matrix(out)
  })
  
  # aggregate
  aggregated <-
    switch(aggregate,
           mean   = Reduce(`+`, preds) / length(preds),
           median = apply(array(unlist(preds), dim = c(nrow(preds[[1]]), ncol(preds[[1]]), length(preds))),
                          c(1, 2), stats::median),
           none   = preds[[1]]
    )
  
  output <- list(prediction = aggregated, per_model = if (aggregate == "none") preds else preds)
  
  if (type == "class") {
    if (!mode %in% c("binary", "multiclass")) {
      stop("Class predictions are only available for classification modes.", call. = FALSE)
    }
    thr_used <- threshold %||%
      attr(model, "chosen_threshold") %||% model$chosen_threshold %||%
      attr(model, "threshold") %||%
      .ddesonn_threshold_default(mode)
    
    if (mode == "binary") {
      output$class <- ifelse(aggregated >= thr_used, 1L, 0L)
      output$chosen_threshold <- thr_used
    } else {
      output$class <- max.col(aggregated, ties.method = "first")
    }
  }
  
  output
}


# new helper – safe fusion writer
.write_fused_consensus <- function(result, run_dir, ts, seeds,
                                   methods = c("avg","wavg","vote_soft","vote_hard"),
                                   weight_column = c("tuned_f1","f1","accuracy")) {
  # only for ensembles
  cfg <- result$configuration %||% list()
  if (!isTRUE(cfg$do_ensemble)) return(invisible(NULL))
  
  s_chr <- as.character(length(seeds))
  agg_file <- file.path(run_dir, sprintf("agg_predictions_test__%s_seeds_%s.rds", s_chr, ts))
  fused_dir <- file.path(run_dir, "fused")
  dir.create(fused_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (!file.exists(agg_file)) {
    # nothing to fuse (shouldn’t happen because we write agg first)
    saveRDS(data.frame(), file.path(fused_dir, sprintf("Fused_EMPTY__%s_seeds_%s.rds", s_chr, ts)))
    return(invisible(NULL))
  }
  
  df <- readRDS(agg_file)
  has_ytrue <- ("y_true" %in% names(df)) && any(is.finite(suppressWarnings(as.numeric(df$y_true))))
  
  # Try legacy fuse (best case: writes metrics too)
  can_legacy <- exists("DDESONN_fuse_from_agg", mode = "function")
  for (i in seq_along(result$runs)) {
    seed_i <- result$runs[[i]]$seed %||% i
    out_base <- sprintf("run%03d_seed%s_%s", i, seed_i, ts)
    
    if (can_legacy) {
      # Use legacy; pass y_true only if it's actually present and numeric
      y_true <- NULL
      if (has_ytrue) {
        # filter only this run/seed test rows and take the longest contiguous vector
        di <- subset(df, (run_index == i | RUN_INDEX == i) & (seed == seed_i | SEED == seed_i))
        y_try <- suppressWarnings(as.numeric(di$y_true))
        if (length(y_try) && any(is.finite(y_try))) y_true <- y_try
      }
      
      fuse_res <- try(DDESONN_fuse_from_agg(
        AGG_PREDICTIONS_FILE = agg_file,
        RUN_INDEX = i,
        SEED = seed_i,
        y_true = y_true,                               # may be NULL; function will error if missing -> fallback below
        methods = methods,
        weight_column = weight_column,
        use_tuned_threshold_for_vote = TRUE,
        default_threshold = 0.5,
        vote_quorum = NULL,
        classification_mode = cfg$classification_mode %||% "binary"
      ), silent = TRUE)
      
      if (!inherits(fuse_res, "try-error")) {
        # Write what legacy gives us
        if (!is.null(fuse_res$metrics)) {
          saveRDS(fuse_res$metrics, file.path(fused_dir, sprintf("Fused_Metrics__%s.rds", out_base)))
        }
        if (is.list(fuse_res$predictions) && length(fuse_res$predictions)) {
          # one file per method (e.g., Ensemble_avg, Ensemble_wavg, ...)
          for (nm in names(fuse_res$predictions)) {
            saveRDS(fuse_res$predictions[[nm]],
                    file.path(fused_dir, sprintf("Fused_%s__%s.rds", nm, out_base)))
          }
        }
        next
      }
      # fallthrough to simple avg on error (e.g., no y_true present)
    }
    
    # Minimal guaranteed output: AVG of per-slot probabilities (no metrics)
    # helper: choose the first existing column name from a set
    .pick_col <- function(d, candidates) {
      hit <- intersect(candidates, names(d))
      if (length(hit) == 0L) {
        stop(sprintf("None of the expected columns found: [%s]. Have: [%s]",
                     paste(candidates, collapse = ", "),
                     paste(names(d), collapse = ", ")), call. = FALSE)
      }
      hit[[1L]]
    }
    
    # BEFORE (causes NSE error if RUN_INDEX/SEED don't exist in df)
    # di <- subset(df, (run_index == i | RUN_INDEX == i) & (seed == seed_i | SEED == seed_i))
    
    # AFTER (NSE-free, case-tolerant)
    ri_col <- .pick_col(df, c("run_index", "RUN_INDEX"))
    sd_col <- .pick_col(df, c("seed", "SEED"))
    
    di <- df[df[[ri_col]] == i & df[[sd_col]] == seed_i, , drop = FALSE]
    
    # require expected columns
    slot_col <- if ("model_slot" %in% names(di)) "model_slot" else if ("MODEL_SLOT" %in% names(di)) "MODEL_SLOT" else NA_character_
    if (is.na(slot_col) || !("y_pred" %in% names(di))) next
    
    # build wide by obs: rowMeans(y_pred by slot)
    # ensure stable order by obs then slot
    di <- di[order(di$obs, di[[slot_col]]), , drop = FALSE]
    # pivot by obs: average across slots
    # (robust way without tidyr)
    obs_vals <- sort(unique(di$obs))
    y_fused <- vapply(obs_vals, function(o) {
      mean(as.numeric(di$y_pred[di$obs == o]), na.rm = TRUE)
    }, numeric(1))
    
    fused_df <- data.frame(obs = obs_vals, y_fused_avg = as.numeric(y_fused))
    saveRDS(fused_df, file.path(fused_dir, sprintf("Fused_Ensemble_avg__%s.rds", out_base)))
  }
  
  invisible(NULL)
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


# replace the old helper with this
.make_dirs_legacy <- function(base, do_ensemble = FALSE) {
  dirs <- c(
    file.path(base, "models", "main"),
    file.path(base, "logs")
  )
  if (isTRUE(do_ensemble)) {
    dirs <- c(dirs, file.path(base, "fused"))
  }
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
  md$best_train_loss <- .take1num(md$best_train_loss)
  md$best_epoch_train_loss <- .int(md$best_epoch_train_loss %||% NA_integer_)
  md$best_val_acc <- .take1num(md$best_val_acc)
  md$best_val_epoch <- .int(md$best_val_epoch %||% NA_integer_)
  md$best_val_prediction_time <- .take1num(md$best_val_prediction_time)
  
  md$predictor <- slot_obj
  md$predictor_fn <- function(X, ...) slot_obj$predict(X, ...)
  md
}

.build_metrics_row <- function(md, run_index, seed, slot, split = "test") {
  # ---------- tiny scalars ----------
  .scalar1 <- function(v) {
    if (is.null(v) || length(v) == 0) return(NA)
    if (is.list(v)) v <- unlist(v, use.names = FALSE, recursive = TRUE)
    if (!length(v)) return(NA)
    v <- v[[1]]
    vn <- suppressWarnings(as.numeric(v))
    if (!is.na(vn)) return(vn)
    as.character(v)
  }
  .num1 <- function(v) { vn <- suppressWarnings(as.numeric(.scalar1(v))); if (is.na(vn)) NA_real_ else vn }
  .int1 <- function(v) { vi <- suppressWarnings(as.integer(.scalar1(v))); if (is.na(vi)) NA_integer_ else vi }
  
  # ---------- metrics from CM (validation-side only) ----------
  .cm_to_metrics <- function(TP, FP, TN, FN) {
    vals <- suppressWarnings(as.numeric(c(TP, FP, TN, FN)))
    if (length(vals) < 4 || any(is.na(vals))) {
      return(list(accuracy=NA_real_, precision=NA_real_, recall=NA_real_, f1=NA_real_))
    }
    TP <- vals[1]; FP <- vals[2]; TN <- vals[3]; FN <- vals[4]
    N <- TP + FP + TN + FN
    if (is.na(N) || N <= 0) {
      return(list(accuracy=NA_real_, precision=NA_real_, recall=NA_real_, f1=NA_real_))
    }
    acc  <- (TP + TN) / N
    prec <- if ((TP + FP) > 0) TP / (TP + FP) else NA_real_
    rec  <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
    f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) (2 * prec * rec) / (prec + rec) else NA_real_
    list(accuracy=acc, precision=prec, recall=rec, f1=f1)
  }
  
  # ---------- flatten helper (length-1 only) ----------
  .flatten1 <- function(x, prefix = NULL) {
    out <- list()
    if (is.null(x)) return(out)
    if (is.list(x)) {
      flat <- tryCatch(rapply(x, f=function(z) z, how="unlist"), error=function(e) NULL)
      if (is.null(flat)) return(out)
      L <- as.list(flat)
    } else {
      L <- as.list(x)
    }
    keep <- vapply(L, is.atomic, logical(1)) & (lengths(L) == 1L)
    L <- L[keep]
    if (!length(L)) return(out)
    nm <- names(L); if (is.null(nm)) nm <- rep("", length(L))
    for (i in seq_along(L)) {
      key <- nm[i]; if (!nzchar(key)) next
      if (!is.null(prefix)) key <- paste0(prefix, ".", key)
      v  <- L[[i]]
      vn <- suppressWarnings(as.numeric(v))
      out[[key]] <- if (!is.na(vn)) vn else as.character(v)
    }
    out
  }
  
  # ---------- tolerant VALIDATION readers ----------
  .get_val_metrics <- function(md) {
    # 1) tuned: nested performance_metric
    pm <- tryCatch(md$accuracy_precision_recall_f1_tuned$performance_metric, error=function(e) NULL)
    if (is.list(pm) && length(pm)) return(pm)
    # 1b) tuned: flat fields
    tflat <- tryCatch(md$accuracy_precision_recall_f1_tuned, error=function(e) NULL)
    if (is.list(tflat) && length(tflat)) {
      cand <- list(
        accuracy  = tflat$accuracy,
        precision = tflat$precision,
        recall    = tflat$recall,
        f1        = tflat$f1,
        f1_score  = tflat$f1_score
      )
      if (any(!vapply(cand, function(z) is.null(z) || (is.atomic(z) && length(z)==1L), logical(1)))) cand <- cand
      if (length(Filter(Negate(is.null), cand))) return(cand)
    }
    # 2) validation_metrics: nested performance_metric
    pm2 <- tryCatch(md$validation_metrics$performance_metric, error=function(e) NULL)
    if (is.list(pm2) && length(pm2)) return(pm2)
    # 2b) validation_metrics: flat fields
    vflat <- tryCatch(md$validation_metrics, error=function(e) NULL)
    if (is.list(vflat) && length(vflat)) {
      cand <- list(
        accuracy  = vflat$accuracy,
        precision = vflat$precision,
        recall    = vflat$recall,
        f1        = vflat$f1,
        f1_score  = vflat$f1_score
      )
      if (length(Filter(Negate(is.null), cand))) return(cand)
    }
    list()
  }
  .get_val_cm <- function(md) {
    # tuned nested CM
    cm <- tryCatch(md$accuracy_precision_recall_f1_tuned$confusion_matrix, error=function(e) NULL)
    if (is.list(cm) && length(cm)) return(cm)
    # tuned flat CM
    tflat <- tryCatch(md$accuracy_precision_recall_f1_tuned, error=function(e) NULL)
    if (is.list(tflat) && length(tflat)) {
      cand <- list(TP = tflat$TP, FP = tflat$FP, TN = tflat$TN, FN = tflat$FN)
      if (length(Filter(Negate(is.null), cand))) return(cand)
    }
    # validation_metrics nested CM
    cm2 <- tryCatch(md$validation_metrics$confusion_matrix, error=function(e) NULL)
    if (is.list(cm2) && length(cm2)) return(cm2)
    # validation_metrics flat CM
    vflat <- tryCatch(md$validation_metrics, error=function(e) NULL)
    if (is.list(vflat) && length(vflat)) {
      cand <- list(TP = vflat$TP, FP = vflat$FP, TN = vflat$TN, FN = vflat$FN)
      if (length(Filter(Negate(is.null), cand))) return(cand)
    }
    list()
  }
  
  # ---------- collect all (for visibility only; not used to fill primary cells) ----------
  bags <- list()
  bags <- c(bags, list(.flatten1(md$performance_metric, "performance_metric")))
  bags <- c(bags, list(.flatten1(md$relevance_metric,   "relevance_metric")))
  bags <- c(bags, list(.flatten1(tryCatch(md$performance_relevance_data$performance_metric, error=function(e) NULL),
                                 "performance_metric")))
  bags <- c(bags, list(.flatten1(tryCatch(md$metrics$performance_metric, error=function(e) NULL),
                                 "performance_metric")))
  if (!is.null(md$accuracy_precision_recall_f1_tuned)) {
    bags <- c(bags, list(.flatten1(md$accuracy_precision_recall_f1_tuned,
                                   "accuracy_precision_recall_f1_tuned")))
    cm_tuned <- tryCatch(md$accuracy_precision_recall_f1_tuned$confusion_matrix, error=function(e) NULL)
    if (is.list(cm_tuned) && length(cm_tuned)) {
      bags <- c(bags, list(.flatten1(cm_tuned, "accuracy_precision_recall_f1_tuned.confusion_matrix")))
    }
  }
  if (!is.null(md$confusion_matrix)) {
    bags <- c(bags, list(.flatten1(md$confusion_matrix, "confusion_matrix")))
  }
  flat_all <- Reduce(function(a, b) { a[names(b)] <- b; a }, bags, init = list())
  
  # ---------- base row ----------
  row <- list(
    run_index = as.integer(run_index),
    seed = as.integer(seed),
    MODEL_SLOT = as.integer(slot),
    model_slot = as.integer(slot),
    split = as.character(split),
    serial = as.character(md$model_serial_num %||% NA_character_),
    model_name = as.character(md$model_name %||% paste0("model_", slot)),
    best_train_acc           = .num1(md$best_train_acc),
    best_epoch_train         = .int1(md$best_epoch_train),
    best_train_loss          = .num1(md$best_train_loss),
    best_epoch_train_loss    = .int1(md$best_epoch_train_loss),
    best_val_acc             = .num1(md$best_val_acc),
    best_val_epoch           = .int1(md$best_val_epoch),
    best_val_prediction_time = .num1(md$best_val_prediction_time)
  )
  
  # Detect mode if present
  mode_md <- tryCatch(as.character(md$classification_mode), error = function(e) NA_character_)
  mode_md <- if (length(mode_md) && nzchar(mode_md)) tolower(mode_md) else NA_character_
  
  if (identical(mode_md, "regression")) {
    # Prefer VALIDATION metrics if available, else TRAIN performance_metric
    vm <- tryCatch(md$validation_metrics, error=function(e) NULL)
    pm <- tryCatch(md$performance_metric, error=function(e) NULL)
    src <- if (is.list(vm) && length(vm)) vm else pm
    
    row$MSE  <- .num1(src$MSE)
    row$RMSE <- .num1(src$RMSE)
    row$MAE  <- .num1(src$MAE)
    row$R2   <- .num1(src$R2)
    
    # wipe classification scalars so they don't show confusing NA columns in sorted blocks
    row$accuracy  <- NA_real_
    row$precision <- NA_real_
    row$recall    <- NA_real_
    row$f1        <- NA_real_
    row$f1_score  <- NA_real_
    row[["confusion_matrix.TP"]] <- NA_real_
    row[["confusion_matrix.FP"]] <- NA_real_
    row[["confusion_matrix.TN"]] <- NA_real_
    row[["confusion_matrix.FN"]] <- NA_real_
  } else {
    # existing classification path remains as-is
    vp <- .get_val_metrics(md)
    row$accuracy  <- .num1(vp$accuracy)
    row$precision <- .num1(vp$precision)
    row$recall    <- .num1(vp$recall)
    row$f1        <- .num1(vp$f1)
    row$f1_score  <- if (!is.na(.num1(vp$f1_score))) .num1(vp$f1_score) else .num1(vp$f1)
    
    vcm <- .get_val_cm(md)
    row[["confusion_matrix.TP"]] <- .num1(vcm$TP)
    row[["confusion_matrix.FP"]] <- .num1(vcm$FP)
    row[["confusion_matrix.TN"]] <- .num1(vcm$TN)
    row[["confusion_matrix.FN"]] <- .num1(vcm$FN)
  }
  
  
  # keep all flattened fields visible
  if (length(flat_all)) for (nm in names(flat_all)) row[[nm]] <- .scalar1(flat_all[[nm]])
  
  # ---------- main scalars: from VALIDATION (tuned > validation_metrics) ----------
  vp <- .get_val_metrics(md)
  row$accuracy  <- .num1(vp$accuracy)
  row$precision <- .num1(vp$precision)
  row$recall    <- .num1(vp$recall)
  row$f1        <- .num1(vp$f1)
  row$f1_score  <- if (!is.na(.num1(vp$f1_score))) .num1(vp$f1_score) else .num1(vp$f1)
  
  # ---------- CM: from VALIDATION (tuned > validation_metrics) ----------
  vcm <- .get_val_cm(md)
  row[["confusion_matrix.TP"]] <- .num1(vcm$TP)
  row[["confusion_matrix.FP"]] <- .num1(vcm$FP)
  row[["confusion_matrix.TN"]] <- .num1(vcm$TN)
  row[["confusion_matrix.FN"]] <- .num1(vcm$FN)
  
  # if any scalar metrics are still NA but we have validation CM, derive them
  if (any(is.na(c(row$accuracy, row$precision, row$recall, row$f1)))) {
    mets <- .cm_to_metrics(row[["confusion_matrix.TP"]],
                           row[["confusion_matrix.FP"]],
                           row[["confusion_matrix.TN"]],
                           row[["confusion_matrix.FN"]])
    if (is.na(row$accuracy))  row$accuracy  <- mets$accuracy
    if (is.na(row$precision)) row$precision <- mets$precision
    if (is.na(row$recall))    row$recall    <- mets$recall
    if (is.na(row$f1))        row$f1        <- mets$f1
  }
  if (is.na(row$f1_score)) row$f1_score <- row$f1
  
  # ---------- final scalar sweep ----------
  for (nm in names(row)) row[[nm]] <- .scalar1(row[[nm]])
  
  as.data.frame(row, check.names = TRUE, stringsAsFactors = FALSE)
}

.metrics_from_labels_probs <- function(y_true, p_hat, threshold = 0.5) {
  y_true <- as.integer(y_true)
  p_hat  <- as.numeric(p_hat)
  if (!length(y_true) || !length(p_hat)) {
    return(list(
      performance_metric = list(accuracy = NA_real_, precision = NA_real_, recall = NA_real_, f1 = NA_real_, f1_score = NA_real_),
      confusion_matrix   = list(TP = NA_real_, FP = NA_real_, TN = NA_real_, FN = NA_real_)
    ))
  }
  y_pred <- as.integer(p_hat >= threshold)
  
  TP <- sum(y_pred == 1L & y_true == 1L, na.rm = TRUE)
  FP <- sum(y_pred == 1L & y_true == 0L, na.rm = TRUE)
  TN <- sum(y_pred == 0L & y_true == 0L, na.rm = TRUE)
  FN <- sum(y_pred == 0L & y_true == 1L, na.rm = TRUE)
  N  <- TP + FP + TN + FN
  
  acc  <- if (N > 0) (TP + TN)/N else NA_real_
  prec <- if ((TP + FP) > 0) TP/(TP + FP) else NA_real_
  rec  <- if ((TP + FN) > 0) TP/(TP + FN) else NA_real_
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) 2*prec*rec/(prec + rec) else NA_real_
  
  list(
    performance_metric = list(accuracy = acc, precision = prec, recall = rec, f1 = f1, f1_score = f1),
    confusion_matrix   = list(TP = TP, FP = FP, TN = TN, FN = FN)
  )
}

.write_single_runs_metrics <- function(result, run_dir, ts, seeds) {
  s_chr <- as.character(length(seeds))
  
  ## existing outputs (legacy)
  test_path  <- file.path(run_dir, sprintf("SingleRun_Test_Metrics_%s_seeds_%s.rds",        s_chr, ts))
  train_path <- file.path(run_dir, sprintf("SingleRun_Train_Acc_Val_Metrics_%s_seeds_%s.rds", s_chr, ts))
  
  rows_train <- list()
  rows_val   <- list()
  rows_test  <- list()
  ptr_tr <- 0L
  ptr_va <- 0L
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
      md$model_name       <- md$model_name       %||% paste0("model_", k)
      
      ## TRAIN row
      ptr_tr <- ptr_tr + 1L
      rows_train[[ptr_tr]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "train")
      
      ## VALIDATION row
      ptr_va <- ptr_va + 1L
      rows_val[[ptr_va]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "validation")
      
      ## TEST row
      ptr_te <- ptr_te + 1L
      rows_test[[ptr_te]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "test")
    }
  }
  
  bind <- function(lst) if (!length(lst)) data.frame() else do.call(rbind, lst)
  df_train <- bind(rows_train)
  df_val   <- bind(rows_val)
  df_test  <- bind(rows_test)
  
  id_order <- c("run_index", "seed", "model_slot", "MODEL_SLOT", "split", "serial", "model_name")
  metric_pref <- c(
    # classification-first
    "accuracy", "precision", "recall", "f1", "f1_score",
    # regression
    "MSE", "MAE", "RMSE", "R2",
    # CM + best_* as you already have
    "confusion_matrix.TP", "confusion_matrix.FP", "confusion_matrix.TN", "confusion_matrix.FN",
    "best_train_acc", "best_epoch_train", "best_train_loss", "best_epoch_train_loss",
    "best_val_acc", "best_val_epoch", "best_val_prediction_time"
  )
  
  ord <- function(df) c(
    intersect(id_order, names(df)),
    intersect(metric_pref, names(df)),
    setdiff(names(df), c(id_order, metric_pref))
  )
  
  if (ncol(df_train)) df_train <- df_train[, ord(df_train), drop = FALSE]
  if (ncol(df_val))   df_val   <- df_val[,   ord(df_val),   drop = FALSE]
  if (ncol(df_test))  df_test  <- df_test[,  ord(df_test),  drop = FALSE]
  
  ## writes (legacy only)
  saveRDS(df_test,  test_path)   # TEST metrics
  saveRDS(df_train, train_path)  # TRAIN/VAL aggregate as before
}

.write_ensemble_runs_metrics <- function(result, run_dir, ts, seeds) {
  s_chr <- as.character(length(seeds))
  
  ## existing outputs (legacy)
  test_path  <- file.path(run_dir, sprintf("Ensemble_Test_Metrics_%s_seeds_%s.rds",        s_chr, ts))
  train_path <- file.path(run_dir, sprintf("Ensemble_Train_Acc_Val_Metrics_%s_seeds_%s.rds", s_chr, ts))
  
  rows_train <- list()
  rows_val   <- list()
  rows_test  <- list()
  ptr_tr <- 0L
  ptr_va <- 0L
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
      md$model_serial_num <- md$model_serial_num %||% sprintf("1.main.%d", k)
      md$model_name       <- md$model_name       %||% paste0("model_", k)
      
      ## TRAIN
      ptr_tr <- ptr_tr + 1L
      rows_train[[ptr_tr]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "train")
      
      ## VALIDATION
      ptr_va <- ptr_va + 1L
      rows_val[[ptr_va]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "validation")
      
      ## TEST
      ptr_te <- ptr_te + 1L
      rows_test[[ptr_te]] <- .build_metrics_row(md, run_index = i, seed = seed_i, slot = k, split = "test")
    }
  }
  
  bind <- function(lst) if (!length(lst)) data.frame() else do.call(rbind, lst)
  df_train <- bind(rows_train)
  df_val   <- bind(rows_val)
  df_test  <- bind(rows_test)
  
  id_order <- c("run_index", "seed", "model_slot", "MODEL_SLOT", "split", "serial", "model_name")
  metric_pref <- c(
    "accuracy", "precision", "recall", "f1", "f1_score", "auc", "balanced_accuracy",
    "specificity", "sensitivity", "logloss", "brier",
    "MSE", "MAE", "RMSE", "R2", "MAPE", "SMAPE", "WMAPE", "MASE",
    "confusion_matrix.TP", "confusion_matrix.FP", "confusion_matrix.TN", "confusion_matrix.FN",
    "generalization_ability", "speed", "speed_learn1", "speed_learn2",
    "memory_usage", "robustness", "hit_rate", "ndcg", "diversity", "serendipity",
    "best_train_acc", "best_epoch_train", "best_train_loss", "best_epoch_train_loss",
    "best_val_acc", "best_val_epoch", "best_val_prediction_time"
  )
  
  ord <- function(df) c(
    intersect(id_order, names(df)),
    intersect(metric_pref, names(df)),
    setdiff(names(df), c(id_order, metric_pref))
  )
  
  if (ncol(df_train)) df_train <- df_train[, ord(df_train), drop = FALSE]
  if (ncol(df_val))   df_val   <- df_val[,   ord(df_val),   drop = FALSE]
  if (ncol(df_test))  df_test  <- df_test[,  ord(df_test),  drop = FALSE]
  
  ## writes (legacy only)
  saveRDS(df_test,  test_path)   # TEST metrics
  saveRDS(df_train, train_path)  # TRAIN/VAL aggregate as before
}

.build_single_pretty_tables <- function(
    run_dir,
    ts,
    seeds,
    CLASSIFICATION_MODE = NULL,
    model_slot = 1L
) {
  .log <- function(...) if (isTRUE(getOption("DDESONN.verbose", FALSE))) cat("[BuildPretty] ", paste0(..., collapse = ""), "\n")
  
  .log("------------------------------------------------------------")
  .log("ENTER .build_single_pretty_tables")
  .log("run_dir = ", run_dir)
  .log("ts      = ", ts)
  .log("seeds   = ", paste(seeds, collapse = ","))
  .log("CLASSIFICATION_MODE (in) = ", CLASSIFICATION_MODE)
  .log("model_slot = ", model_slot)
  .log("------------------------------------------------------------")
  
  # derive n_seeds and s_chr for filenames
  n_seeds <- if (!missing(seeds) && length(seeds)) length(seeds) else 1L
  s_chr   <- as.character(n_seeds)
  
  # resolve CLASSIFICATION_MODE
  if (is.null(CLASSIFICATION_MODE)) {
    CLASSIFICATION_MODE <- get0("CLASSIFICATION_MODE",
                                ifnotfound = "regression",
                                inherits   = TRUE)
    .log("CLASSIFICATION_MODE (resolved from global) = ", CLASSIFICATION_MODE)
  } else {
    .log("CLASSIFICATION_MODE (explicit) = ", CLASSIFICATION_MODE)
  }
  CLASSIFICATION_MODE <- tolower(CLASSIFICATION_MODE)
  
  # expected pretty files
  agg_pred_file_test <- file.path(
    run_dir,
    sprintf("SingleRun_Pretty_Test_Metrics_%s_seeds_%s.rds", s_chr, ts)
  )
  agg_pred_file_train <- file.path(
    run_dir,
    sprintf("SingleRun_Pretty_Train_Metrics_%s_seeds_%s.rds", s_chr, ts)
  )
  agg_pred_file_val <- file.path(
    run_dir,
    sprintf("SingleRun_Pretty_Validation_Metrics_%s_seeds_%s.rds", s_chr, ts)
  )
  
  .log("Expected TEST pretty file:  ", agg_pred_file_test)
  .log("Expected TRAIN pretty file: ", agg_pred_file_train)
  .log("Expected VAL pretty file:   ", agg_pred_file_val)
  
  predictions_main_file <- file.path(run_dir, "predictions_main.rds")
  if (!file.exists(predictions_main_file)) {
    stop("[BuildPretty] predictions_main.rds not present; cannot build pretty tables without stored per-observation predictions.")
  }
  
  .log("Found predictions_main.rds at ", predictions_main_file)
  pred_obj <- try(readRDS(predictions_main_file), silent = TRUE)
  if (inherits(pred_obj, "try-error")) {
    err <- attr(pred_obj, "condition")
    msg <- if (inherits(err, "condition")) conditionMessage(err) else as.character(pred_obj)
    stop("[BuildPretty] Unable to read predictions_main.rds: ", msg)
  }
  if (!is.list(pred_obj)) {
    stop("[BuildPretty] predictions_main.rds did not contain a list structure.")
  }
  
  per_seed_predictions <- pred_obj$per_seed_tables %||% pred_obj$per_seed %||% pred_obj$tables
  if (!length(per_seed_predictions)) {
    stop("[BuildPretty] predictions_main.rds does not contain per-seed per-split tables.")
  }
  
  per_seed_predictions <- lapply(per_seed_predictions, function(seed_entry) {
    if (!is.list(seed_entry)) return(NULL)
    valid <- vapply(seed_entry, function(df) is.data.frame(df) && NROW(df), logical(1))
    if (!any(valid)) return(NULL)
    seed_entry[valid]
  })
  per_seed_predictions <- Filter(Negate(is.null), per_seed_predictions)
  if (!length(per_seed_predictions)) {
    stop("[BuildPretty] No usable per-observation predictions found in predictions_main.rds.")
  }
  
  .gather_split_df <- function(split_label) {
    if (is.null(per_seed_predictions) || !length(per_seed_predictions)) return(NULL)
    rows <- list()
    ptr <- 0L
    for (i in seq_along(per_seed_predictions)) {
      run_pred <- per_seed_predictions[[i]]
      if (!is.list(run_pred)) next
      df <- run_pred[[split_label]]
      if (!is.data.frame(df) || !NROW(df)) next
      rows[[ptr <- ptr + 1L]] <- df
    }
    if (!length(rows)) return(NULL)
    cols <- unique(unlist(lapply(rows, names)))
    rows <- lapply(rows, function(df) {
      miss <- setdiff(cols, names(df))
      for (nm in miss) df[[nm]] <- NA
      df[, cols, drop = FALSE]
    })
    out <- try(do.call(rbind, rows), silent = TRUE)
    if (inherits(out, "try-error")) return(NULL)
    rownames(out) <- NULL
    out
  }
  
  .df_info <- function(d, label) {
    if (!is.data.frame(d)) {
      .log("    [", label, "] not a data.frame")
      return()
    }
    .log("    [", label, "] nrow = ", NROW(d), ", ncol = ", NCOL(d))
    .log("    [", label, "] names = ", paste(names(d), collapse = ", "))
  }
  
  # Normalizer shared by all splits
  .normalize_and_rewrite <- function(path, split_label, df) {
    up <- toupper(split_label)
    
    if (!is.data.frame(df)) {
      stop("[BuildPretty] ", up, " supplied object is not a data.frame.")
    }
    
    if (!NROW(df)) {
      .log("  !! ", up, " df has zero rows; writing back as-is.")
      saveRDS(df, path)
      return(invisible(TRUE))
    }
    
    .df_info(df, paste0(up, "_raw"))
    
    # run_index / RUN_INDEX
    if (!"run_index" %in% names(df) && "RUN_INDEX" %in% names(df)) {
      df$run_index <- suppressWarnings(as.integer(df$RUN_INDEX))
    }
    if (!"RUN_INDEX" %in% names(df) && "run_index" %in% names(df)) {
      df$RUN_INDEX <- suppressWarnings(as.integer(df$run_index))
    }
    
    # seed / SEED
    if (!"seed" %in% names(df) && "SEED" %in% names(df)) {
      df$seed <- suppressWarnings(as.integer(df$SEED))
    }
    if (!"SEED" %in% names(df) && "seed" %in% names(df)) {
      df$SEED <- suppressWarnings(as.integer(df$seed))
    }
    
    # model_slot / MODEL_SLOT
    if (!"model_slot" %in% names(df) && "MODEL_SLOT" %in% names(df)) {
      df$model_slot <- suppressWarnings(as.integer(df$MODEL_SLOT))
    }
    if (!"MODEL_SLOT" %in% names(df) && "model_slot" %in% names(df)) {
      df$MODEL_SLOT <- suppressWarnings(as.integer(df$model_slot))
    }
    
    # split / SPLIT
    if (!"split" %in% names(df)) {
      if ("SPLIT" %in% names(df)) {
        df$split <- tolower(as.character(df$SPLIT))
      } else {
        df$split <- tolower(split_label)
      }
    } else {
      df$split <- tolower(as.character(df$split))
    }
    df$SPLIT <- toupper(df$split)
    
    # ensure .__split__ for backwards compatibility
    if (!".__split__" %in% names(df)) {
      df$.__split__ <- df$split
    }
    
    # CLASSIFICATION_MODE
    if (!"CLASSIFICATION_MODE" %in% names(df)) {
      df$CLASSIFICATION_MODE <- toupper(CLASSIFICATION_MODE)
    } else {
      df$CLASSIFICATION_MODE <- toupper(as.character(df$CLASSIFICATION_MODE))
    }
    
    # y_prob / y_pred / y_true numeric (placeholder: keep NA if missing)
    if (!"y_prob" %in% names(df) && "y_pred" %in% names(df)) {
      df$y_prob <- suppressWarnings(as.numeric(df$y_pred))
    }
    if ("y_prob" %in% names(df)) {
      df$y_prob <- suppressWarnings(as.numeric(df$y_prob))
    }
    if ("y_true" %in% names(df)) {
      df$y_true <- suppressWarnings(as.numeric(df$y_true))
    }
    if (!"y_pred" %in% names(df) && "y_prob" %in% names(df)) {
      df$y_pred <- df$y_prob
    } else if ("y_pred" %in% names(df)) {
      df$y_pred <- suppressWarnings(as.numeric(df$y_pred))
    } else {
      df$y_pred <- NA_real_
    }
    
    # obs_index (if missing, create simple 1:n)
    if (!"obs_index" %in% names(df)) {
      df$obs_index <- seq_len(NROW(df))
    } else {
      df$obs_index <- suppressWarnings(as.integer(df$obs_index))
      na_obs <- is.na(df$obs_index)
      if (any(na_obs)) df$obs_index[na_obs] <- seq_len(sum(na_obs))
    }
    
    .df_info(df, paste0(up, "_final"))
    .log("  -> Writing ", up, " pretty file to: ", path)
    saveRDS(df, path)
    invisible(TRUE)
  }
  
  # make sure base dir exists
  if (!dir.exists(run_dir)) dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  
  write_split <- function(split_label, target_path) {
    up <- toupper(split_label)
    df <- .gather_split_df(split_label)
    if (!is.data.frame(df) || !NROW(df)) {
      stop("[BuildPretty] No stored predictions available for ", up, ".")
    }
    .log("  .. Using stored per-observation predictions for ", up, ".")
    .normalize_and_rewrite(target_path, split_label, df)
  }
  
  write_split("test", agg_pred_file_test)
  write_split("train", agg_pred_file_train)
  write_split("validation", agg_pred_file_val)
  
  .log("EXIT .build_single_pretty_tables")
  .log("------------------------------------------------------------")
  
  invisible(list(
    run_dir = run_dir,
    ts      = ts,
    n_seeds = n_seeds,
    CLASSIFICATION_MODE = CLASSIFICATION_MODE,
    files   = list(
      test  = agg_pred_file_test,
      train = agg_pred_file_train,
      val   = agg_pred_file_val
    )
  ))
}











.build_ensemble_run_pretty_metrics_rds <- function(run_dir,
                                                   ts,
                                                   seeds,
                                                   CLASSIFICATION_MODE = NULL) {
  ## number of seeds for filename stamp
  n_seeds <- if (!missing(seeds) && length(seeds)) length(seeds) else 1L
  s_chr   <- as.character(n_seeds)
  
  ## classification mode fallback (match train code)
  if (is.null(CLASSIFICATION_MODE)) {
    CLASSIFICATION_MODE <- get0("CLASSIFICATION_MODE",
                                ifnotfound = "binary",
                                inherits   = TRUE)
  }
  
  ## Ensemble pretty prediction Metrics paths
  agg_pred_file_test  <- file.path(
    run_dir,
    sprintf("Ensemble_Pretty_Test_Metrics_%s_seeds_%s.rds", s_chr, ts)
  )
  agg_pred_file_train <- file.path(
    run_dir,
    sprintf("Ensemble_Pretty_Train_Metrics_%s_seeds_%s.rds", s_chr, ts)
  )
  agg_pred_file_val   <- file.path(
    run_dir,
    sprintf("Ensemble_Pretty_Validation_Metrics_%s_seeds_%s.rds", s_chr, ts)
  )
  
  ## Helper: choose a source path (metrics file if it exists, otherwise
  ## any Ensemble_Pretty_<SPLIT>_* file that matches this ts).
  .pick_source <- function(target_path, split_label) {
    if (file.exists(target_path)) {
      return(target_path)
    }
    patt <- sprintf("^Ensemble_Pretty_%s_.*%s\\.rds$", split_label, ts)
    cand <- list.files(run_dir, pattern = patt, full.names = TRUE)
    if (length(cand)) cand[[1L]] else NULL
  }
  
  ## ------------------------------
  ## TEST PREDICTIONS (Ensemble_Pretty_Test_Metrics_*.rds)
  ## ------------------------------
  src_test <- .pick_source(agg_pred_file_test, "Test")
  if (!is.null(src_test)) {
    dfp <- try(readRDS(src_test), silent = TRUE)
    if (!inherits(dfp, "try-error") && is.data.frame(dfp) && NROW(dfp)) {
      if (!"run_index" %in% names(dfp) && "RUN_INDEX" %in% names(dfp)) {
        dfp$run_index <- suppressWarnings(as.integer(dfp$RUN_INDEX))
      }
      if (!"RUN_INDEX" %in% names(dfp) && "run_index" %in% names(dfp)) {
        dfp$RUN_INDEX <- suppressWarnings(as.integer(dfp$run_index))
      }
      
      if (!"seed" %in% names(dfp) && "SEED" %in% names(dfp)) {
        dfp$seed <- suppressWarnings(as.integer(dfp$SEED))
      }
      if (!"SEED" %in% names(dfp) && "seed" %in% names(dfp)) {
        dfp$SEED <- suppressWarnings(as.integer(dfp$seed))
      }
      
      if (!"model_slot" %in% names(dfp) && "MODEL_SLOT" %in% names(dfp)) {
        dfp$model_slot <- suppressWarnings(as.integer(dfp$MODEL_SLOT))
      }
      if (!"MODEL_SLOT" %in% names(dfp) && "model_slot" %in% names(dfp)) {
        dfp$MODEL_SLOT <- suppressWarnings(as.integer(dfp$model_slot))
      }
      
      if (!"split" %in% names(dfp)) {
        if ("SPLIT" %in% names(dfp)) {
          dfp$split <- tolower(as.character(dfp$SPLIT))
        } else {
          dfp$split <- "test"
        }
      } else {
        dfp$split <- tolower(as.character(dfp$split))
      }
      dfp$SPLIT <- toupper(dfp$split)
      
      if (!"CLASSIFICATION_MODE" %in% names(dfp)) {
        dfp$CLASSIFICATION_MODE <- toupper(CLASSIFICATION_MODE)
      } else {
        dfp$CLASSIFICATION_MODE <- toupper(as.character(dfp$CLASSIFICATION_MODE))
      }
      
      if (!"y_prob" %in% names(dfp) && "y_pred" %in% names(dfp)) {
        dfp$y_prob <- suppressWarnings(as.numeric(dfp$y_pred))
      }
      if ("y_prob" %in% names(dfp)) {
        dfp$y_prob <- suppressWarnings(as.numeric(dfp$y_prob))
      }
      if ("y_true" %in% names(dfp)) {
        dfp$y_true <- suppressWarnings(as.numeric(dfp$y_true))
      }
      if (!"y_pred" %in% names(dfp) && "y_prob" %in% names(dfp)) {
        dfp$y_pred <- dfp$y_prob
      }
      
      ## Always save to the Metrics path (creates if missing)
      saveRDS(dfp, agg_pred_file_test)
    }
  }
  
  ## ------------------------------
  ## TRAIN PREDICTIONS (Ensemble_Pretty_Train_Metrics_*.rds)
  ## ------------------------------
  src_train <- .pick_source(agg_pred_file_train, "Train")
  if (!is.null(src_train)) {
    dfp_tr <- try(readRDS(src_train), silent = TRUE)
    if (!inherits(dfp_tr, "try-error") && is.data.frame(dfp_tr) && NROW(dfp_tr)) {
      if (!"run_index" %in% names(dfp_tr) && "RUN_INDEX" %in% names(dfp_tr)) {
        dfp_tr$run_index <- suppressWarnings(as.integer(dfp_tr$RUN_INDEX))
      }
      if (!"RUN_INDEX" %in% names(dfp_tr) && "run_index" %in% names(dfp_tr)) {
        dfp_tr$RUN_INDEX <- suppressWarnings(as.integer(dfp_tr$run_index))
      }
      
      if (!"seed" %in% names(dfp_tr) && "SEED" %in% names(dfp_tr)) {
        dfp_tr$seed <- suppressWarnings(as.integer(dfp_tr$SEED))
      }
      if (!"SEED" %in% names(dfp_tr) && "seed" %in% names(dfp_tr)) {
        dfp_tr$SEED <- suppressWarnings(as.integer(dfp_tr$seed))
      }
      
      if (!"model_slot" %in% names(dfp_tr) && "MODEL_SLOT" %in% names(dfp_tr)) {
        dfp_tr$model_slot <- suppressWarnings(as.integer(dfp_tr$MODEL_SLOT))
      }
      if (!"MODEL_SLOT" %in% names(dfp_tr) && "model_slot" %in% names(dfp_tr)) {
        dfp_tr$MODEL_SLOT <- suppressWarnings(as.integer(dfp_tr$model_slot))
      }
      
      if (!"split" %in% names(dfp_tr)) {
        if ("SPLIT" %in% names(dfp_tr)) {
          dfp_tr$split <- tolower(as.character(dfp_tr$SPLIT))
        } else {
          dfp_tr$split <- "train"
        }
      } else {
        dfp_tr$split <- tolower(as.character(dfp_tr$split))
      }
      dfp_tr$SPLIT <- toupper(dfp_tr$split)
      
      if (!"CLASSIFICATION_MODE" %in% names(dfp_tr)) {
        dfp_tr$CLASSIFICATION_MODE <- toupper(CLASSIFICATION_MODE)
      } else {
        dfp_tr$CLASSIFICATION_MODE <- toupper(as.character(dfp_tr$CLASSIFICATION_MODE))
      }
      
      if (!"y_prob" %in% names(dfp_tr) && "y_pred" %in% names(dfp_tr)) {
        dfp_tr$y_prob <- suppressWarnings(as.numeric(dfp_tr$y_pred))
      }
      if ("y_prob" %in% names(dfp_tr)) {
        dfp_tr$y_prob <- suppressWarnings(as.numeric(dfp_tr$y_prob))
      }
      if ("y_true" %in% names(dfp_tr)) {
        dfp_tr$y_true <- suppressWarnings(as.numeric(dfp_tr$y_true))
      }
      if (!"y_pred" %in% names(dfp_tr) && "y_prob" %in% names(dfp_tr)) {
        dfp_tr$y_pred <- dfp_tr$y_prob
      }
      
      saveRDS(dfp_tr, agg_pred_file_train)
    }
  }
  
  ## ------------------------------
  ## VALIDATION PREDICTIONS (Ensemble_Pretty_Validation_Metrics_*.rds)
  ## ------------------------------
  src_val <- .pick_source(agg_pred_file_val, "Validation")
  if (!is.null(src_val)) {
    dfp_val <- try(readRDS(src_val), silent = TRUE)
    if (!inherits(dfp_val, "try-error") && is.data.frame(dfp_val) && NROW(dfp_val)) {
      if (!"run_index" %in% names(dfp_val) && "RUN_INDEX" %in% names(dfp_val)) {
        dfp_val$run_index <- suppressWarnings(as.integer(dfp_val$RUN_INDEX))
      }
      if (!"RUN_INDEX" %in% names(dfp_val) && "run_index" %in% names(dfp_val)) {
        dfp_val$RUN_INDEX <- suppressWarnings(as.integer(dfp_val$run_index))
      }
      
      if (!"seed" %in% names(dfp_val) && "SEED" %in% names(dfp_val)) {
        dfp_val$seed <- suppressWarnings(as.integer(dfp$SEED))
      }
      if (!"SEED" %in% names(dfp_val) && "seed" %in% names(dfp_val)) {
        dfp_val$SEED <- suppressWarnings(as.integer(dfp_val$seed))
      }
      
      if (!"model_slot" %in% names(dfp_val) && "MODEL_SLOT" %in% names(dfp_val)) {
        dfp_val$model_slot <- suppressWarnings(as.integer(dfp_val$MODEL_SLOT))
      }
      if (!"MODEL_SLOT" %in% names(dfp_val) && "model_slot" %in% names(dfp_val)) {
        dfp_val$MODEL_SLOT <- suppressWarnings(as.integer(dfp_val$model_slot))
      }
      
      if (!"split" %in% names(dfp_val)) {
        if ("SPLIT" %in% names(dfp_val)) {
          dfp_val$split <- tolower(as.character(dfp_val$SPLIT))
        } else {
          dfp_val$split <- "validation"
        }
      } else {
        dfp_val$split <- tolower(as.character(dfp_val$split))
      }
      dfp_val$SPLIT <- toupper(dfp_val$split)
      
      if (!"CLASSIFICATION_MODE" %in% names(dfp_val)) {
        dfp_val$CLASSIFICATION_MODE <- toupper(CLASSIFICATION_MODE)
      } else {
        dfp_val$CLASSIFICATION_MODE <- toupper(as.character(dfp_val$CLASSIFICATION_MODE))
      }
      
      if (!"y_prob" %in% names(dfp_val) && "y_pred" %in% names(dfp_val)) {
        dfp_val$y_prob <- suppressWarnings(as.numeric(dfp_val$y_pred))
      }
      if ("y_prob" %in% names(dfp_val)) {
        dfp_val$y_prob <- suppressWarnings(as.numeric(dfp_val$y_prob))
      }
      if ("y_true" %in% names(dfp_val)) {
        dfp_val$y_true <- suppressWarnings(as.numeric(dfp_val$y_true))
      }
      if (!"y_pred" %in% names(dfp_val) && "y_prob" %in% names(dfp_val)) {
        dfp_val$y_pred <- dfp_val$y_prob
      }
      
      saveRDS(dfp_val, agg_pred_file_val)
    }
  }
}



.write_agg_predictions <- function(result, run_dir, ts, seeds) {
  s_chr <- as.character(length(seeds))
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

.write_temp_agg_predictions <- function(result, run_dir, ts, seeds) {
  `%||%` <- get0("%||%", ifnotfound = function(x, y) if (is.null(x)) y else x)
  
  X <- result$`.__prediction_matrix`
  if (is.null(X)) return(invisible(NULL))
  
  # ---------- how many temp iterations exist across runs ----------
  max_temp <- 0L
  for (i in seq_along(result$runs)) {
    ti <- result$runs[[i]]$temp_iterations
    if (!is.null(ti)) max_temp <- max(max_temp, length(ti))
  }
  if (max_temp == 0L) return(invisible(NULL))
  
  s_chr  <- as.character(length(seeds))
  log_dir <- file.path(run_dir, "logs")
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
  
  # ---------- PREDICTIONS per temp_eXX (unchanged behavior) ----------
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
      
      # find entry for this temp iteration e
      entry <- NULL
      for (t in ti) if (identical(as.integer(t$iteration), as.integer(e))) { entry <- t; break }
      if (is.null(entry) || is.null(entry$model)) next
      
      pr <- try(ddesonn_predict(
        model = entry$model, new_data = X, aggregate = "none", type = "response"
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
          seed      = rep.int(seed_i, n),
          MODEL_SLOT = rep.int(k, n),
          model_slot = rep.int(k, n),
          obs   = seq_len(n),
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
    if (ncol(out)) out <- out[, c(id_order, "y_pred", "y_true", rest), drop = FALSE]
    
    saveRDS(out, out_path)
  }
  
  # ---------- NEW: persist movement_log + change_log ----------
  # We try to use result$runs[[i]]$tables$movement_log / change_log if present,
  # otherwise we rebuild by collecting logs from each temp iteration entry.
  build_log_df <- function(run_i, type = c("movement", "change", "main")) {
    type <- match.arg(type)
    # Preferred location (mirrors Train flow)
    tbls <- run_i$tables
    if (is.list(tbls)) {
      if (type == "movement" && is.data.frame(tbls$movement_log) && NROW(tbls$movement_log)) {
        return(tbls$movement_log)
      }
      if (type == "change" && is.data.frame(tbls$change_log) && NROW(tbls$change_log)) {
        return(tbls$change_log)
      }
      if (type == "main" && is.data.frame(tbls$main_log) && NROW(tbls$main_log)) {
        return(tbls$main_log)
      }
    }
    # Fallback: gather from temp_iteration entries if they carry rows
    ti <- run_i$temp_iterations
    if (is.null(ti) || !length(ti)) return(data.frame())
    acc <- list(); p <- 0L
    for (ent in ti) {
      # allow multiple shapes (movement_log/change_log on the entry itself)
      if (type == "movement" && is.data.frame(ent$movement_log) && NROW(ent$movement_log)) {
        acc[[p <- p + 1L]] <- ent$movement_log
      }
      if (type == "change" && is.data.frame(ent$change_log) && NROW(ent$change_log)) {
        acc[[p <- p + 1L]] <- ent$change_log
      }
      if (type == "main" && is.data.frame(ent$main_log) && NROW(ent$main_log)) {
        acc[[p <- p + 1L]] <- ent$main_log
      }
    }
    if (!length(acc)) return(data.frame())
    out <- try(do.call(rbind, acc), silent = TRUE)
    if (inherits(out, "try-error")) out <- data.frame()
    out
  }
  
  # Write one pair of files per run (consistent naming with TestDDESONN.R)
  for (i in seq_along(result$runs)) {
    seed_i <- result$runs[[i]]$seed %||% i
    
    mv <- build_log_df(result$runs[[i]], "movement")
    ch <- build_log_df(result$runs[[i]], "change")
    ml <- build_log_df(result$runs[[i]], "main")
    
    # De-dup (sometimes callers accumulate)
    dedup <- function(df) {
      if (!is.data.frame(df) || !NROW(df)) return(df)
      # try best-effort unique on common columns if present
      key_cols <- intersect(
        c("iteration","phase","slot","role","serial","metric_name","metric_value","message","timestamp"),
        names(df)
      )
      if (!length(key_cols)) return(unique(df))
      df[!duplicated(df[, key_cols, drop = FALSE]), , drop = FALSE]
    }
    mv <- dedup(mv); ch <- dedup(ch)
    ml <- dedup(ml)
    
    mv_path <- file.path(log_dir, sprintf("movement_log_run%03d_seed%s_%s.rds", i, seed_i, ts))
    ch_path <- file.path(log_dir, sprintf("change_log_run%03d_seed%s_%s.rds",   i, seed_i, ts))
    ml_path <- file.path(log_dir, sprintf("main_log_run%03d_seed%s_%s.rds",     i, seed_i, ts))
    
    # Only write if we actually have rows (exactly like your train flow)
    if (is.data.frame(mv) && NROW(mv)) saveRDS(mv, mv_path)
    if (is.data.frame(ch) && NROW(ch)) saveRDS(ch, ch_path)
    if (is.data.frame(ml) && NROW(ml)) saveRDS(ml, ml_path)
  }
  
  invisible(NULL)
}

# ============================================================
# .persist_ddesonn_run (FULL FIXED — preserved)
# -  FIX: guard optional ensemble post-writes when upstream data is empty
# -  FIX: prevent .write_fused_consensus() from crashing when it would read empty df (no cols)
# - PRESERVED: all stamping, directories, model saving, metadata, metrics, logs
# ============================================================

.persist_ddesonn_run <- function(result, output_root, save_models = TRUE) {
  if (is.null(output_root) || !nzchar(output_root)) return(invisible(NULL))
  
  cfg <- result$configuration %||% list()
  
  # --- inline the stamp logic (no external helper) ---
  ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  seeds <- cfg$seeds %||% 1L
  if (!isTRUE(cfg$do_ensemble)) {
    
    # seeds may be NULL, FALSE, numeric(0), 0L, or a vector of seeds
    if (is.null(seeds) || identical(seeds, FALSE) ||
        (is.numeric(seeds) && length(seeds) == 0L) ||
        (is.numeric(seeds) && all(seeds == 0))) {
      seed_tag <- "wNoSeed"
    } else if (length(seeds) == 1L) {
      seed_tag <- "wSeed"
    } else {
      seed_tag <- "wSeeds"
    }
    
    root_dir <- "SingleRuns"
    run_tag  <- sprintf("%s__m%d__%s", ts_stamp, as.integer(cfg$num_networks %||% 1L), seed_tag)
  } else {
    root_dir <- "EnsembleRuns"
    run_tag  <- ts_stamp
  }
  
  art_root <- ddesonn_artifacts_root(output_root)
  run_dir <- file.path(art_root, root_dir, run_tag)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  
  # create base dirs AFTER run_dir exists
  .make_dirs_legacy(run_dir, do_ensemble = isTRUE(cfg$do_ensemble))
  
  ts <- ts_stamp
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
  
  ## local classification mode for pretty-builder functions
  classification_mode <- cfg$classification_mode %||% "binary"
  
  if (isTRUE(cfg$do_ensemble)) {
    .write_ensemble_runs_metrics(result, run_dir, ts, seeds)
    
    # ============================================================
    # Optional pretty tables — do not fail builds                 
    # ============================================================
    .build_ensemble_pretty_tables_fn <- get0(".build_ensemble_pretty_tables", mode = "function") 
    if (!is.null(.build_ensemble_pretty_tables_fn)) {                                           
      .build_ensemble_pretty_tables_fn(                                                         
        run_dir             = run_dir,
        ts                  = ts,
        seeds               = seeds,
        CLASSIFICATION_MODE = classification_mode
      )
    }                                                                                           
    
  } else {
    .write_single_runs_metrics(result, run_dir, ts, seeds)
    
    # ============================================================
    # Optional pretty tables — do not fail builds                 
    # ============================================================
    .build_single_pretty_tables_fn <- get0(".build_single_pretty_tables", mode = "function")    
    if (!is.null(.build_single_pretty_tables_fn)) {                                             
      .build_single_pretty_tables_fn(                                                           
        run_dir             = run_dir,
        ts                  = ts,
        seeds               = seeds,
        CLASSIFICATION_MODE = classification_mode
      )
    }                                                                                           
  }
  
  if (isTRUE(cfg$do_ensemble)) {
    .write_agg_predictions(result, run_dir, ts, seeds)
    .write_temp_agg_predictions(result, run_dir, ts, seeds)
    
    # ============================================================
    #  FIX: Only write fused consensus when inputs exist
    # - Prevents crash when upstream prediction frames are empty (0 cols)
    # - This can happen in vignette/build runs or when no per-seed tables were produced
    # ============================================================
    
    has_any_pred_cols <- FALSE                                                             
    
    # Prefer in-memory tables when available
    if (is.list(result$predictions) && is.list(result$predictions$per_seed_tables)) {       
      tbls <- result$predictions$per_seed_tables                                            
      for (t in tbls) {                                                                     
        if (is.data.frame(t) && ncol(t) > 0L) {                                             
          has_any_pred_cols <- TRUE                                                         
          break                                                                             
        }                                                                                   
      }                                                                                     
    }                                                                                       
    
    # Fallback: check the on-disk agg predictions file if your writer uses one
    if (!isTRUE(has_any_pred_cols)) {                                                       
      # try a couple of common filenames (safe no-op if none exist)
      candidates <- c(                                                                      
        file.path(run_dir, "predictions_main.rds"),                                         
        file.path(run_dir, "predictions_temp.rds")                                          
      )                                                                                     
      for (fp in candidates) {                                                              
        if (file.exists(fp)) {                                                              
          obj <- try(readRDS(fp), silent = TRUE)                                             
          if (!inherits(obj, "try-error") && is.data.frame(obj) && ncol(obj) > 0L) {        
            has_any_pred_cols <- TRUE                                                       
            break                                                                           
          }                                                                                 
        }                                                                                   
      }                                                                                     
    }                                                                                       
    
    if (isTRUE(has_any_pred_cols)) {                                                        
      .write_fused_consensus(result, run_dir, ts, seeds)                                     
    } else {                                                                                
      # optional breadcrumb (kept quiet for CRAN/vignette stability)
      invisible(NULL)                                                                       
    }                                                                                       
  }
  
  
  logs_dir <- file.path(run_dir, "logs")
  dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  
  log_enabled <- isTRUE(cfg$do_ensemble) &&                                          # 
    as.integer(cfg$num_networks %||% 0L) >= 1L &&                                    # 
    as.integer(cfg$num_temp_iterations %||% 0L) >= 1L                                # 
  
  for (i in seq_along(seeds)) {
    seed_i <- seeds[[i]]
    
    main_p <- file.path(logs_dir, sprintf("main_log_run%03d_seed%s_%s.rds", i, seed_i, ts))
    if (!file.exists(main_p)) saveRDS(data.frame(), main_p)
    
    if (isTRUE(log_enabled)) {                                                       # 
      mv_p <- file.path(logs_dir, sprintf("movement_log_run%03d_seed%s_%s.rds", i, seed_i, ts))
      ch_p <- file.path(logs_dir, sprintf("change_log_run%03d_seed%s_%s.rds",   i, seed_i, ts))
      if (!file.exists(mv_p)) saveRDS(data.frame(), mv_p)
      if (!file.exists(ch_p)) saveRDS(data.frame(), ch_p)
    }                                                                                # 
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
#' @param x Training features (the training split) as a data frame, matrix, or tibble.
#' @param y Training labels/targets (the training split).
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
#' @param x_valid Optional validation features. Overrides `validation$x` when set.
#' @param y_valid Optional validation labels. Overrides `validation$y` when set.
#' @param model_overrides Named list forwarded to [ddesonn_model()] allowing
#'   custom architectures.
#' @param training_overrides Named list forwarded to [ddesonn_fit()] for the
#'   main run(s). Any argument accepted by [ddesonn_fit()] may be provided here.
#'   Unspecified values fall back to [ddesonn_training_defaults()] for the given
#'   `classification_mode` and `hidden_sizes`. See **Details** and the example
#'   showing how to inspect defaults.
#' @param temp_overrides Optional named list forwarded to [ddesonn_fit()] for
#'   TEMP iterations. Defaults to `training_overrides`. Use this when TEMP
#'   candidates should train differently than the main model.
#' @param prediction_data Optional features for prediction. When supplied,
#'   predictions are computed for each seed/iteration.
#' @param test Optional test list with elements `x` and `y`. When supplied,
#'   the final model computes test metrics (loss and, for classification,
#'   accuracy) and stores them in `result$test_metrics`. The run history
#'   (`result$history`) mirrors the training metadata (train/validation losses)
#'   and appends `test_loss` when test data is provided.
#' @param x_test Optional test features. Overrides `test$x` when set.
#' @param y_test Optional test labels. Overrides `test$y` when set.
#' @param prediction_type Passed to [ddesonn_predict()].
#' @param aggregate Aggregation strategy within a single model (across ensemble
#'   members).
#' @param seed_aggregate Aggregation strategy across seeds. Set to `"none"` to
#'   keep per-seed prediction matrices.
#' @param threshold Optional threshold override for classification prediction.
#' @param output_root Optional directory where legacy-style artifacts are
#'   written. When `NULL` (default) no files are created.
#' @param plot_controls Optional list passed through to [ddesonn_fit()] as
#'   `plot_controls`. Use this to enable/disable specific report plots or
#'   diagnostics (for example, evaluation report settings). The supported
#'   structure is defined by [ddesonn_fit()]; this function does not create
#'   defaults.
#' @param save_models Logical; if `TRUE` (default) individual models are
#'   persisted when `output_root` is supplied.
#' @param verbose Logical; emit detailed progress output when TRUE.
#' @param verboseLow Logical; emit important progress output when TRUE.
#' @param debug Logical; emit debug diagnostics when TRUE.
#'
#' @details
#' **Discovering available training overrides**
#'
#' `training_overrides` is a direct pass-through to [ddesonn_fit()]. To see the
#' baseline defaults used by `ddesonn_run()`, call:
#'
#' `ddesonn_training_defaults(classification_mode, hidden_sizes)`
#'
#' To see all tunable training arguments, see `?ddesonn_fit`.
#'
#' @return A list (classed as `"ddesonn_run_result"`) containing the
#'   configuration, per-seed models, and optional prediction summaries.
#' @export
#'
#' @examples
#' \donttest{
#' # ============================================================
#' # DDESONN — FULL example using package data in inst/extdata
#' # (binary classification; train/valid/test split; scale train-only)
#' # ============================================================
#'
#' library(DDESONN)
#'
#' set.seed(111)
#'
#' # ------------------------------------------------------------
#' # 1) Locate package extdata folder (robust across check/install)  
#' # ------------------------------------------------------------
#' ext_dir <- system.file("extdata", package = "DDESONN")
#' if (!nzchar(ext_dir)) {
#'   stop("Could not find DDESONN extdata folder. Is the package installed?",
#'        call. = FALSE)
#' }
#'
#' # ------------------------------------------------------------
#' # 1b) Find CSVs (recursive + check-dir edge cases)               
#' # ------------------------------------------------------------
#' csvs <- list.files(
#'   ext_dir,
#'   pattern = "\\\\.csv$",
#'   full.names = TRUE,
#'   recursive = TRUE
#' )
#'
#' # Defensive fallback for rare nested layouts
#' if (!length(csvs)) {                                             
#'   ext_dir2 <- file.path(ext_dir, "inst", "extdata")               
#'   if (dir.exists(ext_dir2)) {                                    
#'     csvs <- list.files(
#'       ext_dir2,
#'       pattern = "\\\\.csv$",
#'       full.names = TRUE,
#'       recursive = TRUE
#'     )
#'   }
#' }
#'
#' if (!length(csvs)) {
#'   message(sprintf(
#'     "No .csv files found under: %s — skipping example.",
#'     ext_dir
#'   ))
#' } else {
#'
#'   hf_path <- file.path(ext_dir, "heart_failure_clinical_records.csv")
#'   data_path <- if (file.exists(hf_path)) hf_path else csvs[[1]]
#'
#'   cat("[extdata] using:", data_path, "\\n")
#'
#' # ------------------------------------------------------------
#' # 2) Load data
#' # ------------------------------------------------------------
#' df <- read.csv(data_path)
#'
#' # Prefer DEATH_EVENT if present; otherwise infer a binary target
#' target_col <- if ("DEATH_EVENT" %in% names(df)) {
#'   "DEATH_EVENT"
#' } else {
#'   cand <- names(df)[vapply(df, function(col) {
#'     v <- suppressWarnings(as.numeric(col))
#'     if (all(is.na(v))) return(FALSE)
#'     u <- unique(v[is.finite(v)])
#'     length(u) <= 2 && all(sort(u) %in% c(0, 1))
#'   }, logical(1))]
#'   if (!length(cand)) {
#'     stop(
#'       "Could not infer a binary target column. ",
#'       "Provide a binary CSV in extdata or rename target to DEATH_EVENT.",
#'       call. = FALSE
#'     )
#'   }
#'   cand[[1]]
#' }
#'
#' cat("[data] target_col =", target_col, "\\n")
#'
#' # ------------------------------------------------------------
#' # 3) Build X and y
#' # ------------------------------------------------------------
#' y_all <- matrix(as.integer(df[[target_col]]), ncol = 1)
#'
#' x_df <- df[, setdiff(names(df), target_col), drop = FALSE]
#' x_all <- as.matrix(x_df)
#' storage.mode(x_all) <- "double"
#'
#' # ------------------------------------------------------------
#' # 4) Split 70 / 15 / 15
#' # ------------------------------------------------------------
#' n <- nrow(x_all)
#' idx <- sample.int(n)
#'
#' n_train <- floor(0.70 * n)
#' n_valid <- floor(0.15 * n)
#'
#' i_tr <- idx[1:n_train]
#' i_va <- idx[(n_train + 1):(n_train + n_valid)]
#' i_te <- idx[(n_train + n_valid + 1):n]
#'
#' x_train <- x_all[i_tr, , drop = FALSE]
#' y_train <- y_all[i_tr, , drop = FALSE]
#'
#' x_valid <- x_all[i_va, , drop = FALSE]
#' y_valid <- y_all[i_va, , drop = FALSE]
#'
#' x_test  <- x_all[i_te, , drop = FALSE]
#' y_test  <- y_all[i_te, , drop = FALSE]
#'
#' cat(sprintf("[split] train=%d valid=%d test=%d\\n",
#'             nrow(x_train), nrow(x_valid), nrow(x_test)))
#'
#' # ------------------------------------------------------------
#' # 5) Scale using TRAIN stats only (no leakage)
#' # ------------------------------------------------------------
#' x_train_s <- scale(x_train)
#' ctr <- attr(x_train_s, "scaled:center")
#' scl <- attr(x_train_s, "scaled:scale")
#' scl[!is.finite(scl) | scl == 0] <- 1
#'
#' x_valid_s <- sweep(sweep(x_valid, 2, ctr, "-"), 2, scl, "/")
#' x_test_s  <- sweep(sweep(x_test,  2, ctr, "-"), 2, scl, "/")
#'
#' mx <- suppressWarnings(max(abs(x_train_s)))
#' if (!is.finite(mx) || mx == 0) mx <- 1
#'
#' x_train <- x_train_s / mx
#' x_valid <- x_valid_s / mx
#' x_test  <- x_test_s  / mx
#'
#' # ------------------------------------------------------------
#' # 6) Run DDESONN
#' # ------------------------------------------------------------
#' res <- ddesonn_run(
#'   x = x_train,
#'   y = y_train,
#'   classification_mode = "binary",
#'
#'   hidden_sizes = c(64, 32),
#'   seeds = 1L,
#'   do_ensemble = FALSE,
#'
#'   validation = list(
#'     x = x_valid,
#'     y = y_valid
#'   ),
#'
#'   test = list(
#'     x = x_test,
#'     y = y_test
#'   ),
#'
#'   training_overrides = list(
#'     init_method = "he",
#'     optimizer = "adagrad",
#'     lr = 0.125,
#'     lambda = 0.00028,
#'
#'     activation_functions = list(relu, relu, sigmoid),
#'     dropout_rates = list(0.10),
#'     loss_type = "CrossEntropy",
#'
#'     validation_metrics = TRUE,
#'     num_epochs = 360,
#'     final_summary_decimals = 6L
#'   ),
#'
#'   plot_controls = list(
#'     evaluate_report = list(
#'       roc_curve = TRUE,
#'       pr_curve  = FALSE
#'     )
#'   )
#' )
#' }
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
                        x_valid = NULL,  
                        y_valid = NULL,  
                        model_overrides = list(),  
                        training_overrides = list(),  
                        temp_overrides = NULL,  
                        prediction_data = NULL,  
                        test = NULL,  
                        x_test = NULL,  
                        y_test = NULL,  
                        prediction_type = c("response", "class"),  
                        aggregate = c("mean", "median", "none"),  
                        seed_aggregate = c("mean", "median", "none"),  
                        threshold = NULL,  
                        output_root = NULL,  
                        plot_controls = NULL,  
                        save_models = TRUE,  
                        verbose = FALSE,  
                        verboseLow = FALSE,  
                        debug = FALSE) {  
  classification_mode <- match.arg(classification_mode)
  aggregate <- match.arg(aggregate)
  seed_aggregate <- match.arg(seed_aggregate)
  prediction_type <- match.arg(prediction_type)
  
  verbose <- isTRUE(verbose %||% FALSE)  
  verboseLow <- isTRUE(verboseLow %||% FALSE)  
  debug <- isTRUE(debug %||% getOption("DDESONN.debug", FALSE))  
  debug <- isTRUE(debug) && identical(Sys.getenv("DDESONN_DEBUG"), "1")  

  # ============================================================  
  # SECTION: Verbosity resolution (run config)  
  # - verbose defaults FALSE unless explicitly TRUE        
  # - verboseLow defaults FALSE unless explicitly TRUE     
  # ============================================================  
  ddesonn_console_log(  
    sprintf("[VERBOSE] resolved: verbose=%s verboseLow=%s", verbose, verboseLow),  
    level = "important",  
    verbose = verbose,  
    verboseLow = verboseLow  
  )  
  
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
  
  # ============================================================
  # SECTION: Prepare training params
  # ============================================================
  base_train_overrides <- ddesonn_training_defaults(classification_mode, hidden_sizes)
  base_train_overrides <- utils::modifyList(base_train_overrides, training_overrides, keep.null = TRUE)
  if (is.null(base_train_overrides$verbose)) base_train_overrides$verbose <- verbose  
  if (is.null(base_train_overrides$verboseLow)) base_train_overrides$verboseLow <- verboseLow  
  if (is.null(base_train_overrides$debug)) base_train_overrides$debug <- debug  
  
  # ============================================================
  # SECTION: NORMALIZATION BRIDGE (Scenario 1 -> canonical)  
  # - Map Scenario-1 training_overrides keys into canonical plot_controls keys
  # - Auto-enable save_per_epoch if any per-epoch plot flag is TRUE and save_per_epoch not set
  # - Do NOT change Scenario-2 behavior
  # ============================================================
  per_epoch_cfg <- NULL
  if (!is.null(training_overrides$per_epoch_view_plots)) {                                           
    per_epoch_cfg <- training_overrides$per_epoch_view_plots                                          
  } else if (!is.null(training_overrides$per_epoch_plots)) {                                         
    per_epoch_cfg <- training_overrides$per_epoch_plots                                               
  }                                                                                                  
  if (!is.null(per_epoch_cfg)) {                                                                     
    if (is.null(base_train_overrides$plot_controls) || !is.list(base_train_overrides$plot_controls)) {  
      base_train_overrides$plot_controls <- list()                                                    
    }                                                                                                 
    if (isTRUE(per_epoch_cfg)) {                                                                     
      per_epoch_cfg <- list(viewAllPlots = TRUE)                                                      
    } else if (isFALSE(per_epoch_cfg)) {                                                             
      per_epoch_cfg <- list(viewAllPlots = FALSE)                                                     
    } else if (is.list(per_epoch_cfg)) {                                                             
      legacy_cfg <- per_epoch_cfg                                                                     
      per_epoch_cfg <- list(                                                                          
        viewAllPlots     = legacy_cfg$viewAllPlots,                                                   
        verbose          = legacy_cfg$verbose,                                                        
        saveEnabled      = legacy_cfg$saveEnabled,                                                    
        accuracy_plot    = legacy_cfg$accuracy_plot %||% legacy_cfg$loss_curve,                       
        saturation_plot  = legacy_cfg$saturation_plot %||% legacy_cfg$probe_plots,                    
        max_weight_plot  = legacy_cfg$max_weight_plot %||% legacy_cfg$probe_plots                      
      )                                                                                               
    }                                                                                                 
    base_train_overrides$plot_controls$per_epoch <- per_epoch_cfg      
  }                                                                                                  
  
  # ============================================================
  #  FIX: Scenario-1 FINAL performance/relevance plots rename
  # - New Scenario-1 key: final_update_performance_relevance_plots
  # - Canonical key used by engine: plot_controls$performance_relevance
  # - Backward compat: if old ..._boxplots exists, accept it too
  # ============================================================
  if (!is.null(training_overrides$final_update_performance_relevance_plots)) {                       
    if (is.null(base_train_overrides$plot_controls) || !is.list(base_train_overrides$plot_controls)) {  
      base_train_overrides$plot_controls <- list()                                                    
    }                                                                                                 
    base_train_overrides$plot_controls$performance_relevance <-                                      
      training_overrides$final_update_performance_relevance_plots                                    
  } else if (!is.null(training_overrides$final_update_performance_relevance_boxplots)) {             
    if (is.null(base_train_overrides$plot_controls) || !is.list(base_train_overrides$plot_controls)) {  
      base_train_overrides$plot_controls <- list()                                                    
    }                                                                                                 
    base_train_overrides$plot_controls$performance_relevance <-                                      
      training_overrides$final_update_performance_relevance_boxplots                                  
  } else if (!is.null(training_overrides$performance_relevance)) {                                   
    if (is.null(base_train_overrides$plot_controls) || !is.list(base_train_overrides$plot_controls)) {  
      base_train_overrides$plot_controls <- list()                                                    
    }                                                                                                 
    base_train_overrides$plot_controls$performance_relevance <-                                      
      training_overrides$performance_relevance                                                       
  }                                                                                                  
  
  if (is.null(base_train_overrides$save_per_epoch) && !is.null(base_train_overrides$plot_controls$per_epoch)) {  
    pe <- base_train_overrides$plot_controls$per_epoch                                                
    pe_any_true <- isTRUE(any(vapply(pe, function(v) isTRUE(v), logical(1)), na.rm = TRUE))           
    if (isTRUE(pe_any_true)) {                                                                        
      base_train_overrides$save_per_epoch <- TRUE                                                     
    }                                                                                                 
  }                                                                                                   
  
  base_train_overrides$output_root <- base_train_overrides$output_root %||% output_root
  
  # ============================================================  
  # SECTION: Plot controls passthrough (Scenario 2)
  # - Do NOT create local defaults here.
  # - If user provided plot_controls, pass it through to training.
  # ============================================================  
  if (!is.null(plot_controls)) {                                                       
    base_train_overrides$plot_controls <- plot_controls                                 
  }                                                                                      

  # ============================================================  
  # SECTION: Plot controls passthrough only (no mutation)  
  # ============================================================  
  
  # ============================================================
  # SECTION: Verbose / EVOKE logger  
  # ============================================================
  verbose_run <- isTRUE(base_train_overrides$verbose %||% FALSE)  # kept (informational)
  
  .evoke_predict_begin <- function(where, why, seed = NA_integer_, run_index = NA_integer_) {  
    ddesonn_console_log(                                                                       
      sprintf(                                                                                  
        "[EVOKE-PREDICT-BEGIN] where=%s | why=%s | seed=%s | run_index=%s\n",                    
        as.character(where), as.character(why), as.character(seed), as.character(run_index)     
      ),                                                                                        
      level = "info",                                                                           
      verbose = verbose,                                                                        
      verboseLow = verboseLow                                                                   
    )                                                                                           
    invisible(NULL)                                                                             
  }                                                                                             
  
  .evoke_predict_end <- function(where, why, seed = NA_integer_, run_index = NA_integer_) {     
    ddesonn_console_log(                                                                       
      sprintf(                                                                                  
        "[EVOKE-PREDICT-END] where=%s | why=%s | seed=%s | run_index=%s\n",                      
        as.character(where), as.character(why), as.character(seed), as.character(run_index)     
      ),                                                                                        
      level = "info",                                                                           
      verbose = verbose,                                                                        
      verboseLow = verboseLow                                                                   
    )                                                                                           
    invisible(NULL)                                                                             
  }                                                                                             
  
  validation_data <- validation
  if (!is.null(x_valid) || !is.null(y_valid)) {
    if (is.null(x_valid) || is.null(y_valid)) {
      stop("Provide both x_valid and y_valid or neither.", call. = FALSE)
    }
    validation_data <- list(x = x_valid, y = y_valid)
  }
  test_data <- test
  if (!is.null(x_test) || !is.null(y_test)) {
    if (is.null(x_test) || is.null(y_test)) {
      stop("Provide both x_test and y_test or neither.", call. = FALSE)
    }
    test_data <- list(x = x_test, y = y_test)
  }
  
  prediction_matrix <- NULL
  if (!is.null(prediction_data)) {
    prediction_matrix <- .as_numeric_matrix(prediction_data)
  }
  test_matrix <- NULL
  test_labels <- NULL
  if (!is.null(test_data)) {
    if (is.list(test_data) && !is.null(test_data$x)) {
      test_matrix <- .as_numeric_matrix(test_data$x)
      test_labels <- test_data$y %||% NULL
    }
  }
  
  target_metric <- {
    default_metric <- if (identical(classification_mode, "regression")) "MSE" else "accuracy"
    get0(
      "metric_name",
      inherits = TRUE,
      ifnotfound = get0("TARGET_METRIC", inherits = TRUE, ifnotfound = default_metric)
    )
  }
  
  metric_minimize <- function(metric) {
    m <- tolower(as.character(metric %||% ""))
    if (!nzchar(m)) return(FALSE)
    m %in% c(
      "mse", "mae", "rmse", "r2", "mape", "smape", "wmape", "mase",
      "logloss", "brier", "quantization_error", "topographic_error",
      "clustering_quality_db", "generalization_ability", "loss"
    )
  }
  
  main_meta_var <- function(i) sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(i))
  temp_meta_var <- function(e, i) sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(e), as.integer(i))
  
  snapshot_main_serials_meta <- function() {
    vars <- grep("^Ensemble_Main_(0|1)_model_\\d+_metadata$", ls(.ddesonn_state), value = TRUE)
    if (!length(vars)) return(character())
    ord <- suppressWarnings(as.integer(sub("^Ensemble_Main_(?:0|1)_model_(\\d+)_metadata$", "\\1", vars)))
    vars <- vars[order(ord)]
    vapply(vars, function(v) {
      md <- get(v, envir = .ddesonn_state)
      as.character(md$model_serial_num %||% NA_character_)
    }, character(1))
  }
  
  get_metric_by_serial <- function(serial, metric_name) {
    vars <- grep(
      "^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
      ls(.ddesonn_state), value = TRUE
    )
    if (!length(vars)) return(NA_real_)
    for (v in vars) {
      md <- get(v, envir = .ddesonn_state)
      if (identical(as.character(md$model_serial_num %||% NA_character_), as.character(serial))) {
        val <- tryCatch(md$performance_metric[[metric_name]], error = function(e) NULL)
        if (is.null(val)) {
          val <- tryCatch(md$relevance_metric[[metric_name]], error = function(e) NULL)
        }
        vn <- suppressWarnings(as.numeric(val))
        if (length(vn) && is.finite(vn[1])) return(vn[1])
        return(NA_real_)
      }
    }
    NA_real_
  }
  
  get_temp_serials_meta <- function(iter_j) {
    e <- as.integer(iter_j) + 1L
    vars <- grep(sprintf("^Ensemble_Temp_%d_model_\\d+_metadata$", e), ls(.ddesonn_state), value = TRUE)
    if (!length(vars)) return(character())
    ord <- suppressWarnings(as.integer(sub(sprintf("^Ensemble_Temp_%d_model_(\\d+)_metadata$", e), "\\1", vars)))
    vars <- vars[order(ord)]
    vapply(vars, function(v) {
      md <- get(v, envir = .ddesonn_state)
      s <- md$model_serial_num
      if (!is.null(s) && nzchar(as.character(s))) as.character(s) else NA_character_
    }, character(1))
  }
  
  extract_y_true <- function(y_source) {
    if (is.null(y_source)) return(NULL)
    y_mat <- try(.as_numeric_matrix(y_source), silent = TRUE)
    if (inherits(y_mat, "try-error") || !is.matrix(y_mat) || !nrow(y_mat)) {
      return(NULL)
    }
    if (classification_mode == "binary") {
      return(as.numeric(y_mat[, 1]))
    }
    if (classification_mode == "multiclass") {
      if (ncol(y_mat) <= 1L) {
        return(suppressWarnings(as.integer(y_mat[, 1])))
      }
      return(max.col(y_mat, ties.method = "first"))
    }
    as.numeric(y_mat[, 1])
  }
  
  # ============================================================
  # SECTION: build_split_predictions() — EVOKE prints for split-driven predict calls  
  # ============================================================
  build_split_predictions <- function(model,
                                      new_data,
                                      y_source,
                                      split_label,
                                      run_idx,
                                      seed_val) {
    if (is.null(new_data)) return(NULL)
    
    #  FIX: Explicit BEGIN/END around the exact ddesonn_predict() call that triggers net$predict()
    .evoke_predict_begin(                                                                                 
      where = sprintf("ddesonn_run::build_split_predictions(split=%s)", tolower(split_label)),            
      why   = "Split diagnostics: ddesonn_predict(aggregate='none', type='response') -> per-model tables",
      seed  = seed_val,                                                                                   
      run_index = run_idx                                                                                 
    )                                                                                                     
    
    pr <- try(                                                                                               
      ddesonn_predict(                                                                                       
        model,                                                                                                
        new_data,                                                                                             
        aggregate = "none",                                                                                   
        type = "response",                                                                                    
        verbose = verbose,                                                                                    
        verboseLow = verboseLow,                                                                              
        debug = debug                                                                                          
      ),                                                                                                      
      silent = TRUE                                                                                           
    )                                                                                                         
    
    .evoke_predict_end(                                                                                   
      where = sprintf("ddesonn_run::build_split_predictions(split=%s)", tolower(split_label)),            
      why   = "Split diagnostics predict finished (if you saw predict() above, it came from this block)", 
      seed  = seed_val,                                                                                   
      run_index = run_idx                                                                                 
    )                                                                                                     
    
    if (inherits(pr, "try-error") || is.null(pr$per_model) || !length(pr$per_model)) {
      return(NULL)
    }
    
    y_true_vec <- extract_y_true(y_source)
    rows <- list()
    ptr <- 0L
    
    for (k in seq_along(pr$per_model)) {
      mat <- try(.as_numeric_matrix(pr$per_model[[k]]), silent = TRUE)
      if (inherits(mat, "try-error") || !is.matrix(mat)) next
      
      n <- nrow(mat)
      if (!n) next
      
      if (!is.null(y_true_vec) && length(y_true_vec) != n) {
        y_slot <- rep(NA_real_, n)
      } else if (!is.null(y_true_vec)) {
        y_slot <- as.numeric(y_true_vec)
      } else {
        y_slot <- rep(NA_real_, n)
      }
      
      if (classification_mode == "binary") {
        y_prob <- as.numeric(mat[, 1])
        thr <- threshold %||% attr(model, "chosen_threshold") %||% model$chosen_threshold %||%
          attr(model, "threshold") %||% .ddesonn_threshold_default(classification_mode)
        y_pred <- ifelse(is.na(y_prob), NA_integer_, as.integer(y_prob >= thr))
      } else if (classification_mode == "multiclass") {
        y_prob <- apply(mat, 1, max)
        y_pred <- max.col(mat, ties.method = "first")
      } else {
        y_prob <- as.numeric(mat[, 1])
        y_pred <- y_prob
      }
      
      obs_idx <- seq_len(n)
      
      df <- data.frame(
        run_index = rep.int(as.integer(run_idx), n),
        RUN_INDEX = rep.int(as.integer(run_idx), n),
        seed = rep.int(as.integer(seed_val), n),
        SEED = rep.int(as.integer(seed_val), n),
        model_slot = rep.int(as.integer(k), n),
        MODEL_SLOT = rep.int(as.integer(k), n),
        slot = rep.int(as.integer(k), n),
        obs_index = obs_idx,
        obs = obs_idx,
        split = tolower(split_label),
        SPLIT = toupper(split_label),
        `.__split__` = tolower(split_label),
        CLASSIFICATION_MODE = toupper(classification_mode),
        y_true = y_slot,
        y_prob = y_prob,
        y_pred = y_pred,
        stringsAsFactors = FALSE
      )
      
      if (classification_mode == "multiclass") {
        df$y_prob_full <- lapply(seq_len(n), function(i) as.numeric(mat[i, , drop = TRUE]))
      }
      
      rows[[ptr <- ptr + 1L]] <- df
    }
    
    if (!length(rows)) return(NULL)
    out <- try(do.call(rbind, rows), silent = TRUE)
    if (inherits(out, "try-error")) return(NULL)
    rownames(out) <- NULL
    out
  }
  
  empty_log_tables <- function(log_enabled) {  # 
    out <- list(
      main_log = data.frame(
        iteration = integer(), phase = character(), slot = integer(),
        serial = character(), metric_name = character(),
        metric_value = numeric(), message = character(),
        timestamp = as.POSIXct(character()), stringsAsFactors = FALSE
      )
    )
    
    if (isTRUE(log_enabled)) {                # 
      out$movement_log <- data.frame(
        iteration = integer(), phase = character(), slot = integer(),
        role = character(), serial = character(), metric_name = character(),
        metric_value = numeric(), message = character(),
        timestamp = as.POSIXct(character()), stringsAsFactors = FALSE
      )
      out$change_log <- data.frame(
        iteration = integer(), role = character(), serial = character(),
        metric_name = character(), metric_value = numeric(),
        message = character(), timestamp = as.POSIXct(character()),
        stringsAsFactors = FALSE
      )
    }
    
    out
  }
  
  
  
  record_main_snapshot <- function(log_tables, iteration, phase) {
    serials <- snapshot_main_serials_meta()
    if (!length(serials)) return(log_tables)
    vals <- vapply(serials, get_metric_by_serial, numeric(1), metric_name = target_metric)
    rows <- data.frame(
      iteration = if (is.null(iteration)) NA_integer_ else as.integer(iteration),
      phase = as.character(phase),
      slot = seq_along(serials),
      serial = as.character(serials),
      metric_name = rep.int(target_metric, length(serials)),
      metric_value = suppressWarnings(as.numeric(vals)),
      message = rep.int("", length(serials)),
      timestamp = rep.int(Sys.time(), length(serials)),
      stringsAsFactors = FALSE
    )
    log_tables$main_log <- rbind(log_tables$main_log, rows)
    log_tables
  }
  
  append_movement_entries <- function(log_tables, iteration, removed_info, added_slot, added_serial) {
    ts <- Sys.time()
    if (!is.null(removed_info) && !is.null(removed_info$worst_serial)) {
      row_removed <- data.frame(
        iteration = as.integer(iteration),
        phase = "removed",
        slot = as.integer(removed_info$worst_slot %||% removed_info$worst_model_index %||% NA_integer_),
        role = "removed",
        serial = as.character(removed_info$worst_serial %||% NA_character_),
        metric_name = target_metric,
        metric_value = suppressWarnings(as.numeric(removed_info$worst_value %||% NA_real_)),
        message = if (!is.null(added_slot)) sprintf("%s replaced", removed_info$worst_serial) else "removed (no replacement)",
        timestamp = ts,
        stringsAsFactors = FALSE
      )
      log_tables$movement_log <- rbind(log_tables$movement_log, row_removed)
      log_tables$change_log <- rbind(
        log_tables$change_log,
        data.frame(
          iteration = as.integer(iteration),
          role = "removed",
          serial = as.character(removed_info$worst_serial %||% NA_character_),
          metric_name = target_metric,
          metric_value = suppressWarnings(as.numeric(removed_info$worst_value %||% NA_real_)),
          message = "model removed from main",
          timestamp = ts,
          stringsAsFactors = FALSE
        )
      )
    }
    if (!is.null(added_slot)) {
      row_added <- data.frame(
        iteration = as.integer(iteration),
        phase = "added",
        slot = as.integer(added_slot),
        role = "added",
        serial = as.character(added_serial %||% NA_character_),
        metric_name = target_metric,
        metric_value = NA_real_,
        message = "candidate moved into main",
        timestamp = ts,
        stringsAsFactors = FALSE
      )
      log_tables$movement_log <- rbind(log_tables$movement_log, row_added)
      log_tables$change_log <- rbind(
        log_tables$change_log,
        data.frame(
          iteration = as.integer(iteration),
          role = "added",
          serial = as.character(added_serial %||% NA_character_),
          metric_name = target_metric,
          metric_value = NA_real_,
          message = sprintf("slot %s filled from TEMP", as.integer(added_slot)),
          timestamp = ts,
          stringsAsFactors = FALSE
        )
      )
    }
    log_tables
  }
  
  prune_network_from_main <- function(main_model, target_metric_name) {
    main_serials <- snapshot_main_serials_meta()
    if (!length(main_serials)) return(NULL)
    vals <- vapply(main_serials, get_metric_by_serial, numeric(1), metric_name = target_metric_name)
    if (all(!is.finite(vals))) return(NULL)
    minimize <- metric_minimize(target_metric_name)
    worst_idx <- if (minimize) which.max(vals) else which.min(vals)
    worst_idx <- worst_idx[1]
    if (!length(main_model$ensemble) || worst_idx < 1L || worst_idx > length(main_model$ensemble)) {
      return(NULL)
    }
    list(
      removed_network = main_model$ensemble[[worst_idx]],
      worst_model_index = as.integer(worst_idx),
      worst_slot = as.integer(worst_idx),
      worst_serial = as.character(main_serials[worst_idx]),
      worst_value = as.numeric(vals[worst_idx])
    )
  }
  
  add_network_to_main <- function(main_model,
                                  temp_model,
                                  iteration_index,
                                  target_metric_name,
                                  worst_slot) {
    temp_serials <- get_temp_serials_meta(iteration_index)
    if (!length(temp_serials)) {
      return(list(model = main_model, slot = NULL, serial = NULL))
    }
    vals <- vapply(temp_serials, get_metric_by_serial, numeric(1), metric_name = target_metric_name)
    if (all(!is.finite(vals))) {
      return(list(model = main_model, slot = NULL, serial = NULL))
    }
    minimize <- metric_minimize(target_metric_name)
    best_idx <- if (minimize) which.min(vals) else which.max(vals)
    best_idx <- best_idx[1]
    best_serial <- as.character(temp_serials[best_idx])
    parts <- strsplit(best_serial, "\\.")[[1]]
    temp_model_index <- suppressWarnings(as.integer(tail(parts, 1)))
    if (!is.finite(temp_model_index) || temp_model_index < 1L) {
      return(list(model = main_model, slot = NULL, serial = NULL))
    }
    if (temp_model_index > length(temp_model$ensemble)) {
      return(list(model = main_model, slot = NULL, serial = NULL))
    }
    candidate <- temp_model$ensemble[[temp_model_index]]
    if (is.null(candidate)) {
      return(list(model = main_model, slot = NULL, serial = NULL))
    }
    main_model$ensemble[[worst_slot]] <- candidate
    temp_env <- temp_meta_var(iteration_index + 1L, temp_model_index)
    main_env <- main_meta_var(worst_slot)
    if (exists(temp_env, envir = .ddesonn_state)) {
      md <- get(temp_env, envir = .ddesonn_state)
      md$model_serial_num <- best_serial
      assign(main_env, md, envir = .ddesonn_state)
    }
    list(model = main_model, slot = as.integer(worst_slot), serial = best_serial)
  }
  
  # ============================================================
  # SECTION: Per-seed main runs
  # ============================================================
  runs <- lapply(seq_along(seeds), function(i) {
    seed_i <- seeds[[i]]
    if (is.null(seed_i) || length(seed_i) == 0L) {
      seed_i <- i
    }
    set.seed(seed_i)
    
    if (isTRUE(do_ensemble)) {
      vars <- grep(
        "^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
        ls(.ddesonn_state), value = TRUE
      )
      if (length(vars)) rm(list = vars, envir = .ddesonn_state)
    }
    
    log_enabled <- isTRUE(do_ensemble) &&                                              # 
      as.integer(num_networks %||% 0L) >= 1L &&                                        # 
      as.integer(num_temp_iterations %||% 0L) >= 1L                                    # 
    
    log_tables <- empty_log_tables(log_enabled)                                        # 
    
    main_model_args <- base_model_args
    if (isTRUE(do_ensemble)) {
      main_model_args$ensemble_number <- 1L
    }
    
    mdl <- do.call(ddesonn_model, main_model_args)
    
    val <- NULL
    if (!is.null(validation_data)) {
      val <- list(x = validation_data$x, y = validation_data$y)
    }
    
    if (isTRUE(debug)) {                                                                # 
      ddesonn_console_log(                                                              # 
        sprintf(                                                                        # 
          "[EVOKE-FIT-BEGIN] where=%s | why=%s | seed=%s | run_index=%s\n",              # 
          "ddesonn_run::per_seed(main_fit)->ddesonn_fit",                                # 
          "Training main model; ddesonn_fit() may invoke predict() internally via validation/metrics/snapshots", # 
          as.character(seed_i),                                                         # 
          as.character(i)                                                               # 
        ),                                                                              # 
        level = "important",                                                            # 
        verbose = verbose,                                                              # 
        verboseLow = verboseLow,                                                        # 
        debug = debug                                                                   # 
      )                                                                                # 
    }                                                                                   # 
    
    if (isTRUE(debug)) {                                                               
      ddesonn_debug(                                                                   
        "[DBG] per-epoch plot controls (base_train_overrides$plot_controls$per_epoch)",
        debug = debug                                                                  
      )                                                                                
      ddesonn_debug(                                                                   
        paste(utils::capture.output(utils::str(base_train_overrides$plot_controls$per_epoch)), collapse = "\n"), 
        debug = debug                                                                  
      )                                                                                
      ddesonn_debug(                                                                   
        paste("save_per_epoch:", base_train_overrides$save_per_epoch),                 
        debug = debug                                                                  
      )                                                                                
      pe_cfg <- base_train_overrides$plot_controls$per_epoch                           
      saveEnabled <- if (is.list(pe_cfg)) pe_cfg$saveEnabled else NULL                 
      ddesonn_debug(                                                                   
        paste("saveEnabled:", if (is.null(saveEnabled)) "NULL" else saveEnabled),      
        debug = debug                                                                  
      )                                                                                
      plots_root <- base_train_overrides$output_root %||% output_root                  
      ddesonn_debug(                                                                   
        paste("plots_dir:", ddesonn_plots_dir(plots_root)),                            
        debug = debug                                                                  
      )                                                                                
    }                                                                                  
    
    do.call(ddesonn_fit, c(list(model = mdl, x = x_matrix, y = y_matrix, validation = val), base_train_overrides))
    
    if (isTRUE(debug)) {                                                                # 
      ddesonn_console_log(                                                              # 
        sprintf(                                                                        # 
          "[EVOKE-FIT-END] where=%s | why=%s | seed=%s | run_index=%s\n",                # 
          "ddesonn_run::per_seed(main_fit)->ddesonn_fit",                                # 
          "ddesonn_fit() returned to ddesonn_run; any predict() emitted above could have originated inside training/metrics/validation", # 
          as.character(seed_i),                                                         # 
          as.character(i)                                                               # 
        ),                                                                              # 
        level = "important",                                                            # 
        verbose = verbose,                                                              # 
        verboseLow = verboseLow,                                                        # 
        debug = debug                                                                   # 
      )                                                                                # 
    }                                                                                   # 
    
    main_pred <- NULL
    aggregate_pred <- NULL
    
    if (!is.null(prediction_matrix)) {
      
      if (isTRUE(debug)) {                                                              # 
        .evoke_predict_begin(                                                           # 
          where = "ddesonn_run::per_seed(main_fit)->main_pred",
          why   = "User prediction_data provided; ddesonn_predict() will call net$predict() across ensemble",
          seed  = seed_i,
          run_index = i
        )
      }                                                                                 # 
      
      preds <- ddesonn_predict(                                                         
        mdl,                                                                            
        prediction_matrix,                                                              
        aggregate = aggregate,                                                          
        type = prediction_type,                                                         
        threshold = threshold,                                                          
        verbose = verbose,                                                              
        verboseLow = verboseLow,                                                        
        debug = debug                                                                   
      )                                                                                 
      main_pred <- preds
      
      if (isTRUE(debug)) {                                                              # 
        .evoke_predict_end(                                                             # 
          where = "ddesonn_run::per_seed(main_fit)->main_pred",
          why   = "Main prediction finished",
          seed  = seed_i,
          run_index = i
        )
      }                                                                                 # 
    }
    
    temp_summary <- list()
    temp_models <- vector("list", length = if (isTRUE(do_ensemble)) num_temp_iterations else 0L)
    
    if (isTRUE(do_ensemble) && num_temp_iterations > 0L) {
      tmp_overrides <- temp_overrides %||% base_train_overrides
      tmp_overrides$output_root <- tmp_overrides$output_root %||% output_root
      
      temp_list <- vector("list", length = num_temp_iterations)
      
      for (iter in seq_len(num_temp_iterations)) {
        log_tables <- record_main_snapshot(log_tables, iteration = iter, phase = "main_before")
        
        temp_model_args <- base_model_args
        temp_model_args$ensemble_number <- iter + 1L
        tmp_model <- do.call(ddesonn_model, temp_model_args)
        
        if (isTRUE(debug)) {                                                            # 
          ddesonn_console_log(                                                          # 
            sprintf(                                                                    # 
              "[EVOKE-FIT-BEGIN] where=%s | why=%s | seed=%s | run_index=%s\n",          # 
              sprintf("ddesonn_run::TEMP(iter=%d)->ddesonn_fit", as.integer(iter)),      # 
              "Training TEMP candidate model; ddesonn_fit() may invoke predict() internally via validation/metrics/snapshots", # 
              as.character(seed_i),                                                     # 
              as.character(i)                                                           # 
            ),                                                                          # 
            level = "important",                                                        # 
            verbose = verbose,                                                          # 
            verboseLow = verboseLow,                                                    # 
            debug = debug                                                               # 
          )                                                                             # 
        }                                                                               # 
        
        do.call(ddesonn_fit, c(list(model = tmp_model, x = x_matrix, y = y_matrix, validation = val), tmp_overrides))
        
        if (isTRUE(debug)) {                                                            # 
          ddesonn_console_log(                                                          # 
            sprintf(                                                                    # 
              "[EVOKE-FIT-END] where=%s | why=%s | seed=%s | run_index=%s\n",            # 
              sprintf("ddesonn_run::TEMP(iter=%d)->ddesonn_fit", as.integer(iter)),      # 
              "ddesonn_fit() returned to ddesonn_run for TEMP candidate; any predict() emitted above could have originated inside training/metrics/validation", # 
              as.character(seed_i),                                                     # 
              as.character(i)                                                           # 
            ),                                                                          # 
            level = "important",                                                        # 
            verbose = verbose,                                                          # 
            verboseLow = verboseLow,                                                    # 
            debug = debug                                                               # 
          )                                                                             # 
        }                                                                               # 
        
        per_seed <- NULL
        aggregate_tmp <- NULL
        
        if (!is.null(prediction_matrix)) {
          
          if (isTRUE(debug)) {                                                          # 
            .evoke_predict_begin(                                                       # 
              where = sprintf("ddesonn_run::TEMP(iter=%d)->candidate_eval", iter),
              why   = "TEMP evaluation on prediction_data: ddesonn_predict(aggregate='none', type='response')",
              seed  = seed_i,
              run_index = i
            )
          }                                                                             # 
          
          tpred <- ddesonn_predict(                                                     
            tmp_model,                                                                  
            prediction_matrix,                                                          
            aggregate = "none",                                                         
            type = "response",                                                          
            verbose = verbose,                                                          
            verboseLow = verboseLow,                                                    
            debug = debug                                                               
          )                                                                             
          per_seed <- tpred$per_model
          aggregate_tmp <- .aggregate_predictions(per_seed, aggregate)
          
          if (isTRUE(debug)) {                                                          # 
            .evoke_predict_end(                                                         # 
              where = sprintf("ddesonn_run::TEMP(iter=%d)->candidate_eval", iter),
              why   = "TEMP evaluation finished",
              seed  = seed_i,
              run_index = i
            )
          }                                                                             # 
        }
        
        temp_list[[iter]] <- list(iteration = iter, model = tmp_model, per_seed = per_seed, aggregate = aggregate_tmp)
        temp_models[[iter]] <- tmp_model
        
        removed <- prune_network_from_main(mdl, target_metric)
        added <- list(model = mdl, slot = NULL, serial = NULL)
        
        if (!is.null(removed)) {
          added <- add_network_to_main(
            mdl, tmp_model,
            iteration_index = iter,
            target_metric_name = target_metric,
            worst_slot = removed$worst_slot
          )
          mdl <- added$model
        }
        
        log_tables <- append_movement_entries(log_tables, iter, removed, added$slot, added$serial)
        log_tables <- record_main_snapshot(log_tables, iteration = iter, phase = "main_after")
      }
      
      temp_summary <- temp_list
    }
    
    if (isTRUE(do_ensemble) && num_temp_iterations == 0L) {
      log_tables <- record_main_snapshot(log_tables, iteration = NULL, phase = "main_only")
    }
    
    if (!is.null(prediction_matrix)) {
      if (isTRUE(do_ensemble) && num_temp_iterations > 0L) {
        
        if (isTRUE(debug)) {                                                            # 
          .evoke_predict_begin(                                                         # 
            where = "ddesonn_run::post_TEMP(main_mutated)->main_pred",
            why   = "Recompute main predictions after TEMP-based mutation: ddesonn_predict() calls net$predict()",
            seed  = seed_i,
            run_index = i
          )
        }                                                                               # 
        
        preds <- ddesonn_predict(                                                       
          mdl,                                                                          
          prediction_matrix,                                                            
          aggregate = aggregate,                                                        
          type = prediction_type,                                                       
          threshold = threshold,                                                        
          verbose = verbose,                                                            
          verboseLow = verboseLow,                                                      
          debug = debug                                                                 
        )                                                                               
        main_pred <- preds
        
        if (isTRUE(debug)) {                                                            # 
          .evoke_predict_end(                                                           # 
            where = "ddesonn_run::post_TEMP(main_mutated)->main_pred",
            why   = "Post-TEMP main prediction finished",
            seed  = seed_i,
            run_index = i
          )
        }                                                                               # 
      }
    }
    
    run_predictions <- list(
      train = build_split_predictions(mdl, x_matrix, y_matrix, "train", i, seed_i),
      validation = if (!is.null(validation_data) && !is.null(validation_data$x)) {
        build_split_predictions(mdl, validation_data$x, validation_data$y, "validation", i, seed_i)
      } else {
        NULL
      },
      test = if (!is.null(test_matrix)) {
        build_split_predictions(mdl, test_matrix, test_labels, "test", i, seed_i)
      } else if (!is.null(prediction_matrix)) {
        build_split_predictions(mdl, prediction_matrix, NULL, "test", i, seed_i)
      } else {
        NULL
      }
    )
    
    list(
      seed = seed_i,
      main = list(model = mdl, predictions = main_pred),
      temp_iterations = temp_summary,
      tables = log_tables,
      predictions = run_predictions
    )
  })
  
  
  # ============================================================
  # SECTION: Aggregate across seeds (if prediction_matrix provided)
  # ============================================================
  main_seed_predictions <- NULL
  main_aggregate <- NULL
  temp_summary <- list()
  if (!is.null(prediction_matrix)) {
    per_seed_preds <- lapply(runs, function(run) {
      if (is.null(run$main$predictions)) return(NULL)
      if (identical(aggregate, "none")) {
        run$main$predictions$per_model
      } else {
        list(run$main$predictions$prediction)
      }
    })
    per_seed_preds <- Filter(Negate(is.null), per_seed_preds)
    main_seed_predictions <- per_seed_preds
    
    if (!identical(seed_aggregate, "none") && length(per_seed_preds)) {
      mats <- lapply(per_seed_preds, function(pe) {
        if (is.list(pe) && length(pe) > 1L) {
          .aggregate_predictions(pe, aggregate = aggregate)
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
  
  if (isTRUE(do_ensemble) && num_temp_iterations > 0L) {
    temp_summary <- lapply(seq_len(num_temp_iterations), function(iter) {
      preds <- lapply(runs, function(run) {
        ent <- NULL
        for (e in run$temp_iterations) if (identical(e$iteration, iter)) { ent <- e; break }
        ent
      })
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
  
  final_model <- NULL
  final_training <- NULL
  if (length(runs)) {
    final_run <- runs[[length(runs)]]
    if (is.list(final_run) &&
        !is.null(final_run$main) &&
        !is.null(final_run$main$model)) {
      final_model <- final_run$main$model
      final_training <- final_model$last_training
    }
  }
  result$model <- final_model
  if (!is.null(final_training)) {
    result$metrics <- final_training$performance_relevance_data %||% NULL
    result$history <- final_training$predicted_outputAndTime %||% NULL
  }
  
  per_seed_tables <- lapply(runs, function(run) {
    preds <- run$predictions
    if (!is.list(preds) || !length(preds)) {
      return(NULL)
    }
    keep <- vapply(preds, function(x) is.data.frame(x) && NROW(x), logical(1))
    if (!any(keep)) {
      return(NULL)
    }
    preds[keep]
  })
  per_seed_tables <- Filter(Negate(is.null), per_seed_tables)
  
  predictions_payload <- list()
  if (!is.null(prediction_matrix)) {
    predictions_payload$per_seed_raw <- main_seed_predictions
    predictions_payload$aggregate <- main_aggregate
    result$`.__prediction_matrix` <- prediction_matrix
  }
  if (length(per_seed_tables)) {
    predictions_payload$per_seed_tables <- per_seed_tables
  }
  if (length(predictions_payload)) {
    result$predictions <- predictions_payload
  }
  
  if (length(temp_summary)) {
    result$temp_predictions <- temp_summary
  }
  
  class(result) <- unique(c("ddesonn_run_result", class(result)))
  
  if (!is.null(output_root)) {
    .persist_ddesonn_run(result, output_root = output_root, save_models = save_models)
  }
  if (!is.null(result$`.__prediction_matrix`)) {
    result$`.__prediction_matrix` <- NULL
  }
  
  test_metrics <- NULL
  if (!is.null(test_matrix) && !is.null(test_labels) && length(runs)) {
    final_model <- final_model %||% (runs[[length(runs)]]$main$model %||% NULL)
    summary_threshold <- NULL
    if (!is.null(final_training) && is.list(final_training$performance_relevance_data)) {
      summary_threshold <- final_training$performance_relevance_data$threshold %||% NULL
      
      thr_num <- suppressWarnings(as.numeric(summary_threshold))                             
      summary_threshold <- if (length(thr_num) == 1L && is.finite(thr_num)) thr_num else NULL
    }
    test_threshold <- summary_threshold %||% threshold
    test_metrics_result <- .compute_test_metrics(
      model = final_model,
      x_test = test_matrix,
      y_test = test_labels,
      classification_mode = classification_mode,
      cfg = base_train_overrides,
      threshold = test_threshold
    )
    if (is.list(test_metrics_result) && isTRUE(test_metrics_result$ok)) {
      test_metrics <- test_metrics_result
      result$test_metrics <- test_metrics
      if (!is.null(final_training) && is.list(final_training$predicted_outputAndTime)) {
        final_training$predicted_outputAndTime$test_loss <- test_metrics$loss
        final_training$predicted_outputAndTime$test_accuracy <- test_metrics$accuracy
        final_training$predicted_outputAndTime$test_loss_type <- test_metrics$loss_type
        if (!is.null(final_model)) {
          final_model$last_training <- final_training
        }
        result$history <- final_training$predicted_outputAndTime
      }
    } else {
      test_metrics <- NULL
      ddesonn_console_log(                                                                   
        sprintf("Test metrics unavailable: %s", test_metrics_result$reason %||% "unknown error"),  
        level = "important",                                                                  
        verbose = verbose,                                                                    
        verboseLow = verboseLow                                                               
      )                                                                                       
    }
  }
  if (!is.null(final_training)) {
    .emit_final_run_summary(
      pred_summary_final = final_training$predicted_outputAndTime,
      performance_relevance_data = final_training$performance_relevance_data,
      cfg = base_train_overrides,
      mode = tolower(classification_mode),
      test_metrics = test_metrics
    )
  }
  if (!is.null(final_model) && identical(tolower(classification_mode), "binary")) {
    
    dec_out <- base_train_overrides$final_summary_decimals %||% NULL                        
    
    thr_used <- threshold %||%
      attr(final_model, "chosen_threshold") %||% final_model$chosen_threshold %||%
      attr(final_model, "threshold") %||%
      .ddesonn_threshold_default("binary")
    
    train_probs <- try(ddesonn_predict(final_model, x_matrix, aggregate = aggregate, type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
    if (!inherits(train_probs, "try-error") && !is.null(train_probs$prediction)) {
      y_train_vec <- .coerce_binary_labels(y_matrix)
      .emit_binary_classification_report(
        "Train", y_train_vec, train_probs$prediction, thr_used,
        final_summary_decimals = dec_out,
        viewTables = isTRUE(base_train_overrides$viewTables)
      )
    }
    if (!is.null(validation_data) && !is.null(validation_data$x)) {
      val_probs <- try(ddesonn_predict(final_model, validation_data$x, aggregate = aggregate, type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
      if (!inherits(val_probs, "try-error") && !is.null(val_probs$prediction)) {
        y_val_vec <- .coerce_binary_labels(validation_data$y)
        .emit_binary_classification_report(
          "Validation", y_val_vec, val_probs$prediction, thr_used,
          final_summary_decimals = dec_out,
          viewTables = isTRUE(base_train_overrides$viewTables)
        )
      }
    }
    if (!is.null(test_matrix) && !is.null(test_labels)) {
      test_probs <- try(ddesonn_predict(final_model, test_matrix, aggregate = aggregate, type = "response", verbose = verbose, verboseLow = verboseLow, debug = debug), silent = TRUE)  
      if (!inherits(test_probs, "try-error") && !is.null(test_probs$prediction)) {
        y_test_vec <- .coerce_binary_labels(test_labels)
        .emit_binary_classification_report(
          "Test", y_test_vec, test_probs$prediction, thr_used,
          final_summary_decimals = dec_out,
          viewTables = isTRUE(base_train_overrides$viewTables)
        )
      }
    }
  }
  
  result
}




#' Print a summary of a DDESONN run result
#'
#' @param x A \code{ddesonn_run_result} object.
#' @param ... Unused.
#' @return \code{x}, invisibly.
#' @method print ddesonn_run_result
#' @export
print.ddesonn_run_result <- function(x, ...) {
  cfg <- x$configuration %||% list()
  cat("DDESONN run result\n")
  if (length(cfg)) {
    cat(sprintf("  Mode: %s\n", cfg$classification_mode %||% "unknown"))
    if (!is.null(cfg$hidden_sizes)) {
      cat(sprintf("  Hidden sizes: %s\n", paste(cfg$hidden_sizes, collapse = ", ")))
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
