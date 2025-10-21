source("R/utils.R")

#' Internal package environment used to lazily load the legacy DDESONN stack.
.ddesonn_env <- new.env(parent = .GlobalEnv)

#' Null-coalescing helper used across the high-level API.
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

.ddesonn_find_root <- function() {
  pkg_root <- system.file(package = "DDESONN")
  if (nzchar(pkg_root)) return(pkg_root)
  getOption("DDESONN_ROOT", default = getwd())  # should be the *repo root*
}

.ddesonn_source_legacy <- function() {
  if (exists("DDESONN", envir = .ddesonn_env, inherits = FALSE)) {
    return(invisible(.ddesonn_env))
  }
  
  root <- .ddesonn_find_root()
  
  # now expect DDESONN.R inside R/
  target <- file.path(root, "R", "DDESONN.R")
  if (!file.exists(target)) {
    stop("Unable to locate 'R/DDESONN.R'. Set options(DDESONN_ROOT=...) to the repository root before calling the API.")
  }
  
  base_source <- base::source
  assign(
    "source",
    function(file, ...) {
      # resolve to R/<file> first, then fallback to root/<file>
      cand1 <- file.path(root, "R", file)
      cand2 <- file.path(root, file)
      resolved <- if (file.exists(cand1)) cand1 else cand2
      if (!file.exists(resolved)) {
        stop(sprintf("Unable to locate dependency file '%s' under '%s' or '%s'",
                     file, file.path(root, "R"), root), call. = FALSE)
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

.get_output_head_from_mode <- function(mode, num_classes = NULL) {
  mode <- tolower(mode %||% "")
  .ddesonn_source_legacy()
  
  fetch_activation <- function(name) {
    fn <- get0(name, envir = .ddesonn_env, inherits = TRUE)
    if (!is.function(fn)) {
      fn <- get0(name, envir = baseenv(), inherits = TRUE)
    }
    if (!is.function(fn)) {
      stop(sprintf("Activation '%s' is not available.", name), call. = FALSE)
    }
    if (is.null(attr(fn, "name"))) attr(fn, "name") <- name
    fn
  }
  
  res <- switch(
    mode,
    binary = list(act = fetch_activation("sigmoid"), loss = "bce"),
    multiclass = list(act = fetch_activation("softmax"), loss = "cce"),
    regression = list(act = fetch_activation("identity"), loss = "mse"),
    stop(sprintf("Unsupported mode '%s'", mode), call. = FALSE)
  )
  
  res$num_classes <- if (!is.null(num_classes)) as.integer(num_classes) else NULL
  res
}

.loss_name_to_training <- function(loss_name) {
  if (is.null(loss_name) || !length(loss_name)) return(NULL)
  key <- tolower(as.character(loss_name[[1]]))
  switch(
    key,
    bce = "CrossEntropy",
    "binary_crossentropy" = "CrossEntropy",
    cce = "CategoricalCrossEntropy",
    "categorical_crossentropy" = "CategoricalCrossEntropy",
    mse = "MSE",
    mae = "MAE",
    loss_name
  )
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
  
  defaults <- list(
    lr = 0.125,
    lr_decay_rate = 0.5,
    lr_decay_epoch = 20L,
    lr_min = 1e-5,
    num_epochs = 3L,
    self_org = FALSE,
    threshold = .ddesonn_threshold_default(mode),
    reg_type = "L1",
    lambda = NA_real_,
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
  
  if (identical(mode, "regression")) {
    len_hidden <- length(hidden_sizes %||% integer())
    defaults$lr <- 1e-3
    defaults$lr_decay_rate <- 0
    defaults$lr_decay_epoch <- 0L
    defaults$lr_min <- 0
    defaults$num_epochs <- max(as.integer(defaults$num_epochs), 50L)
    defaults$reg_type <- "none"
    defaults$lambda <- 0
    defaults$dropout_rates <- if (len_hidden) as.list(rep(0, len_hidden)) else list()
    defaults$optimizer <- "adam"
  }
  
  defaults
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
    init_method = init_method,
    custom_scale = custom_scale
  )
  
  attr(model, "classification_mode") <- classification_mode
  attr(model, "activation_functions") <- activation_functions
  attr(model, "activation_functions_predict") <- activation_functions_predict
  attr(model, "hidden_sizes") <- hidden_sizes
  attr(model, "lambda") <- lambda
  attr(model, "ML_NN") <- ML_NN
  attr(model, "output_size") <- output_size
  
  head_info <- .get_output_head_from_mode(classification_mode, output_size)
  model$output_activation <- head_info$act
  model$loss_name <- head_info$loss
  attr(model, "output_activation") <- head_info$act
  attr(model, "loss_name") <- head_info$loss
  model$target_transform <- NULL
  attr(model, "target_transform") <- NULL
  
  if (length(model$ensemble)) {
    for (net in model$ensemble) {
      try(net$output_activation <- head_info$act, silent = TRUE)
      try(net$loss_name <- head_info$loss, silent = TRUE)
    }
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
  
  overrides <- list(...)  # <-- move earlier so we can use it for mode
  # 1) Resolve mode with explicit override first, then model attr, then default
  mode <- tolower(overrides$classification_mode %||% attr(model, "classification_mode") %||% "binary")
  hidden_sizes <- attr(model, "hidden_sizes") %||% NULL
  
  model_mode <- attr(model, "classification_mode") %||% "binary"
  if (!identical(model_mode, mode)) {
    attr(model, "classification_mode") <- mode
  }
  output_size <- attr(model, "output_size") %||%
    tryCatch(model$ensemble[[1]]$output_size, error = function(e) NULL)
  head_info <- .get_output_head_from_mode(mode, output_size)
  model$output_activation <- head_info$act
  model$loss_name <- head_info$loss
  attr(model, "output_activation") <- head_info$act
  attr(model, "loss_name") <- head_info$loss
  if (length(model$ensemble)) {
    for (net in model$ensemble) {
      try(net$output_activation <- head_info$act, silent = TRUE)
      try(net$loss_name <- head_info$loss, silent = TRUE)
    }
  }
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
  cfg$loss_type <- overrides$loss_type %||% .loss_name_to_training(head_info$loss) %||% cfg$loss_type
  if (mode == "regression") {
    cfg$threshold <- NA_real_
  }
  
  # 2) Threshold tuner only for binary; NULL otherwise (prevents downstream “tuned” bundles)
  if (identical(mode, "binary")) {
    cfg$threshold_function <- cfg$threshold_function %||% .ddesonn_get("tune_threshold_accuracy")
  } else {
    cfg$threshold_function <- NULL
  }
  
  cfg$ML_NN <- isTRUE(cfg$ML_NN %||% attr(model, "ML_NN"))
  cfg$ensemble_number <- overrides$ensemble_number %||% cfg$ensemble_number %||% 0L
  
  lambda_override <- overrides$lambda %||% cfg$lambda
  if (!is.null(lambda_override) && length(lambda_override) && is.finite(lambda_override)) {
    try(model$lambda <- lambda_override, silent = TRUE)
    attr(model, "lambda") <- lambda_override
    if (length(model$ensemble)) {
      for (net in model$ensemble) {
        try(net$lambda <- lambda_override, silent = TRUE)
      }
    }
  }
  
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
  
  target_transform <- NULL
  if (mode == "regression") {
    scaled_train <- maybe_scale_y(labels)
    labels <- scaled_train$values
    target_transform <- scaled_train$transform
    
    if (!is.null(cfg$y_validation)) {
      scaled_val <- maybe_scale_y(cfg$y_validation, transform = target_transform)
      cfg$y_validation <- scaled_val$values
    }
    
    preprocess <- cfg$preprocessScaledData %||% list()
    preprocess$target_transform <- target_transform
    preprocess$reg_target_mode <- target_transform$type %||% "identity"
    preprocess$reg_target_mode_applied <- !identical(preprocess$reg_target_mode, "identity")
    cfg$preprocessScaledData <- preprocess
    
    model$target_transform <- target_transform
    attr(model, "target_transform") <- target_transform
    meta_preprocess <- attr(model, "preprocess") %||% list()
    meta_preprocess$target_transform <- target_transform
    attr(model, "preprocess") <- meta_preprocess
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
    verbose = cfg$verbose
  )
  
  result <- do.call(model$train, train_args)
  model$last_training <- result
  attr(model, "threshold") <- cfg$threshold
  
  # =========================
  # Attach per-slot metrics
  # =========================
  if (mode %in% c("binary", "multiclass")) {
    thr_used <- cfg$threshold %||% .ddesonn_threshold_default(mode)
    
    # TRAIN predictions (per-member)
    pr_train <- try(ddesonn_predict(model, x, aggregate = "none", type = "response"), silent = TRUE)
    per_model_train <- if (!inherits(pr_train, "try-error")) pr_train$per_model else NULL
    
    # VALID predictions (per-member) if present
    per_model_valid <- NULL
    if (!is.null(validation)) {
      pr_valid <- try(ddesonn_predict(model, validation$x, aggregate = "none", type = "response"), silent = TRUE)
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
            # keep validation macro metrics merged into performance_metric if you want:
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
    pr_train <- try(ddesonn_predict(model, x, aggregate = "none", type = "response"), silent = TRUE)
    per_model_train <- if (!inherits(pr_train, "try-error")) pr_train$per_model else NULL
    
    # VALID predictions (per-member) if present
    per_model_valid <- NULL
    if (!is.null(validation)) {
      pr_valid <- try(ddesonn_predict(model, validation$x, aggregate = "none", type = "response"), silent = TRUE)
      per_model_valid <- if (!inherits(pr_valid, "try-error")) pr_valid$per_model else NULL
    }
    
    # true labels as numeric vectors
    y_train_vec <- as.numeric(labels[, 1])
    y_valid_vec <- if (!is.null(cfg$y_validation)) as.numeric(cfg$y_validation[, 1]) else NULL
    
    inverse_labels <- NULL
    if (is.list(target_transform)) {
      inverse_labels <- target_transform$invert %||% target_transform$inverse %||% target_transform$restore
    }
    if (is.function(inverse_labels)) {
      y_train_vec <- inverse_labels(y_train_vec)
      if (!is.null(y_valid_vec)) {
        y_valid_vec <- inverse_labels(y_valid_vec)
      }
    }
    
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
          slot_obj$metadata$performance_metric <- compute_regression_metrics(y_train_vec, pt)
        }
        
        # VALID metrics per-slot (optional)
        if (!is.null(per_model_valid) && length(per_model_valid) >= k && !is.null(y_valid_vec)) {
          pv <- as.numeric(per_model_valid[[k]][, 1])
          slot_obj$metadata$validation_metrics <- compute_regression_metrics(y_valid_vec, pv)
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
ddesonn_predict <- function(model, new_data,
                            aggregate = c("mean", "median", "none"),
                            type = c("response", "class"),
                            threshold = NULL) {
  if (!inherits(model, "ddesonn_model")) {
    stop("'model' must be created with ddesonn_model().", call. = FALSE)
  }
  aggregate <- match.arg(aggregate)
  type <- match.arg(type)
  
  X <- .as_numeric_matrix(new_data)
  mode <- attr(model, "classification_mode") %||% "binary"
  
  target_transform <- attr(model, "target_transform") %||%
    model$target_transform %||%
    ((attr(model, "preprocess") %||% list())$target_transform %||% NULL)
  inverse_transform <- NULL
  if (is.list(target_transform)) {
    inverse_transform <- target_transform$invert %||%
      target_transform$inverse %||%
      target_transform$restore
    if (!is.function(inverse_transform)) {
      center <- target_transform$params$center %||% 0
      scale  <- target_transform$params$scale %||% 1
      if (is.numeric(scale) && length(scale) == 1 && is.finite(scale) && !identical(scale, 0)) {
        inverse_transform <- function(v) center + as.numeric(v) * scale
      }
    }
  }
  
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
      verbose = FALSE,
      debug   = FALSE
    )
    
    out <- res$predicted_output %||% res$prediction %||% res
    mat <- .as_numeric_matrix(out)
    storage.mode(mat) <- "double"
    mat
  })
  
  # aggregate
  aggregated <-
    switch(aggregate,
           mean   = Reduce(`+`, preds) / length(preds),
           median = apply(array(unlist(preds), dim = c(nrow(preds[[1]]), ncol(preds[[1]]), length(preds))),
                          c(1, 2), stats::median),
           none   = preds[[1]]
    )
  
  if (identical(mode, "regression")) {
    preds <- lapply(preds, function(mat) {
      mat_num <- matrix(as.numeric(mat), nrow = nrow(mat), ncol = ncol(mat))
      if (is.function(inverse_transform)) {
        mat_num <- matrix(inverse_transform(as.numeric(mat_num)), nrow = nrow(mat_num), ncol = ncol(mat_num))
      }
      mat_num
    })
    
    agg_nrow <- nrow(aggregated)
    agg_ncol <- ncol(aggregated)
    if (is.null(agg_nrow) || is.null(agg_ncol)) {
      agg_len <- length(aggregated)
      agg_nrow <- agg_len
      agg_ncol <- 1L
    }
    aggregated <- matrix(as.numeric(aggregated), nrow = agg_nrow, ncol = agg_ncol)
    if (is.function(inverse_transform)) {
      aggregated <- matrix(inverse_transform(as.numeric(aggregated)), nrow = agg_nrow, ncol = agg_ncol)
    }
    if (aggregate == "none") {
      aggregated <- preds[[1]]
    }
  }
  
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
      str(slot_obj)
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
    # classification-first (kept)
    "accuracy", "precision", "recall", "f1", "f1_score",
    # add regression here:
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
  if (ncol(df_test)) df_test <- df_test[, ord(df_test), drop = FALSE]
  
  saveRDS(df_test, pretty_test_path)
  saveRDS(df_test, test_path)
  saveRDS(df_train, train_path)
}

.write_ensemble_runs_metrics <- function(result, run_dir, ts, seeds) {
  s_chr <- as.character(length(seeds))
  pretty_test_path <- file.path(run_dir, sprintf("Ensemble_Pretty_Test_Metrics_%s_seeds_%s.rds", s_chr, ts))
  test_path        <- file.path(run_dir, sprintf("Ensemble_Test_Metrics_%s_seeds_%s.rds", s_chr, ts))
  train_path       <- file.path(run_dir, sprintf("Ensemble_Train_Acc_Val_Metrics_%s_seeds_%s.rds", s_chr, ts))
  
  rows_train <- list()
  rows_test  <- list()
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
      md$model_serial_num <- md$model_serial_num %||% sprintf("1.main.%d", k)
      md$model_name       <- md$model_name %||% paste0("model_", k)
      
      # Train metrics
      ptr_tr <- ptr_tr + 1L
      rows_train[[ptr_tr]] <- .build_metrics_row(
        md, run_index = i, seed = seed_i, slot = k, split = "train"
      )
      
      # Test metrics
      ptr_te <- ptr_te + 1L
      rows_test[[ptr_te]] <- .build_metrics_row(
        md, run_index = i, seed = seed_i, slot = k, split = "test"
      )
    }
  }
  
  bind <- function(lst) if (!length(lst)) data.frame() else do.call(rbind, lst)
  df_train <- bind(rows_train)
  df_test  <- bind(rows_test)
  
  id_order <- c("run_index", "seed", "model_slot", "MODEL_SLOT", "split", "serial", "model_name")
  metric_pref <- c(
    "accuracy", "precision", "recall", "f1", "f1_score", "auc", "balanced_accuracy",
    "specificity", "sensitivity", "logloss", "brier",
    "MSE", "MAE", "RMSE", "R2", "MAPE", "SMAPE", "WMAPE", "MASE",
    "confusion_matrix.TP", "confusion_matrix.FP", "confusion_matrix.TN", "confusion_matrix.FN",
    "generalization_ability", "speed", "speed_learn1", "speed_learn2",
    "memory_usage", "robustness", "hit_rate", "ndcg", "diversity", "serendipity",
    "best_train_acc", "best_epoch_train", "best_train_loss", "best_epoch_train_loss", "best_val_acc", "best_val_epoch", "best_val_prediction_time"
  )
  
  ord <- function(df) c(
    intersect(id_order, names(df)),
    intersect(metric_pref, names(df)),
    setdiff(names(df), c(id_order, metric_pref))
  )
  
  if (ncol(df_train)) df_train <- df_train[, ord(df_train), drop = FALSE]
  if (ncol(df_test))  df_test  <- df_test[,  ord(df_test),  drop = FALSE]
  
  saveRDS(df_test,  pretty_test_path)
  saveRDS(df_test,  test_path)
  saveRDS(df_train, train_path)
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
  
  # choose a stable base: repo root → artifacts → (SingleRuns|EnsembleRuns)
  # tip: pass output_root = .ddesonn_find_root()
  art_root <- {
    nr <- normalizePath(output_root, winslash = "/", mustWork = FALSE)
    if (basename(nr) == "artifacts") nr else file.path(output_root, "artifacts")
  }
  run_dir <- file.path(art_root, root_dir, run_tag)
  
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
  
  if (isTRUE(cfg$do_ensemble)) {
    .write_ensemble_runs_metrics(result, run_dir, ts, seeds)
  } else {
    .write_single_runs_metrics(result, run_dir, ts, seeds)
  }
  
  if (isTRUE(cfg$do_ensemble)) {
    .write_agg_predictions(result, run_dir, ts, seeds)
    .write_temp_agg_predictions(result, run_dir, ts, seeds)
    .write_fused_consensus(result, run_dir, ts, seeds)
  }
  
  logs_dir <- file.path(run_dir, "logs")
  dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_along(seeds)) {
    seed_i <- seeds[[i]]
    paths <- c(
      movement = file.path(logs_dir, sprintf("movement_log_run%03d_seed%s_%s.rds", i, seed_i, ts)),
      change   = file.path(logs_dir, sprintf("change_log_run%03d_seed%s_%s.rds",   i, seed_i, ts)),
      main     = file.path(logs_dir, sprintf("main_log_run%03d_seed%s_%s.rds",     i, seed_i, ts))
    )
    for (p in paths) {
      if (!file.exists(p)) saveRDS(data.frame(), p)
    }
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
    vars <- grep("^Ensemble_Main_(0|1)_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
    if (!length(vars)) return(character())
    ord <- suppressWarnings(as.integer(sub("^Ensemble_Main_(?:0|1)_model_(\\d+)_metadata$", "\\1", vars)))
    vars <- vars[order(ord)]
    vapply(vars, function(v) {
      md <- get(v, envir = .GlobalEnv)
      as.character(md$model_serial_num %||% NA_character_)
    }, character(1))
  }
  
  get_metric_by_serial <- function(serial, metric_name) {
    vars <- grep(
      "^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
      ls(.GlobalEnv), value = TRUE
    )
    if (!length(vars)) return(NA_real_)
    for (v in vars) {
      md <- get(v, envir = .GlobalEnv)
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
    vars <- grep(sprintf("^Ensemble_Temp_%d_model_\\d+_metadata$", e), ls(.GlobalEnv), value = TRUE)
    if (!length(vars)) return(character())
    ord <- suppressWarnings(as.integer(sub(sprintf("^Ensemble_Temp_%d_model_(\\d+)_metadata$", e), "\\1", vars)))
    vars <- vars[order(ord)]
    vapply(vars, function(v) {
      md <- get(v, envir = .GlobalEnv)
      s <- md$model_serial_num
      if (!is.null(s) && nzchar(as.character(s))) as.character(s) else NA_character_
    }, character(1))
  }
  
  empty_log_tables <- function() {
    list(
      main_log = data.frame(
        iteration = integer(), phase = character(), slot = integer(),
        serial = character(), metric_name = character(),
        metric_value = numeric(), message = character(),
        timestamp = as.POSIXct(character()), stringsAsFactors = FALSE
      ),
      movement_log = data.frame(
        iteration = integer(), phase = character(), slot = integer(),
        role = character(), serial = character(), metric_name = character(),
        metric_value = numeric(), message = character(),
        timestamp = as.POSIXct(character()), stringsAsFactors = FALSE
      ),
      change_log = data.frame(
        iteration = integer(), role = character(), serial = character(),
        metric_name = character(), metric_value = numeric(),
        message = character(), timestamp = as.POSIXct(character()),
        stringsAsFactors = FALSE
      )
    )
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
    if (exists(temp_env, envir = .GlobalEnv)) {
      md <- get(temp_env, envir = .GlobalEnv)
      md$model_serial_num <- best_serial
      assign(main_env, md, envir = .GlobalEnv)
    }
    list(model = main_model, slot = as.integer(worst_slot), serial = best_serial)
  }
  
  # Per-seed main runs
  runs <- lapply(seq_along(seeds), function(i) {
    set.seed(seeds[[i]])
    if (isTRUE(do_ensemble)) {
      vars <- grep(
        "^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
        ls(.GlobalEnv), value = TRUE
      )
      if (length(vars)) rm(list = vars, envir = .GlobalEnv)
    }
    
    log_tables <- empty_log_tables()
    
    main_model_args <- base_model_args
    if (isTRUE(do_ensemble)) {
      main_model_args$ensemble_number <- 1L
    }
    
    mdl <- do.call(ddesonn_model, main_model_args)
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
    temp_models <- vector("list", length = if (isTRUE(do_ensemble)) num_temp_iterations else 0L)
    if (isTRUE(do_ensemble) && num_temp_iterations > 0L) {
      tmp_overrides <- temp_overrides %||% base_train_overrides
      temp_list <- vector("list", length = num_temp_iterations)
      for (iter in seq_len(num_temp_iterations)) {
        log_tables <- record_main_snapshot(log_tables, iteration = iter, phase = "main_before")
        # TEMP iteration: clone model or reuse with potential tweaks
        temp_model_args <- base_model_args
        temp_model_args$ensemble_number <- iter + 1L
        tmp_model <- do.call(ddesonn_model, temp_model_args)
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
        temp_models[[iter]] <- tmp_model
        
        removed <- prune_network_from_main(mdl, target_metric)
        added <- list(model = mdl, slot = NULL, serial = NULL)
        if (!is.null(removed)) {
          added <- add_network_to_main(mdl, tmp_model, iteration_index = iter, target_metric_name = target_metric, worst_slot = removed$worst_slot)
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
        preds <- ddesonn_predict(mdl, prediction_matrix, aggregate = aggregate, type = prediction_type, threshold = threshold)
        main_pred <- preds
      }
    }
    
    list(
      seed = seeds[[i]],
      main = list(model = mdl, predictions = main_pred),
      temp_iterations = temp_summary,
      tables = log_tables
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
