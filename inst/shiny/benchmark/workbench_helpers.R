# Model-workbench helpers. Training mathematics stays in the public package API.
wb_seed <- function(seed, code) {
  present <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (present) old <- get(".Random.seed", envir = .GlobalEnv)
  on.exit(if (present) assign(".Random.seed", old, envir = .GlobalEnv) else
    if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
      rm(".Random.seed", envir = .GlobalEnv))
  set.seed(seed)
  force(code)
}

wb_prepare <- function(data, target, features, mode, seed = 42L,
                       train_fraction = .6, validation_fraction = .2,
                       standardize = TRUE, split = "random") {
  if (!is.data.frame(data) || nrow(data) < 10 || anyDuplicated(names(data)))
    stop("Use at least 10 rows and unique column names.")
  if (!mode %in% c("binary", "multiclass", "regression")) stop("Choose a supported task.")
  if (length(target) != 1 || !target %in% names(data) || !length(features) ||
      any(!features %in% names(data)) || target %in% features || anyDuplicated(features))
    stop("Choose a target and at least one distinct predictor.")
  x <- data[features]
  if (!all(vapply(x, function(v) is.numeric(v) && all(is.finite(v)), logical(1))))
    stop("Predictors must be numeric and finite. Encode categories and handle missing values first.")
  y <- data[[target]]
  if (anyNA(y) || any(trimws(as.character(y)) == "")) stop("The target contains missing values.")
  if (mode == "regression" && (!is.numeric(y) || any(!is.finite(y))))
    stop("Regression requires a finite numeric target.")
  if (mode != "regression" && is.numeric(y) && any(!is.finite(y))) stop("Target values must be finite.")
  if (!is.finite(seed) || seed < 0 || seed > 2147483646 || seed %% 1 != 0)
    stop("Seed must be an integer between 0 and 2147483646.")
  if (!is.finite(train_fraction) || !is.finite(validation_fraction) ||
      train_fraction < .1 || validation_fraction < .05 ||
      train_fraction + validation_fraction > .95) stop("Leave at least 5% for testing and validation.")
  if (!split %in% c("random", "ordered")) stop("Choose random or ordered splitting.")
  groups <- if (mode == "regression" || split == "ordered") list(seq_len(nrow(data))) else
    unname(split(seq_len(nrow(data)), as.character(y)))
  # Random classification splits are stratified. Ordered splitting preserves time order.
  ids <- wb_seed(seed, {
    out <- list(train = integer(), validation = integer(), test = integer())
    for (g in groups) {
      if (length(g) < 5) stop("Each class needs at least five rows for this split.")
      if (split == "random") g <- sample(g, length(g))
      a <- max(1L, floor(length(g) * train_fraction))
      b <- max(1L, floor(length(g) * validation_fraction))
      if (a + b >= length(g)) stop("Split leaves no test rows.")
      out$train <- c(out$train, g[seq_len(a)])
      out$validation <- c(out$validation, g[a + seq_len(b)])
      out$test <- c(out$test, g[seq.int(a + b + 1L, length(g))])
    }
    out
  })
  classes <- if (mode != "regression") sort(unique(as.character(y[ids$train]))) else NULL
  if (mode == "binary" && length(classes) != 2) stop("Binary classification needs exactly two training classes.")
  if (mode == "multiclass" && (length(classes) < 2 || length(classes) > 100))
    stop("Multiclass classification needs 2–100 training classes.")
  if (mode != "regression" && any(!as.character(y) %in% classes))
    stop("Validation/test contains a class absent from training. Change the split.")
  x <- as.matrix(x)
  center <- if (standardize) colMeans(x[ids$train, , drop = FALSE]) else rep(0, ncol(x))
  scale <- if (standardize) apply(x[ids$train, , drop = FALSE], 2, stats::sd) else rep(1, ncol(x))
  scale[!is.finite(scale) | scale == 0] <- 1
  x <- sweep(sweep(x, 2, center, "-"), 2, scale, "/")
  encode <- function(z) {
    if (mode == "regression") return(matrix(as.numeric(z), ncol = 1))
    k <- match(as.character(z), classes)
    if (mode == "binary") return(matrix(k - 1, ncol = 1))
    m <- matrix(0, nrow = length(k), ncol = length(classes))
    m[cbind(seq_along(k), k)] <- 1
    m
  }
  parts <- lapply(ids, function(i) list(x = x[i, , drop = FALSE], y = encode(y[i]),
    actual = y[i], row = i))
  list(parts = parts, features = features, target = target, mode = mode,
    classes = classes, center = center, scale = scale, seed = seed,
    split = split, standardize = standardize, rows = nrow(data))
}

wb_config <- function(hidden = "16,8", epochs = 50, lr = .01,
                      optimizer = "adagrad", lambda = 0, reg = "none") {
  h <- if (!nzchar(trimws(hidden))) integer() else suppressWarnings(as.numeric(strsplit(hidden, ",", fixed = TRUE)[[1]]))
  if (length(h) > 5 || any(!is.finite(h) | h < 1 | h > 512 | h %% 1 != 0))
    stop("Use up to five hidden layers, each 1–512 units, separated by commas; leave blank for a single-layer model.")
  if (length(epochs) != 1 || !is.finite(epochs) || epochs < 1 || epochs > 5000 || epochs %% 1 != 0)
    stop("Choose 1–5,000 epochs.")
  if (!is.finite(lr) || lr <= 0 || lr > 1) stop("Learning rate must be greater than zero and at most 1.")
  if (!optimizer %in% c("adagrad", "adam", "rmsprop", "sgd")) stop("Unsupported optimizer.")
  if (!is.finite(lambda) || lambda < 0 || !reg %in% c("none", "L1", "L2")) stop("Invalid regularization.")
  list(hidden = as.integer(h), epochs = as.integer(epochs), lr = lr,
    optimizer = optimizer, lambda = lambda, reg = reg)
}

wb_predictions <- function(model, prepared, part) {
  p <- as.matrix(DDESONN::ddesonn_predict(model, part$x)$prediction)
  expected <- if (prepared$mode == "multiclass") length(prepared$classes) else 1L
  if (!identical(dim(p), c(nrow(part$x), as.integer(expected))) || any(!is.finite(p)))
    stop("The model returned invalid predictions.")
  if (prepared$mode != "regression" && any(p < 0 | p > 1)) stop("Classification probabilities are outside 0–1.")
  p
}

wb_evaluate <- function(prepared, part, p) {
  if (prepared$mode == "regression") {
    y <- as.numeric(part$y); pred <- as.numeric(p)
    metrics <- data.frame(Metric = c("RMSE", "MAE", "R squared"), Value = c(
      sqrt(mean((y - pred)^2)), mean(abs(y - pred)),
      if (sum((y - mean(y))^2) > 0) 1 - sum((y - pred)^2) / sum((y - mean(y))^2) else NA_real_))
    return(list(metrics = metrics, predictions = data.frame(row = part$row, actual = part$actual, prediction = pred), confusion = NULL))
  }
  if (prepared$mode == "binary") {
    probabilities <- cbind(1 - p[, 1], p[, 1])
    chosen <- ifelse(p[, 1] >= .5, 2L, 1L)
  } else {
    probabilities <- p
    if (any(abs(rowSums(p) - 1) > 1e-5)) stop("Multiclass probabilities do not sum to one.")
    chosen <- max.col(p, ties.method = "first")
  }
  actual <- match(as.character(part$actual), prepared$classes)
  correct_prob <- probabilities[cbind(seq_along(actual), actual)]
  truth <- matrix(0, nrow = length(actual), ncol = length(prepared$classes))
  truth[cbind(seq_along(actual), actual)] <- 1
  brier <- if (prepared$mode == "binary") mean((p[, 1] - part$y[, 1])^2) else mean(rowSums((probabilities - truth)^2))
  metrics <- data.frame(Metric = c("Accuracy", "Log loss", "Brier score"), Value = c(
    mean(actual == chosen), -mean(log(pmax(correct_prob, 1e-15))), brier))
  confusion <- table(Actual = factor(prepared$classes[actual], levels = prepared$classes),
    Predicted = factor(prepared$classes[chosen], levels = prepared$classes))
  pred <- data.frame(row = part$row, actual = part$actual, predicted = prepared$classes[chosen])
  prob <- as.data.frame(probabilities); names(prob) <- paste0("probability_", prepared$classes)
  list(metrics = metrics, predictions = cbind(pred, prob), confusion = as.data.frame.matrix(confusion))
}

wb_run <- function(prepared, config, previous = NULL) {
  # callr isolates RNG, package globals, graphics and all mutable model state.
  set.seed(prepared$seed)
  model <- if (is.null(previous)) DDESONN::ddesonn_model(
    input_size = length(prepared$features),
    output_size = if (prepared$mode == "multiclass") length(prepared$classes) else 1,
    hidden_sizes = config$hidden, classification_mode = prepared$mode,
    ML_NN = length(config$hidden) > 0, lambda = config$lambda,
    num_networks = 1) else previous$model
  tr <- prepared$parts$train; va <- prepared$parts$validation
  DDESONN::ddesonn_fit(model, tr$x, tr$y, validation = va,
    num_epochs = config$epochs, optimizer = config$optimizer, lr = config$lr,
    lr_decay_rate = 1, lr_min = config$lr, self_org = FALSE,
    reg_type = if (config$reg == "none") NULL else config$reg,
    dropout_rates = NULL, numeric_columns = NULL, batch_normalize_data = FALSE,
    validation_metrics = FALSE, best_weights_on_latest_weights_off = FALSE,
    viewTables = FALSE, verboseLow = TRUE)
  evaluation <- lapply(prepared$parts, function(part)
    wb_evaluate(prepared, part, wb_predictions(model, prepared, part)))
  list(format = "DDESONN-workbench-1", package_version = as.character(utils::packageVersion("DDESONN")),
    model = model, prepared = prepared, config = config, evaluation = evaluation,
    completed = format(Sys.time(), tz = "UTC", usetz = TRUE),
    fit_runs = if (is.null(previous)) 1L else previous$fit_runs + 1L)
}

wb_validate_bundle <- function(bundle) {
  if (!is.list(bundle) || !identical(bundle$format, "DDESONN-workbench-1") ||
      !inherits(bundle$model, "ddesonn_model") || !is.list(bundle$prepared) ||
      !is.list(bundle$config) || !all(c("train", "validation", "test") %in% names(bundle$evaluation)))
    stop("Load a completed model bundle saved by this workbench.")
  p <- bundle$prepared
  if (!p$mode %in% c("binary", "multiclass", "regression") ||
      !length(p$features) || length(p$center) != length(p$features) ||
      length(p$scale) != length(p$features) || any(!is.finite(p$center)) ||
      any(!is.finite(p$scale) | p$scale <= 0)) stop("The saved preprocessing is invalid.")
  # Exercise the model rather than trusting a stale saved evaluation.
  for (part in names(p$parts)) bundle$evaluation[[part]] <- wb_evaluate(p, p$parts[[part]],
    wb_predictions(bundle$model, p, p$parts[[part]]))
  bundle
}

wb_new_predictions <- function(bundle, data) {
  p <- bundle$prepared
  if (!is.data.frame(data) || !nrow(data) || anyDuplicated(names(data)) ||
      any(!p$features %in% names(data))) stop("Prediction data must include every saved predictor, with unique column names.")
  x <- data[p$features]
  if (!all(vapply(x, function(v) is.numeric(v) && all(is.finite(v)), logical(1))))
    stop("Prediction features must be finite numeric values.")
  x <- sweep(sweep(as.matrix(x), 2, p$center, "-"), 2, p$scale, "/")
  predictions <- wb_predictions(bundle$model, p, list(x = x))
  if (p$mode == "regression") return(data.frame(row = seq_len(nrow(x)), prediction = predictions[, 1]))
  probs <- if (p$mode == "binary") cbind(1 - predictions[, 1], predictions[, 1]) else predictions
  chosen <- if (p$mode == "binary") ifelse(predictions[, 1] >= .5, 2L, 1L) else max.col(probs, ties.method = "first")
  out <- as.data.frame(probs); names(out) <- paste0("probability_", p$classes)
  cbind(data.frame(row = seq_len(nrow(x)), predicted = p$classes[chosen]), out)
}

wb_log_history <- function(path) {
  lines <- if (file.exists(path)) readLines(path, warn = FALSE) else character()
  lines <- lines[grepl("Epoch [0-9]+ .*Loss: [-+0-9.eE]+", lines)]
  data.frame(epoch = as.integer(sub(".*Epoch ([0-9]+).*", "\\1", lines)),
    loss = as.numeric(sub(".*Loss: ([-+0-9.eE]+).*", "\\1", lines)))
}

wb_start <- function(helper, prepared, config, previous = NULL) {
  dir <- tempfile("ddesonn-session-"); dir.create(dir)
  log <- file.path(dir, "training.log")
  error <- file.path(dir, "stderr.log")
  process <- tryCatch(callr::r_bg(function(helper, prepared, config, previous) {
    env <- new.env(parent = globalenv()); sys.source(helper, envir = env)
    env$wb_run(prepared, config, previous)
  }, args = list(helper = helper, prepared = prepared, config = config, previous = previous),
    libpath = .libPaths(), wd = dir, stdout = log, stderr = error, supervise = TRUE),
    error = function(e) { unlink(dir, recursive = TRUE); stop(e) })
  list(process = process, dir = dir, log = log, error = error,
    started = Sys.time(), epochs = config$epochs)
}

wb_stop <- function(job) {
  if (!is.null(job)) {
    if (job$process$is_alive()) {
      job$process$kill_tree()
      job$process$wait(5000)
      if (job$process$is_alive()) {
        job$process$kill()
        job$process$wait(1000)
      }
      if (job$process$is_alive()) stop("The training worker has not stopped. Try Cancel again.")
    }
    unlink(job$dir, recursive = TRUE)
  }
  invisible(NULL)
}
