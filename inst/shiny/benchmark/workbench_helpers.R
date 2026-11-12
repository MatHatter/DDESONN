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
    split = split, standardize = standardize, train_fraction = train_fraction,
    validation_fraction = validation_fraction, rows = nrow(data))
}

wb_config <- function(hidden = "16,8", epochs = 50, lr = .01,
                      optimizer = "adagrad", lambda = 0, reg = "none", self_org = FALSE, checkpoint = "best_validation") {
  h <- if (!nzchar(trimws(hidden))) integer() else suppressWarnings(as.numeric(strsplit(hidden, ",", fixed = TRUE)[[1]]))
  if (length(h) > 5 || any(!is.finite(h) | h < 1 | h > 512 | h %% 1 != 0))
    stop("Use up to five hidden layers, each 1–512 units, separated by commas; leave blank for a single-layer model.")
  if (length(epochs) != 1 || !is.finite(epochs) || epochs < 1 || epochs > 5000 || epochs %% 1 != 0)
    stop("Choose 1–5,000 epochs.")
  if (!is.finite(lr) || lr <= 0 || lr > 1) stop("Learning rate must be greater than zero and at most 1.")
  if (!optimizer %in% c("adagrad", "adam", "rmsprop", "sgd")) stop("Unsupported optimizer.")
  if (!is.finite(lambda) || lambda < 0 || !reg %in% c("none", "L1", "L2")) stop("Invalid regularization.")
  if (!is.logical(self_org) || length(self_org) != 1 || is.na(self_org))
    stop("Self-organization must be TRUE or FALSE.")
  checkpoint <- match.arg(checkpoint, c("best_validation", "final"))
  list(hidden = as.integer(h), epochs = as.integer(epochs), lr = lr,
    optimizer = optimizer, lambda = lambda, reg = reg, self_org = self_org, checkpoint = checkpoint)
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
  # Older workbench bundles predate the checkbox and used FALSE.
  config$self_org <- isTRUE(config$self_org)
  if (is.null(config$checkpoint)) config$checkpoint <- "final" # Older bundles used final weights.
  config$checkpoint <- match.arg(config$checkpoint, c("best_validation", "final"))
  use_best <- identical(config$checkpoint, "best_validation")
  model <- if (is.null(previous)) DDESONN::ddesonn_model(
    input_size = length(prepared$features),
    output_size = if (prepared$mode == "multiclass") length(prepared$classes) else 1,
    hidden_sizes = config$hidden, classification_mode = prepared$mode,
    ML_NN = length(config$hidden) > 0, lambda = config$lambda,
    num_networks = 1) else previous$model
  tr <- prepared$parts$train; va <- prepared$parts$validation
  # Warm starts preserve the learned organization instead of organizing again.
  self_org_applied <- is.null(previous) && config$self_org
  DDESONN::ddesonn_fit(model, tr$x, tr$y, validation = va,
    num_epochs = config$epochs, optimizer = config$optimizer, lr = config$lr,
    lr_decay_rate = 1, lr_min = config$lr, self_org = self_org_applied,
    reg_type = if (config$reg == "none") NULL else config$reg,
    dropout_rates = NULL, numeric_columns = NULL, batch_normalize_data = FALSE,
    validation_metrics = use_best, best_weights_on_latest_weights_off = use_best,
    threshold = .5,
    viewTables = FALSE, verboseLow = TRUE)
  selected_epoch <- config$epochs
  if (use_best) {
    network <- model$ensemble[[1]]
    summary <- model$last_training$predicted_outputAndTime
    selected_epoch <- if (prepared$mode == "regression") summary$best_val_epoch_loss else summary$best_val_epoch
    if (is.null(network$best_weights) || is.null(network$best_biases) ||
        length(selected_epoch) != 1 || !is.finite(selected_epoch))
      stop("No valid validation checkpoint was recorded; no completed model was replaced.")
    # Reporting a best checkpoint is not enough: inference must use its tensors.
    network$weights <- network$best_weights
    network$biases <- network$best_biases
  }
  evaluation <- lapply(prepared$parts, function(part)
    wb_evaluate(prepared, part, wb_predictions(model, prepared, part)))
  list(format = "DDESONN-workbench-1", package_version = as.character(utils::packageVersion("DDESONN")),
    model = model, prepared = prepared, config = config, self_org_applied = self_org_applied,
    evaluation = evaluation, selected_epoch = as.integer(selected_epoch),
    recipe = if (!is.null(previous$recipe)) paste(previous$recipe,
      "\n# Additional fit from the previously selected weights\n", wb_fit_code(prepared, config, FALSE),
      wb_evaluation_code(prepared, "test"), sep = "\n") else if (!is.null(previous))
      wb_resume_recipe(prepared, config) else wb_recipe(prepared, config),
    completed = format(Sys.time(), tz = "UTC", usetz = TRUE),
    fit_runs = if (is.null(previous)) 1L else previous$fit_runs + 1L)
}

# Scores describe the tensors actually used for inference, including final mode.
wb_checkpoint <- function(bundle) {
  regression <- identical(bundle$prepared$mode, "regression")
  metrics <- bundle$evaluation$validation$metrics
  score <- metrics$Value[match(if (regression) "RMSE" else "Accuracy", metrics$Metric)]
  list(mode = if (is.null(bundle$config$checkpoint)) "final" else bundle$config$checkpoint,
    epoch = bundle$selected_epoch, epochs = bundle$config$epochs,
    criterion = if (regression) "validation loss (MSE)" else "validation accuracy",
    score = if (regression) score^2 else score)
}

wb_checkpoint_summary <- function(bundle) {
  if (is.null(bundle)) return("No completed checkpoint yet.")
  checkpoint <- wb_checkpoint(bundle)
  score <- if (length(checkpoint$score) != 1 || !is.finite(checkpoint$score)) "not recorded" else
    if (identical(bundle$prepared$mode, "regression")) format(signif(checkpoint$score, 6), trim = TRUE) else
      paste0(format(round(100 * checkpoint$score, 2), trim = TRUE), "%")
  epoch <- if (length(checkpoint$epoch) == 1 && is.finite(checkpoint$epoch)) checkpoint$epoch else "not recorded"
  if (identical(checkpoint$mode, "best_validation"))
    sprintf("Epoch %s of %s · best %s: %s · Best validation checkpoint", epoch,
      checkpoint$epochs, checkpoint$criterion, score) else
    sprintf("Epoch %s of %s · final epoch · %s: %s", epoch,
      checkpoint$epochs, checkpoint$criterion, score)
}

wb_training_error <- function(error, stderr = character()) {
  details <- conditionMessage(error)
  guidance <- if (grepl("finite|NaN|Inf|invalid predictions|probabilities|checkpoint", details, ignore.case = TRUE))
    "Training could not produce a usable model. Try a smaller learning rate, enable standardization on Features and prepare again, or turn off initial self-organization." else
    if (grepl("colMeans|two dimensions|non-conformable|dimension", details, ignore.case = TRUE))
      "Training could not use this data shape. Try two or more numeric predictors; if this repeats, share the technical details when reporting the model-shape problem." else
    if (grepl("there is no package|not installed|could not start|failed to start|cannot open.*connection", details, ignore.case = TRUE))
      "The training worker could not finish. Try training again; if this repeats, check that DDESONN and the optional callr package are installed in the same R library." else
      "Training could not start or finish. Check the model settings (hidden sizes, epochs and learning rate), then try again. A smaller model or learning rate may help."
  list(message = guidance, details = paste(c(details, utils::tail(stderr, 40)), collapse = "\n"))
}

wb_validate_bundle <- function(bundle) {
  if (!is.list(bundle) || !identical(bundle$format, "DDESONN-workbench-1") ||
      !inherits(bundle$model, "ddesonn_model") || !is.list(bundle$prepared) ||
      !is.list(bundle$config) || !all(c("train", "validation", "test") %in% names(bundle$evaluation)))
    stop("Load a completed model bundle saved by this workbench.")
  if (is.null(bundle$config$checkpoint)) bundle$config$checkpoint <- "final"
  if (is.null(bundle$selected_epoch) && identical(bundle$config$checkpoint, "final"))
    bundle$selected_epoch <- bundle$config$epochs
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

# Public-API recipes use base R preprocessing; no private workbench helpers needed.
wb_literal <- function(value) paste(capture.output(dput(value)), collapse = "\n")

wb_data_code <- function(source = NULL) {
  if (identical(source$example, "iris")) "library(DDESONN)\ndata <- datasets::iris" else
    paste0("library(DDESONN)\ndata <- read.csv(",
      wb_literal(if (is.null(source$file)) "training-data.csv" else source$file),
      ", check.names = FALSE)\n# Use the original CSV with the same rows and row order.")
}

wb_features_code <- function(p) {
  header <- paste0("features <- ", wb_literal(p$features), "\ntarget <- ", wb_literal(p$target),
    "\nmode <- ", wb_literal(p$mode), "\nstopifnot(nrow(data) == ", p$rows,
    ", all(c(features, target) %in% names(data)))\n")
  if (!is.null(p$train_fraction) && !is.null(p$validation_fraction)) {
    split_code <- paste0("set.seed(", p$seed, ")\n",
      if (p$mode == "regression" || p$split == "ordered")
        "groups <- list(seq_len(nrow(data)))\n" else
        "groups <- unname(split(seq_len(nrow(data)), as.character(data[[target]])))\n",
      "split_rows <- list(train = integer(), validation = integer(), test = integer())\n",
      "for (g in groups) {\n",
      if (p$split == "random") "  g <- sample(g, length(g))\n" else "",
      "  a <- max(1L, floor(length(g) * ", wb_literal(p$train_fraction), "))\n",
      "  b <- max(1L, floor(length(g) * ", wb_literal(p$validation_fraction), "))\n",
      "  split_rows$train <- c(split_rows$train, g[seq_len(a)])\n",
      "  split_rows$validation <- c(split_rows$validation, g[a + seq_len(b)])\n",
      "  split_rows$test <- c(split_rows$test, g[seq.int(a + b + 1L, length(g))])\n}\n")
  } else {
    split_code <- paste0("# Frozen row indices from an older bundle\nsplit_rows <- ",
      wb_literal(lapply(p$parts, `[[`, "row")), "\n")
  }
  scale_code <- paste0("x <- as.matrix(data[features])\n",
    if (isTRUE(p$standardize)) paste0(
      "center <- colMeans(x[split_rows$train, , drop = FALSE])\n",
      "scale <- apply(x[split_rows$train, , drop = FALSE], 2, sd)\n",
      "scale[!is.finite(scale) | scale == 0] <- 1\n") else
      "center <- rep(0, length(features))\nscale <- rep(1, length(features))\n",
    "x <- sweep(sweep(x, 2, center, '-'), 2, scale, '/')\n")
  labels <- switch(p$mode,
    regression = "y <- matrix(as.numeric(data[[target]]), ncol = 1)\n",
    binary = paste0("classes <- sort(unique(as.character(data[[target]][split_rows$train])))\n",
      "y <- matrix(match(as.character(data[[target]]), classes) - 1L, ncol = 1)\n"),
    multiclass = paste0("classes <- sort(unique(as.character(data[[target]][split_rows$train])))\n",
      "y <- matrix(0, nrow = nrow(data), ncol = length(classes))\n",
      "y[cbind(seq_len(nrow(data)), match(as.character(data[[target]]), classes))] <- 1\n"))
  paste0(header, split_code, scale_code, labels,
    "parts <- lapply(split_rows, function(i) list(x = x[i, , drop = FALSE],\n",
    "  y = y[i, , drop = FALSE], actual = data[[target]][i], row = i))")
}

wb_fit_code <- function(p, cfg, create = TRUE) {
  best <- identical(cfg$checkpoint, "best_validation")
  start <- paste0("set.seed(", p$seed, ")\n")
  if (create) start <- paste0(start,
    "model <- ddesonn_model(input_size = length(features), output_size = ",
    if (p$mode == "multiclass") "length(classes)" else "1", ",\n",
    "  hidden_sizes = ", wb_literal(cfg$hidden), ", classification_mode = mode,\n",
    "  ML_NN = ", wb_literal(length(cfg$hidden) > 0), ", lambda = ", wb_literal(cfg$lambda), ", num_networks = 1)\n")
  code <- paste0(start,
    "ddesonn_fit(model, parts$train$x, parts$train$y, validation = parts$validation,\n",
    "  num_epochs = ", cfg$epochs, ", optimizer = ", wb_literal(cfg$optimizer), ", lr = ", wb_literal(cfg$lr), ",\n",
    "  lr_decay_rate = 1, lr_min = ", wb_literal(cfg$lr), ", self_org = ", wb_literal(create && isTRUE(cfg$self_org)), ",\n",
    "  reg_type = ", wb_literal(if (cfg$reg == "none") NULL else cfg$reg), ", dropout_rates = NULL,\n",
    "  numeric_columns = NULL, batch_normalize_data = FALSE, threshold = 0.5,\n",
    "  validation_metrics = ", wb_literal(best), ", best_weights_on_latest_weights_off = ", wb_literal(best), ",\n",
    "  viewTables = FALSE, verboseLow = TRUE)\n")
  if (best) paste0(code,
    "# Explicitly restore both checkpoint tensors before inference.\n",
    "network <- model$ensemble[[1]]\n",
    "selected_epoch <- model$last_training$predicted_outputAndTime$",
    if (p$mode == "regression") "best_val_epoch_loss" else "best_val_epoch", "\n",
    "stopifnot(length(selected_epoch) == 1, is.finite(selected_epoch),\n",
    "  !is.null(network$best_weights), !is.null(network$best_biases))\n",
    "network$weights <- network$best_weights\nnetwork$biases <- network$best_biases\n") else
    paste0(code, "selected_epoch <- ", cfg$epochs, "L # final epoch\n")
}

wb_evaluation_code <- function(p, split = "test") {
  stopifnot(split %in% c("train", "validation", "test"))
  start <- paste0("part <- parts[[", wb_literal(split), "]]\n",
    "probabilities <- as.matrix(ddesonn_predict(model, part$x)$prediction)\n")
  code <- switch(p$mode,
    regression = paste0("actual <- as.numeric(part$y)\nprediction <- probabilities[, 1]\n",
      "metrics <- c(RMSE = sqrt(mean((actual - prediction)^2)),\n",
      "  MAE = mean(abs(actual - prediction)),\n",
      "  R_squared = if (sum((actual - mean(actual))^2) > 0)\n",
      "    1 - sum((actual - prediction)^2) / sum((actual - mean(actual))^2) else NA_real_)"),
    binary = paste0("p <- probabilities[, 1]\nactual <- part$y[, 1]\nprediction <- as.integer(p >= 0.5)\n",
      "correct_probability <- ifelse(actual == 1, p, 1 - p)\n",
      "metrics <- c(Accuracy = mean(prediction == actual),\n",
      "  Log_loss = -mean(log(pmax(correct_probability, 1e-15))),\n",
      "  Brier_score = mean((p - actual)^2))\n",
      "confusion <- table(Actual = factor(classes[actual + 1], levels = classes),\n",
      "  Predicted = factor(classes[prediction + 1], levels = classes))"),
    multiclass = paste0("actual <- max.col(part$y, ties.method = 'first')\n",
      "prediction <- max.col(probabilities, ties.method = 'first')\n",
      "correct_probability <- probabilities[cbind(seq_along(actual), actual)]\n",
      "metrics <- c(Accuracy = mean(prediction == actual),\n",
      "  Log_loss = -mean(log(pmax(correct_probability, 1e-15))),\n",
      "  Brier_score = mean(rowSums((probabilities - part$y)^2)))\n",
      "confusion <- table(Actual = factor(classes[actual], levels = classes),\n",
      "  Predicted = factor(classes[prediction], levels = classes))"))
  paste(start, code, "print(metrics)", sep = "\n")
}

wb_recipe <- function(p, cfg) {
  paste("# Equivalent R recipe for this prepared data and fit configuration.",
    wb_data_code(p$source), wb_features_code(p), wb_fit_code(p, cfg),
    wb_evaluation_code(p), sep = "\n\n")
}

wb_bundle_code <- function() {
  paste0("library(DDESONN)\n# Use the bundle downloaded from this workbench.\n",
    "bundle <- readRDS('ddesonn-workbench-model.rds')\nmodel <- bundle$model\n",
    "new_data <- read.csv('new-data.csv', check.names = FALSE)\n",
    "settings <- bundle$prepared\nx_new <- as.matrix(new_data[settings$features])\n",
    "x_new <- sweep(sweep(x_new, 2, settings$center, '-'), 2, settings$scale, '/')\n",
    "probabilities <- ddesonn_predict(model, x_new)$prediction\n",
    "# Preserve the workbench bundle (model + preprocessing + split + settings):\n",
    "saveRDS(bundle, 'my-workbench-model.rds')\n",
    "# A standalone API model is also possible; keep its preprocessing separately:\n",
    "saveRDS(model, 'api-model.rds')")
}


wb_resume_recipe <- function(p, cfg) {
  paste("# Requires the workbench bundle saved BEFORE this additional fit.",
    "library(DDESONN)", "bundle <- readRDS('previous-workbench-model.rds')",
    "model <- bundle$model", "parts <- bundle$prepared$parts",
    "mode <- bundle$prepared$mode", "classes <- bundle$prepared$classes",
    wb_fit_code(p, cfg, FALSE), wb_evaluation_code(p), sep = "\n\n")
}

# Code in help uses the same public-API generators as downloadable recipes.
wb_help_topics <- function() c(
  "Model performance", "Evaluation split", "Confusion matrix (classification)",
  "Save or restore", "Another fit from saved weights", "Additional epochs",
  "Predict on new rows", "Start with Data", "Inspect your dataset",
  "Choose what the model learns", "Task", "Train / validation / test split",
  "Training fraction", "Validation fraction", "Random seed",
  "Standardize using training means and SDs", "Prepare for training", "About your split",
  "Build & Train", "Hidden-layer sizes", "Epochs", "Optimizer", "Learning rate",
  "Self-organize before initial training", "Weights used for results",
  "Regularization", "Regularization strength", "Training progress",
  "Current completed model", "Target", "Predictors")

wb_help_key <- function(topic) paste0("wb_help_", gsub("[^a-z0-9]+", "_", tolower(topic)))

wb_help_code <- function(topic, prepared = NULL, config = NULL, completed = NULL,
                         source = list(example = "iris"), evaluation_split = "test",
                         more_epochs = 10L) {
  if (!topic %in% wb_help_topics()) stop("Unknown workbench help topic: ", topic)
  p <- if (is.null(prepared)) wb_prepare(datasets::iris, "Species", names(datasets::iris)[1:4], "multiclass") else prepared
  cfg <- if (is.null(config)) wb_config() else config
  value <- wb_literal
  snippet <- switch(topic,
    "Start with Data" = wb_data_code(source),
    "Inspect your dataset" = "head(data, 10)\nstr(data)\ncolSums(is.na(data))",
    "Choose what the model learns" = paste0("target <- ", value(p$target), "\nfeatures <- ", value(p$features),
      "\nstopifnot(!target %in% features)"),
    "Target" = paste0("target <- ", value(p$target), "\noutcome <- data[[target]]"),
    "Predictors" = paste0("features <- ", value(p$features), "\nx <- as.matrix(data[features])"),
    "Task" = paste0("mode <- ", value(p$mode), "\n", switch(p$mode,
      regression = "y <- matrix(as.numeric(data[[target]]), ncol = 1)",
      binary = "classes <- sort(unique(as.character(data[[target]][split_rows$train])))\ny <- matrix(match(as.character(data[[target]]), classes) - 1L, ncol = 1)",
      multiclass = "classes <- sort(unique(as.character(data[[target]][split_rows$train])))\ny <- matrix(0, nrow(data), length(classes))\ny[cbind(seq_len(nrow(data)), match(as.character(data[[target]]), classes))] <- 1")),
    "Train / validation / test split" = paste0("# ", p$split, " split; fractions are applied within each group.\n",
      if (p$split == "random" && p$mode != "regression")
        "groups <- unname(split(seq_len(nrow(data)), as.character(data[[target]])))\n" else
        "groups <- list(seq_len(nrow(data)))\n",
      if (p$split == "random") paste0("set.seed(", p$seed, ")\ngroups <- lapply(groups, function(g) sample(g, length(g)))") else
        "# Keep the original order within each group."),
    "Training fraction" = paste0("train_fraction <- ", value(if (is.null(p$train_fraction)) .6 else p$train_fraction),
      "\n# g is one group from the split step.\nn_train <- max(1L, floor(length(g) * train_fraction))"),
    "Validation fraction" = paste0("validation_fraction <- ", value(if (is.null(p$validation_fraction)) .2 else p$validation_fraction),
      "\n# Remaining rows after training and validation become Test.\nn_validation <- max(1L, floor(length(g) * validation_fraction))"),
    "Random seed" = paste0("seed <- ", p$seed, "L\nset.seed(seed)\n# Reset this seed for splitting and again before model initialization."),
    "Standardize using training means and SDs" = if (isTRUE(p$standardize)) paste(
      "# During preparation, before x has been scaled:",
      "center <- colMeans(x[split_rows$train, , drop = FALSE])",
      "scale <- apply(x[split_rows$train, , drop = FALSE], 2, sd)",
      "scale[!is.finite(scale) | scale == 0] <- 1",
      "x <- sweep(sweep(x, 2, center, '-'), 2, scale, '/')", sep = "\n") else
        "# Standardization is off.\ncenter <- rep(0, ncol(x))\nscale <- rep(1, ncol(x))\n# Keep x on its original scale.",
    "Prepare for training" = paste("# Assemble the already split, encoded and scaled arrays:",
      "parts <- lapply(split_rows, function(i) list(",
      "  x = x[i, , drop = FALSE], y = y[i, , drop = FALSE],",
      "  actual = data[[target]][i], row = i))", sep = "\n"),
    "About your split" = "lapply(parts, function(part) part$row)\n# Disjoint row indices; this does not detect related subjects or duplicates.\nintersect(parts$train$row, parts$test$row)",
    "Build & Train" = paste0("model <- ddesonn_model(input_size = length(features),\n  output_size = ",
      if (p$mode == "multiclass") "length(classes)" else "1", ", classification_mode = mode,\n  hidden_sizes = ",
      value(cfg$hidden), ", ML_NN = ", value(length(cfg$hidden) > 0), ",\n  lambda = ", value(cfg$lambda), ", num_networks = 1)"),
    "Hidden-layer sizes" = paste0("hidden_sizes <- ", value(cfg$hidden),
      "\nML_NN <- length(hidden_sizes) > 0\n# Arguments to ddesonn_model(); integer(0) means no hidden layers."),
    "Epochs" = paste0("num_epochs <- ", cfg$epochs, "L\n# ddesonn_fit(..., num_epochs = num_epochs)\n# Both checkpoint modes run this full budget; no automatic early stopping."),
    "Optimizer" = paste0("optimizer <- ", value(cfg$optimizer), "\n# Pass optimizer to ddesonn_fit(); compare choices using Validation."),
    "Learning rate" = paste0("lr <- ", value(cfg$lr), "\nlr_min <- lr\nlr_decay_rate <- 1\n# Fixed-rate arguments to ddesonn_fit()."),
    "Self-organize before initial training" = paste0("self_org <- ", value(isTRUE(cfg$self_org)),
      "\n# New fit: ddesonn_fit(..., self_org = self_org)\n# Additional fit from saved weights: self_org = FALSE"),
    "Weights used for results" = if (identical(cfg$checkpoint, "best_validation")) paste(
      "validation_metrics <- TRUE", "best_weights_on_latest_weights_off <- TRUE",
      "# After the fit, restore both tensors before inference:", "network <- model$ensemble[[1]]",
      "network$weights <- network$best_weights", "network$biases <- network$best_biases", sep = "\n") else
        "validation_metrics <- FALSE\nbest_weights_on_latest_weights_off <- FALSE\n# Use final weights and biases; no checkpoint restoration.",
    "Regularization" = paste0("reg_type <- ", value(if (cfg$reg == "none") NULL else cfg$reg),
      "\n# Pass reg_type to ddesonn_fit(); NULL means no penalty."),
    "Regularization strength" = paste0("lambda <- ", value(cfg$lambda),
      "\n# Pass lambda to ddesonn_model(); the penalty applies when L1 or L2 is selected."),
    "Evaluation split" = paste0("part <- parts[[", value(evaluation_split),
      "]]\n# This selects rows for reporting; Test never selects training weights."),
    "Model performance" = paste("# After predicting the selected part, compute only the metrics:",
      paste(deparse(Filter(function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
        identical(e[[2]], as.name("metrics")), parse(text = wb_evaluation_code(
          if (is.null(completed)) p else completed$prepared, evaluation_split)))[[1]]), collapse = "\n"), sep = "\n"),
    "Confusion matrix (classification)" = {
      mode <- if (is.null(completed)) p$mode else completed$prepared$mode
      if (mode == "regression") "# Regression has no confusion matrix.\nplot(as.numeric(part$y), as.numeric(probabilities), xlab = 'Actual', ylab = 'Prediction')" else paste(
        if (mode == "binary") "actual_index <- part$y[, 1] + 1L\npredicted_index <- ifelse(probabilities[, 1] >= 0.5, 2L, 1L)" else
          "actual_index <- max.col(part$y, ties.method = 'first')\npredicted_index <- max.col(probabilities, ties.method = 'first')",
        "table(Actual = factor(classes[actual_index], levels = classes),",
        "      Predicted = factor(classes[predicted_index], levels = classes))", sep = "\n")
    },
    "Save or restore" = "# The trusted workbench bundle retains model, preprocessing and settings.\nbundle <- readRDS('ddesonn-workbench-model.rds')\nsaveRDS(bundle, 'my-workbench-model.rds')",
    "Predict on new rows" = paste("# Requires the saved workbench bundle and a new CSV:",
      "settings <- bundle$prepared", "new_data <- read.csv('new-data.csv', check.names = FALSE)",
      "x_new <- as.matrix(new_data[settings$features])",
      "x_new <- sweep(sweep(x_new, 2, settings$center, '-'), 2, settings$scale, '/')",
      "predictions <- ddesonn_predict(bundle$model, x_new)$prediction", sep = "\n"),
    "Another fit from saved weights" = "model <- bundle$model\nparts <- bundle$prepared$parts\n# Reuse saved data/settings; restart optimizer schedules.\nself_org <- FALSE",
    "Additional epochs" = {
      epochs <- tryCatch(wb_config(epochs = more_epochs)$epochs, error = function(e) NULL)
      if (is.null(epochs)) "# Choose 1–5,000 additional epochs." else
        paste0("num_epochs <- ", epochs, "L\n# Epoch budget for the additional fit; other settings come from bundle$config.")
    },
    "Training progress" = "history <- bundle$history\nplot(history$epoch, history$loss, type = 'l', xlab = 'Epoch', ylab = 'Training loss')\nabline(v = bundle$selected_epoch, lty = 2)\n# Inspecting the curve does not select weights.",
    "Current completed model" = "bundle$package_version\nbundle$config\nbundle$selected_epoch\nbundle$fit_runs\n# Show full R code to inspect this model's saved recipe.")
  prefix <- if (is.null(prepared) && !topic %in% c("Start with Data", "Inspect your dataset",
      "Current completed model", "Save or restore", "Predict on new rows", "Training progress",
      "Another fit from saved weights", "Additional epochs")) "# Iris/default illustration until data is prepared.\n" else ""
  if (!is.null(completed) && topic %in% c("Model performance", "Confusion matrix (classification)")) prefix <- ""
  if (is.null(config) && topic %in% c("Build & Train", "Hidden-layer sizes", "Epochs", "Optimizer",
      "Learning rate", "Self-organize before initial training", "Weights used for results",
      "Regularization", "Regularization strength"))
    prefix <- paste0(prefix, "# Default example values; correct invalid model settings for current values.\n")
  paste0(prefix, snippet)
}

# Complete scripts live in one explicit viewer, not in individual help icons.
wb_full_code <- function(prepared = NULL, config = NULL, completed = NULL,
                         which = c("next", "completed", "prediction")) {
  which <- match.arg(which)
  if (which == "completed") {
    if (is.null(completed$recipe)) stop("This model has no saved full training recipe.")
    return(list(code = completed$recipe, filename = "ddesonn-completed-model.R",
      notice = "Saved model recipe, including its original settings and Test evaluation. Later control changes do not affect it."))
  }
  if (which == "prediction") return(list(code = wb_bundle_code(), filename = "ddesonn-saved-prediction.R",
    notice = "Requires the trusted downloaded workbench bundle and new-data.csv. This loads and predicts with an existing model."))
  example <- is.null(prepared)
  if (!example && is.null(config)) stop("Correct the model settings to generate the full script.")
  p <- if (example) wb_prepare(datasets::iris, "Species", names(datasets::iris)[1:4], "multiclass") else prepared
  if (example) p$source <- list(example = "iris")
  list(code = wb_recipe(p, if (is.null(config)) wb_config() else config),
    filename = "ddesonn-next-model.R", notice = if (example)
      paste("Iris illustration. Prepare data for a full script using your dataset.",
        if (is.null(config)) "Default model settings are shown while controls are invalid." else "Current valid model settings apply to this illustration.") else
      "Prepared dataset and current valid fit settings, followed by Test evaluation. Use the original CSV and row order.")
}
