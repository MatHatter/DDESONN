# Model-workbench helpers. Training mathematics stays in the public package API.
wb_example_catalog <- function() list(
  iris = list(label = "Iris — flower classification", builtin = TRUE,
    target = "Species", mode = "multiclass", features = names(datasets::iris)[1:4],
    note = "Target: Species. Four numeric flower measurements. Ready to prepare on Features."),
  mtcars = list(label = "mtcars — fuel-economy regression", builtin = TRUE,
    target = "mpg", mode = "regression", features = c("wt", "hp"),
    note = "Target: mpg. All other columns are selected by default; review them on Features."),
  heart_failure = list(label = "Heart failure — binary classification",
    file = "heart_failure_clinical_records.csv", target = "DEATH_EVENT", mode = "binary",
    note = "Target: DEATH_EVENT. Uses the 12 numeric predictors. This CSV has repeated inputs; read the benchmark limitations in Vignettes. Loading it does not run or reproduce the research benchmark."),
  wmt = list(label = "WMT prices — packaged stock snapshot",
    file = "WMT_1970-10-01_2025-03-15.csv", target = "close", mode = "regression",
    features = c("open", "high", "low", "volume"), split = "ordered",
    note = "Same-row price regression illustration, not a validated forecast. Target: close. All other columns are selected by default; review date and adjusted close for preprocessing and leakage. Ordered splitting is suggested."),
  customer_train = list(label = "Customer segmentation — labeled dataset",
    file = "train_multiclass_customer_segmentation.csv", target = "Segmentation", mode = "multiclass",
    features = "Age", exclude = "ID",
    note = "One labeled dataset: Shiny splits it into training, validation and test sets on Features. Target: Segmentation. All other columns are selected by default, including ID. Review the Auto preprocessing plan for identifier exclusions, category encoding and missing-value treatment before preparation. Original source rows and columns remain available for inspection and download."),
  customer_test = list(label = "Customer segmentation — unlabeled prediction data",
    file = "test_multiclass_customer_segmentation.csv", features = "Age", exclude = "ID",
    prediction_only = TRUE,
    note = "This file has no Segmentation labels. Inspect or download it, or predict with a compatible model you have trained or loaded. It cannot evaluate segmentation accuracy without true labels."))

wb_example_choices <- function() {
  catalog <- Filter(function(x) !isTRUE(x$prediction_only), wb_example_catalog())
  stats::setNames(names(catalog), vapply(catalog, `[[`, character(1), "label"))
}

wb_example <- function(id) {
  catalog <- wb_example_catalog()
  if (length(id) != 1L || is.na(id) || !id %in% names(catalog)) stop("Choose a packaged example.")
  entry <- catalog[[id]]
  if (isTRUE(entry$builtin)) {
    entry$data <- if (id == "iris") datasets::iris else datasets::mtcars
    entry$source <- list(example = id)
    entry$path <- NULL
  } else {
    path <- system.file("extdata", entry$file, package = "DDESONN")
    if (!nzchar(path)) stop("This example is missing from the installed package. Reinstall DDESONN with its data or inspect the dataset source link.")
    entry$data <- utils::read.csv(path, check.names = FALSE)
    entry$path <- path
    entry$source <- list(package_file = entry$file,
      sha256 = digest::digest(file = path, algo = "sha256"),
      package_version = as.character(utils::packageVersion("DDESONN")))
  }
  entry$id <- id
  entry
}

wb_default_features <- function(data, target, preset = NULL) {
  # Selection is explicit and schema-based; preparation validates suitability.
  setdiff(names(data), target)
}

wb_write_example <- function(id, file) {
  example <- wb_example(id)
  if (!is.null(example$path)) {
    if (!file.copy(example$path, file, overwrite = TRUE)) stop("Could not copy the packaged CSV.")
  } else utils::write.csv(example$data, file, row.names = FALSE)
  invisible(file)
}

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
                       standardize = TRUE, split = "random", preprocessing = NULL) {
  if (!is.data.frame(data) || nrow(data) < 10 || anyDuplicated(names(data)))
    stop("Use at least 10 rows and unique column names.")
  if (!mode %in% c("binary", "multiclass", "regression")) stop("Choose a supported task.")
  if (length(target) != 1 || !target %in% names(data) || !length(features) ||
      any(!features %in% names(data)) || target %in% features || anyDuplicated(features))
    stop("Choose a target and at least one distinct predictor.")
  x <- data[features]
  if (is.null(preprocessing) && !all(vapply(x, function(v) is.numeric(v) && all(is.finite(v)), logical(1))))
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
  fitted <- if (is.null(preprocessing)) NULL else wb_pp_fit(x[ids$train, , drop = FALSE], features, preprocessing)
  if (!is.null(fitted)) {
    features <- fitted$features
    x <- wb_pp_apply(data, fitted)
  } else x <- as.matrix(x)
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
    preprocessing = fitted, encoded_features = colnames(x),
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
  if (!optimizer %in% DDESONN::ddesonn_optimizer_options()) stop("Unsupported optimizer.")
  if (!is.finite(lambda) || lambda < 0 || !reg %in% c("none", "L1", "L2", "L1_L2", "Group_Lasso", "Max_Norm")) stop("Invalid regularization.")
  if (!is.logical(self_org) || length(self_org) != 1 || is.na(self_org))
    stop("Self-organization must be TRUE or FALSE.")
  checkpoint <- match.arg(checkpoint, c("best_validation", "final"))
  list(hidden = as.integer(h), epochs = as.integer(epochs), lr = lr,
    optimizer = optimizer, lambda = lambda, reg = reg, self_org = self_org, checkpoint = checkpoint)
}

wb_predictions <- function(model, prepared, part) {
  p <- if (!is.null(attr(model, "workbench_custom"))) wb_custom_predictions(model, prepared, part) else
    as.matrix(DDESONN::ddesonn_predict(model, part$x)$prediction)
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
  if (identical(config$workflow, "custom")) return(wb_custom_run(prepared, config, previous))
  # callr isolates RNG, package globals, graphics and all mutable model state.
  set.seed(prepared$seed)
  # Older workbench bundles predate the checkbox and used FALSE.
  config$self_org <- isTRUE(config$self_org)
  if (is.null(config$checkpoint)) config$checkpoint <- "final" # Older bundles used final weights.
  config$checkpoint <- match.arg(config$checkpoint, c("best_validation", "final"))
  use_best <- identical(config$checkpoint, "best_validation")
  model <- if (is.null(previous)) DDESONN::ddesonn_model(
    input_size = ncol(prepared$parts$train$x),
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
    dropout_rates = NULL, numeric_columns = if (is.null(prepared$preprocessing)) NULL else integer(0), batch_normalize_data = FALSE,
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
  if (identical(bundle$config$workflow, "custom")) return(paste(
    if (identical(bundle$config$checkpoint, "best_validation")) "Each member uses its own best validation checkpoint (weights and biases)." else "Each member uses the final epoch.",
    "All requested epochs ran; results combine members within seeds, then across seeds."))
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
  guidance <- if (grepl("Binary mode expects 1-column probabilities", details, fixed = TRUE))
    "The installed training engine treated multiclass predictions as binary. Reinstall DDESONN from the same updated source as this app, then restart R. Check packageVersion('DDESONN'); this is a task-type mismatch, not a learning-rate problem." else
    if (grepl("No valid validation checkpoint", details, ignore.case = TRUE))
    "No usable validation checkpoint was recorded. Review the technical details and validation data. Try a smaller learning rate, or choose Final epoch explicitly for a separate run; it may succeed with its last weights. No checkpoint is substituted automatically." else
    if (grepl("finite|NaN|Inf|invalid predictions|probabilities|checkpoint", details, ignore.case = TRUE))
    "Training could not produce a usable model. Try a smaller learning rate, enable standardization on Features and prepare again, or turn off initial self-organization." else
    if (grepl("colMeans|two dimensions|non-conformable|dimension", details, ignore.case = TRUE))
      "Training could not use this data shape. Try two or more numeric predictors; if this repeats, share the technical details when reporting the model-shape problem." else
    if (grepl("there is no package|not installed|could not start|failed to start|cannot open.*connection", details, ignore.case = TRUE))
      "The training worker could not finish. Try training again; if this repeats, check that DDESONN and the optional callr package are installed in the same R library." else
      "Training could not start or finish. Check the model settings (hidden sizes, epochs and learning rate), then try again. A smaller model or learning rate may help."
  list(message = guidance, details = paste(c(
    paste0("Installed DDESONN engine: ", utils::packageVersion("DDESONN")),
    paste0("R version: ", getRversion()), details, utils::tail(stderr, 40)), collapse = "\n"))
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
      !length(p$features) || length(p$center) != ncol(p$parts$train$x) ||
      length(p$scale) != ncol(p$parts$train$x) || any(!is.finite(p$center)) ||
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
  x <- wb_prediction_inputs(data, p)
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
  epochs <- as.integer(sub(".*Epoch ([0-9]+).*", "\\1", lines))
  # Orchestration restarts epoch numbering per member. Use cumulative member
  # epochs for the chart; a single-member Basic history remains unchanged.
  data.frame(epoch = if (anyDuplicated(epochs)) seq_along(epochs) else epochs,
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
    started = Sys.time(), epochs = if (identical(config$workflow, "custom"))
      config$epochs * length(config$seeds) * config$members * (if (is.null(previous)) 1L + config$temp else 1L) else config$epochs)
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
  if (identical(source$example, "iris")) return("library(DDESONN)\n\n# Read the training data.\ndata <- datasets::iris")
  if (identical(source$example, "mtcars")) return("library(DDESONN)\n\n# Read the training data.\ndata <- datasets::mtcars")
  if (!is.null(source$package_file)) return(paste0(
    "library(DDESONN)\n\n# Read the same CSV from the installed package.\n",
    "data_path <- system.file('extdata', ", wb_literal(source$package_file), ", package = 'DDESONN')\n",
    "stopifnot(nzchar(data_path))\n",
    if (!is.null(source$sha256)) paste0("stopifnot(identical(digest::digest(file = data_path, algo = 'sha256'), ", wb_literal(source$sha256), "))\n") else "",
    "data <- read.csv(data_path, check.names = FALSE)"))
  paste0("library(DDESONN)\n\n# Read the training data.\ndata <- read.csv(",
      wb_literal(if (is.null(source$file)) "training-data.csv" else source$file),
      ", check.names = FALSE)\n# Use the original CSV with the same rows and row order.")
}

wb_features_code <- function(p) {
  header <- paste0("features <- ", wb_literal(if (is.null(p$preprocessing)) p$features else p$preprocessing$plan$Column), "\ntarget <- ", wb_literal(p$target),
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
  input_code <- if (is.null(p$preprocessing)) "x <- as.matrix(data[features])\n" else paste0(
    "# Portable categorical encoding and missing-value rules.\n", wb_pp_code(), "\n\n",
    "preprocessing_config <- ", wb_literal(p$preprocessing$config), "\n",
    "preprocessing <- wb_pp_fit(data[split_rows$train, , drop = FALSE], features, preprocessing_config)\n",
    "x <- wb_pp_apply(data, preprocessing)\n")
  scale_code <- paste0(input_code,
    if (isTRUE(p$standardize)) paste0(
      "center <- colMeans(x[split_rows$train, , drop = FALSE])\n",
      "scale <- apply(x[split_rows$train, , drop = FALSE], 2, sd)\n",
      "scale[!is.finite(scale) | scale == 0] <- 1\n") else
      "center <- rep(0, ncol(x))\nscale <- rep(1, ncol(x))\n",
    "x <- sweep(sweep(x, 2, center, '-'), 2, scale, '/')\n")
  labels <- switch(p$mode,
    regression = "y <- matrix(as.numeric(data[[target]]), ncol = 1)\n",
    binary = paste0("classes <- sort(unique(as.character(data[[target]][split_rows$train])))\n",
      "y <- matrix(match(as.character(data[[target]]), classes) - 1L, ncol = 1)\n"),
    multiclass = paste0("classes <- sort(unique(as.character(data[[target]][split_rows$train])))\n",
      "y <- matrix(0, nrow = nrow(data), ncol = length(classes))\n",
      "y[cbind(seq_len(nrow(data)), match(as.character(data[[target]]), classes))] <- 1\n"))
  paste0("# Choose predictors, target and task.\n", header, "\n# Split rows before learning preprocessing.\n", split_code,
    "\n# Scale using training rows only.\n", scale_code, "\n# Encode the target.\n", labels,
    "\n# Assemble the three data partitions.\n",
    "parts <- lapply(split_rows, function(i) list(x = x[i, , drop = FALSE],\n",
    "  y = y[i, , drop = FALSE], actual = data[[target]][i], row = i))")
}

# Shared explanation for the focused help and every newly generated fit recipe.
wb_lr_comments <- function(lr) {
  paste(
    "# lr_decay_rate multiplies; it is not an amount subtracted.",
    paste0("# 1 keeps 100%: ", wb_literal(lr), " * 1 = ", wb_literal(lr), " (no decay)."),
    "# Before the floor: 0.5 halves the rate; 0.9 keeps 90%.",
    "# Every lr_decay_epoch epochs the factor is applied again.",
    "# API default interval: classification 1 epoch; regression 20.", "",
    "# lr_min is the floor; setting it to lr also prevents decay.",
    "# To decay: 0 < lr_decay_rate < 1 AND lr_min < lr.", "",
    "# Formula: max(lr_min,",
    "#   lr * lr_decay_rate^floor(epoch / lr_decay_epoch))", sep = "\n")
}

wb_fit_code <- function(p, cfg, create = TRUE) {
  if (identical(cfg$workflow, "custom")) {
    metadata <- p; metadata$parts <- NULL
    return(paste(
      "# Use the installed workbench's native orchestration and saved fusion rules.",
      "workbench <- new.env(parent = globalenv())",
      "sys.source(system.file('shiny', 'benchmark', 'workbench_helpers.R', package = 'DDESONN'), workbench)",
      paste0("config <- ", wb_literal(cfg)),
      if (create) paste0("prepared <- ", wb_literal(metadata), "\nprepared$parts <- parts\nprepared$center <- center\nprepared$scale <- scale\n",
        if (!is.null(p$preprocessing)) "prepared$preprocessing <- preprocessing\nprepared$features <- preprocessing$features\nprepared$encoded_features <- colnames(parts$train$x)\n" else "",
        "bundle <- workbench$wb_run(prepared, config)") else
        "bundle <- workbench$wb_run(bundle$prepared, config, bundle)",
      "model <- bundle$model", sep = "\n\n"))
  }
  best <- identical(cfg$checkpoint, "best_validation")
  start <- paste0("# Reproduce model initialization and training randomness.\nset.seed(", p$seed, ")\n\n")
  if (create) start <- paste0(start,
    "# Create a new, untrained model.\nmodel <- ddesonn_model(input_size = ncol(parts$train$x), output_size = ",
    if (p$mode == "multiclass") "length(classes)" else "1", ",\n",
    "  hidden_sizes = ", wb_literal(cfg$hidden), ", classification_mode = mode,\n",
    "  ML_NN = ", wb_literal(length(cfg$hidden) > 0), ", lambda = ", wb_literal(cfg$lambda), ", num_networks = 1)\n")
  code <- paste0(start, "\n# Train with a fixed learning rate.\n", wb_lr_comments(cfg$lr), "\n\n",
    "ddesonn_fit(model, parts$train$x, parts$train$y, validation = parts$validation,\n",
    "  num_epochs = ", cfg$epochs, ", optimizer = ", wb_literal(cfg$optimizer), ", lr = ", wb_literal(cfg$lr), ",\n",
    "  lr_decay_rate = 1, lr_min = ", wb_literal(cfg$lr), ", self_org = ", wb_literal(create && isTRUE(cfg$self_org)), ",\n",
    "  reg_type = ", wb_literal(if (cfg$reg == "none") NULL else cfg$reg), ", dropout_rates = NULL,\n",
    "  numeric_columns = ", if (is.null(p$preprocessing)) "NULL" else "integer(0)", ", batch_normalize_data = FALSE, threshold = 0.5,\n",
    "  validation_metrics = ", wb_literal(best), ", best_weights_on_latest_weights_off = ", wb_literal(best), ",\n",
    "  viewTables = FALSE, verboseLow = TRUE)\n")
  if (best) paste0(code,
    "\n# Explicitly restore both checkpoint tensors before inference.\n",
    "network <- model$ensemble[[1]]\n",
    "selected_epoch <- model$last_training$predicted_outputAndTime$",
    if (p$mode == "regression") "best_val_epoch_loss" else "best_val_epoch", "\n",
    "stopifnot(length(selected_epoch) == 1, is.finite(selected_epoch),\n",
    "  !is.null(network$best_weights), !is.null(network$best_biases))\n",
    "network$weights <- network$best_weights\nnetwork$biases <- network$best_biases\n") else
    paste0(code, "\n# Keep the weights from the final requested epoch.\nselected_epoch <- ", cfg$epochs, "L # final epoch\n")
}

wb_evaluation_code <- function(p, split = "test") {
  stopifnot(split %in% c("train", "validation", "test"))
  start <- paste0("# Predict the selected evaluation rows with the trained model.\npart <- parts[[", wb_literal(split), "]]\n",
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
  paste(start, paste("# Compare predictions with known outcomes.", code, sep = "\n"),
    "# Display the evaluation scores.\nprint(metrics)", sep = "\n\n")
}

wb_recipe <- function(p, cfg) {
  paste("# Equivalent R recipe for this prepared data and fit configuration.",
    wb_data_code(p$source), wb_features_code(p), wb_fit_code(p, cfg),
    if (identical(cfg$workflow, "custom")) "print(bundle$evaluation$test)\nprint(bundle$seed_comparison)\nprint(bundle$member_weights)" else wb_evaluation_code(p), sep = "\n\n")
}

wb_bundle_code <- function() {
  paste(
    "# Purpose: reload a saved model with its original settings and use it again.\n# This began as a standalone script before the package API existed.\n# This example uses the current API; no training runs here.\n# A bundle is the .rds file from Save model in this workbench.\n# It contains the model, feature names and training preprocessing settings.",
    "library(DDESONN)",
    "# 1. Load the saved bundle (replace this path with your downloaded file).\nbundle <- readRDS('ddesonn-workbench-model.rds')\nmodel <- bundle$model",
    "# 2. Read new rows to predict (replace this path with your CSV).\n# Include the original predictor column names; the target is not required.\nnew_data <- read.csv('new-data.csv', check.names = FALSE)",
    "# 3. Reuse saved encoding, imputation, predictor order and training scaling.\n# Do not calculate new means or SDs from the prediction rows.\nsettings <- bundle$prepared\nworkbench <- new.env(parent = globalenv())\nsys.source(system.file('shiny', 'benchmark', 'workbench_helpers.R', package = 'DDESONN'), workbench)\nx_new <- workbench$wb_prediction_inputs(new_data, settings)",
    "# 4. Predict without changing or retraining the saved model.\n# Classification returns class probabilities; regression returns numbers.\nworkbench <- new.env(parent = globalenv())\nsys.source(system.file('shiny', 'benchmark', 'workbench_helpers.R', package = 'DDESONN'), workbench)\n# This preserves Custom member/seed combination and Basic behavior.\nprobabilities <- workbench$wb_predictions(model, settings, list(x = x_new))\nprint(probabilities)",
    "# Optional: save a copy of the whole bundle for use again later.\n# This retains the model AND its preprocessing, split and settings.\nsaveRDS(bundle, 'my-workbench-model.rds')",
    "# Optional: export only the API model.\n# This file omits preprocessing and the Custom orchestration workflow; keep the bundle.\nsaveRDS(model, 'api-model.rds')", sep = "\n\n")
}


wb_resume_recipe <- function(p, cfg) {
  paste("# Requires the workbench bundle saved BEFORE this additional fit.",
    "library(DDESONN)", "bundle <- readRDS('previous-workbench-model.rds')",
    "model <- bundle$model", "parts <- bundle$prepared$parts",
    "mode <- bundle$prepared$mode", "classes <- bundle$prepared$classes",
    wb_fit_code(p, cfg, FALSE), if (identical(cfg$workflow, "custom")) "print(bundle$evaluation$test)" else wb_evaluation_code(p), sep = "\n\n")
}

# Separate explanatory blocks without expanding every individual statement.
wb_space_comments <- function(code) {
  lines <- strsplit(code, "\n", fixed = TRUE)[[1]]
  out <- character()
  for (line in lines) {
    if (grepl("^#", line) && length(out) && nzchar(tail(out, 1)) &&
        !grepl("^#", tail(out, 1))) out <- c(out, "")
    out <- c(out, line)
  }
  paste(out, collapse = "\n")
}

# Code in help uses the same public-API generators as downloadable recipes.
wb_help_topics <- function() c(
  "Model performance", "Evaluation split", "Confusion matrix (classification)",
  "Save or restore", "Another fit from saved weights", "Additional epochs",
  "Predict on new rows", "Start with Data", "Inspect your dataset",
  "Choose what the model learns", "Task", "Train / validation / test split",
  "Training fraction", "Validation fraction", "Data-split seed",
  "Standardize using training means and SDs", "Prepare for training", "About your split",
  "Build & Train", "Hidden-layer sizes", "Epochs per model fit", "Optimizer", "Learning rate",
  "Self-organize before initial training", "Weights used for results",
  "Regularization", "Regularization strength", "Training progress",
  "Current completed model", "Target", "Predictors",
  "Workflow", "Run structure", "Members per model", "Temporary iterations", "Training-run seed values",
  "Hidden activation", "Weight initialization", "Enable regularization", "Enable dropout", "Dropout rate",
  "Enable learning-rate scheduling", "Schedule", "Minimum rate", "Decay multiplier", "Decay interval (epochs)",
  "Combine members within each training seed", "Combine across training seeds", "Detailed training diagnostics")

wb_help_key <- function(topic) paste0("wb_help_", gsub("[^a-z0-9]+", "_", tolower(topic)))

wb_help_code <- function(topic, prepared = NULL, config = NULL, completed = NULL,
                         source = list(example = "iris"), evaluation_split = "test",
                         more_epochs = 10L) {
  if (!topic %in% wb_help_topics()) stop("Unknown workbench help topic: ", topic)
  p <- if (is.null(prepared)) wb_prepare(datasets::iris, "Species", names(datasets::iris)[1:4], "multiclass") else prepared
  cfg <- if (is.null(config)) wb_config() else config
  custom <- if (identical(cfg$workflow, "custom")) cfg else wb_custom_config(cfg)
  value <- wb_literal
  snippet <- switch(topic,
    "Workflow" = "# Basic: create and fit a model with ddesonn_model() and ddesonn_fit().\n# Custom: orchestrate members/seeds/MAIN/TEMP with ddesonn_run().\n# Both use the same native R engine; Keras is not used.",
    "Run structure" = paste0("# Single: 1 member, no orchestration; members: several, no MAIN/TEMP.\n# MAIN: incumbent/champion; TEMP: challenger/candidate/competitor.\ndo_ensemble <- ", value(custom$structure %in% c("main", "temp")),
      "\nnum_networks <- ", custom$members, "L\nnum_temp_iterations <- ", custom$temp,
      "L\n# ddesonn_run(..., do_ensemble = do_ensemble,\n#   num_networks = num_networks, num_temp_iterations = num_temp_iterations)"),
    "Members per model" = paste0("num_networks <- ", custom$members, "L\n# Members inside each model; independent of length(seeds)."),
    "Temporary iterations" = paste0("num_temp_iterations <- ", custom$temp, "L\n# Each TEMP model supplies challengers to the MAIN incumbent ensemble.\n# A warm start does not repeat this search."),
    "Training-run seed values" = paste0("seeds <- ", value(custom$seeds), "\n# length(seeds) is the number of training runs, not the number of epochs.\n# Each run completes num_epochs for each model fit before the next run starts.\n# Reuse the same prepared rows and scaling for every run."),
    "Hidden activation" = paste0("hidden_activation <- get(", value(custom$activation), ", asNamespace('DDESONN'))\noutput_name <- switch(mode, binary = 'sigmoid', multiclass = 'softmax', regression = 'identity')\noutput_activation <- get(output_name, asNamespace('DDESONN'))\nactivation_functions <- c(rep(list(hidden_activation), length(hidden_sizes)), list(output_activation))\nactivation_functions_predict <- activation_functions"),
    "Weight initialization" = paste0("init_method <- ", value(custom$init), "\n# ddesonn_model(..., init_method = init_method)\n# Warm starts retain the saved weights and biases."),
    "Enable regularization" = paste0("regularization_enabled <- ", value(cfg$reg != "none"), "\nreg_type <- ", value(if (cfg$reg == "none") NULL else cfg$reg), "\n# Off: NULL penalty and zero effective strength."),
    "Enable dropout" = paste0("dropout_enabled <- ", value(custom$dropout > 0), "\n# Off passes zero dropout to the training configuration."),
    "Dropout rate" = paste0("dropout_rate <- ", value(custom$dropout), "\ndropout_rates <- as.list(rep(dropout_rate, max(1L, length(hidden_sizes))))\n# Training only; prediction uses the full network."),
    "Enable learning-rate scheduling" = paste0("schedule_enabled <- ", value(custom$schedule != "fixed"), "\n# Off: lr_decay_rate = 1 and lr_min = lr.\n# On: use the selected Schedule and its decay controls."),
    "Schedule" = paste0("initial_lr <- ", value(custom$lr), "\nminimum_rate <- ", value(custom$minimum), "\nmultiplier <- ", value(custom$decay), "\ninterval <- ", custom$interval,
      "L\n# At epoch: max(minimum_rate, initial_lr * multiplier^floor(epoch / interval))"),
    "Minimum rate" = paste0("lr_min <- ", value(custom$minimum), "\n# Positive floor, at most the initial learning rate."),
    "Decay multiplier" = paste0("lr_decay_rate <- ", value(custom$decay), "\n# Retained fraction per interval, before applying lr_min."),
    "Decay interval (epochs)" = paste0("lr_decay_epoch <- ", custom$interval, "L\n# The engine exponent is floor(epoch / lr_decay_epoch)."),
    "Combine members within each training seed" = paste0("# Requires a completed Custom bundle and its prepared parts.\npayload <- attr(bundle$model, 'workbench_custom')\nmodel <- payload$models[[1]]\nmembers <- ddesonn_predict(model, parts$test$x, aggregate = 'none')$per_model\nmember_prediction <- DDESONN:::.ddesonn_combine_predictions(\n  members, mode, method = ", value(custom$fusion), ", weights = payload$weights[[1]])\n# Saved member weights came from validation, not test labels."),
    "Combine across training seeds" = paste0("# seed_predictions contains one already-combined N x K matrix per seed.\nseed_method <- ", value(custom$seed_aggregate), "\nprediction <- DDESONN:::.ddesonn_combine_predictions(\n  seed_predictions, mode, method = seed_method)\n# This stage combines seeds equally, after within-seed member combination."),
    "Detailed training diagnostics" = paste0("verbose <- ", value(isTRUE(custom$diagnostics)), "\nverboseLow <- TRUE\n# verbose adds tensor, gradient/error and optimizer summaries.\n# verboseLow retains ordinary epoch progress.\n# The app shows the most recent 2,000 lines under Training log.\n# Completed bundle$training_log retains the displayed log excerpts."),
    "Start with Data" = wb_data_code(source),
    "Inspect your dataset" = "# Preview rows.\nhead(data, 10)\n\n# Check column types.\nstr(data)\n\n# Count missing values in each column.\ncolSums(is.na(data))",
    "Choose what the model learns" = paste0("target <- ", value(p$target), "\nfeatures <- ", value(p$features),
      "\nstopifnot(!target %in% features)"),
    "Target" = paste0("target <- ", value(p$target), "\noutcome <- data[[target]]"),
    "Predictors" = paste0("features <- ", value(p$features), "\npredictor_data <- data[features]\n# Raw attributes: Prepare data applies encoding and missing-value rules."),
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
    "Data-split seed" = paste0("seed <- ", p$seed, "L\nset.seed(seed)\n# Random splitting uses this seed; ordered splitting keeps row order.\n# Basic reuses it for its one training run. Custom uses its separate seeds list.\n# 42 is one seed value, not 42 runs."),
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
    "Build & Train" = paste0("model <- ddesonn_model(input_size = ncol(parts$train$x),\n  output_size = ",
      if (p$mode == "multiclass") "length(classes)" else "1", ", classification_mode = mode,\n  hidden_sizes = ",
      value(cfg$hidden), ", ML_NN = ", value(length(cfg$hidden) > 0), ",\n  lambda = ", value(cfg$lambda), ", num_networks = 1)"),
    "Hidden-layer sizes" = paste0("hidden_sizes <- ", value(cfg$hidden),
      "\nML_NN <- length(hidden_sizes) > 0\n# Arguments to ddesonn_model(); integer(0) means no hidden layers."),
    "Epochs per model fit" = paste0("num_epochs <- ", cfg$epochs, "L\n# ddesonn_fit(..., num_epochs = num_epochs)\n# Each model fit completes this budget. Both checkpoint modes run all epochs.\n# Example: 2 seeds, Single model, 100 epochs = 2 complete runs and 200 total epochs."),
    "Optimizer" = paste0("optimizer <- ", value(cfg$optimizer), "\n# Pass optimizer to ddesonn_fit(); compare choices using Validation."),
    "Learning rate" = if (identical(cfg$workflow, "custom")) paste0("lr <- ", value(cfg$lr),
      "\nlr_min <- ", value(cfg$minimum), "\nlr_decay_rate <- ", value(cfg$decay),
      "\nlr_decay_epoch <- ", cfg$interval,
      "L\n# Scheduled rate: max(lr_min, lr * lr_decay_rate^floor(epoch / lr_decay_epoch))\n# The multiplier retains this fraction of the rate at each interval.") else
      paste0("lr <- ", value(cfg$lr), "\nlr_min <- lr\nlr_decay_rate <- 1\n\n", wb_lr_comments(cfg$lr)),
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
      "\n# Pass lambda to ddesonn_model(); the selected L1, L2, L1_L2, Group_Lasso or Max_Norm penalty uses this strength."),
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
    "Save or restore" = "# Load the .rds file downloaded with Save model.\n# The bundle contains the trained model and preprocessing settings.\nbundle <- readRDS('ddesonn-workbench-model.rds')\n\n# Save a copy for later; this does not train or predict.\nsaveRDS(bundle, 'my-workbench-model.rds')",
    "Predict on new rows" = paste(
      "# Run the complete saved-model prediction example first to create x_new.",
      "# That script reuses fitted encoding, imputation and scaling; it does not refit.",
      "settings <- bundle$prepared",
      "stopifnot(is.matrix(x_new), ncol(x_new) == length(settings$center))",
      "# Basic API probabilities; the full script also preserves Custom fusion.",
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
  if (is.null(config) && topic %in% c("Build & Train", "Hidden-layer sizes", "Epochs per model fit", "Optimizer",
      "Learning rate", "Self-organize before initial training", "Weights used for results",
      "Regularization", "Regularization strength"))
    prefix <- paste0(prefix, "# Default example values; correct invalid model settings for current values.\n")
  paste0(if (nzchar(prefix)) paste0(prefix, "\n") else "",
    wb_space_comments(snippet))
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
    notice = paste("Purpose: reload a saved model with its original settings and use it again. This began as a standalone script before the package API existed.",
      "This R example uses the current API to predict with an already trained model in a new session; it does not train again.",
      "A bundle is the .rds file downloaded with Save model, containing the model and preprocessing settings.",
      "Replace the two input paths with your saved bundle and a CSV of new rows containing the same predictor columns.",
      "The output is class probabilities for classification or numeric predictions for regression; saving copies is optional.", sep = "\n\n")))
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

# Custom controls are normalized at the server boundary. Basic uses the original
# configuration and execution path, including its RNG and numerical defaults.
wb_custom_config <- function(config, structure = "single", members = 1, seeds = "42",
                             temp = 0, activation = "relu", init = "he", dropout = 0,
                             schedule = "fixed", minimum = .00001, decay = .5,
                             interval = 20, fusion = "mean", seed_aggregate = "mean",
                             diagnostics = FALSE) {
  integer_in <- function(x, lo, hi, label) {
    if (length(x) != 1 || !is.finite(x) || x %% 1 != 0 || x < lo || x > hi)
      stop(label, " must be an integer from ", lo, " to ", hi, ".")
    as.integer(x)
  }
  structure <- match.arg(structure, c("single", "members", "main", "temp"))
  seeds <- suppressWarnings(as.numeric(trimws(strsplit(seeds, ",", fixed = TRUE)[[1]])))
  if (!length(seeds) || length(seeds) > 100 || any(!is.finite(seeds) | seeds < 0 |
      seeds > 2147483646 | seeds %% 1 != 0) || anyDuplicated(seeds))
    stop("Use 1–100 distinct training seeds, separated by commas.")
  config$workflow <- "custom"; config$structure <- structure
  config$members <- if (structure == "single") 1L else integer_in(members, 1, 100, "Members")
  config$seeds <- as.integer(seeds)
  config$temp <- if (structure == "temp") integer_in(temp, 1, 100, "Temporary iterations") else 0L
  config$activation <- match.arg(activation, c("relu", "tanh", "sigmoid", "selu", "leaky_relu"))
  config$init <- match.arg(init, c("he", "xavier", "lecun", "orthogonal", "variance_scaling", "glorot_uniform"))
  if (length(dropout) != 1 || !is.finite(dropout) || dropout < 0 || dropout >= 1)
    stop("Dropout must be at least 0 and below 1.")
  config$dropout <- dropout
  config$schedule <- match.arg(schedule, c("fixed", "exponential"))
  if (config$schedule == "exponential" && (length(minimum) != 1 || !is.finite(minimum) ||
      minimum <= 0 || minimum > config$lr || length(decay) != 1 || !is.finite(decay) || decay <= 0 || decay > 1))
    stop("Use 0 < minimum rate <= initial rate and 0 < decay multiplier <= 1.")
  config$minimum <- if (config$schedule == "fixed") config$lr else minimum
  config$decay <- if (config$schedule == "fixed") 1 else decay
  config$interval <- integer_in(interval, 1, 5000, "Decay interval")
  config$fusion <- match.arg(fusion, c("mean", "weighted", "median", "majority", "proportion"))
  config$seed_aggregate <- match.arg(seed_aggregate, c("mean", "median"))
  config$diagnostics <- isTRUE(diagnostics)
  config
}

wb_workload <- function(config) {
  number <- function(x) format(x, scientific = FALSE, trim = TRUE, big.mark = ",")
  if (!identical(config$workflow, "custom")) return(paste0(
    "Basic: 1 training run, 1 model fit, ", number(config$epochs), " epochs.\n",
    "Uses the Data-split seed from Features. The seed value is not a run count."))
  runs <- length(config$seeds)
  fits <- as.double(runs) * config$members * (1 + config$temp)
  total <- fits * config$epochs
  intro <- paste0(number(runs), " training run", if (runs == 1L) "" else "s",
    " (one complete run per listed seed value).")
  if (config$members == 1L && config$temp == 0L) {
    shown <- seq_len(min(runs, 3L))
    plan <- paste0("Run ", shown, " (seed ", config$seeds[shown], "): ",
      number(config$epochs), " epochs.", collapse = "\n")
    if (runs > 3L) plan <- paste0(plan, "\n", runs - 3L, " more runs with the same epoch budget.")
    budget <- paste0("Total: ", number(runs), " × ", number(config$epochs), " = ",
      number(total), " epochs.")
  } else {
    plan <- paste0("Each run trains ", number(config$members), " member networks for ",
      number(config$epochs), " epochs each.")
    if (config$temp > 0L) plan <- paste0(plan, " This repeats for the MAIN fit and each of the ",
      number(config$temp), " TEMP iterations.")
    budget <- paste0("Total: ", number(fits), " model fits × ", number(config$epochs),
      " epochs = ", number(total), " total member epochs.")
  }
  paste(intro, plan, budget,
    "Each run completes its full epoch budget before the next run starts. The prepared split stays fixed.",
    if (isTRUE(config$self_org)) "Initial self-organization adds work before each new network." else NULL,
    sep = "\n")
}

# Use a single calculation for evaluation, saved bundles and new predictions.
# A majority result is encoded as a one-hot decision (not a calibrated probability).
wb_combine <- function(predictions, mode, method = "mean", weights = NULL) {
  DDESONN:::.ddesonn_combine_predictions(predictions, mode, method, weights)
}

wb_custom_predictions <- function(model, prepared, part) {
  payload <- attr(model, "workbench_custom")
  predictions <- lapply(seq_along(payload$models), function(i) {
    raw <- DDESONN::ddesonn_predict(payload$models[[i]], part$x, aggregate = "none")$per_model
    wb_combine(raw, prepared$mode, payload$fusion, payload$weights[[i]])
  })
  wb_combine(predictions, prepared$mode, payload$seed_aggregate)
}

wb_custom_run <- function(prepared, config, previous = NULL) {
  if (prepared$mode == "regression" && config$fusion %in% c("majority", "proportion"))
    stop("Voting applies only to classification.")
  if (prepared$mode != "binary" && config$fusion == "proportion")
    stop("Positive-vote proportion applies only to binary classification.")
  tr <- prepared$parts$train; va <- prepared$parts$validation
  best <- identical(config$checkpoint, "best_validation")
  acts <- c(rep(config$activation, length(config$hidden)), switch(prepared$mode,
    binary = "sigmoid", multiclass = "softmax", regression = "identity"))
  # Resolve supported activation names to functions through the package namespace.
  acts <- lapply(acts, function(name) get(name, envir = asNamespace("DDESONN")))
  train <- list(num_epochs = config$epochs, optimizer = config$optimizer, lr = config$lr,
    lr_min = config$minimum, lr_decay_rate = config$decay, lr_decay_epoch = config$interval,
    self_org = is.null(previous) && config$self_org,
    do_ensemble = config$structure %in% c("main", "temp"),
    reg_type = if (config$reg == "none") NULL else config$reg,
    dropout_rates = as.list(rep(config$dropout, max(1, length(config$hidden)))),
    numeric_columns = if (is.null(prepared$preprocessing)) NULL else integer(0), batch_normalize_data = FALSE,
    validation_metrics = best, best_weights_on_latest_weights_off = best,
    threshold = .5, viewTables = FALSE, verboseLow = TRUE, verbose = config$diagnostics)
  if (is.null(previous)) {
    run <- DDESONN::ddesonn_run(tr$x, tr$y, classification_mode = prepared$mode,
      hidden_sizes = config$hidden, seeds = config$seeds,
      do_ensemble = config$structure %in% c("main", "temp"), num_networks = config$members,
      num_temp_iterations = config$temp, validation = va,
      model_overrides = list(ML_NN = length(config$hidden) > 0, lambda = config$lambda,
        activation_functions = acts, activation_functions_predict = acts, init_method = config$init),
      training_overrides = train, save_models = FALSE, verboseLow = TRUE,
      verbose = config$diagnostics)
    models <- lapply(run$runs, function(r) r$main$model)
    orchestration <- lapply(run$runs, `[[`, "tables")
  } else {
    payload <- attr(previous$model, "workbench_custom")
    if (is.null(payload)) stop("This saved model has no Custom training state.")
    # Native recipe calls also preserve the saved source bundle. The app adds
    # process isolation on top of this copy.
    models <- lapply(payload$models, wb_clone_model)
    for (i in seq_along(models)) {
      set.seed(config$seeds[i])
      do.call(DDESONN::ddesonn_fit, c(list(model = models[[i]], x = tr$x, y = tr$y, validation = va), train))
    }
    orchestration <- previous$orchestration
  }
  # Every member has its own best tensors; never reuse the last member's tensors.
  if (best) for (model in models) for (network in model$ensemble) {
    if (is.null(network$best_weights) || is.null(network$best_biases))
      stop("A member has no valid validation checkpoint; no completed model was replaced.")
    network$weights <- network$best_weights; network$biases <- network$best_biases
  }
  weights <- lapply(models, function(model) {
    preds <- DDESONN::ddesonn_predict(model, va$x, aggregate = "none")$per_model
    score <- vapply(preds, function(p) {
      if (prepared$mode == "regression") 1 / max(mean((p - va$y)^2), 1e-12) else
        wb_evaluate(prepared, va, p)$metrics$Value[1]
    }, numeric(1))
    if (sum(score) == 0) score[] <- 1
    score / sum(score)
  })
  per_seed <- lapply(seq_along(models), function(i) {
    p <- wb_combine(DDESONN::ddesonn_predict(models[[i]], va$x, aggregate = "none")$per_model,
      prepared$mode, config$fusion, weights[[i]])
    m <- wb_evaluate(prepared, va, p)$metrics
    data.frame(seed = config$seeds[i], m)
  })
  # Keep the root model API-compatible; serialize the other seeds as model state.
  root <- wb_clone_model(models[[1]])
  attr(root, "workbench_custom") <- list(models = models, weights = weights,
    fusion = config$fusion, seed_aggregate = config$seed_aggregate)
  evaluation <- lapply(prepared$parts, function(part)
    wb_evaluate(prepared, part, wb_custom_predictions(root, prepared, part)))
  list(format = "DDESONN-workbench-1", package_version = as.character(utils::packageVersion("DDESONN")),
    model = root, prepared = prepared, config = config, evaluation = evaluation,
    self_org_applied = is.null(previous) && config$self_org,
    selected_epoch = if (best) NA_integer_ else config$epochs,
    seed_comparison = do.call(rbind, per_seed), member_weights = weights, orchestration = orchestration,
    recipe = if (is.null(previous)) wb_recipe(prepared, config) else wb_resume_recipe(prepared, config), completed = format(Sys.time(), tz = "UTC", usetz = TRUE),
    fit_runs = if (is.null(previous)) 1L else previous$fit_runs + 1L)
}


wb_fusion_choices <- function(mode) {
  if (identical(mode, "regression")) return(c("Average predictions" = "mean",
    "Validation-weighted predictions" = "weighted", "Median predictions" = "median"))
  choices <- c("Average probabilities" = "mean", "Validation-weighted probabilities" = "weighted",
    "Median probabilities" = "median", stats::setNames("majority",
      if (identical(mode, "binary")) "Majority voting" else "Majority voting (plurality)"))
  if (identical(mode, "binary")) choices <- c(choices, "Positive-vote proportion" = "proportion")
  choices
}


wb_clone_model <- function(model) {
  # R6 deep cloning does not recursively copy the networks inside ensemble's
  # list. Serialization preserves API attributes AND isolates nested R6 state,
  # matching the worker boundary used by the app.
  unserialize(serialize(model, NULL))
}

wb_bundle_manifest <- function(bundle) {
  p <- bundle$prepared; cfg <- bundle$config
  data.frame(Component = c("Saved run", "Task and predictors", "Trained parameters", "Prepared records", "Preprocessing", "Settings", "Results and history", "Logs", "Recipe", "Top-level fields"),
    Contents = c(paste("DDESONN", bundle$package_version, "completed", bundle$completed),
      paste(p$mode, "·", paste(p$features, collapse = ", ")),
      if (identical(cfg$workflow, "custom")) paste(length(cfg$seeds), "seeds ×", cfg$members, if (cfg$structure %in% c("main", "temp")) "MAIN members; complete weights and biases; saved member/seed combination and weights" else "members per model; complete weights and biases; saved member/seed combination and weights") else "Selected model's complete weights and biases",
      paste(paste(names(p$parts), vapply(p$parts, function(part) nrow(part$x), integer(1)), sep = ": "), collapse = "; "),
      "Raw/encoded predictor order, fitted encoding and imputation rules, training centers/scales, row indices, split data and labels",
      paste(vapply(names(cfg), function(name) paste0(name, " = ", paste(cfg[[name]], collapse = ",")), character(1)), collapse = "; "), paste(length(bundle$evaluation), "evaluation splits;", if (is.data.frame(bundle$history)) nrow(bundle$history) else 0L, "recorded loss rows"),
      paste(length(bundle$training_log), "retained log lines"),
      if (is.null(bundle$recipe)) "No saved replay recipe in this older bundle" else paste(length(strsplit(bundle$recipe, "\n")[[1]]), "lines of saved R code"),
      paste(names(bundle), collapse = ", ")), stringsAsFactors = FALSE)
}


# Inspection never changes the data supplied to preparation or prediction.
wb_inspect_rows <- function(data, view = "head", n = 5, start = 1, columns = names(data)) {
  bounded <- function(x, default, maximum) {
    if (length(x) != 1L || !is.finite(x)) x <- default
    as.integer(max(1, min(maximum, floor(x))))
  }
  n <- bounded(n, 5, 100)
  start <- bounded(start, 1, max(1, nrow(data)))
  ids <- seq_len(nrow(data))
  if (identical(view, "tail")) ids <- utils::tail(ids, n) else {
    if (!identical(view, "range")) start <- 1L
    ids <- ids[ids >= start & ids < start + n]
  }
  columns <- intersect(columns, names(data))
  list(data = data[ids, columns, drop = FALSE], rows = ids, columns = columns,
    expression = if (identical(view, "head") || identical(view, "tail"))
      paste0(view, "(data[, ", wb_literal(columns), ", drop = FALSE], ", n, ")") else
      paste0("data[", wb_literal(ids), ", ", wb_literal(columns), ", drop = FALSE]"))
}

wb_data_profile <- function(data) {
  rows <- lapply(names(data), function(name) {
    x <- data[[name]]
    missing <- is.na(x) | (is.character(x) & trimws(x) == "")
    values <- if (is.numeric(x)) x[is.finite(x)] else numeric()
    q <- if (length(values)) stats::quantile(values, c(0, .25, .5, .75, 1), names = FALSE) else rep(NA_real_, 5)
    data.frame(Column = name, Type = class(x)[1], Missing = sum(missing),
      Nonfinite = if (is.numeric(x)) sum(!is.finite(x) & !is.na(x)) else 0L,
      Distinct = length(unique(x[!missing])), Mean = if (length(values)) mean(values) else NA_real_,
      SD = if (length(values) > 1) stats::sd(values) else NA_real_,
      Min = q[1], Q1 = q[2], Median = q[3], Q3 = q[4], Max = q[5], check.names = FALSE)
  })
  do.call(rbind, rows)
}

# Portable preprocessing: these functions are also embedded in generated recipes.
wb_pp_config <- function(mode = "auto", numeric = "auto", encoding = "auto",
                         categorical_missing = "auto", identifiers = "auto", overrides = list()) {
  mode <- match.arg(mode, c("auto", "manual"))
  resolve <- function(x, choices, automatic, manual) {
    x <- match.arg(x, c("auto", choices))
    if (x == "auto") if (mode == "auto") automatic else manual else x
  }
  list(mode = mode,
    numeric = resolve(numeric, c("median", "mean", "zero", "error"), "median", "error"),
    encoding = resolve(encoding, c("onehot", "ordinal", "error"), "onehot", "error"),
    categorical_missing = resolve(categorical_missing, c("level", "mode", "error"), "level", "error"),
    identifiers = resolve(identifiers, c("exclude", "keep"), "exclude", "keep"), overrides = overrides)
}

wb_pp_plan <- function(data, features, config) {
  do.call(rbind, lapply(features, function(name) {
    x <- data[[name]]; o <- config$overrides[[name]]
    action <- if (is.null(o$action) || o$action == "auto") {
      if (config$identifiers == "exclude" && grepl("(^id$|_id$|^id_)", name, ignore.case = TRUE)) "exclude" else "keep"
    } else match.arg(o$action, c("exclude", "keep"))
    numeric <- if (is.null(o$numeric) || o$numeric == "auto") config$numeric else o$numeric
    encoding <- if (is.null(o$encoding) || o$encoding == "auto") config$encoding else o$encoding
    missing <- if (is.null(o$missing) || o$missing == "auto") config$categorical_missing else o$missing
    data.frame(Column = name, Type = if (is.numeric(x)) "numeric" else "categorical",
      Role = action, Encoding = if (is.numeric(x) || action == "exclude") "none" else encoding,
      Missing = if (action == "exclude") "not used" else if (is.numeric(x)) numeric else missing,
      Reason = if (action == "exclude") "Identifier-name rule or explicit column override" else if (is.numeric(x))
        if (numeric == "zero") "Retain numeric values; fill missing/nonfinite values with zero" else if (numeric == "error") "Require complete, finite numeric values" else
          "Learn replacement from training rows only" else if (encoding == "ordinal")
        "Use explicit low-to-high order; missing = 0, unseen = -1" else "Use training categories; reserve Missing/Unknown inputs", stringsAsFactors = FALSE)
  }))
}

wb_pp_fit <- function(data, features, config) {
  plan <- wb_pp_plan(data, features, config)
  features <- plan$Column[plan$Role == "keep"]
  if (!length(features)) stop("Preprocessing excludes every predictor. Keep at least one column.")
  specs <- lapply(features, function(name) {
    x <- data[[name]]; row <- plan[plan$Column == name, ]; o <- config$overrides[[name]]
    if (is.numeric(x)) {
      method <- row$Missing
      if (!method %in% c("median", "mean", "zero", "error")) stop("Unsupported numeric missing-value method.")
      values <- x[is.finite(x)]
      if (!length(values) && method %in% c("median", "mean"))
        stop(paste(name, "has no finite training values. Exclude it or choose zero filling."))
      fill <- switch(method, median = stats::median(values), mean = mean(values), zero = 0, error = NA_real_)
      return(list(name = name, type = "numeric", missing = method, fill = fill, columns = name))
    }
    if (!is.character(x) && !is.factor(x) && !is.logical(x)) stop(paste("Unsupported column type:", name))
    x <- trimws(as.character(x)); x[is.na(x) | x == ""] <- NA_character_
    method <- row$Missing
    if (!method %in% c("level", "mode", "error")) stop("Unsupported categorical missing-value method.")
    values <- sort(unique(x[!is.na(x)]))
    if (length(values) > 200) stop(paste(name, "has over 200 training categories. Exclude identifiers/free text or preprocess it separately."))
    if (method == "mode" && !length(values)) stop(paste(name, "has no observed training category. Choose a Missing level or exclude it."))
    fill <- if (method == "mode") names(sort(table(x), decreasing = TRUE))[1] else NA_character_
    encoding <- row$Encoding
    if (!encoding %in% c("onehot", "ordinal")) stop(paste("Choose one-hot or ordinal encoding for", name))
    if (encoding == "ordinal") {
      order <- o$order
      if (!length(order) || anyNA(order) || any(!nzchar(order)) || anyDuplicated(order) || any(!values %in% order))
        stop(paste("Specify a unique low-to-high ordinal order containing every training category for", name))
      values <- order
    }
    columns <- if (encoding == "ordinal") name else paste0(name, "__", c(values, "[Missing]", "[Unknown]"))
    list(name = name, type = "categorical", missing = method, fill = fill,
      encoding = encoding, levels = values, columns = columns)
  })
  columns <- make.unique(unlist(lapply(specs, `[[`, "columns")), sep = "__")
  if (length(columns) > 2000) stop("Preprocessing creates over 2,000 inputs. Reduce categorical cardinality or selected columns.")
  at <- 0L
  for (i in seq_along(specs)) {
    n <- length(specs[[i]]$columns); specs[[i]]$columns <- columns[at + seq_len(n)]; at <- at + n
  }
  list(version = 1L, features = features, specs = specs, columns = columns, config = config, plan = plan)
}

wb_pp_apply <- function(data, preprocessing) {
  if (any(!preprocessing$features %in% names(data))) stop("Prediction data must include every saved predictor.")
  matrices <- lapply(preprocessing$specs, function(s) {
    x <- data[[s$name]]
    if (s$type == "numeric") {
      if (!is.numeric(x)) stop(paste(s$name, "must remain numeric, as in the training data."))
      bad <- !is.finite(x)
      if (any(bad) && s$missing == "error") stop(paste(s$name, "contains missing/nonfinite values. Choose a missing-value treatment."))
      x[bad] <- s$fill
      return(matrix(x, ncol = 1, dimnames = list(NULL, s$columns)))
    }
    x <- trimws(as.character(x)); bad <- is.na(x) | x == ""
    if (any(bad) && s$missing == "error") stop(paste(s$name, "contains missing categories. Choose a Missing level or mode filling."))
    if (s$missing == "mode") { x[bad] <- s$fill; bad[] <- FALSE }
    k <- match(x, s$levels)
    # Ordinal: 0 means missing and -1 means unseen; known levels are 1..K.
    if (s$encoding == "ordinal") {
      k[is.na(k)] <- -1L; k[bad] <- 0L
      return(matrix(as.numeric(k), ncol = 1, dimnames = list(NULL, s$columns)))
    }
    k[is.na(k)] <- length(s$levels) + 2L; k[bad] <- length(s$levels) + 1L
    m <- matrix(0, nrow = length(x), ncol = length(s$columns), dimnames = list(NULL, s$columns))
    m[cbind(seq_along(k), k)] <- 1
    m
  })
  out <- do.call(cbind, matrices)
  if (any(!is.finite(out))) stop("Preprocessing produced nonfinite inputs.")
  out
}

wb_pp_code <- function() {
  functions <- c("wb_pp_config", "wb_pp_plan", "wb_pp_fit", "wb_pp_apply")
  paste(vapply(functions, function(name) paste0(name, " <- ", paste(deparse(get(name)), collapse = "\n")), character(1)), collapse = "\n\n")
}

wb_prediction_inputs <- function(data, p) {
  x <- if (!is.null(p$preprocessing)) wb_pp_apply(data, p$preprocessing) else {
    x <- data[p$features]
    if (!all(vapply(x, function(v) is.numeric(v) && all(is.finite(v)), logical(1))))
      stop("Prediction features must be finite numeric values.")
    as.matrix(x)
  }
  sweep(sweep(x, 2, p$center, "-"), 2, p$scale, "/")
}

wb_pp_summary <- function(preprocessing) {
  do.call(rbind, lapply(preprocessing$specs, function(s) data.frame(
    Column = s$name, Encoding = if (s$type == "numeric") "numeric" else s$encoding,
    Missing = s$missing,
    Replacement = if (!is.na(s$fill)) as.character(s$fill) else if (s$missing == "level") "Separate Missing level" else "Reject missing",
    Training_categories = if (s$type == "numeric") "not applicable" else paste(utils::head(s$levels, 20), collapse = " | "),
    Output_inputs = length(s$columns), stringsAsFactors = FALSE)))
}
