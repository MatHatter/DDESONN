# Reproducible DDESONN pilot with exact predictor groups kept in one split.
# Source this file to use the helpers; execute it with Rscript to run a pilot.
clean_split_prepare <- function(data, split_seed = 42L) {
  helper <- system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN")
  h <- new.env(parent = globalenv()); sys.source(helper, h)
  features <- setdiff(names(data), "DEATH_EVENT")
  p <- h$wb_prepare(data, "DEATH_EVENT", features, "binary", seed = split_seed,
    train_fraction = .7, validation_fraction = .15)
  keys <- apply(as.matrix(data[features]), 1, function(row)
    digest::digest(unname(row), algo = "sha256"))
  sets <- lapply(p$parts, function(part) unique(keys[part$row]))
  overlaps <- c(train_validation = length(intersect(sets$train, sets$validation)),
    train_test = length(intersect(sets$train, sets$test)),
    validation_test = length(intersect(sets$validation, sets$test)))
  stopifnot(all(overlaps == 0L),
    identical(sort(unlist(lapply(p$parts, `[[`, "row"), use.names = FALSE)), seq_len(nrow(data))))
  # One shared divisor, fitted only on training rows, following the historical
  # preprocessing convention after training-derived mean/SD standardization.
  divisor <- max(abs(p$parts$train$x))
  if (!is.finite(divisor) || divisor <= 0) stop("Invalid training scale.")
  for (name in names(p$parts)) {
    p$parts[[name]]$x <- p$parts[[name]]$x / divisor
    p$parts[[name]]$group <- keys[p$parts[[name]]$row]
  }
  p$max_abs_training_divisor <- divisor
  p$independent_overlap_audit <- overlaps
  p
}

clean_split_metrics <- function(actual, probability, groups) {
  actual <- as.numeric(actual); probability <- as.numeric(probability)
  if (!length(actual) || length(actual) != length(probability) ||
      length(groups) != length(actual) || anyNA(groups) ||
      any(!actual %in% c(0, 1)) || any(!is.finite(probability)) ||
      any(probability < 0 | probability > 1)) stop("Invalid evaluation inputs.")
  prediction <- as.integer(probability >= .5)
  correct <- prediction == actual
  p <- pmin(pmax(probability, 1e-15), 1 - 1e-15)
  data.frame(rows = length(actual), predictor_groups = length(unique(groups)),
    accuracy = mean(correct), group_weighted_accuracy = mean(tapply(correct, groups, mean)),
    majority_baseline_accuracy = max(mean(actual == 0), mean(actual == 1)),
    log_loss = -mean(actual * log(p) + (1 - actual) * log1p(-p)),
    brier = mean((probability - actual)^2),
    TP = sum(prediction == 1 & actual == 1), TN = sum(prediction == 0 & actual == 0),
    FP = sum(prediction == 1 & actual == 0), FN = sum(prediction == 0 & actual == 1))
}

clean_split_fit <- function(prepared, seed, epochs = 360L, observe_model = NULL, average_gradients = NULL) {
  stopifnot(length(seed) == 1L, is.finite(seed), seed >= 0, seed == floor(seed),
    length(epochs) == 1L, is.finite(epochs), epochs >= 1, epochs == floor(epochs))
  # Legacy diagnostic plotting must not create Rplots.pdf in the caller directory.
  previous_device <- grDevices::dev.cur()
  previous_devices <- grDevices::dev.list()
  grDevices::pdf(file = NULL)
  on.exit({
    created_devices <- setdiff(grDevices::dev.list(), previous_devices)
    for (device in rev(created_devices)) grDevices::dev.off(device)
    if (previous_device %in% grDevices::dev.list()) grDevices::dev.set(previous_device)
  }, add = TRUE)
  set.seed(seed)
  model <- DDESONN::ddesonn_model(ncol(prepared$parts$train$x), 1,
    hidden_sizes = c(64L, 32L), classification_mode = "binary", lambda = .00028,
    init_method = "he", custom_scale = 1.04349, num_networks = 1L)
  # Optional experiment observer may record tensors/masks without changing them.
  if (!is.null(observe_model)) {
    cleanup <- observe_model(model)
    if (!is.function(cleanup)) stop("Model observer must return a cleanup function.")
    on.exit(cleanup(), add = TRUE)
  }
  tr <- prepared$parts$train; va <- prepared$parts$validation
  started <- proc.time()[["elapsed"]]
  invisible(capture.output(DDESONN::ddesonn_fit(model, tr$x, tr$y,
    validation = va[c("x", "y")], num_epochs = as.integer(epochs),
    optimizer = "adagrad", gradient_reduction = "legacy_mixed",
    average_gradients = average_gradients,
    lr = .125, lr_decay_rate = 1, lr_min = .125,
    reg_type = "L1", dropout_rates = list(.1, NULL, NULL), epsilon = 1e-7,
    self_org = FALSE, numeric_columns = integer(0), batch_normalize_data = FALSE,
    validation_metrics = TRUE, best_weights_on_latest_weights_off = TRUE,
    threshold = .5, viewTables = FALSE, verbose = FALSE, verboseLow = FALSE)))
  net <- model$ensemble[[1L]]
  epoch <- model$last_training$predicted_outputAndTime$best_val_epoch
  if (is.null(net$best_weights) || is.null(net$best_biases) ||
      length(epoch) != 1L || !is.finite(epoch)) stop("Missing validation checkpoint.")
  net$weights <- net$best_weights; net$biases <- net$best_biases
  predictions <- lapply(prepared$parts, function(part)
    as.numeric(DDESONN::ddesonn_predict(model, part$x)$prediction))
  metrics <- do.call(rbind, lapply(names(prepared$parts), function(name) {
    part <- prepared$parts[[name]]
    cbind(seed = seed, split = name, selected_epoch = epoch, requested_epochs = epochs,
      clean_split_metrics(part$y, predictions[[name]], part$group))
  }))
  list(model = model, predictions = predictions, metrics = metrics,
    elapsed_seconds = proc.time()[["elapsed"]] - started)
}

clean_split_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) < 1L || length(args) > 5L)
    stop("Usage: Rscript heart_failure_clean_split.R OUTPUT [SEEDS=1,2,3] [EPOCHS=360] [SPLIT_SEED=42] [AVERAGE_GRADIENTS=legacy_mixed|FALSE|TRUE]")
  output <- args[1L]
  seeds <- if (length(args) >= 2L) suppressWarnings(as.numeric(strsplit(args[2L], ",", fixed = TRUE)[[1L]])) else 1:3
  epochs <- if (length(args) >= 3L) suppressWarnings(as.numeric(args[3L])) else 360L
  split_seed <- if (length(args) >= 4L) suppressWarnings(as.numeric(args[4L])) else 42L
  reduction <- if (length(args) >= 5L) tolower(args[5L]) else "legacy_mixed"
  if (!reduction %in% c("legacy_mixed", "false", "true"))
    stop("Average gradients must be FALSE, TRUE or legacy_mixed.")
  average_gradients <- switch(reduction, "false" = FALSE, "true" = TRUE, legacy_mixed = NULL)
  reduction <- switch(reduction, "false" = "sum", "true" = "mean", legacy_mixed = "legacy_mixed")
  if (!length(seeds) || any(!is.finite(seeds)) || any(seeds < 0 | seeds > 2147483646 | seeds != floor(seeds)) ||
      anyDuplicated(seeds) || length(epochs) != 1L || !is.finite(epochs) || epochs < 1 || epochs != floor(epochs))
    stop("Seeds must be unique nonnegative integers; epochs must be a positive integer.")
  if (dir.exists(output) && length(list.files(output, all.files = TRUE, no.. = TRUE)))
    stop("Output directory must be new or empty; existing experiments are never overwritten.")
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  dataset <- system.file("extdata", "heart_failure_clinical_records.csv", package = "DDESONN")
  if (!requireNamespace("callr", quietly = TRUE)) stop("Install optional package 'callr' to run isolated seeds.")
  script <- system.file("scripts", "heart_failure_clean_split.R", package = "DDESONN")
  frozen_script <- file.path(output, "runner.R")
  if (!file.copy(script, frozen_script)) stop("Could not freeze the installed runner.")
  data <- utils::read.csv(dataset)
  prepared <- clean_split_prepare(data, split_seed)
  config <- list(package_version = as.character(utils::packageVersion("DDESONN")),
    dataset_sha256 = digest::digest(file = dataset, algo = "sha256"),
    runner_sha256 = digest::digest(file = frozen_script, algo = "sha256"),
    seed_execution = "fresh_R_process", native_binary_class_weights = c(negative = 1, positive = 2),
    seeds = seeds, epochs = epochs, split_seed = split_seed,
    train_group_fraction = .7, validation_group_fraction = .15,
    threshold = .5, hidden_sizes = c(64L,32L), optimizer = "adagrad",
    gradient_reduction = reduction, average_gradients = average_gradients,
    effective_dropout = c(.1,.1,0),
    lr = .125, lambda = .00028, reg_type = "L1", epsilon = 1e-7,
    dropout = list(.1,NULL,NULL), init_method = "he", custom_scale = 1.04349,
    checkpoint = "best_validation_accuracy", self_org = FALSE,
    note = "Preliminary current-API experiment; not a matched Keras comparison or exact replay of frozen historical training.")
  saveRDS(list(config = config, prepared = prepared), file.path(output, "split.rds"))
  utils::write.csv(data.frame(row = unlist(lapply(prepared$parts, `[[`, "row")),
    split = rep(names(prepared$parts), vapply(prepared$parts, function(p) length(p$row), integer(1)))),
    file.path(output, "split-rows.csv"), row.names = FALSE)
  writeLines(capture.output(dput(config)), file.path(output, "config.R"))
  writeLines(capture.output(sessionInfo()), file.path(output, "session-info.txt"))
  results <- list()
  for (seed in seeds) {
    cat("Training seed", seed, "for", epochs, "epochs; fixed group-disjoint split", split_seed, "\n")
    fit <- callr::r(function(script, prepared, seed, epochs, average_gradients) {
      h <- new.env(parent = globalenv()); sys.source(script, h)
      h$clean_split_fit(prepared, seed, epochs, average_gradients = average_gradients)
    }, args = list(script = normalizePath(frozen_script), prepared = prepared,
      seed = seed, epochs = epochs, average_gradients = average_gradients), libpath = .libPaths())
    artifact <- file.path(output, paste0("seed-", seed, ".rds"))
    saveRDS(fit, artifact)
    restored <- readRDS(artifact)
    for (name in names(prepared$parts)) {
      probabilities <- as.numeric(DDESONN::ddesonn_predict(restored$model, prepared$parts[[name]]$x)$prediction)
      stopifnot(identical(probabilities, fit$predictions[[name]]))
    }
    results[[length(results) + 1L]] <- fit$metrics
    utils::write.csv(do.call(rbind, results), file.path(output, "metrics.csv"), row.names = FALSE)
    cat("Seed", seed, "test accuracy:", fit$metrics$accuracy[fit$metrics$split == "test"],
      "; seconds:", fit$elapsed_seconds, "\n")
    rm(fit, restored); invisible(gc())
  }
  files <- list.files(output, full.names = TRUE)
  utils::write.csv(data.frame(file = basename(files), sha256 = vapply(files, function(f)
    digest::digest(file = f, algo = "sha256"), character(1))), file.path(output, "manifest.csv"), row.names = FALSE)
  writeLines("All requested seeds completed; saved-model predictions verified.", file.path(output, "COMPLETE.txt"))
}

if (sys.nframe() == 0L) clean_split_main()
