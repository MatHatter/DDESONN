# Store only report metadata, never input data, environments or model tensors.
.report_value <- function(x) {
  if (!length(x) || all(is.na(x))) return("Unavailable")
  if (is.logical(x)) x <- ifelse(is.na(x), "Unavailable", ifelse(x, "Enabled", "Disabled"))
  paste(x, collapse = ", ")
}

.report_dataset <- function(name) {
  if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) return(NULL)
  basename(chartr(intToUtf8(92L), "/", name))
}

.report_activations <- function(activations) {
  ns <- asNamespace("DDESONN")
  known <- c(list(identity = base::identity), mget(getNamespaceExports(ns), envir = ns))
  if (is.function(activations)) activations <- list(activations)
  vapply(activations, function(fn) {
    if (is.character(fn)) return(paste(fn, collapse = ", "))
    matches <- names(Filter(function(candidate) identical(candidate, fn), known))
    if (length(matches)) matches[1L] else {
      label <- attr(fn, "name")
      if (is.character(label) && length(label) == 1L) label else "Custom function"
    }
  }, character(1))
}

.report_fit <- function(model, cfg, result, elapsed, data, output_size, mode, reduction) {
  hidden <- if (isTRUE(attr(model, "ML_NN"))) attr(model, "hidden_sizes")
  lambda <- attr(model, "lambda")
  normalization <- if (length(cfg$numeric_columns)) .report_value(cfg$numeric_columns) else "Disabled"
  sequence <- attr(model, "sequence_config")
  monitor_validation <- isTRUE(cfg$validation_metrics) && NROW(cfg$X_validation) > 0L
  validation_loss <- if (!monitor_validation) "Disabled" else if (mode != "regression" &&
      identical(cfg$loss_type, "cross_entropy")) "Cross-entropy" else "Mean squared error"
  settings <- list(
    Architecture = paste(c(ncol(data), hidden, output_size), collapse = " \u2192 "),
    `Training activations` = .report_activations(cfg$activation_functions),
    `Prediction activations` = .report_activations(cfg$activation_functions_predict),
    Initialization = attr(model, "report_initialization"),
    Optimizer = cfg$optimizer,
    `Optimizer beta 1` = cfg$beta1, `Optimizer beta 2` = cfg$beta2,
    `Optimizer epsilon` = cfg$epsilon, `Lookahead interval` = cfg$lookahead_step,
    `Initial learning rate` = cfg$lr,
    `Learning rate schedule` = if (isTRUE(cfg$lr_decay_rate == 1))
      sprintf("Fixed at %s", max(cfg$lr, cfg$lr_min)) else
      sprintf("Multiply by %s every %s epochs (minimum %s)", cfg$lr_decay_rate, cfg$lr_decay_epoch, cfg$lr_min),
    `Training batch size` = paste(nrow(data), "(full training split)"),
    `Normalization batch size` = if (isTRUE(cfg$batch_normalize_data)) {
      batch <- model$calculate_batch_size(data)
      if (batch %% 1 == 0) min(nrow(data), batch) else paste("Up to", ceiling(batch))
    } else "Disabled",
    `Gradient reduction` = .ddesonn_gradient_label(reduction),
    `Training loss` = cfg$loss_type,
    Regularization = if (!length(cfg$reg_type) || isTRUE(lambda == 0)) "Disabled" else
      sprintf("%s (lambda %s)", .report_value(cfg$reg_type), .report_value(lambda)),
    Dropout = if (length(unlist(cfg$dropout_rates)) && any(unlist(cfg$dropout_rates) > 0, na.rm = TRUE))
      unlist(cfg$dropout_rates) else "Disabled",
    `Early stopping` = "Disabled",
    `Validation monitoring` = monitor_validation,
    `Validation loss measure` = validation_loss,
    `Best-weight evaluation option` = isTRUE(cfg$best_weights_on_latest_weights_off),
    `Input normalization columns` = normalization,
    `Batch normalization` = cfg$batch_normalize_data,
    `Batch normalization gamma / beta` = if (isTRUE(cfg$batch_normalize_data))
      paste(.report_value(cfg$gamma_bn), "/", .report_value(cfg$beta_bn)) else "Disabled",
    `Batch normalization momentum / epsilon` = if (isTRUE(cfg$batch_normalize_data))
      paste(.report_value(cfg$momentum_bn), "/", .report_value(cfg$epsilon_bn)) else "Disabled",
    `Shuffle for batch normalization` = if (isTRUE(cfg$batch_normalize_data)) cfg$shuffle_bn else FALSE,
    `Self-organization` = cfg$self_org,
    `Update weights` = cfg$update_weights, `Update biases` = cfg$update_biases,
    `Sample weighting` = !is.null(cfg$sample_weights),
    `Supplied preprocessing statistics` = !is.null(cfg$preprocessScaledData),
    `Binary threshold setting` = if (identical(mode, "binary")) cfg$threshold else "Not applicable",
    `Tuned binary threshold` = if (identical(mode, "binary") && isTRUE(cfg$validation_metrics))
      result$performance_relevance_data$threshold %||% "Unavailable" else "Disabled",
    `Sequence encoder` = if (identical(sequence$encoder, "ssm"))
      sprintf("SSM, length %s, state dimension %s, convolution width %s",
        sequence$sequence_length, sequence$state_dim, sequence$conv_width) else "Disabled"
  )
  list(settings = settings, members = result$report_members, requested = cfg$num_epochs,
    elapsed = elapsed, trained = isTRUE(cfg$train_flag), validation_rows = NROW(cfg$X_validation),
    checkpoint = if (isTRUE(cfg$train_flag)) "Final epoch" else "Existing weights")
}

.report_fit_checkpoint <- function(fit, checkpoint = NULL) {
  if (is.null(fit)) return(NULL)
  if (!is.null(checkpoint)) {
    fit$checkpoint <- checkpoint
    if (identical(checkpoint, "Best validation checkpoint"))
      fit$members <- lapply(fit$members, function(member) {
        member$prediction_epoch <- member$best_checkpoint_epoch
        member
      })
  }
  fit
}

.report_model_fit <- function(model, label, checkpoint = NULL) {
  fit <- .report_fit_checkpoint(model$last_training$run_report_fit, checkpoint)
  if (!is.null(fit)) fit$label <- label
  fit
}

.report_scenario <- function(ensemble, members, temp) {
  if (isTRUE(ensemble) && isTRUE(temp > 0L)) return("D | Main ensemble with TEMP models")
  if (isTRUE(ensemble)) return("C | Main ensemble")
  if (isTRUE(members > 1L)) return("B | Multiple independently trained models")
  "A | Single model"
}

.report_field_lines <- function(fields) {
  if (!length(fields)) return(character())
  paste0(names(fields), ": ", vapply(fields, .report_value, character(1)))
}

.report_epoch_range <- function(values) {
  if (!length(values) || anyNA(values)) return("Unavailable")
  paste(unique(range(values)), collapse = "\u2013")
}

.report_best_loss <- function(member) {
  if (is.finite(member$best_loss) && !is.na(member$best_epoch))
    sprintf("%.7g at epoch %s", member$best_loss, member$best_epoch) else NA_character_
}

.report_heading <- function(title) paste("=====", title, "=====")

# Fit intervals include training-time validation, and prediction_elapsed counts
# only separately measured prediction or scoring work outside those intervals.
.report_timing <- function(stage) {
  fits <- stage$fits
  if (!length(fits) || !length(stage$elapsed) || !length(stage$prediction_elapsed))
    return(character())
  finite_seconds <- function(x) length(x) == 1L && is.finite(x) && x >= 0
  times <- lapply(fits, `[[`, "elapsed")
  if (!all(vapply(c(times, list(stage$elapsed, stage$prediction_elapsed)), finite_seconds, logical(1))))
    return(character())
  times <- unlist(times, use.names = FALSE)
  labels <- vapply(fits, `[[`, character(1), "label")
  other <- stage$elapsed - sum(times) - stage$prediction_elapsed
  if (other < -1e-8) return(character())
  main <- grepl(" MAIN$", labels)
  temp <- grepl(" TEMP ", labels, fixed = TRUE)
  lines <- function(label, seconds) sprintf("%s: %.2f s", label, seconds)
  c(
    stage$timing_scope,
    if (any(main)) lines("**MAIN fits**", sum(times[main])),
    if (any(temp)) lines("TEMP fits", sum(times[temp])),
    if (any(!(main | temp))) lines("**Model fits**", sum(times[!(main | temp)])),
    lines("Measured prediction / scoring outside fits", stage$prediction_elapsed),
    lines("Other run work", max(0, other)),
    lines("Total measured", stage$elapsed),
    "Fits include any prediction and validation performed inside training."
  )
}

.report_format <- function(mode, scenario, dataset, counts, stages, provenance,
                           preprocessing = list(), excluded = NULL, overlap = NULL) {
  task <- switch(mode %||% "", binary = "Binary classification",
    multiclass = "Multiclass classification", regression = "Regression", "Unavailable")
  latest <- if (length(stages)) stages[[length(stages)]] else list()
  total <- sum(counts, na.rm = TRUE)
  split <- if (length(counts)) {
    shares <- if (total > 0L) 100 * counts / total else rep(0, length(counts))
    paste0(names(counts), ": ", ifelse(is.na(counts), "Unavailable",
      paste0(counts, " | ", format(round(shares, 2), trim = TRUE), "%")))
  }
  fits <- latest$fits
  members <- unlist(lapply(fits, `[[`, "members"), recursive = FALSE)
  completed <- vapply(members, function(m) m$completed, integer(1))
  requested <- vapply(fits, function(fit) as.integer(fit$requested), integer(1)) %||% latest$requested
  seconds <- latest$elapsed
  stopping <- latest$stopping %||% if (length(fits)) {
    if (!any(vapply(fits, `[[`, logical(1), "trained"))) "Training disabled"
    else if (!length(members)) "Unavailable" else "Epoch budget completed"
  } else "Unavailable"
  best <- vapply(members, .report_best_loss, character(1))
  best_summary <- if (length(best) == 1L) best else if (any(!is.na(best)))
    "See per-model results in the training stages"
  cumulative <- latest$cumulative
  if (is.null(cumulative) && length(stages) == 1L) cumulative <- .report_epoch_range(completed)
  epoch_pair <- paste(.report_epoch_range(completed), "/", .report_epoch_range(requested))
  checkpoint <- latest$checkpoint %||% unique(vapply(fits, `[[`, character(1), "checkpoint"))
  overlap_line <- paste("Split overlap:", .overlap_text(overlap))
  identity <- .report_field_lines(list(Scenario = scenario, Task = task, Dataset = dataset))
  latest_timing <- .report_timing(latest)
  overview <- c(identity, "", .report_heading("Data splits"), split, excluded, overlap_line,
    "", .report_heading("Training and checkpoint"), latest$overview,
    .report_field_lines(list(
      `Epochs completed / requested per model (latest stage)` = epoch_pair,
      `Cumulative epochs per model across recorded stages` = cumulative,
      `Stopping reason` = stopping,
      `Best validation loss` = best_summary,
      `Prediction checkpoint` = checkpoint)),
    if (length(latest_timing)) c("", .report_heading("Timing"), latest_timing)
    else if (length(seconds) && is.finite(seconds)) sprintf("Elapsed time (latest stage): %.2f s", seconds),
    if (length(provenance)) c("", .report_heading("Provenance"), .report_field_lines(provenance)))

  details <- c(.report_heading("Effective configuration"),
    .report_field_lines(preprocessing))
  for (i in seq_along(stages)) {
    stage <- stages[[i]]
    stage_title <- stage$title %||%
      paste("Training stage", i, if (i == 1L) "(initial)" else "(continued)")
    timing <- .report_timing(stage)
    details <- c(details, "", .report_heading(stage_title),
      if (!length(timing) && length(stage$elapsed)) sprintf("Elapsed time: %.2f s", stage$elapsed),
      .report_field_lines(c(list(`DDESONN version` = stage$package_version,
        `R version` = stage$r_version), stage$settings)))
    settings <- lapply(stage$fits, `[[`, "settings")
    groups <- unique(settings)
    for (group in groups) {
      same <- vapply(settings, identical, logical(1), y = group)
      labels <- vapply(stage$fits[same], `[[`, character(1), "label")
      details <- c(details, paste(labels, collapse = ", "), .report_field_lines(group))
    }
    if (i < length(stages) && length(timing))
      details <- c(details, "", .report_heading("Timing"), timing)
    if (length(stage$fits)) details <- c(details, "", .report_heading("Per-model outcomes"))
    for (fit in stage$fits) {
      label <- if (grepl(" TEMP ", fit$label, fixed = TRUE)) fit$label else paste0("**", fit$label, "**")
      details <- c(details, sprintf("%s \u2014 %s (fit time %.2f s)", label, fit$checkpoint, fit$elapsed))
      for (j in seq_along(fit$members)) {
        member <- fit$members[[j]]
        loss <- .report_value(.report_best_loss(member))
        details <- c(details, sprintf("  Model %s: %s / %s epochs | Best validation loss: %s | Retained weights from epoch: %s",
          j, member$completed, fit$requested, loss, .report_value(member$prediction_epoch)),
          if (length(member$elapsed) && is.finite(member$elapsed))
            sprintf("    Network fit: %.2f s (included in fit time above)", member$elapsed))
      }
    }
  }
  compact <- c(identity, split, excluded, .report_field_lines(list(
    `Epochs completed / requested per model` = epoch_pair, `Prediction checkpoint` = checkpoint)),
    overlap_line)
  list(overview = overview, details = details, compact = compact, overlap = overlap)
}

.report_text <- function(report) c(.report_heading("DDESONN run report"), report$overview, "", report$details)

.report_console_text <- function(report, verbose, verbose_low) {
  if (isTRUE(verbose)) return(.report_text(report))
  if (isTRUE(verbose_low)) return(c("", .report_heading("DDESONN run summary"), "", report$compact))
  character()
}
