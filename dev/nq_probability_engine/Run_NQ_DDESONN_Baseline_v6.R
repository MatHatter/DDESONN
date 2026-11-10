#!/usr/bin/env Rscript

# ============================================================
# NQ INTRADAY PROBABILITY ENGINE — NORMAL DDESONN BASELINE
# ============================================================
# This script runs DDESONN WITHOUT the SSM sequence encoder.
# It uses chronological ObservationId-level partitions:
#   Train:      through 2024-12-24
#   Validation: 2025-01-08 through 2025-12-24
#   Holdout:    2026-01-08 onward
#
# Easiest use from RStudio:
#   1. Put this script and DDESONN_NQ_FeatureDataset.csv in one R project.
#   2. Open this script.
#   3. Click Source.
#
# Optional usage from PowerShell:
#   Rscript .\Run_NQ_DDESONN_Baseline.R `
#     .\DDESONN_NQ_FeatureDataset.csv `
#     .\NQ_DDESONN_Baseline_Results
# ============================================================

arguments <- commandArgs(trailingOnly = TRUE)
feature_input <- if (length(arguments) >= 1L) {
  arguments[[1L]]
} else {
  "DDESONN_NQ_FeatureDataset.csv"
}
feature_path <- normalizePath(feature_input, mustWork = TRUE)
results_path <- if (length(arguments) >= 2L) {
  arguments[[2L]]
} else {
  "NQ_DDESONN_Baseline_Results"
}
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("DDESONN", quietly = TRUE)) {
  stop(
    "The DDESONN package is not installed in this R environment.",
    call. = FALSE
  )
}

message("Reading NQ feature dataset: ", feature_path)
dataset <- utils::read.csv(
  feature_path,
  stringsAsFactors = FALSE,
  na.strings = c("NA", "")
)

required_columns <- c(
  "ObservationId", "ObservationTimeET", "Direction", "EntryPrice",
  "MatrixType", "TargetPct", "StopPct", "Horizon", "Outcome",
  "LabelTargetFirst", "DayOfWeek", "MinuteOfDay", "Atr",
  "DailyOpen", "DailyHigh", "DailyLow", "DailyClose",
  "DailyBodyAtr", "DailyRangeAtr", "Open5m", "High5m", "Low5m",
  "Close5m", "Return5m", "Return30m", "NearSupport",
  "NearResistance", "BullFlag", "BearFlag", "SupportDistanceAtr",
  "ResistanceDistanceAtr"
)
missing_columns <- setdiff(required_columns, names(dataset))
if (length(missing_columns)) {
  stop(
    "The feature dataset is missing: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

dataset$ObservationTime <- as.POSIXct(
  dataset$ObservationTimeET,
  format = "%Y-%m-%d %H:%M:%S",
  tz = "America/New_York"
)
if (anyNA(dataset$ObservationTime)) {
  stop("At least one ObservationTimeET value could not be parsed.", call. = FALSE)
}

# The supplied feature schema does not yet include Instrument. Protect this
# NQ experiment using the observed price range instead.
median_entry <- stats::median(dataset$EntryPrice, na.rm = TRUE)
if (!is.finite(median_entry) || median_entry < 10000) {
  stop(
    sprintf(
      "This does not appear to be NQ data. Median EntryPrice is %.2f.",
      median_entry
    ),
    call. = FALSE
  )
}

# Collision outcomes intentionally have no binary label. They are excluded.
dataset <- dataset[
  is.finite(dataset$LabelTargetFirst) & dataset$Outcome != "Collision",
  ,
  drop = FALSE
]
dataset <- dataset[order(dataset$ObservationTime, dataset$ObservationId), , drop = FALSE]
row.names(dataset) <- NULL

if (anyDuplicated(dataset[c(
  "ObservationId", "Direction", "MatrixType", "TargetPct", "StopPct", "Horizon"
)])) {
  stop("Duplicate modeling keys were found in the feature dataset.", call. = FALSE)
}

# Seven-day gaps protect the longest swing labels from crossing year splits.
train_end <- as.POSIXct("2024-12-24 23:59:59", tz = "America/New_York")
validation_start <- as.POSIXct("2025-01-08 00:00:00", tz = "America/New_York")
validation_end <- as.POSIXct("2025-12-24 23:59:59", tz = "America/New_York")
holdout_start <- as.POSIXct("2026-01-08 00:00:00", tz = "America/New_York")

dataset$Split <- NA_character_
dataset$Split[dataset$ObservationTime <= train_end] <- "Train"
dataset$Split[
  dataset$ObservationTime >= validation_start &
    dataset$ObservationTime <= validation_end
] <- "Validation"
dataset$Split[dataset$ObservationTime >= holdout_start] <- "Holdout"
dataset <- dataset[!is.na(dataset$Split), , drop = FALSE]

# All rows belonging to one market observation must remain in one partition.
split_count_by_observation <- tapply(
  dataset$Split,
  dataset$ObservationId,
  function(value) length(unique(value))
)
if (any(split_count_by_observation != 1L)) {
  stop("An ObservationId crossed chronological split boundaries.", call. = FALSE)
}

train_rows <- which(dataset$Split == "Train")
validation_rows <- which(dataset$Split == "Validation")
holdout_rows <- which(dataset$Split == "Holdout")
if (!length(train_rows) || !length(validation_rows) || !length(holdout_rows)) {
  stop("Train, validation, and holdout must all contain rows.", call. = FALSE)
}

# ============================================================
# CAUSAL CURRENT-STATE FEATURES
# ============================================================
# Outcome and LabelTargetFirst are never inputs. Raw price levels are converted
# into relative values so the model does not learn the calendar price regime.

relative_to_entry <- function(value, entry) {
  value / entry - 1
}

numeric_features <- data.frame(
  TargetPct = dataset$TargetPct,
  StopPct = dataset$StopPct,
  DaySin = sin(2 * pi * dataset$DayOfWeek / 7),
  DayCos = cos(2 * pi * dataset$DayOfWeek / 7),
  TimeSin = sin(2 * pi * dataset$MinuteOfDay / 1440),
  TimeCos = cos(2 * pi * dataset$MinuteOfDay / 1440),
  AtrPct = dataset$Atr / dataset$EntryPrice,
  DailyOpenRel = relative_to_entry(dataset$DailyOpen, dataset$EntryPrice),
  DailyHighRel = relative_to_entry(dataset$DailyHigh, dataset$EntryPrice),
  DailyLowRel = relative_to_entry(dataset$DailyLow, dataset$EntryPrice),
  DailyCloseRel = relative_to_entry(dataset$DailyClose, dataset$EntryPrice),
  DailyBodyAtr = dataset$DailyBodyAtr,
  DailyRangeAtr = dataset$DailyRangeAtr,
  Open5mRel = relative_to_entry(dataset$Open5m, dataset$EntryPrice),
  High5mRel = relative_to_entry(dataset$High5m, dataset$EntryPrice),
  Low5mRel = relative_to_entry(dataset$Low5m, dataset$EntryPrice),
  Close5mRel = relative_to_entry(dataset$Close5m, dataset$EntryPrice),
  Return5m = dataset$Return5m,
  Return30m = dataset$Return30m,
  NearSupport = dataset$NearSupport,
  NearResistance = dataset$NearResistance,
  BullFlag = dataset$BullFlag,
  BearFlag = dataset$BearFlag,
  SupportDistanceAtr = dataset$SupportDistanceAtr,
  ResistanceDistanceAtr = dataset$ResistanceDistanceAtr,
  stringsAsFactors = FALSE
)

categorical_features <- data.frame(
  Direction = factor(dataset$Direction, levels = c("Bearish", "Bullish")),
  MatrixType = factor(dataset$MatrixType, levels = c("Intraday", "Swing")),
  Horizon = factor(
    dataset$Horizon,
    levels = c("15m", "30m", "60m", "2h", "4h", "Close",
      "1D", "2D", "3D", "4D", "5D", "6D", "7D")
  )
)
categorical_matrix <- stats::model.matrix(
  ~ Direction + MatrixType + Horizon - 1,
  data = categorical_features
)

feature_matrix <- cbind(as.matrix(numeric_features), categorical_matrix)
storage.mode(feature_matrix) <- "double"

# Imputation and standardization are fitted on training data only.
training_medians <- apply(
  feature_matrix[train_rows, , drop = FALSE],
  2L,
  stats::median,
  na.rm = TRUE
)
training_medians[!is.finite(training_medians)] <- 0

for (column_index in seq_len(ncol(feature_matrix))) {
  missing <- !is.finite(feature_matrix[, column_index])
  feature_matrix[missing, column_index] <- training_medians[[column_index]]
}

training_centers <- colMeans(feature_matrix[train_rows, , drop = FALSE])
training_scales <- apply(
  feature_matrix[train_rows, , drop = FALSE],
  2L,
  stats::sd
)
training_scales[!is.finite(training_scales) | training_scales < 1e-12] <- 1

feature_matrix <- sweep(feature_matrix, 2L, training_centers, "-")
feature_matrix <- sweep(feature_matrix, 2L, training_scales, "/")
# Clip extreme standardized values without dropping the matrix dimensions.
feature_matrix[feature_matrix < -8] <- -8
feature_matrix[feature_matrix > 8] <- 8
# DDESONN's full-batch training path is most stable with inputs bounded to
# approximately [-1, 1]. This fixed divisor does not use holdout statistics.
feature_matrix <- feature_matrix / 8

if (any(!is.finite(feature_matrix))) {
  stop("The prepared feature matrix contains non-finite values.", call. = FALSE)
}

# DDESONN performs its own normalization when batch normalization is disabled.
# A zero-variance column would therefore be divided by zero inside the package.
training_feature_sd <- apply(
  feature_matrix[train_rows, , drop = FALSE],
  2L,
  stats::sd
)
keep_feature <- is.finite(training_feature_sd) & training_feature_sd > 1e-12
if (any(!keep_feature)) {
  message(
    "Removing zero-variance training features: ",
    paste(colnames(feature_matrix)[!keep_feature], collapse = ", ")
  )
  feature_matrix <- feature_matrix[, keep_feature, drop = FALSE]
  training_medians <- training_medians[keep_feature]
  training_centers <- training_centers[keep_feature]
  training_scales <- training_scales[keep_feature]
}

if (any(!is.finite(feature_matrix)) || any(apply(
  feature_matrix[train_rows, , drop = FALSE], 2L, stats::sd
) <= 1e-12)) {
  stop("Feature stability validation failed before DDESONN training.", call. = FALSE)
}

labels <- matrix(as.numeric(dataset$LabelTargetFirst), ncol = 1L)
colnames(labels) <- "TargetFirst"

x_train <- feature_matrix[train_rows, , drop = FALSE]
y_train <- labels[train_rows, , drop = FALSE]
x_validation <- feature_matrix[validation_rows, , drop = FALSE]
y_validation <- labels[validation_rows, , drop = FALSE]
x_holdout <- feature_matrix[holdout_rows, , drop = FALSE]
y_holdout <- labels[holdout_rows, , drop = FALSE]

audit <- data.frame(
  Split = c("Train", "Validation", "Holdout"),
  Rows = c(length(train_rows), length(validation_rows), length(holdout_rows)),
  IndependentObservations = c(
    length(unique(dataset$ObservationId[train_rows])),
    length(unique(dataset$ObservationId[validation_rows])),
    length(unique(dataset$ObservationId[holdout_rows]))
  ),
  TargetFirstRate = c(mean(y_train), mean(y_validation), mean(y_holdout)),
  stringsAsFactors = FALSE
)
utils::write.csv(
  audit,
  file.path(results_path, "NQ_DDESONN_split_audit.csv"),
  row.names = FALSE
)
print(audit)

# ============================================================
# NORMAL DDESONN — SSM EXPLICITLY DISABLED
# ============================================================

run_arguments <- list(
  x = x_train,
  y = y_train,
  classification_mode = "binary",
  hidden_sizes = c(32L, 16L),
  seeds = 111L,
  do_ensemble = FALSE,
  num_networks = 1L,
  x_valid = x_validation,
  y_valid = y_validation,
  x_test = x_holdout,
  y_test = y_holdout,
  aggregate = "mean",
  seed_aggregate = "mean",
  model_overrides = list(
    init_method = "he",
    custom_scale = 1
  ),
  training_overrides = list(
    num_epochs = 3L,
    lr = 0.001,
    optimizer = "adagrad",
    dropout_rates = list(0.10, 0.10),
    loss_type = "CrossEntropy",
    # The same training-fitted transformation has already been applied to
    # train, validation, and holdout. DDESONN must not rescale train alone.
    preprocessScaledData = list(already_scaled = TRUE),
    batch_normalize_data = FALSE,
    validation_metrics = TRUE,
    viewTables = FALSE,
    verbose = FALSE
  ),
  save_models = FALSE,
  verbose = TRUE
)

# DDESONN versions before the optional encoder need no encoder argument.
# Integrated versions are explicitly placed in their original no-SSM mode.
run_parameter_names <- names(formals(DDESONN::ddesonn_run))
if ("sequence_encoder" %in% run_parameter_names) {
  run_arguments$sequence_encoder <- "none"
}
if ("temporal_encoder" %in% run_parameter_names) {
  run_arguments$temporal_encoder <- "none"
}

message(sprintf(
  "Training input: %d rows x %d features; finite=%s; range=[%.4f, %.4f]",
  nrow(x_train), ncol(x_train), all(is.finite(x_train)), min(x_train), max(x_train)
))
message("Training normal DDESONN without SSM (matched preprocessing v6)...")
fit <- do.call(DDESONN::ddesonn_run, run_arguments)

if (is.null(fit$model)) {
  stop("DDESONN completed without returning a final model.", call. = FALSE)
}

prediction_result <- DDESONN::ddesonn_predict(
  fit$model,
  x_holdout,
  aggregate = "mean",
  type = "response"
)
prediction_matrix <- as.matrix(prediction_result$prediction)
if (nrow(prediction_matrix) != nrow(x_holdout) || ncol(prediction_matrix) < 1L) {
  stop("DDESONN returned an unexpected holdout prediction shape.", call. = FALSE)
}
holdout_probability <- as.numeric(prediction_matrix[, 1L])
holdout_probability <- pmax(1e-6, pmin(1 - 1e-6, holdout_probability))
holdout_actual <- as.numeric(y_holdout[, 1L])

auc_score <- function(actual, probability) {
  positive <- actual == 1
  positive_count <- sum(positive)
  negative_count <- sum(!positive)
  if (!positive_count || !negative_count) return(NA_real_)
  probability_ranks <- rank(probability, ties.method = "average")
  (sum(probability_ranks[positive]) -
    positive_count * (positive_count + 1) / 2) /
    (positive_count * negative_count)
}

holdout_metrics <- data.frame(
  Model = "Normal DDESONN — No SSM",
  Rows = length(holdout_actual),
  IndependentObservations = length(unique(dataset$ObservationId[holdout_rows])),
  ActualTargetFirstRate = mean(holdout_actual),
  MeanPredictedProbability = mean(holdout_probability),
  BrierScore = mean((holdout_probability - holdout_actual)^2),
  LogLoss = -mean(
    holdout_actual * log(holdout_probability) +
      (1 - holdout_actual) * log(1 - holdout_probability)
  ),
  AUC = auc_score(holdout_actual, holdout_probability),
  AccuracyAt50 = mean(as.integer(holdout_probability >= 0.50) == holdout_actual),
  PredictionsAt65 = sum(holdout_probability >= 0.65),
  PrecisionAt65 = if (any(holdout_probability >= 0.65)) {
    mean(holdout_actual[holdout_probability >= 0.65])
  } else {
    NA_real_
  },
  stringsAsFactors = FALSE
)

holdout_predictions <- data.frame(
  ObservationId = dataset$ObservationId[holdout_rows],
  ObservationTimeET = dataset$ObservationTimeET[holdout_rows],
  Direction = dataset$Direction[holdout_rows],
  MatrixType = dataset$MatrixType[holdout_rows],
  TargetPct = dataset$TargetPct[holdout_rows],
  StopPct = dataset$StopPct[holdout_rows],
  Horizon = dataset$Horizon[holdout_rows],
  ActualOutcome = dataset$Outcome[holdout_rows],
  ActualTargetFirst = holdout_actual,
  PredictedProbability = holdout_probability,
  PredictedAt50 = as.integer(holdout_probability >= 0.50),
  stringsAsFactors = FALSE
)

metric_group <- interaction(
  holdout_predictions$MatrixType,
  holdout_predictions$Horizon,
  drop = TRUE
)
per_horizon_metrics <- do.call(
  rbind,
  lapply(split(seq_len(nrow(holdout_predictions)), metric_group), function(index) {
    actual <- holdout_actual[index]
    probability <- holdout_probability[index]
    data.frame(
      MatrixType = holdout_predictions$MatrixType[index[[1L]]],
      Horizon = holdout_predictions$Horizon[index[[1L]]],
      Rows = length(index),
      ActualTargetFirstRate = mean(actual),
      MeanPredictedProbability = mean(probability),
      BrierScore = mean((probability - actual)^2),
      AUC = auc_score(actual, probability),
      stringsAsFactors = FALSE
    )
  })
)
row.names(per_horizon_metrics) <- NULL

calibration_breaks <- seq(0, 1, by = 0.10)
calibration_bin <- cut(
  holdout_probability,
  breaks = calibration_breaks,
  include.lowest = TRUE,
  right = TRUE
)
calibration <- do.call(
  rbind,
  lapply(split(
    seq_along(holdout_probability),
    calibration_bin,
    drop = TRUE
  ), function(index) {
    data.frame(
      ProbabilityBin = as.character(calibration_bin[index[[1L]]]),
      Rows = length(index),
      MeanPredictedProbability = mean(holdout_probability[index]),
      ActualTargetFirstRate = mean(holdout_actual[index]),
      stringsAsFactors = FALSE
    )
  })
)
row.names(calibration) <- NULL

utils::write.csv(
  holdout_metrics,
  file.path(results_path, "NQ_DDESONN_holdout_metrics.csv"),
  row.names = FALSE
)
utils::write.csv(
  per_horizon_metrics,
  file.path(results_path, "NQ_DDESONN_holdout_metrics_by_horizon.csv"),
  row.names = FALSE
)
utils::write.csv(
  calibration,
  file.path(results_path, "NQ_DDESONN_holdout_calibration.csv"),
  row.names = FALSE
)
utils::write.csv(
  holdout_predictions,
  file.path(results_path, "NQ_DDESONN_holdout_predictions.csv"),
  row.names = FALSE
)

saveRDS(
  list(
    fit = fit,
    feature_columns = colnames(feature_matrix),
    training_medians = training_medians,
    training_centers = training_centers,
    training_scales = training_scales,
    split_dates = list(
      train_end = train_end,
      validation_start = validation_start,
      validation_end = validation_end,
      holdout_start = holdout_start
    ),
    ssm_enabled = FALSE
  ),
  file.path(results_path, "NQ_DDESONN_baseline_model.rds")
)

message("Normal DDESONN holdout run completed.")
print(holdout_metrics)
message("Results written to: ", normalizePath(results_path, mustWork = TRUE))
