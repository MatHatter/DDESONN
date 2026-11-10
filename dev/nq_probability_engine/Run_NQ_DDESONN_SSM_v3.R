#!/usr/bin/env Rscript

# ============================================================================
# NQ INTRADAY PROBABILITY ENGINE — DDESONN + CAUSAL SSM v3
# ============================================================================
# Controlled follow-up to Run_NQ_DDESONN_Baseline_v6.R.
#
# Locked items (unchanged from v6):
#   * NQ feature rows, labels, features, chronological splits and embargo
#   * DDESONN architecture, seed, optimizer, learning rate and epochs
#   * training-only preprocessing and the 2026 holdout
#
# Only experimental change:
#   * add a fixed 16-value SSM embedding of the previous 48 five-minute bars
#
# RStudio use:
#   1. Put this file, DDESONN_NQ_FeatureDataset.csv, and
#      NQ_5Minute_History.csv in dev/nq_probability_engine.
#   2. Run devtools::load_all() from the DDESONN project.
#   3. Open this file and click Source.
# ============================================================================

arguments <- commandArgs(trailingOnly = TRUE)
feature_input <- if (length(arguments) >= 1L) arguments[[1L]] else {
  "DDESONN_NQ_FeatureDataset.csv"
}
history_input <- if (length(arguments) >= 2L) arguments[[2L]] else {
  "NQ_5Minute_History.csv"
}
results_path <- if (length(arguments) >= 3L) arguments[[3L]] else {
  "NQ_DDESONN_SSM_Results"
}

feature_path <- normalizePath(feature_input, mustWork = TRUE)
history_path <- normalizePath(history_input, mustWork = TRUE)
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("DDESONN", quietly = TRUE)) {
  stop("The DDESONN package is not loaded or installed.", call. = FALSE)
}

# Use the package encoder when it is present. The fallback is the same small,
# fixed causal selective state-space encoder, kept here so this experiment does
# not require another package-source change after the v6 preprocessing fix.
ssm_exports <- c("ddesonn_ssm_init", "ddesonn_ssm_encode")
package_has_ssm_exports <- all(
  ssm_exports %in% getNamespaceExports("DDESONN")
)
package_ssm_init <- if (package_has_ssm_exports) {
  getExportedValue("DDESONN", "ddesonn_ssm_init")
} else {
  NULL
}
package_ssm_init_arguments <- if (!is.null(package_ssm_init)) {
  names(formals(package_ssm_init))
} else {
  character()
}
package_ssm_encode <- if (package_has_ssm_exports) {
  getExportedValue("DDESONN", "ddesonn_ssm_encode")
} else {
  NULL
}
package_ssm_encode_arguments <- if (!is.null(package_ssm_encode)) {
  names(formals(package_ssm_encode))
} else {
  character()
}

package_ssm_api_v1 <- all(
  c("input_size", "state_size", "output_size", "conv_width") %in%
    package_ssm_init_arguments
)
package_ssm_api_current <-
  all(c("sequence_features", "state_dim", "conv_width", "seed") %in%
    package_ssm_init_arguments) &&
  all(c("encoder", "sequence_data", "fit_scale") %in%
    package_ssm_encode_arguments)

if (package_has_ssm_exports && package_ssm_api_v1) {
  ssm_init <- package_ssm_init
  ssm_encode <- package_ssm_encode
  ssm_source <- "DDESONN package encoder"
} else if (package_has_ssm_exports && package_ssm_api_current) {
  # Adapter for the current trainable DDESONN SSM API. In this API the
  # state dimension is also the output embedding dimension.
  ssm_init <- function(input_size,
                       state_size = 16L,
                       output_size = 16L,
                       conv_width = 4L,
                       seed = 1L,
                       input_scale = 0.20) {
    if (as.integer(output_size) != as.integer(state_size)) {
      stop(
        "For the current DDESONN SSM API, output_size must equal state_size.",
        call. = FALSE
      )
    }
    package_ssm_init(
      sequence_features = as.integer(input_size),
      state_dim = as.integer(state_size),
      conv_width = as.integer(conv_width),
      seed = as.integer(seed)
    )
  }
  ssm_encode <- function(encoder, sequences) {
    package_ssm_encode(
      encoder = encoder,
      sequence_data = sequences,
      fit_scale = FALSE
    )
  }
  ssm_source <- "DDESONN package encoder (current API)"
} else {
  ssm_source <- if (package_has_ssm_exports) {
    "self-contained experiment encoder (package SSM API differs)"
  } else {
    "self-contained experiment encoder"
  }

  ssm_init <- function(input_size,
                       state_size = 16L,
                       output_size = 16L,
                       conv_width = 4L,
                       seed = 1L,
                       input_scale = 0.20) {
    input_size <- as.integer(input_size)
    state_size <- as.integer(state_size)
    output_size <- as.integer(output_size)
    conv_width <- as.integer(conv_width)
    if (any(c(input_size, state_size, output_size, conv_width) < 1L)) {
      stop("SSM dimensions must be positive integers.", call. = FALSE)
    }

    old_seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (old_seed_exists) old_seed <- get(".Random.seed", envir = .GlobalEnv)
    on.exit({
      if (old_seed_exists) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(as.integer(seed))

    convolution_kernel <- exp(-seq.int(0L, conv_width - 1L))
    convolution_kernel <- convolution_kernel / sum(convolution_kernel)
    projection_sd <- as.numeric(input_scale) / sqrt(input_size)
    encoder <- list(
      input_size = input_size,
      state_size = state_size,
      output_size = output_size,
      conv_width = conv_width,
      seed = as.integer(seed),
      conv_kernel = convolution_kernel,
      decay_rate = stats::runif(state_size, min = 0.05, max = 0.35),
      input_projection = matrix(
        stats::rnorm(input_size * state_size, sd = projection_sd),
        nrow = input_size,
        ncol = state_size
      ),
      delta_projection = matrix(
        stats::rnorm(input_size * state_size, sd = projection_sd * 0.50),
        nrow = input_size,
        ncol = state_size
      ),
      delta_bias = stats::runif(state_size, min = -2.5, max = -1.0),
      gate_projection = matrix(
        stats::rnorm(input_size * state_size, sd = projection_sd),
        nrow = input_size,
        ncol = state_size
      ),
      state_projection = matrix(
        stats::rnorm(state_size * output_size, sd = 1 / sqrt(state_size)),
        nrow = state_size,
        ncol = output_size
      ),
      skip_projection = matrix(
        stats::rnorm(input_size * output_size, sd = projection_sd),
        nrow = input_size,
        ncol = output_size
      )
    )
    class(encoder) <- "ddesonn_ssm_encoder"
    encoder
  }

  ssm_forward <- function(encoder, sequence) {
    sequence <- as.matrix(sequence)
    storage.mode(sequence) <- "double"
    sequence[!is.finite(sequence)] <- 0
    state <- numeric(encoder$state_size)
    output <- numeric(encoder$output_size)

    for (time_index in seq_len(nrow(sequence))) {
      first_index <- max(1L, time_index - encoder$conv_width + 1L)
      source_index <- seq.int(time_index, first_index)
      kernel <- encoder$conv_kernel[seq_along(source_index)]
      convolved <- colSums(sequence[source_index, , drop = FALSE] * kernel)

      delta_input <-
        as.numeric(convolved %*% encoder$delta_projection) +
          encoder$delta_bias
      delta <- pmax(delta_input, 0) + log1p(exp(-abs(delta_input)))
      delta <- pmax(1e-4, pmin(2, delta))
      decay <- exp(-delta * encoder$decay_rate)

      gate_input <- as.numeric(convolved %*% encoder$gate_projection)
      gate <- 1 / (1 + exp(-pmax(-35, pmin(35, gate_input))))
      drive <- tanh(
        as.numeric(convolved %*% encoder$input_projection)
      ) * gate
      state <- decay * state + (1 - decay) * drive
      output <- tanh(
        as.numeric(state %*% encoder$state_projection) +
          as.numeric(
            sequence[time_index, , drop = FALSE] %*%
              encoder$skip_projection
          )
      )
    }
    output
  }

  ssm_encode <- function(encoder, sequences) {
    dimensions <- dim(sequences)
    if (length(dimensions) != 3L || dimensions[3L] != encoder$input_size) {
      stop(
        "SSM sequences must be samples x timesteps x input features.",
        call. = FALSE
      )
    }
    output <- matrix(
      0,
      nrow = dimensions[1L],
      ncol = encoder$output_size
    )
    for (sample_index in seq_len(dimensions[1L])) {
      output[sample_index, ] <- ssm_forward(
        encoder,
        sequences[sample_index, , , drop = TRUE]
      )
    }
    colnames(output) <- sprintf("SSM_%02d", seq_len(ncol(output)))
    output
  }
}

message("Reading NQ feature dataset: ", feature_path)
message("RUN SCRIPT VERSION: SSM v3 — TRAIN-ONLY ENCODER SCALING")
dataset <- utils::read.csv(
  feature_path,
  stringsAsFactors = FALSE,
  na.strings = c("NA", "")
)
message("Reading dense NQ five-minute history: ", history_path)
bars <- utils::read.csv(
  history_path,
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

lower_bar_names <- tolower(names(bars))
pick_bar_column <- function(candidates, label) {
  match_index <- match(tolower(candidates), lower_bar_names, nomatch = 0L)
  match_index <- match_index[match_index > 0L]
  if (!length(match_index)) {
    stop("The history CSV is missing ", label, ".", call. = FALSE)
  }
  names(bars)[match_index[[1L]]]
}

bar_time_column <- pick_bar_column(
  c("timestamp", "BarTimeET", "ObservationTimeET"),
  "a timestamp column"
)
bar_open_column <- pick_bar_column(c("open", "Open5m"), "open")
bar_high_column <- pick_bar_column(c("high", "High5m"), "high")
bar_low_column <- pick_bar_column(c("low", "Low5m"), "low")
bar_close_column <- pick_bar_column(c("close", "Close5m"), "close")
bar_volume_column <- pick_bar_column(c("volume", "Volume5m"), "volume")

parse_et_time <- function(value) {
  as.POSIXct(
    value,
    format = "%Y-%m-%d %H:%M:%S",
    tz = "America/New_York"
  )
}

dataset$ObservationTime <- parse_et_time(dataset$ObservationTimeET)
bars$BarTime <- parse_et_time(bars[[bar_time_column]])
if (anyNA(dataset$ObservationTime) || anyNA(bars$BarTime)) {
  stop("At least one timestamp could not be parsed.", call. = FALSE)
}

for (column_name in c(
  bar_open_column, bar_high_column, bar_low_column,
  bar_close_column, bar_volume_column
)) {
  bars[[column_name]] <- suppressWarnings(as.numeric(bars[[column_name]]))
}
if (any(!is.finite(as.matrix(bars[c(
  bar_open_column, bar_high_column, bar_low_column, bar_close_column
)])))) {
  stop("The history CSV contains non-finite OHLC values.", call. = FALSE)
}
bars <- bars[order(bars$BarTime), , drop = FALSE]
if (anyDuplicated(bars$BarTime)) {
  stop("The history CSV contains duplicate timestamps.", call. = FALSE)
}

# NQ identity guard. This prevents accidentally combining ES features/history.
median_entry <- stats::median(dataset$EntryPrice, na.rm = TRUE)
median_history_close <- stats::median(bars[[bar_close_column]], na.rm = TRUE)
if (!is.finite(median_entry) || median_entry < 10000 ||
    !is.finite(median_history_close) || median_history_close < 10000) {
  stop("Feature data and five-minute history must both be NQ-like.", call. = FALSE)
}

# Collision outcomes intentionally have no binary target.
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
  stop("Duplicate modeling keys were found.", call. = FALSE)
}

# Same chronological split and seven-day purge as the no-SSM v6 baseline.
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

split_count_by_observation <- tapply(
  dataset$Split,
  dataset$ObservationId,
  function(value) length(unique(value))
)
if (any(split_count_by_observation != 1L)) {
  stop("An ObservationId crossed a split boundary.", call. = FALSE)
}

train_rows <- which(dataset$Split == "Train")
validation_rows <- which(dataset$Split == "Validation")
holdout_rows <- which(dataset$Split == "Holdout")
if (!length(train_rows) || !length(validation_rows) || !length(holdout_rows)) {
  stop("Train, validation and holdout must all contain rows.", call. = FALSE)
}

# Verify that feature observations align exactly to the dense NQ history.
observation_table <- dataset[
  !duplicated(dataset$ObservationId),
  c("ObservationId", "ObservationTime", "Close5m", "Split"),
  drop = FALSE
]
observation_bar_index <- match(observation_table$ObservationTime, bars$BarTime)
if (anyNA(observation_bar_index)) {
  stop(
    sprintf(
      "%d feature observations do not have an exact five-minute history match.",
      sum(is.na(observation_bar_index))
    ),
    call. = FALSE
  )
}
matched_close <- bars[[bar_close_column]][observation_bar_index]
relative_close_error <- abs(matched_close / observation_table$Close5m - 1)
if (stats::median(relative_close_error, na.rm = TRUE) > 0.01) {
  stop("The feature file and history file appear to be different instruments.", call. = FALSE)
}

# ============================================================================
# SAME STATIC FEATURE MATRIX AS NORMAL DDESONN v6
# ============================================================================

relative_to_entry <- function(value, entry) value / entry - 1

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
    levels = c(
      "15m", "30m", "60m", "2h", "4h", "Close",
      "1D", "2D", "3D", "4D", "5D", "6D", "7D"
    )
  )
)
categorical_matrix <- stats::model.matrix(
  ~ Direction + MatrixType + Horizon - 1,
  data = categorical_features
)

feature_matrix <- cbind(as.matrix(numeric_features), categorical_matrix)
storage.mode(feature_matrix) <- "double"

training_medians <- apply(
  feature_matrix[train_rows, , drop = FALSE],
  2L,
  stats::median,
  na.rm = TRUE
)
training_medians[!is.finite(training_medians)] <- 0
for (column_index in seq_len(ncol(feature_matrix))) {
  missing_value <- !is.finite(feature_matrix[, column_index])
  feature_matrix[missing_value, column_index] <- training_medians[[column_index]]
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
feature_matrix[feature_matrix < -8] <- -8
feature_matrix[feature_matrix > 8] <- 8
feature_matrix <- feature_matrix / 8

training_feature_sd <- apply(
  feature_matrix[train_rows, , drop = FALSE],
  2L,
  stats::sd
)
keep_feature <- is.finite(training_feature_sd) & training_feature_sd > 1e-12
if (any(!keep_feature)) {
  message(
    "Removing zero-variance static features: ",
    paste(colnames(feature_matrix)[!keep_feature], collapse = ", ")
  )
  feature_matrix <- feature_matrix[, keep_feature, drop = FALSE]
  training_medians <- training_medians[keep_feature]
  training_centers <- training_centers[keep_feature]
  training_scales <- training_scales[keep_feature]
}

# ============================================================================
# CAUSAL 48-BAR SSM SEQUENCE — THE ONLY EXPERIMENTAL ADDITION
# ============================================================================

sequence_length <- 48L
bar_open <- bars[[bar_open_column]]
bar_high <- bars[[bar_high_column]]
bar_low <- bars[[bar_low_column]]
bar_close <- bars[[bar_close_column]]
bar_volume <- pmax(0, bars[[bar_volume_column]])

lag_return <- function(values, lag_count) {
  result <- rep(NA_real_, length(values))
  if (length(values) > lag_count) {
    result[(lag_count + 1L):length(values)] <-
      values[(lag_count + 1L):length(values)] /
        values[1L:(length(values) - lag_count)] - 1
  }
  result
}

wilder_atr <- function(high, low, close, period = 14L) {
  previous_close <- c(NA_real_, head(close, -1L))
  true_range <- pmax(
    high - low,
    abs(high - previous_close),
    abs(low - previous_close),
    na.rm = TRUE
  )
  true_range[[1L]] <- high[[1L]] - low[[1L]]
  result <- rep(NA_real_, length(close))
  if (length(close) < period) return(result)
  result[[period]] <- mean(true_range[seq_len(period)])
  if (length(close) > period) {
    for (index in seq.int(period + 1L, length(close))) {
      result[[index]] <-
        ((period - 1) * result[[index - 1L]] + true_range[[index]]) / period
    }
  }
  result
}

bar_range <- bar_high - bar_low
bar_atr <- wilder_atr(bar_high, bar_low, bar_close, 14L)
bar_minute <- as.integer(format(bars$BarTime, "%H")) * 60L +
  as.integer(format(bars$BarTime, "%M"))

sequence_feature_matrix <- cbind(
  Return5m = lag_return(bar_close, 1L),
  Return15m = lag_return(bar_close, 3L),
  Return30m = lag_return(bar_close, 6L),
  Return60m = lag_return(bar_close, 12L),
  RangePct = bar_range / bar_close,
  BodyPct = abs(bar_close - bar_open) / bar_close,
  UpperWickPct = (bar_high - pmax(bar_open, bar_close)) / bar_close,
  LowerWickPct = (pmin(bar_open, bar_close) - bar_low) / bar_close,
  AtrPct = bar_atr / bar_close,
  CloseLocation = ifelse(bar_range > 0, (bar_close - bar_low) / bar_range, 0.5),
  LogVolume = log1p(bar_volume),
  TimeSin = sin(2 * pi * bar_minute / 1440),
  TimeCos = cos(2 * pi * bar_minute / 1440)
)
storage.mode(sequence_feature_matrix) <- "double"

# Fit sequence preprocessing only on bars ending before the train boundary.
training_bar <- bars$BarTime <= train_end
sequence_medians <- apply(
  sequence_feature_matrix[training_bar, , drop = FALSE],
  2L,
  stats::median,
  na.rm = TRUE
)
sequence_medians[!is.finite(sequence_medians)] <- 0
for (column_index in seq_len(ncol(sequence_feature_matrix))) {
  missing_value <- !is.finite(sequence_feature_matrix[, column_index])
  sequence_feature_matrix[missing_value, column_index] <-
    sequence_medians[[column_index]]
}
sequence_centers <- colMeans(
  sequence_feature_matrix[training_bar, , drop = FALSE]
)
sequence_scales <- apply(
  sequence_feature_matrix[training_bar, , drop = FALSE],
  2L,
  stats::sd
)
sequence_scales[
  !is.finite(sequence_scales) | sequence_scales < 1e-12
] <- 1
sequence_feature_matrix <- sweep(
  sequence_feature_matrix,
  2L,
  sequence_centers,
  "-"
)
sequence_feature_matrix <- sweep(
  sequence_feature_matrix,
  2L,
  sequence_scales,
  "/"
)
sequence_feature_matrix[sequence_feature_matrix < -8] <- -8
sequence_feature_matrix[sequence_feature_matrix > 8] <- 8
sequence_feature_matrix <- sequence_feature_matrix / 8

if (any(observation_bar_index < sequence_length)) {
  stop("At least one observation lacks 48 preceding five-minute bars.", call. = FALSE)
}

# Encode only unique ObservationIds, then map embeddings to their modeling rows.
# This avoids allocating the same 48-bar sequence hundreds of times.
observation_sequences <- array(
  0,
  dim = c(
    nrow(observation_table),
    sequence_length,
    ncol(sequence_feature_matrix)
  )
)
for (observation_index in seq_len(nrow(observation_table))) {
  end_index <- observation_bar_index[[observation_index]]
  start_index <- end_index - sequence_length + 1L
  observation_sequences[observation_index, , ] <-
    sequence_feature_matrix[start_index:end_index, , drop = FALSE]
}

message("SSM implementation: DDESONN package encoder (confirmed current API)")
ssm_encoder <- DDESONN::ddesonn_ssm_init(
  sequence_features = ncol(sequence_feature_matrix),
  state_dim = 16L,
  conv_width = 4L,
  seed = 707L
)
message(
  "Encoding ", nrow(observation_table),
  " unique 48-bar observation sequences..."
)

# Fit the package encoder's internal scaling on TRAIN observations only.
# The fitted encoder is returned as an attribute on the embedding matrix.
training_observation <- which(observation_table$Split == "Train")
validation_observation <- which(observation_table$Split == "Validation")
holdout_observation <- which(observation_table$Split == "Holdout")

message("Fitting SSM scaling and encoding Train observations...")
training_embedding <- DDESONN::ddesonn_ssm_encode(
  encoder = ssm_encoder,
  sequence_data = observation_sequences[
    training_observation, , , drop = FALSE
  ],
  fit_scale = TRUE
)
fitted_ssm_encoder <- attr(training_embedding, "encoder", exact = TRUE)
if (is.null(fitted_ssm_encoder)) {
  stop(
    "DDESONN did not return the fitted SSM encoder attribute.",
    call. = FALSE
  )
}
ssm_encoder <- fitted_ssm_encoder

message("Encoding Validation with frozen Train scaling...")
validation_embedding <- DDESONN::ddesonn_ssm_encode(
  encoder = ssm_encoder,
  sequence_data = observation_sequences[
    validation_observation, , , drop = FALSE
  ],
  fit_scale = FALSE
)
message("Encoding Holdout with frozen Train scaling...")
holdout_embedding <- DDESONN::ddesonn_ssm_encode(
  encoder = ssm_encoder,
  sequence_data = observation_sequences[
    holdout_observation, , , drop = FALSE
  ],
  fit_scale = FALSE
)

observation_embedding <- matrix(
  NA_real_,
  nrow = nrow(observation_table),
  ncol = ncol(training_embedding)
)
observation_embedding[training_observation, ] <- training_embedding
observation_embedding[validation_observation, ] <- validation_embedding
observation_embedding[holdout_observation, ] <- holdout_embedding
if (any(!is.finite(observation_embedding))) {
  stop("The SSM encoder returned non-finite embeddings.", call. = FALSE)
}

# Give SSM features the same train-only scaling convention as static features.
embedding_train <- observation_table$Split == "Train"
embedding_centers <- colMeans(
  observation_embedding[embedding_train, , drop = FALSE]
)
embedding_scales <- apply(
  observation_embedding[embedding_train, , drop = FALSE],
  2L,
  stats::sd
)
embedding_scales[
  !is.finite(embedding_scales) | embedding_scales < 1e-12
] <- 1
observation_embedding <- sweep(
  observation_embedding,
  2L,
  embedding_centers,
  "-"
)
observation_embedding <- sweep(
  observation_embedding,
  2L,
  embedding_scales,
  "/"
)
observation_embedding[observation_embedding < -8] <- -8
observation_embedding[observation_embedding > 8] <- 8
observation_embedding <- observation_embedding / 8
colnames(observation_embedding) <- sprintf(
  "SSM_%02d",
  seq_len(ncol(observation_embedding))
)

embedding_row <- match(dataset$ObservationId, observation_table$ObservationId)
if (anyNA(embedding_row)) {
  stop("An SSM embedding could not be mapped to a modeling row.", call. = FALSE)
}
model_matrix <- cbind(feature_matrix, observation_embedding[embedding_row, , drop = FALSE])
if (any(!is.finite(model_matrix))) {
  stop("The combined static + SSM matrix contains non-finite values.", call. = FALSE)
}

labels <- matrix(as.numeric(dataset$LabelTargetFirst), ncol = 1L)
colnames(labels) <- "TargetFirst"
x_train <- model_matrix[train_rows, , drop = FALSE]
y_train <- labels[train_rows, , drop = FALSE]
x_validation <- model_matrix[validation_rows, , drop = FALSE]
y_validation <- labels[validation_rows, , drop = FALSE]
x_holdout <- model_matrix[holdout_rows, , drop = FALSE]
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
  StaticFeatures = ncol(feature_matrix),
  SSMFeatures = ncol(observation_embedding),
  SequenceBars = sequence_length,
  stringsAsFactors = FALSE
)
utils::write.csv(
  audit,
  file.path(results_path, "NQ_DDESONN_SSM_split_audit.csv"),
  row.names = FALSE
)
print(audit)

# ============================================================================
# DDESONN SETTINGS LOCKED TO THE SUCCESSFUL NO-SSM v6 BASELINE
# ============================================================================

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
    preprocessScaledData = list(already_scaled = TRUE),
    batch_normalize_data = FALSE,
    validation_metrics = TRUE,
    viewTables = FALSE,
    verbose = FALSE
  ),
  save_models = FALSE,
  verbose = TRUE
)

# The temporal data are already encoded above, once per unique observation.
# Tell integrated package versions not to add another SSM pass.
run_parameter_names <- names(formals(DDESONN::ddesonn_run))
if ("sequence_encoder" %in% run_parameter_names) {
  run_arguments$sequence_encoder <- "none"
}

message(sprintf(
  paste(
    "Training input: %d rows x %d features",
    "(%d static + %d SSM); range=[%.4f, %.4f]"
  ),
  nrow(x_train), ncol(x_train), ncol(feature_matrix),
  ncol(observation_embedding), min(x_train), max(x_train)
))
message("Training DDESONN with causal SSM features...")
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
  stop("DDESONN returned an unexpected prediction shape.", call. = FALSE)
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
  Model = "DDESONN + causal SSM",
  Rows = length(holdout_actual),
  IndependentObservations = length(unique(dataset$ObservationId[holdout_rows])),
  ActualTargetFirstRate = mean(holdout_actual),
  MeanPredictedProbability = mean(holdout_probability),
  MinimumProbability = min(holdout_probability),
  MaximumProbability = max(holdout_probability),
  BrierScore = mean((holdout_probability - holdout_actual)^2),
  LogLoss = -mean(
    holdout_actual * log(holdout_probability) +
      (1 - holdout_actual) * log(1 - holdout_probability)
  ),
  AUC = auc_score(holdout_actual, holdout_probability),
  AccuracyAt50 = mean(as.integer(holdout_probability >= 0.50) == holdout_actual),
  AccuracyAt55 = mean(as.integer(holdout_probability >= 0.55) == holdout_actual),
  PredictionsAt55 = sum(holdout_probability >= 0.55),
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

calibration_bin <- cut(
  holdout_probability,
  breaks = seq(0, 1, by = 0.10),
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
  file.path(results_path, "NQ_DDESONN_SSM_holdout_metrics.csv"),
  row.names = FALSE
)
utils::write.csv(
  per_horizon_metrics,
  file.path(results_path, "NQ_DDESONN_SSM_holdout_metrics_by_horizon.csv"),
  row.names = FALSE
)
utils::write.csv(
  calibration,
  file.path(results_path, "NQ_DDESONN_SSM_holdout_calibration.csv"),
  row.names = FALSE
)
utils::write.csv(
  holdout_predictions,
  file.path(results_path, "NQ_DDESONN_SSM_holdout_predictions.csv"),
  row.names = FALSE
)

saveRDS(
  list(
    fit = fit,
    ssm_encoder = ssm_encoder,
    static_feature_columns = colnames(feature_matrix),
    ssm_feature_columns = colnames(observation_embedding),
    training_medians = training_medians,
    training_centers = training_centers,
    training_scales = training_scales,
    sequence_medians = sequence_medians,
    sequence_centers = sequence_centers,
    sequence_scales = sequence_scales,
    embedding_centers = embedding_centers,
    embedding_scales = embedding_scales,
    sequence_length = sequence_length,
    split_dates = list(
      train_end = train_end,
      validation_start = validation_start,
      validation_end = validation_end,
      holdout_start = holdout_start
    ),
    ssm_enabled = TRUE
  ),
  file.path(results_path, "NQ_DDESONN_SSM_model.rds")
)

message("DDESONN + SSM holdout run completed.")
print(holdout_metrics)
message("Results written to: ", normalizePath(results_path, mustWork = TRUE))
