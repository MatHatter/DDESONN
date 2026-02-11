#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(DDESONN))

`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

# ------------------------------------------------------------------------------
# $$$ PHASE 1 START $$$
# WMT next-day log return regression
# ------------------------------------------------------------------------------
CLASSIFICATION_MODE <- "regression"
PREDICT_NEXT_DAY    <- TRUE
REG_TARGET_MODE     <- "return_log"  # predict log-return of future_close
REDUCE_DATA         <- FALSE

num_epochs <- 2
lr         <- 0.01
hidden_sizes <- c(128, 64, 32)
RESTORE_BEST_WEIGHTS <- TRUE
TARGET_SCALE <- 1

suppressPackageStartupMessages({
  library(dplyr)
  library(zoo)
  library(quantmod)
})

# ------------------------------------------------------------------------------
# Load raw historical WMT candles
# ------------------------------------------------------------------------------
data <- read.csv("data/WMT_1970-10-01_2025-03-15.csv", stringsAsFactors = FALSE)
stopifnot("date" %in% names(data))
data <- data %>% arrange(date)

if (isTRUE(PREDICT_NEXT_DAY)) {
  data <- data %>%
    mutate(future_close = dplyr::lead(close, 1L)) %>%
    filter(!is.na(future_close))
  dependent_variable <- "future_close"
} else {
  dependent_variable <- "close"
}

# ------------------------------------------------------------------------------
# $$$ PHASE 2 START $$$
# ------------------------------------------------------------------------------
data <- data %>%
  arrange(date) %>%
  mutate(
    ret_1d   = log(close / dplyr::lag(close, 1)),
    ret_5d   = log(close / dplyr::lag(close, 5)),
    ret_10d  = log(close / dplyr::lag(close, 10)),
    sma_10   = zoo::rollmean(close, 10, fill = NA),
    mean_dev_10d = close / sma_10,
    intraday_range = (high - low) / pmax(close, 1e-12),
    vol_10d = zoo::rollapply(
      log(close / dplyr::lag(close, 1)),
      10,
      sd,
      fill = NA,
      align = "right"
    ),
    vol_sma_10   = zoo::rollmean(volume, 10, fill = NA),
    vol_ratio_10 = volume / pmax(vol_sma_10, 1e-12),
    gap_overnight = (open - dplyr::lag(close, 1)) / pmax(dplyr::lag(close, 1), 1e-12),
    intraday_strength = (close - open) / pmax((high - low), 1e-8),
    range_ma_10   = zoo::rollmean(high - low, 10, fill = NA),
    range_ratio_10 = (high - low) / pmax(range_ma_10, 1e-12)
  )

# ------------------------------------------------------------------------------
# $$$ PHASE 2 END $$$
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# $$$ PHASE 3 START $$$
# ------------------------------------------------------------------------------
tickers <- c("SPY", "XLP", "^VIX")
for (tkr in tickers) {
  tryCatch({
    getSymbols(
      tkr,
      from = "1970-01-01",
      to   = "2025-03-15",
      auto.assign = TRUE
    )
  }, error = function(e) {
    warning(paste("Could not download:", tkr))
  })
}

.safe_cl <- function(sym) {
  tryCatch(Cl(get(sym)), error = function(e) NULL)
}

.safe_daily_return_xts <- function(sym, diff_mode = c("logret","rel_change")) {
  diff_mode <- match.arg(diff_mode)
  cl <- .safe_cl(sym)
  if (is.null(cl)) return(NULL)
  
  cl_filled <- zoo::na.locf(cl, na.rm = FALSE)
  cl_filled <- zoo::na.approx(cl_filled, na.rm = FALSE)
  
  if (is.null(cl_filled) || NROW(cl_filled) < 2L) return(NULL)
  if (all(!is.finite(as.numeric(cl_filled)))) return(NULL)
  
  out <- if (identical(diff_mode, "logret")) {
    diff(log(cl_filled))
  } else {
    diff(cl_filled) / lag(cl_filled)
  }
  
  out <- out[is.finite(out)]
  if (NROW(out) == 0L) return(NULL)
  out
}

SPY_ret_1d    <- .safe_daily_return_xts("SPY", diff_mode = "logret")
XLP_ret_1d    <- .safe_daily_return_xts("XLP", diff_mode = "logret")
VIX_change_1d <- .safe_daily_return_xts("VIX", diff_mode = "rel_change")

market_df_list <- list()

if (!is.null(SPY_ret_1d) && NROW(SPY_ret_1d) > 0L) {
  tmp <- data.frame(
    date = as.Date(index(SPY_ret_1d)),
    SPY_ret_1d = as.numeric(SPY_ret_1d)
  )
  tmp <- tmp[is.finite(tmp$SPY_ret_1d), , drop = FALSE]
  if (nrow(tmp) > 0L) {
    market_df_list[["SPY"]] <- tmp
  }
}

if (!is.null(XLP_ret_1d) && NROW(XLP_ret_1d) > 0L) {
  tmp <- data.frame(
    date = as.Date(index(XLP_ret_1d)),
    XLP_ret_1d = as.numeric(XLP_ret_1d)
  )
  tmp <- tmp[is.finite(tmp$XLP_ret_1d), , drop = FALSE]
  if (nrow(tmp) > 0L) {
    market_df_list[["XLP"]] <- tmp
  }
}

if (!is.null(VIX_change_1d) && NROW(VIX_change_1d) > 0L) {
  tmp <- data.frame(
    date = as.Date(index(VIX_change_1d)),
    VIX_change_1d = as.numeric(VIX_change_1d)
  )
  tmp <- tmp[is.finite(tmp$VIX_change_1d), , drop = FALSE]
  if (nrow(tmp) > 0L) {
    market_df_list[["VIX"]] <- tmp
  }
}

if (length(market_df_list) > 0) {
  market_df <- Reduce(function(x, y) merge(x, y, by = "date", all = TRUE),
                      market_df_list)
  
  market_df <- market_df[order(market_df$date), ]
  if ("SPY_ret_1d"    %in% names(market_df)) market_df$SPY_ret_1d    <- zoo::na.locf(market_df$SPY_ret_1d,    na.rm = FALSE)
  if ("XLP_ret_1d"    %in% names(market_df)) market_df$XLP_ret_1d    <- zoo::na.locf(market_df$XLP_ret_1d,    na.rm = FALSE)
  if ("VIX_change_1d" %in% names(market_df)) market_df$VIX_change_1d <- zoo::na.locf(market_df$VIX_change_1d, na.rm = FALSE)
  
  market_df <- market_df[
    apply(market_df[, setdiff(names(market_df), "date"), drop = FALSE],
          1,
          function(r) any(is.finite(r))),
    ,
    drop = FALSE
  ]
  
  if (nrow(market_df) > 0L) {
    data$date <- as.Date(data$date)
    data <- merge(data, market_df, by = "date", all.x = TRUE)
  } else {
    warning("Market df built but ended up empty after cleaning")
  }
} else {
  warning("No market data fetched. Proceeding without SPY/XLP/VIX context.")
}

# ------------------------------------------------------------------------------
# $$$ PHASE 3 END $$$
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# CLEANUP AFTER PHASE 2 + 3
# ------------------------------------------------------------------------------
numeric_cols <- names(data)[vapply(data, is.numeric, TRUE)]

core_required <- unique(c(
  dependent_variable,
  c("open","high","low","close","volume",
    "ret_1d","ret_5d","ret_10d",
    "intraday_range","vol_10d",
    "gap_overnight","intraday_strength")
))
core_required <- intersect(core_required, numeric_cols)

data <- data[
  apply(
    data[, core_required, drop = FALSE],
    1,
    function(r) all(is.finite(r))
  ),
  ,
  drop = FALSE
]

### >>> FIX: drop early warmup rows where rolling windows are NA-heavy
### this limits chrono split getting garbage at the front
data <- data[stats::complete.cases(data[, core_required, drop = FALSE]), , drop = FALSE]

# ------------------------------------------------------------------------------
# Build X_full (features) and y_full (target)
# ------------------------------------------------------------------------------
if (CLASSIFICATION_MODE == "regression" && isTRUE(REDUCE_DATA)) {
  base_keep <- c("date", "open", "high", "low", "close", "volume")
  keep_cols <- unique(c(base_keep, dependent_variable))
  data_reduced <- data %>% dplyr::select(dplyr::any_of(keep_cols))
  X_full <- data_reduced %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y_full <- data_reduced %>% dplyr::select(dplyr::all_of(dependent_variable))
} else {
  X_full <- data %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y_full <- data %>% dplyr::select(dplyr::all_of(dependent_variable))
}

colname_y <- colnames(y_full)

# ------------------------------------------------------------------------------
# Chronological split (70 / 15 / 15)
# ------------------------------------------------------------------------------
stopifnot(nrow(X_full) == nrow(y_full))
total_num_samples <- nrow(X_full)

p_train <- 0.70
p_val   <- 0.15

num_training_samples   <- max(1L, floor(p_train * total_num_samples))
num_validation_samples <- max(1L, floor(p_val   * total_num_samples))
num_test_samples       <- max(
  0L,
  total_num_samples - num_training_samples - num_validation_samples
)

train_indices      <- seq_len(num_training_samples)
validation_indices <- if (num_validation_samples > 0L) {
  seq(from = max(train_indices) + 1L, length.out = num_validation_samples)
} else integer()

test_indices <- if (num_test_samples > 0L) {
  seq(from = max(c(train_indices, validation_indices)) + 1L,
      length.out = num_test_samples)
} else integer()

X_train      <- X_full[train_indices,      , drop = FALSE]
X_validation <- X_full[validation_indices, , drop = FALSE]
X_test       <- X_full[test_indices,       , drop = FALSE]

y_train      <- y_full[train_indices,      , drop = FALSE]
y_validation <- y_full[validation_indices, , drop = FALSE]
y_test       <- y_full[test_indices,       , drop = FALSE]

cat(sprintf("[SPLIT chrono] train=%d val=%d test=%d\n",
            nrow(X_train), nrow(X_validation), nrow(X_test)))

if (!identical(tolower(CLASSIFICATION_MODE), "regression")) {
  stop("regression mode only script")
}

# ------------------------------------------------------------------------------
# date -> numeric days
# ------------------------------------------------------------------------------
make_date_numeric <- function(df) {
  if (!"date" %in% names(df)) return(df)
  if (nrow(df) == 0L) return(df)
  
  d <- df[["date"]]
  
  if (inherits(d, "POSIXt")) {
    df[["date"]] <- as.numeric(as.Date(d))
  } else if (inherits(d, "Date")) {
    df[["date"]] <- as.numeric(d)
  } else {
    suppressWarnings({ parsed <- as.Date(d) })
    if (all(is.na(parsed))) {
      df[["date"]] <- rep(NA_real_, nrow(df))
    } else {
      df[["date"]] <- as.numeric(parsed)
    }
  }
  df
}

X_train      <- make_date_numeric(X_train)
X_validation <- make_date_numeric(X_validation)
X_test       <- make_date_numeric(X_test)

# ------------------------------------------------------------------------------
# Impute missing numeric cols using TRAIN medians only
# ------------------------------------------------------------------------------
impute_with_train_median <- function(df_train, df_other) {
  num_cols <- names(df_train)[vapply(df_train, is.numeric, TRUE)]
  for (nm in num_cols) {
    med <- suppressWarnings(median(df_train[[nm]], na.rm = TRUE))
    if (!is.finite(med) || is.na(med)) med <- 0
    if (nm %in% names(df_train))  df_train[[nm]][is.na(df_train[[nm]])]  <- med
    if (nm %in% names(df_other))  df_other[[nm]][is.na(df_other[[nm]])]  <- med
  }
  list(train = df_train, other = df_other)
}

tmp <- impute_with_train_median(X_train, X_validation)
X_train      <- tmp$train
X_validation <- tmp$other
tmp <- impute_with_train_median(X_train, X_test)
X_test       <- tmp$other

# ------------------------------------------------------------------------------
# Numeric-only predictor matrices
# ------------------------------------------------------------------------------
X_train_df <- as.data.frame(X_train)
X_val_df   <- as.data.frame(X_validation)
X_test_df  <- as.data.frame(X_test)

num_mask <- vapply(X_train_df, is.numeric, TRUE)
if (!any(num_mask)) stop("no numeric predictors after preprocessing")

X_train_num <- as.matrix(X_train_df[, num_mask, drop = FALSE])
X_val_num   <- as.matrix(X_val_df[,   num_mask, drop = FALSE])
X_test_num  <- as.matrix(X_test_df[,  num_mask, drop = FALSE])

# ------------------------------------------------------------------------------
# Standardize using TRAIN stats only
# ------------------------------------------------------------------------------
X_train_scaled <- scale(X_train_num)
center <- attr(X_train_scaled, "scaled:center")
scale_ <- attr(X_train_scaled, "scaled:scale")
scale_[!is.finite(scale_) | scale_ == 0] <- 1

X_validation_scaled <- sweep(
  sweep(X_val_num,  2, center, "-"),
  2, scale_, "/"
)
X_test_scaled <- sweep(
  sweep(X_test_num, 2, center, "-"),
  2, scale_, "/"
)

max_val <- suppressWarnings(max(abs(X_train_scaled)))
if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1

drop_first_row_safe <- function(obj) {
  if (is.null(obj) || NROW(obj) == 0L) return(obj)
  if (is.matrix(obj))     return(obj[-1, , drop = FALSE])
  if (is.data.frame(obj)) return(obj[-1, , drop = FALSE])
  obj[-1]
}

to_logret <- function(v) {
  vv <- as.numeric(if (is.matrix(v) || is.data.frame(v)) v[,1] else v)
  c(NA_real_, diff(log(pmax(vv, 1e-12))))
}

# ------------------------------------------------------------------------------
# y as next-day log return
# ------------------------------------------------------------------------------
if (identical(tolower(REG_TARGET_MODE), "return_log")) {
  y_train      <- to_logret(y_train);      y_train      <- drop_first_row_safe(y_train)
  if (NROW(y_validation)) {
    y_validation <- to_logret(y_validation); y_validation <- drop_first_row_safe(y_validation)
  }
  if (NROW(y_test)) {
    y_test <- to_logret(y_test); y_test <- drop_first_row_safe(y_test)
  }
  
  X_train_scaled      <- drop_first_row_safe(X_train_scaled)
  X_validation_scaled <- drop_first_row_safe(X_validation_scaled)
  X_test_scaled       <- drop_first_row_safe(X_test_scaled)
  
} else if (!identical(tolower(REG_TARGET_MODE), "price")) {
  stop("REG_TARGET_MODE must be price or return_log")
}

X_train_scaled_final      <- X_train_scaled      / max_val
X_validation_scaled_final <- X_validation_scaled / max_val
X_test_scaled_final       <- X_test_scaled       / max_val

# ------------------------------------------------------------------------------
# y scaling config (identity)
# ------------------------------------------------------------------------------
SCALE_Y_WITH_ZSCORE <- FALSE

y_vec_train <- if (is.matrix(y_train) || is.data.frame(y_train)) {
  as.numeric(y_train[,1])
} else {
  as.numeric(y_train)
}

stopifnot(length(y_vec_train) == NROW(X_train_scaled_final))

if (isTRUE(SCALE_Y_WITH_ZSCORE)) {
  y_center <- mean(y_vec_train, na.rm = TRUE)
  y_scale  <- stats::sd(y_vec_train, na.rm = TRUE)
  if (!is.finite(y_scale) || y_scale == 0) y_scale <- 1
  y_vec_scaled <- (y_vec_train - y_center) / y_scale
  target_transform <- list(
    type   = "zscore",
    params = list(center = y_center, scale = y_scale)
  )
  y_trained_scaled <- TRUE
} else {
  y_vec_scaled <- y_vec_train
  target_transform <- list(
    type   = "identity",
    params = list(center = 0, scale = 1)
  )
  y_trained_scaled <- FALSE
}

y_train_mat <- matrix(as.numeric(y_vec_scaled), ncol = 1L)
storage.mode(y_train_mat) <- "double"
colnames(y_train_mat) <- colname_y

y_train_mat <- y_train_mat * TARGET_SCALE
if (NROW(y_validation)) {
  if (is.matrix(y_validation) || is.data.frame(y_validation)) {
    y_validation <- as.matrix(y_validation)
  }
  y_validation <- y_validation * TARGET_SCALE
}
if (NROW(y_test)) {
  if (is.matrix(y_test) || is.data.frame(y_test)) {
    y_test <- as.matrix(y_test)
  }
  y_test <- y_test * TARGET_SCALE
}

# ------------------------------------------------------------------------------
# Align lengths
# ------------------------------------------------------------------------------
align_n <- min(
  nrow(X_train_scaled_final),
  nrow(y_train_mat)
)

if (align_n != nrow(X_train_scaled_final) ||
    align_n != nrow(y_train_mat)) {
  X_train_scaled_final      <- X_train_scaled_final[seq_len(align_n), , drop = FALSE]
  y_train_mat               <- y_train_mat[seq_len(align_n), , drop = FALSE]
  X_validation_scaled_final <- X_validation_scaled_final[
    seq_len(min(nrow(X_validation_scaled_final), align_n)), , drop = FALSE
  ]
  y_validation <- y_validation[
    seq_len(min(nrow(as.data.frame(y_validation)), align_n)), , drop = FALSE
  ]
  X_test_scaled_final <- X_test_scaled_final[
    seq_len(min(nrow(X_test_scaled_final), align_n)), , drop = FALSE
  ]
  y_test <- y_test[
    seq_len(min(nrow(as.data.frame(y_test)), align_n)), , drop = FALSE
  ]
  cat(sprintf("[reg] Adjusted alignment to n=%d rows\n", align_n))
}

# ------------------------------------------------------------------------------
# Final matrices
# ------------------------------------------------------------------------------
train_x <- as.matrix(X_train_scaled_final)
train_y <- y_train_mat

valid_x <- as.matrix(X_validation_scaled_final)
valid_y <- as.matrix(y_validation)
test_x  <- as.matrix(X_test_scaled_final)
test_y  <- as.matrix(y_test)

# ------------------------------------------------------------------------------
# FINAL HARD FILTER
# ------------------------------------------------------------------------------
is_finite_row <- function(X, y) {
  apply(
    cbind(X, y),
    1,
    function(r) all(is.finite(r))
  )
}

good_train <- is_finite_row(train_x, train_y)
good_valid <- if (NROW(valid_x) > 0L && NROW(valid_y) > 0L) {
  is_finite_row(valid_x, valid_y)
} else logical(0)
good_test  <- if (NROW(test_x)  > 0L && NROW(test_y)  > 0L) {
  is_finite_row(test_x,  test_y)
} else logical(0)

if (length(good_train) && !all(good_train)) {
  cat("Filtering", sum(!good_train), "bad train rows due to non-finite values\n")
}
if (length(good_valid) && !all(good_valid)) {
  cat("Filtering", sum(!good_valid), "bad val rows due to non-finite values\n")
}
if (length(good_test) && !all(good_test)) {
  cat("Filtering", sum(!good_test), "bad test rows due to non-finite values\n")
}

train_x <- train_x[good_train, , drop = FALSE]
train_y <- train_y[good_train, , drop = FALSE]

if (length(good_valid)) {
  valid_x <- valid_x[good_valid, , drop = FALSE]
  valid_y <- valid_y[good_valid, , drop = FALSE]
} else {
  ### >>> FIX: allow empty validation
  valid_x <- matrix(numeric(0), nrow = 0, ncol = ncol(train_x),
                    dimnames = list(NULL, colnames(train_x)))
  valid_y <- matrix(numeric(0), nrow = 0, ncol = 1L,
                    dimnames = list(NULL, colname_y))
}

if (length(good_test)) {
  test_x  <- test_x[good_test, , drop = FALSE]
  test_y  <- test_y[good_test, , drop = FALSE]
} else {
  ### >>> FIX: allow empty test
  test_x <- matrix(numeric(0), nrow = 0, ncol = ncol(train_x),
                   dimnames = list(NULL, colnames(train_x)))
  test_y <- matrix(numeric(0), nrow = 0, ncol = 1L,
                   dimnames = list(NULL, colname_y))
}

### >>> FIX: don't die unless it's truly unusable
if (nrow(train_x) < 20L) {
  stop(paste0(
    "Training set too small after cleaning (", nrow(train_x),
    " rows). Phase 3 merge ate too many rows. ",
    "Comment out Phase 3 or widen acceptable rows."
  ))
}

if (!all(is.finite(train_x)) ||
    !all(is.finite(train_y))) {
  stop("Non-finite values in training data after cleaning.")
}

input_size  <- ncol(train_x)
output_size <- 1L

# ------------------------------------------------------------------------------
# Save preprocess details
# ------------------------------------------------------------------------------
feature_names <- colnames(X_train_num)
center_vec <- setNames(as.numeric(center[feature_names]), feature_names)
scale_vec  <- setNames(as.numeric(scale_[feature_names]), feature_names)

train_medians <- vapply(
  as.data.frame(X_train_df[, feature_names, drop = FALSE]),
  function(col) suppressWarnings(median(col, na.rm = TRUE)),
  numeric(1)
)
train_medians[!is.finite(train_medians)] <- 0

preprocessScaledData <- list(
  feature_names     = as.character(feature_names),
  center            = center_vec,
  scale             = scale_vec,
  max_val           = as.numeric(max_val),
  divide_by_max_val = TRUE,
  train_medians     = setNames(as.numeric(train_medians[feature_names]), feature_names),
  date_policy       = "date->numeric",
  used_scaled_X     = TRUE,
  scaler            = "standardize+divide_by_max",
  imputer           = "train_median",
  input_size        = input_size,
  target_transform  = list(
    type   = "identity",
    params = list(center = 0, scale = 1)
  ),
  y_trained_scaled  = FALSE
)

assign("preprocessScaledData", preprocessScaledData, inherits = TRUE)
assign("target_transform",     preprocessScaledData$target_transform, inherits = TRUE)

cat("=== [reg] Diagnostics ===\n")
cat("train_x dim:", paste(dim(train_x), collapse="x"), "\n")
cat("valid_x dim:", paste(dim(valid_x), collapse="x"), "\n")
cat("test_x  dim:", paste(dim(test_x),  collapse="x"), "\n")
cat("NAs? train:", anyNA(train_x),
    " val:", anyNA(valid_x),
    " test:", anyNA(test_x), "\n")

# ------------------------------------------------------------------------------
# Build / Fit model
# ------------------------------------------------------------------------------
model <- ddesonn_model(
  input_size          = input_size,
  output_size         = output_size,
  hidden_sizes        = hidden_sizes,
  num_networks        = 1L,
  classification_mode = "regression",
  ML_NN               = TRUE,
  custom_scale        = 10
)

ddesonn_fit(
  model,
  train_x,
  train_y,
  validation = list(x = valid_x, y = valid_y),
  num_epochs = num_epochs,
  lr = lr,
  validation_metrics = TRUE,
  verbose = TRUE,
  best_weights_on_latest_weights_off = RESTORE_BEST_WEIGHTS,
  classification_mode = "regression",
  batch_normalize_data = FALSE
)

# ------------------------------------------------------------------------------
# Validation metrics (only if we actually have val rows)
# ------------------------------------------------------------------------------
if (nrow(valid_x) > 0L && nrow(valid_y) > 0L) {
  pred_valid <- ddesonn_predict(model, valid_x, aggregate = "mean")
  
  y_hat_val  <- as.numeric(pred_valid$prediction)
  y_true_val <- as.numeric(valid_y[,1])
  
  stopifnot(length(y_hat_val) == length(y_true_val))
  
  mse_val  <- mean((y_hat_val - y_true_val)^2)
  rmse_val <- sqrt(mse_val)
  mae_val  <- mean(abs(y_hat_val - y_true_val))
  r2_val   <- 1 - sum((y_true_val - y_hat_val)^2) /
    sum((y_true_val - mean(y_true_val))^2)
  
  cat("\n--- VALIDATION METRICS (regression) ---\n")
  cat("RMSE:", round(rmse_val, 4),
      " MAE:", round(mae_val, 4),
      " R2:",  round(r2_val, 4), "\n")
  
  comparison_valid <- data.frame(
    actual    = round(y_true_val, 6),
    predicted = round(y_hat_val, 6),
    error     = round(y_hat_val - y_true_val, 6)
  )
  print(utils::head(comparison_valid, 20))
  
  cat("\nValidation pred mean/sd:\n")
  cat(mean(y_hat_val), sd(y_hat_val), "\n")
  cat("Validation true mean/sd:\n")
  cat(mean(y_true_val), sd(y_true_val), "\n")
  cat("Validation cor(pred,true):\n")
  cat(cor(y_hat_val, y_true_val), "\n")
} else {
  ### >>> FIX: fallback if no val set
  rmse_val <- NA_real_
  mae_val  <- NA_real_
  r2_val   <- NA_real_
  comparison_valid <- data.frame()
  cat("\n--- VALIDATION METRICS (regression) ---\n(no validation rows)\n")
}

# ------------------------------------------------------------------------------
# Test / hold-out metrics (only if we actually have test rows)
# ------------------------------------------------------------------------------
if (nrow(test_x) > 0L && nrow(test_y) > 0L) {
  pred_test <- ddesonn_predict(model, test_x, aggregate = "mean")
  
  y_hat_test  <- as.numeric(pred_test$prediction)
  y_true_test <- as.numeric(test_y[,1])
  
  stopifnot(length(y_hat_test) == length(y_true_test))
  
  mse_test  <- mean((y_hat_test - y_true_test)^2)
  rmse_test <- sqrt(mse_test)
  mae_test  <- mean(abs(y_hat_test - y_true_test))
  r2_test   <- 1 - sum((y_true_test - y_hat_test)^2) /
    sum((y_true_test - mean(y_true_test))^2)
  
  cat("\n--- TEST METRICS (regression, hold-out) ---\n")
  cat("RMSE:", round(rmse_test, 4),
      " MAE:", round(mae_test, 4),
      " R2:",  round(r2_test, 4), "\n")
  
  comparison_test <- data.frame(
    actual    = round(y_true_test, 6),
    predicted = round(y_hat_test, 6),
    error     = round(y_hat_test - y_true_test, 6)
  )
  print(utils::head(comparison_test, 20))
} else {
  ### >>> FIX: fallback if no test set
  rmse_test <- NA_real_
  mae_test  <- NA_real_
  r2_test   <- NA_real_
  comparison_test <- data.frame()
  cat("\n--- TEST METRICS (regression, hold-out) ---\n(no test rows)\n")
}

VALID_RMSE <- rmse_val
VALID_MAE  <- mae_val
VALID_R2   <- r2_val
TEST_RMSE  <- rmse_test
TEST_MAE   <- mae_test
TEST_R2    <- r2_test
COMPARISON_VALID <- comparison_valid
COMPARISON_TEST  <- comparison_test

# ------------------------------------------------------------------------------
# Scenario runner (ensembles)
# ------------------------------------------------------------------------------
scenario_presets <- list(
  A = list(
    label="Scenario A",
    do_ensemble=FALSE,
    num_networks=1L,
    aggregate="mean",
    prediction_type="response",
    seeds = 1L
  ),
  B = list(
    label="Scenario B",
    do_ensemble=FALSE,
    num_networks=4L,
    aggregate="mean",
    prediction_type="response",
    seeds = 1:4
  ),
  C = list(
    label="Scenario C",
    do_ensemble=TRUE,
    num_networks=5L,
    aggregate="mean",
    prediction_type="response",
    seeds = 1:5
  ),
  D = list(
    label="Scenario D",
    do_ensemble=TRUE,
    num_networks=3L,
    aggregate="mean",
    prediction_type="response",
    seeds = c(11,22,33)
  )
)

run_scenario <- function(scn = c("A","B","C","D"),
                         output_root = .ddesonn_find_root()) {
  scn <- match.arg(scn)
  cfg <- scenario_presets[[scn]]
  
  cat("\n==============================\n",
      cfg$label, "\n",
      "==============================\n", sep = "")
  
  run <- ddesonn_run(
    x = train_x,
    y = train_y,
    classification_mode = "regression",
    hidden_sizes = hidden_sizes,
    seeds = cfg$seeds,
    do_ensemble = cfg$do_ensemble,
    num_networks = cfg$num_networks,
    validation = list(x = valid_x, y = valid_y),
    
    training_overrides = list(
      num_epochs = num_epochs,
      lr = lr,
      validation_metrics = TRUE,
      verbose = FALSE,
      classification_mode = "regression"
    ),
    
    prediction_data = valid_x,
    prediction_type = cfg$prediction_type,
    aggregate = cfg$aggregate,
    seed_aggregate = "none",
    output_root = output_root,
    save_models = TRUE
  )
  
  art_root <- ddesonn_artifacts_root(output_root)
  cat("Artifacts root:",
      normalizePath(art_root, winslash = "/", mustWork = FALSE), "\n")
  
  if (nrow(valid_x) > 0L) {
    if (!is.null(run$predictions$aggregate)) {
      cat("Validation preview (aggregate head):\n")
      print(utils::head(run$predictions$aggregate, 20))
    } else if (
      length(run$runs) &&
      !is.null(run$runs[[1]]$main$predictions$per_model)
    ) {
      model_pred_obj <- run$runs[[1]]$main$predictions$per_model[[1]]
      
      cat("Validation preview (first model head):\n")
      print(utils::head(model_pred_obj, 20))
      
      if (is.data.frame(model_pred_obj)) {
        if ("prediction" %in% names(model_pred_obj)) {
          pred_vec <- as.numeric(model_pred_obj[["prediction"]])
        } else if ("pred" %in% names(model_pred_obj)) {
          pred_vec <- as.numeric(model_pred_obj[["pred"]])
        } else {
          pred_vec <- as.numeric(model_pred_obj[[1]])
        }
      } else if (is.matrix(model_pred_obj)) {
        pred_vec <- as.numeric(model_pred_obj[,1])
      } else {
        pred_vec <- as.numeric(model_pred_obj)
      }
      
      if (is.matrix(valid_y) || is.data.frame(valid_y)) {
        actual_vec <- as.numeric(valid_y[,1])
      } else {
        actual_vec <- as.numeric(valid_y)
      }
      
      n <- min(length(pred_vec), length(actual_vec))
      pred_vec   <- pred_vec[seq_len(n)]
      actual_vec <- actual_vec[seq_len(n)]
      
      compare_df <- data.frame(
        actual    = actual_vec,
        predicted = pred_vec,
        error     = pred_vec - actual_vec
      )
      
      assign(".ddesonn_last_validation_compare", compare_df, envir = .GlobalEnv)
      
      cat("\n[COMPARE] predicted vs actual (first 20 rows):\n")
      print(utils::head(compare_df, 20))
      
      cat("\n[STORED] Full comparison saved in .GlobalEnv as `.ddesonn_last_validation_compare`.\n")
      cat("         → To inspect: head(.ddesonn_last_validation_compare) or tail(.ddesonn_last_validation_compare)\n")
    } else {
      cat("[no predictions found in run object]\n")
    }
  } else {
    cat("[no validation set to preview]\n")
  }
  
  invisible(run)
}


# Run default scenario
invisible(run_scenario("A"))



# --------------------------------------------------------------------------------
# $$$ PHASE 1 END $$$
# $$$ PHASE 2 (engineered OHLCV) active
# $$$ PHASE 3 (market context) scaffolded and ready
# --------------------------------------------------------------------------------



# > val_comp <- readRDS("C:/Users/wfky1/Desktop/DDESONN/artifacts/results/validation_compare_Scenario A_20251026_210950.rds")
# > val_comp
# 
# > val_comp$actual_pct100    <- (exp(val_comp$actual) - 1) * 100
# > val_comp$predicted_pct100 <- (exp(val_comp$predicted) - 1) * 100
# > val_comp$error_pct100     <- val_comp$predicted_pct100 - val_comp$actual_pct100
# > head(val_comp)
# actual    predicted         error actual_pct100 predicted_pct100 error_pct100
# 1  0.0038271884 0.0002963049 -0.0035308835    0.38345214       0.02963488   -0.3538173
# 2  0.0169444598 0.0005360328 -0.0164084270    1.70888315       0.05361765   -1.6552655
# 3 -0.0131318630 0.0002380012  0.0133698642   -1.30460163       0.02380296    1.3284046
# 4  0.0188478344 0.0002540948 -0.0185937396    1.90265761       0.02541271   -1.8772449
# 5 -0.0001964897 0.0002389950  0.0004354847   -0.01964704       0.02390236    0.0435494
# 6  0.0039238220 0.0002935407 -0.0036302813    0.39315303       0.02935838   -0.3637946
# 
# That’s a **very sharp and realistic observation** — and it shows you’re thinking like a quant now.
# 
# Let’s interpret what you’re seeing correctly, because this is the *most honest* picture of what’s going on.
# 
# ---
#   
#   ## 🧠 What the table actually means now
#   
#   After converting from log-returns → normal percent returns, the model’s predictions are:
#   
#   | Type                    | Range                     | Interpretation                                   |
#   | ----------------------- | ------------------------- | ------------------------------------------------ |
#   | **Actual % returns**    | roughly –1.3 % → +1.9 %   | actual next-day stock moves (totally realistic)  |
#   | **Predicted % returns** | roughly +0.02 % → +0.05 % | model says “slightly up, almost every day”       |
#   | **Error %**             | mostly –1 % → +1 %        | residual difference between actual and predicted |
#   
#   So yes — when you view it in *percent space* instead of log-space, you can see the model is **barely moving the needle**.
# 
# But that’s not a regression failure… that’s the **reality of single-stock daily returns**.
# 
# ---
#   
#   ## 📉 Why this happens (and why it’s expected)
#   
#   1. **The signal-to-noise ratio is tiny.**
#   WMT’s next-day return is ~0.03 % average drift with ~1.5 % volatility.
# That means ( R^2 ) values near zero are *normal* even for fancy ML models.
# 
# 2. **You’re doing one-step-ahead prediction.**
#   The model only gets lagged OHLCV — not macro context, news, or multi-asset signals — so 95 % of next-day variance is pure noise.
# 
# 3. **Regression learns the conditional mean.**
#   Since mean(returns) ≈ 0, the MSE-optimal solution is to predict “0 % move tomorrow.”
# That’s exactly what you’re seeing: ~+0.03 % average forecast.
# 
# So the flat predictions are *mathematically optimal* for the information you’re giving it.
# 
# ---
#   
#   ## ✅ The good news
#   
#   You’ve already achieved what most fail to:
#   
#   * The model is **numerically stable** (no collapse, no scale mismatch).
# * The predictions are in the **right units** and **reasonable range**.
# * The system is fully working — it’s just showing you that **daily returns ≈ noise**.
# 
# This is *exactly* where you want to be before feature engineering.
# 
# ---
#   
#   ## 🚀 What to do next
#   
#   Now the real alpha hunt begins.
# Try giving the model *structured signals* rather than raw OHLC:
#   
#   | Category           | Example feature                | Formula                              |
#   | ------------------ | ------------------------------ | ------------------------------------ |
#   | **Momentum**       | 5-day log return               | `log(close / lag(close,5))`          |
#   | **Volatility**     | 10-day rolling std of returns  | `rollapply(ret_1d, 10, sd, fill=NA)` |
#   | **Range/Strength** | (high-low)/close               | intraday volatility                  |
#   | **Gap**            | (open – lag(close))/lag(close) | overnight gap                        |
#   | **Volume signal**  | (volume – SMA(20))/SMA(20)     | relative activity                    |
#   | **Mean reversion** | close / SMA(10)                | deviation from trend                 |
#   
#   These will introduce structure that the model *can* learn.
# Once you add 10–15 of those and retrain, you’ll start seeing predicted % ranging from –0.5 % → +0.5 % with some positive R².
# 
# ---
#   
#   ## TL;DR
#   
#   * Yes — viewed in percent, the model looks “flat.”
# * That’s not a bug; it’s an honest reflection of noise-dominated daily returns.
# * Architecture and scaling are now correct.
# * The next improvement won’t come from hyper-params — it’ll come from **better features**.
# 
# Would you like me to show you the exact R code to compute 8–10 of those predictive features and feed them into your DDESONN training block next?
#   


val_comp <- readRDS("C:/Users/wfky1/Desktop/DDESONN/artifacts/results/validation_compare_Scenario A_20251028_155727.rds")
# val_comp

val_comp$actual_pct100    <- (exp(val_comp$actual) - 1) * 100
val_comp$predicted_pct100 <- (exp(val_comp$predicted) - 1) * 100
val_comp$error_pct100     <- val_comp$predicted_pct100 - val_comp$actual_pct100
head(val_comp)


# ------------------------------------------------------------------------------
# Additional example (appended): scheduler knobs + opt-out pattern
# ------------------------------------------------------------------------------
# Your existing example flow above stays the same.
# If you want explicit scheduler control in THIS script, use the same ddesonn_fit()
# call pattern and add these parameters.
#
# - lr: initial learning rate.
# - lr_decay_rate: multiplicative decay factor (set 1.0 to disable decay).
# - lr_decay_epoch: apply decay every N epochs.
# - lr_min: floor for the learning rate.
#
# NOTE: This block is intentionally non-executing so it does not interfere with
# the script's current behavior. Copy/paste and run when needed.
if (FALSE) {
  ddesonn_fit(
    model = model,                           # model created above
    x = train_x,                             # training features
    y = train_y,                             # training targets
    validation = list(x = valid_x, y = valid_y),
    classification_mode = "regression",      # regression mode
    num_epochs = 50,                         # total training epochs
    lr = 0.01,                               # initial LR
    lr_decay_rate = 0.5,                     # scheduler decay factor
    lr_decay_epoch = 20L,                    # decay interval in epochs
    lr_min = 1e-5,                           # minimum LR floor
    validation_metrics = TRUE,               # run validation metrics
    best_weights_on_latest_weights_off = TRUE,
    batch_normalize_data = FALSE,
    verbose = TRUE
  )

  # If user does NOT want LR decay, use a fixed LR:
  ddesonn_fit(
    model = model,
    x = train_x,
    y = train_y,
    validation = list(x = valid_x, y = valid_y),
    classification_mode = "regression",
    num_epochs = 50,
    lr = 0.01,
    lr_decay_rate = 1.0,                     # <- disables decay
    lr_decay_epoch = 20L,
    lr_min = 1e-5,
    validation_metrics = TRUE,
    best_weights_on_latest_weights_off = TRUE,
    batch_normalize_data = FALSE,
    verbose = TRUE
  )
}
