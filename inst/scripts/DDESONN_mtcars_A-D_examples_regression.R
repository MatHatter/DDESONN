#!/usr/bin/env Rscript

suppressPackageStartupMessages(source("R/api.R"))

`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

CLASSIFICATION_MODE <- "regression"
PREDICT_NEXT_DAY    <- TRUE
REG_TARGET_MODE     <- "return_log"
REDUCE_DATA         <- TRUE

num_epochs <- 200
lr         <- 0.01
hidden_sizes <- c(128, 64, 32)
RESTORE_BEST_WEIGHTS <- TRUE

suppressPackageStartupMessages({
  library(dplyr)
})

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

make_date_numeric <- function(df) {
  if (!"date" %in% names(df)) return(df)
  d <- df[["date"]]
  if (inherits(d, "POSIXt")) {
    df[["date"]] <- as.numeric(as.Date(d))
  } else if (inherits(d, "Date")) {
    df[["date"]] <- as.numeric(d)
  } else {
    suppressWarnings({ parsed <- as.Date(d) })
    if (all(is.na(parsed))) {
      df[["date"]] <- NA_real_
    } else {
      df[["date"]] <- as.numeric(parsed)
    }
  }
  df
}

X_train      <- make_date_numeric(X_train)
X_validation <- make_date_numeric(X_validation)
X_test       <- make_date_numeric(X_test)

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

X_train_df <- as.data.frame(X_train)
X_val_df   <- as.data.frame(X_validation)
X_test_df  <- as.data.frame(X_test)

num_mask <- vapply(X_train_df, is.numeric, TRUE)
if (!any(num_mask)) stop("no numeric predictors")

X_train_num <- as.matrix(X_train_df[, num_mask, drop = FALSE])
X_val_num   <- as.matrix(X_val_df[,   num_mask, drop = FALSE])
X_test_num  <- as.matrix(X_test_df[,  num_mask, drop = FALSE])

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

# ---------------------------------------------------------------
# SCALE UP TARGETS to strengthen gradient signal
# ---------------------------------------------------------------
TARGET_SCALE <- 1  # <<< new hyperparam

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

train_x <- as.matrix(X_train_scaled_final)
train_y <- y_train_mat

valid_x <- as.matrix(X_validation_scaled_final)
valid_y <- as.matrix(y_validation)

test_x  <- as.matrix(X_test_scaled_final)
test_y  <- as.matrix(y_test)

input_size  <- ncol(train_x)
output_size <- 1L

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
  target_transform  = target_transform,
  y_trained_scaled  = isTRUE(y_trained_scaled)
)

assign("preprocessScaledData", preprocessScaledData, inherits = TRUE)
assign("target_transform",     target_transform,     inherits = TRUE)

cat("=== [reg] Diagnostics ===\n")
cat("train_x dim:", paste(dim(train_x), collapse="x"), "\n")
cat("valid_x dim:", paste(dim(valid_x), collapse="x"), "\n")
cat("test_x  dim:", paste(dim(test_x),  collapse="x"), "\n")
cat("NAs? train:", anyNA(train_x), " val:", anyNA(valid_x), " test:", anyNA(test_x), "\n")

model <- ddesonn_model(
  input_size          = input_size,
  output_size         = output_size,
  hidden_sizes        = hidden_sizes,
  num_networks        = 1L,
  classification_mode = "regression",
  ML_NN               = TRUE,
  custom_scale = 10
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

VALID_RMSE <- rmse_val
VALID_MAE  <- mae_val
VALID_R2   <- r2_val
TEST_RMSE  <- rmse_test
TEST_MAE   <- mae_test
TEST_R2    <- r2_test
COMPARISON_VALID <- comparison_valid
COMPARISON_TEST  <- comparison_test

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
  
  art_root <- {
    nr <- normalizePath(output_root, winslash = "/", mustWork = FALSE)
    if (basename(nr) == "artifacts") nr else file.path(output_root, "artifacts")
  }
  cat("Artifacts root:",
      normalizePath(art_root, winslash = "/", mustWork = FALSE), "\n")
  
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
    
    cat("\n[COMPARE] predicted vs actual (first 20 rows):\n")
    print(utils::head(compare_df, 20))
    
    run_dir <- NULL
    if (!is.null(run$meta) && !is.null(run$meta$root_dir)) {
      run_dir <- run$meta$root_dir
    } else if (!is.null(run$runs) &&
               length(run$runs) &&
               !is.null(run$runs[[1]]$meta$root_dir)) {
      run_dir <- run$runs[[1]]$meta$root_dir
    }
    if (is.null(run_dir)) {
      run_dir <- art_root
    }
    
    results_dir <- file.path(run_dir, "results")
    dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
    
    tstamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    compare_path <- file.path(
      results_dir,
      paste0("validation_compare_", cfg$label, "_", tstamp, ".rds")
    )
    
    saveRDS(compare_df, compare_path)
    cat("\n[SAVED] validation comparison RDS:\n",
        normalizePath(compare_path), "\n")
    
  } else {
    cat("[no predictions found in run object]\n")
  }
  
  invisible(run)
}

invisible(run_scenario("A"))
