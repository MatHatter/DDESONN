#!/usr/bin/env Rscript
## ============================================================
## DDESONN LOCAL MVP (zero Techila, zero foreach)
## ============================================================

suppressPackageStartupMessages({
  library(DDESONN)
})

## 1. Load your code -------------------------------------------------
# Resolve the DDESONN package root dynamically (works from any subdir),
# then source from <root>/R. No hardcoded local machine paths.
pkg_root <- rprojroot::find_root(rprojroot::is_r_package)
r_path   <- file.path(pkg_root, "R")

r_files <- list.files(r_path, pattern = "\\.R$", recursive = TRUE, full.names = TRUE)

cat("[LOCAL] getwd()    = ", getwd(), "\n", sep = "")
cat("[LOCAL] pkg_root   = ", pkg_root, "\n", sep = "")
cat("[LOCAL] sourcing R = ", r_path, "\n", sep = "")

invisible(lapply(r_files, sys.source, envir = environment()))

## 2. Minimal hyperparameters ---------------------------------------
CLASSIFICATION_MODE <- "binary"
do_ensemble         <- FALSE
num_networks        <- 1L
seed_value          <- 1L  # single run, one seed

set.seed(seed_value)  # ensures reproducibility for all random ops

# constructor args for DDESONN$new()
N               <- 1L
lambda          <- 0
ensemble_number <- 0L
ensembles       <- NULL
ML_NN           <- TRUE
activation_functions <- list(relu, relu, sigmoid)
activation_functions_predict <- activation_functions
init_method     <- "xavier"
custom_scale    <- NULL

# train() args your model expects
lr                        <- 0.001
lr_decay_rate             <- 0
lr_decay_epoch            <- 0
lr_min                    <- 0
self_org                  <- FALSE
threshold                 <- NULL
reg_type                  <- NULL
numeric_columns           <- NULL
dropout_rates             <- NULL
optimizer                 <- "sgd"
beta1                     <- 0.9
beta2                     <- 0.999
epsilon                   <- 1e-8
lookahead_step            <- 0L
batch_normalize_data      <- FALSE
gamma_bn                  <- 1
beta_bn                   <- 0
epsilon_bn                <- 1e-5
momentum_bn               <- 0.9
is_training_bn            <- TRUE
shuffle_bn                <- FALSE
loss_type                 <- "mse"
update_weights            <- TRUE
update_biases             <- TRUE
sample_weights            <- NULL
preprocessScaledData      <- FALSE
validation_metrics        <- TRUE
threshold_function        <- NULL
best_weights_on_latest_weights_off <- FALSE
train_flag                <- TRUE
grouped_metrics           <- FALSE
viewTables                <- FALSE
verbose                   <- TRUE

## 3. Dummy data (train == val for now) ------------------------------
X <- matrix(rnorm(100 * 12), ncol = 12)
y <- sample(0:1, 100, replace = TRUE)

X_train      <- X
y_train      <- y
X_validation <- X
y_validation <- y

## 4. Output dirs / file paths --------------------------------------
ts_stamp  <- format(Sys.time(), "%Y%m%d_%H%M%S")
OUT_ROOT  <- file.path(ddesonn_artifacts_root(NULL), "singleRuns")
RUN_DIR   <- normalizePath(
  file.path(OUT_ROOT, paste0("Local_MVP_", ts_stamp)),
  winslash = "/",
  mustWork = FALSE
)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

agg_metrics_file_train <- file.path(RUN_DIR, "Train_Metrics.rds")
agg_metrics_file_test  <- file.path(RUN_DIR, "Test_Metrics.rds")

cat("[LOCAL] RUN_DIR = ", RUN_DIR, "\n", sep = "")

## helper: flatten + filter metrics ---------------------------------
## input: pm_list, rm_list
## output: named list of columns we want to keep
flatten_and_filter_metrics <- function(pm_list, rm_list) {
  
  # flatten the nested lists to a single named vector
  raw_flat <- tryCatch(
    rapply(
      list(performance_metric = pm_list,
           relevance_metric   = rm_list),
      f   = function(z) z,
      how = "unlist"
    ),
    error = function(e) setNames(vector("list", 0L), character(0))
  )
  
  # Keep only atomic scalars length 1
  if (length(raw_flat)) {
    L <- as.list(raw_flat)
    raw_flat <- raw_flat[
      vapply(L, is.atomic, logical(1)) &
        lengths(L) == 1L
    ]
  }
  
  nms <- names(raw_flat)
  
  if (length(nms)) {
    # we ALWAYS allow these two detail fields
    whitelist_details <- grepl("^details\\.best_threshold$", nms) |
      grepl("^details\\.tuned_by$",       nms)
    
    # fields to drop
    is_custom_rel_err <- grepl("custom_relative_error_binned", nms, fixed = TRUE)
    is_grid_used      <- grepl("grid_used",                    nms, fixed = TRUE)
    
    # anything with "details" in the name...
    is_details        <- grepl("(^|\\.)details(\\.|$)", nms)
    
    # ...but we only drop details.* if it's NOT whitelisted
    drop_details      <- is_details & !whitelist_details
    
    drop <- is_custom_rel_err | is_grid_used | drop_details
    
    keep <- !drop  # IMPORTANT: do not filter out NA values, just drop by name
    
    raw_flat <- raw_flat[keep]
    nms      <- names(raw_flat)
  }
  
  # coerce to simple atomic values for data.frame
  out_list <- if (length(raw_flat)) as.list(raw_flat) else list()
  num_coerced <- suppressWarnings(as.numeric(raw_flat))
  
  for (jj in seq_along(raw_flat)) {
    out_list[[names(raw_flat)[jj]]] <-
      if (!is.na(num_coerced[jj])) num_coerced[jj] else as.character(raw_flat[[jj]])
  }
  
  out_list
}

## 5. Build model ---------------------------------------------------
cat("[LOCAL] constructing model...\n")

run_model <- DDESONN$new(
  num_networks    = max(1L, as.integer(num_networks)),
  input_size      = ncol(X),
  hidden_sizes    = c(64, 32),   # your real hidden sizes
  output_size     = 1L,
  N               = N,
  lambda          = lambda,
  ensemble_number = 0L,
  ensembles       = ensembles,
  ML_NN           = ML_NN,
  activation_functions           = activation_functions,
  activation_functions_predict   = activation_functions_predict,
  init_method     = init_method,
  custom_scale    = custom_scale
)

## 6. Train model ---------------------------------------------------
cat("[LOCAL] training...\n")

model_results <- run_model$train(
  Rdata                       = X,
  labels                      = y,
  X_train                     = X_train,
  y_train                     = y_train,
  lr                          = lr,
  lr_decay_rate               = lr_decay_rate,
  lr_decay_epoch              = lr_decay_epoch,
  lr_min                      = lr_min,
  num_networks                = num_networks,
  ensemble_number             = 0L,
  do_ensemble                 = do_ensemble,
  num_epochs                  = 360,  # your real epoch count
  self_org                    = self_org,
  threshold                   = threshold,
  reg_type                    = reg_type,
  numeric_columns             = numeric_columns,
  CLASSIFICATION_MODE         = CLASSIFICATION_MODE,
  activation_functions        = activation_functions,
  activation_functions_predict= activation_functions_predict,
  dropout_rates               = dropout_rates,
  optimizer                   = optimizer,
  beta1                       = beta1,
  beta2                       = beta2,
  epsilon                     = epsilon,
  lookahead_step              = lookahead_step,
  batch_normalize_data        = batch_normalize_data,
  gamma_bn                    = gamma_bn,
  beta_bn                     = beta_bn,
  epsilon_bn                  = epsilon_bn,
  momentum_bn                 = momentum_bn,
  is_training_bn              = is_training_bn,
  shuffle_bn                  = shuffle_bn,
  loss_type                   = loss_type,
  update_weights              = update_weights,
  update_biases               = update_biases,
  sample_weights              = sample_weights,
  preprocessScaledData        = preprocessScaledData,
  X_validation                = X_validation,
  y_validation                = y_validation,
  validation_metrics          = validation_metrics,
  threshold_function          = threshold_function,
  best_weights_on_latest_weights_off = best_weights_on_latest_weights_off,
  ML_NN                       = ML_NN,
  train                       = train_flag,
  grouped_metrics             = grouped_metrics,
  viewTables                  = viewTables,
  verbose                     = verbose
)

cat("[LOCAL] train() finished.\n")

## 7. Extract training metrics --------------------------------------
pm <- try(model_results$performance_relevance_data$performance_metric,
          silent = TRUE)
rm <- try(model_results$performance_relevance_data$relevance_metric,
          silent = TRUE)

train_metrics_list <- flatten_and_filter_metrics(pm, rm)

train_row <- data.frame(
  seed      = seed_value,
  split     = "train",
  status    = if (!inherits(pm, "try-error")) "ok" else "fail",
  as.data.frame(train_metrics_list, check.names = TRUE),
  row.names = NULL
)

## 8. Evaluate "test" via DDESONN_predict_eval ----------------------
cat("[LOCAL] running predict_eval...\n")

test_eval <- try(
  DDESONN_predict_eval(
    LOAD_FROM_RDS        = FALSE,
    ENV_META_NAME        = "MVP_eval_env",
    INPUT_SPLIT          = "test",
    CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
    RUN_INDEX            = 1L,
    SEED                 = seed_value,
    OUTPUT_DIR           = RUN_DIR,
    OUT_DIR_ASSERT       = RUN_DIR,
    SAVE_METRICS_RDS     = FALSE,
    METRICS_PREFIX       = "metrics_test",
    MODEL_SLOT           = 1L
  ),
  silent = TRUE
)

if (!inherits(test_eval, "try-error") &&
    is.list(test_eval$performance_metric)) {
  
  test_pm <- test_eval$performance_metric
  test_rm <- test_eval$relevance_metric
  
  test_metrics_list <- flatten_and_filter_metrics(test_pm, test_rm)
  
  test_row <- data.frame(
    seed      = seed_value,
    split     = "test",
    status    = "ok",
    as.data.frame(test_metrics_list, check.names = TRUE),
    row.names = NULL
  )
} else {
  test_row <- data.frame(
    seed      = seed_value,
    split     = "test",
    status    = "no-metrics",
    row.names = NULL
  )
}

## 9. Save results ---------------------------------------------------
saveRDS(train_row, agg_metrics_file_train)
saveRDS(test_row,  agg_metrics_file_test)

## 10. Print summary ------------------------------------------------
cat("\n=== TRAIN ROW ===\n")
print(train_row)

cat("\n=== TEST ROW ===\n")
print(test_row)

cat(
  "\nSaved:\n- ", agg_metrics_file_train,
  "\n- ", agg_metrics_file_test,
  "\n", sep = ""
)

cat("\nDone.\n")
