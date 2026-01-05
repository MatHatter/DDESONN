#!/usr/bin/env Rscript
## ============================================================
## DDESONN TECHILA MVP (foreach-based, full flatten/filter)
## ============================================================

suppressPackageStartupMessages({
  library(foreach)
  library(techila)
  library(R6)
})

## ------------------------------------------------------------
## 1. Load core code (LOCAL DEV ONLY)
##    On Techila workers these are auto-sourced via .options.files (flattened)
## ------------------------------------------------------------
r_files <- list.files("R", pattern = "\\.R$", recursive = TRUE, full.names = TRUE)
invisible(lapply(r_files, sys.source, envir = environment()))

## ------------------------------------------------------------
## 1a. Files & packages for workers (FLATTENED paths)
##     Techila flattens directories, so drop the leading 'R/' on workers.
## ------------------------------------------------------------
files_to_source <- basename(list.files("R", pattern = "\\.R$", recursive = TRUE))

pkgs_for_workers <- c(
  "R6",
  "data.table",
  "dplyr",
  "ggplot2",
  "reshape2"
  # "techila"  # not strictly required in worker R session for foreach payload
)

## If you need binary deps (Linux only), you could add a databundle like:
## databundles_linux <- list("/usr/lib/libR.so")
## On Windows, set databundles_linux <- NULL

## ------------------------------------------------------------
## 2. Global config
## ------------------------------------------------------------
CLASSIFICATION_MODE <- "binary"
train          <- TRUE
test           <- TRUE
do_ensemble    <- FALSE
num_networks   <- 1L
seeds          <- 1L   # vector OK too: c(1L)

## reproducibility
set.seed(1)

## model + training params
N               <- 1L
lambda          <- 0
ensemble_number <- 0L
ensembles       <- NULL
ML_NN           <- TRUE
activation_functions <- list(relu, relu, sigmoid)
activation_functions_predict <- activation_functions
init_method     <- "xavier"
custom_scale    <- NULL

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

## ------------------------------------------------------------
## 3. Dummy dataset
## ------------------------------------------------------------
X <- matrix(rnorm(100 * 12), ncol = 12)
y <- sample(0:1, 100, replace = TRUE)
X_train <- X
y_train <- y
X_validation <- X
y_validation <- y

## ------------------------------------------------------------
## 4. Output dirs / filenames
## ------------------------------------------------------------
ts_stamp  <- format(Sys.time(), "%Y%m%d_%H%M%S")
OUT_ROOT  <- "artifacts/SingleRuns"
RUN_DIR   <- normalizePath(file.path(OUT_ROOT, paste0("Techila_MVP_", ts_stamp)),
                           winslash = "/", mustWork = FALSE)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

agg_metrics_file_train <- file.path(RUN_DIR, "Train_Metrics.rds")
agg_metrics_file_test  <- file.path(RUN_DIR, "Test_Metrics.rds")

cat("[TECHILA] RUN_DIR = ", RUN_DIR, "\n", sep = "")

## ------------------------------------------------------------
## 5. Helper: flatten + filter metrics
## ------------------------------------------------------------
flatten_and_filter_metrics <- function(pm_list, rm_list) {
  raw_flat <- tryCatch(
    rapply(
      list(performance_metric = pm_list,
           relevance_metric   = rm_list),
      f   = function(z) z,
      how = "unlist"
    ),
    error = function(e) setNames(vector("list", 0L), character(0))
  )
  
  if (length(raw_flat)) {
    L <- as.list(raw_flat)
    raw_flat <- raw_flat[
      vapply(L, is.atomic, logical(1)) &
        lengths(L) == 1L
    ]
  }
  
  nms <- names(raw_flat)
  
  if (length(nms)) {
    whitelist_details <- grepl("^details\\.best_threshold$", nms) |
      grepl("^details\\.tuned_by$",       nms)
    
    is_custom_rel_err <- grepl("custom_relative_error_binned", nms, fixed = TRUE)
    is_grid_used      <- grepl("grid_used",                    nms, fixed = TRUE)
    is_details        <- grepl("(^|\\.)details(\\.|$)",        nms)
    
    drop_details      <- is_details & !whitelist_details
    drop <- is_custom_rel_err | is_grid_used | drop_details
    
    keep <- !drop
    raw_flat <- raw_flat[keep]
    nms      <- names(raw_flat)
  }
  
  out_list <- if (length(raw_flat)) as.list(raw_flat) else list()
  num_coerced <- suppressWarnings(as.numeric(raw_flat))
  for (jj in seq_along(raw_flat)) {
    out_list[[names(raw_flat)[jj]]] <-
      if (!is.na(num_coerced[jj])) num_coerced[jj] else as.character(raw_flat[[jj]])
  }
  
  out_list
}

## ------------------------------------------------------------
## 6. Initialize Techila backend
## ------------------------------------------------------------
## IMPORTANT: foreach+doTechila will init/uninit automatically.
## Avoid explicit techila::init(); just register backend.
techila::registerDoTechila()

cat("[TECHILA] Backend registered: ", foreach::getDoParName(),
    " | workers=", foreach::getDoParWorkers(), "\n", sep = "")

## ------------------------------------------------------------
## 7. foreach + Techila workers
##    - Removed the warm-up `peach` block entirely.
##    - Added .options.files / .options.packages for worker sourcing.
## ------------------------------------------------------------
res_list <- foreach::foreach(
  i = seq_along(seeds),
  .combine = "c",  # list-concatenate; we unpack later
  .options.files = files_to_source,
  .options.packages = pkgs_for_workers
  # , .options.databundles = databundles_linux   # <- enable on Linux if needed
  # Export everything by default; you can narrow if desired:
  , .export = ls()
) %dopar% {
  
  s <- seeds[i]; set.seed(s)
  cat(sprintf("[WORKER %d] seed %d\n", i, s))
  
  ## model constructor
  run_model <- DDESONN$new(
    num_networks    = max(1L, as.integer(num_networks)),
    input_size      = ncol(X),
    hidden_sizes    = c(64, 32),
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
  
  ## train model
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
    num_epochs                  = 360,
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
    epsilon                     <- epsilon,
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
  
  ## flatten + filter metrics
  pm <- try(model_results$performance_relevance_data$performance_metric, silent = TRUE)
  rm <- try(model_results$performance_relevance_data$relevance_metric,   silent = TRUE)
  train_metrics_list <- flatten_and_filter_metrics(pm, rm)
  
  train_row <- data.frame(
    run_index = i,
    seed      = s,
    split     = "train",
    status    = if (!inherits(pm, "try-error")) "ok" else "fail",
    as.data.frame(train_metrics_list, check.names = TRUE),
    row.names = NULL
  )
  
  ## predict/eval for test
  test_eval <- try(
    DDESONN_predict_eval(
      LOAD_FROM_RDS        = FALSE,
      ENV_META_NAME        = "MVP_eval_env",
      INPUT_SPLIT          = "test",
      CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
      RUN_INDEX            = i,
      SEED                 = s,
      OUTPUT_DIR           = RUN_DIR,
      OUT_DIR_ASSERT       = RUN_DIR,
      SAVE_METRICS_RDS     = FALSE,
      METRICS_PREFIX       = "metrics_test",
      MODEL_SLOT           = 1L
    ),
    silent = TRUE
  )
  
  if (!inherits(test_eval, "try-error") && is.list(test_eval$performance_metric)) {
    test_metrics_list <- flatten_and_filter_metrics(
      test_eval$performance_metric, test_eval$relevance_metric
    )
    test_row <- data.frame(
      run_index = i,
      seed      = s,
      split     = "test",
      status    = "ok",
      as.data.frame(test_metrics_list, check.names = TRUE),
      row.names = NULL
    )
  } else {
    test_row <- data.frame(
      run_index = i,
      seed      = s,
      split     = "test",
      status    = "no-metrics",
      row.names = NULL
    )
  }
  
  list(train_rows = train_row, test_rows = test_row)
}

## ------------------------------------------------------------
## 8. Aggregate and save results
## ------------------------------------------------------------
all_tr <- unlist(lapply(res_list, `[[`, "train_rows"), recursive = FALSE)
all_te <- unlist(lapply(res_list, `[[`, "test_rows"),  recursive = FALSE)

df_tr <- if (length(all_tr)) do.call(rbind, all_tr) else data.frame()
df_te <- if (length(all_te)) do.call(rbind, all_te) else data.frame()

saveRDS(df_tr, agg_metrics_file_train)
saveRDS(df_te, agg_metrics_file_test)

cat("\n=== TRAIN TABLE ===\n")
if (nrow(df_tr)) print(df_tr) else cat("[EMPTY]\n")

cat("\n=== TEST TABLE ===\n")
if (nrow(df_te)) print(df_te) else cat("[EMPTY]\n")

cat("\nSaved:\n- ", agg_metrics_file_train, "\n- ", agg_metrics_file_test, "\n", sep = "")
cat("\n[TECHILA] Done.\n")
