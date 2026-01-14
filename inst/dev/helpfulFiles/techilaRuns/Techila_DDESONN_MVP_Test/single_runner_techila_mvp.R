#!/usr/bin/env Rscript

## ============================================================
## DDESONN TECHILA MVP (foreach-based) — FULL FIXED PARITY       #$$$$$$$$$$$$$
## - Matches working local single-run behavior:
##   * predict_eval is a WRITER -> read AGG_METRICS_FILE          #$$$$$$$$$$$$$
##   * force list-shaped weight/bias records for eval             #$$$$$$$$$$$$$
##   * predictor_fn_safe always passes weights/biases + AF_predict#$$$$$$$$$$$$$
##   * per-worker run dir to avoid collisions                     #$$$$$$$$$$$$$
## ============================================================

suppressPackageStartupMessages({
  library(foreach)
  library(techila)
  library(R6)
})

## ============================================================
## SECTION: 0) Resolve runner root (Techila-safe)                #$$$$$$$$$$$$$
## ============================================================
.get_runner_root <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", cmd[grep("^--file=", cmd)])
  if (length(file_arg) == 1L && nzchar(file_arg)) {
    return(dirname(normalizePath(file_arg, winslash = "/", mustWork = TRUE)))
  }
  of <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(of) && nzchar(of)) {
    return(dirname(normalizePath(of, winslash = "/", mustWork = TRUE)))
  }
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

RUNNER_ROOT <- .get_runner_root()
setwd(RUNNER_ROOT)
cat("[TECHILA] RUNNER_ROOT = ", RUNNER_ROOT, "\n", sep = "")

## ============================================================
## SECTION: 1) Source code (LOCAL DEV)                           #$$$$$$$$$$$$$
## - On Techila workers these are sourced via .options.files     #$$$$$$$$$$$$$
## - Ensure report version is sourced LAST                       #$$$$$$$$$$$$$
## ============================================================
r_files <- list.files("R", pattern = "\\.R$", recursive = TRUE, full.names = TRUE)

## Force the LOCAL runner to use the SAME report file as your single-run block  #$$$$$$$$$$$$$
report_file <- file.path(RUNNER_ROOT, "R", "reports", "evaluate_predictions_report_original.R")  #$$$$$$$$$$$$$
if (!file.exists(report_file)) {
  stop("Techila runner report file not found: ", report_file)
}

r_files <- sort(r_files)
r_files <- c(setdiff(r_files, report_file), report_file)

invisible(lapply(r_files, sys.source, envir = environment()))

## If package namespace is loaded, force it to use this report function too     #$$$$$$$$$$$$$
if ("DDESONN" %in% loadedNamespaces() && exists("EvaluatePredictionsReport", inherits = TRUE)) {  #$$$$$$$$$$$$$
  try(assign("EvaluatePredictionsReport", get("EvaluatePredictionsReport", inherits = TRUE), envir = asNamespace("DDESONN")), silent = TRUE) #$$$$$$$$$$$$$
}

## ============================================================
## SECTION: 1a) Files & packages for Techila workers (flattened)  #$$$$$$$$$$$$$
## ============================================================
files_to_source <- basename(list.files("R", pattern = "\\.R$", recursive = TRUE))                 #$$$$$$$$$$$$$

pkgs_for_workers <- c(
  "R6",
  "data.table",
  "dplyr",
  "ggplot2",
  "reshape2"
)

## ============================================================
## SECTION: 2) Hyperparameters (match local single-run intent)   #$$$$$$$$$$$$$
## ============================================================
CLASSIFICATION_MODE <- "binary"
self_org <- FALSE
train <- TRUE
test <- TRUE

do_ensemble         <- FALSE
num_networks        <- 1L
num_temp_iterations <- 0L

ML_NN <- TRUE
grouped_metrics <- FALSE
update_weights <- TRUE
update_biases  <- TRUE

hidden_sizes <- c(64, 32)

init_method <- "he"
optimizer   <- "adagrad"
lookahead_step <- 5L

batch_normalize_data <- TRUE
shuffle_bn <- FALSE
gamma_bn <- .6
beta_bn  <- .6
epsilon_bn <- 1e-6
momentum_bn <- 0.9
is_training_bn <- TRUE

beta1 <- .9
beta2 <- 0.8
epsilon <- 1e-7

lr <- .125
lambda <- 0.00028
custom_scale <- 1.04349

lr_decay_rate  <- 0.5
lr_decay_epoch <- 20
lr_min <- 1e-5

validation_metrics <- TRUE
best_weights_on_latest_weights_off <- TRUE

dropout_rates <- list(0.10)
threshold_function <- tune_threshold_accuracy
threshold <- .5

loss_type <- "CrossEntropy"

activation_functions <- list(relu, relu, sigmoid)
activation_functions_predict <- activation_functions

sample_weights <- NULL
preprocessScaledData <- FALSE

viewTables <- FALSE
verbose <- TRUE

## ============================================================
## SECTION: 3) Dataset load + split + scale (Techila-safe)       #$$$$$$$$$$$$$
## - Mirrors your local single-run HF loader                      #$$$$$$$$$$$$$
## ============================================================
csv_path <- system.file("extdata", "heart_failure_clinical_records.csv", package = "DDESONN")

project_root <- getwd()
if (!nzchar(csv_path)) {
  for (i in 1:10) {
    candidate <- file.path(project_root, "inst", "extdata", "heart_failure_clinical_records.csv")
    if (file.exists(candidate)) break
    parent <- dirname(project_root)
    if (identical(parent, project_root)) break
    project_root <- parent
  }
  
  csv_path <- normalizePath(
    file.path(project_root, "inst", "extdata", "heart_failure_clinical_records.csv"),
    winslash = "/",
    mustWork = FALSE
  )
}

if (!file.exists(csv_path)) {
  stop(
    "heart_failure_clinical_records.csv not found.\n",
    "Tried:\n",
    "- installed package extdata via system.file(package='DDESONN')\n",
    "- walking up to inst/extdata from getwd()\n",
    "Current getwd(): ", getwd(), "\n",
    "Resolved project_root: ", project_root, "\n",
    call. = FALSE
  )
}

cat("[DATA] Using CSV: ", csv_path, "\n", sep = "")

data <- read.csv(csv_path, stringsAsFactors = FALSE)
dependent_variable <- "DEATH_EVENT"

na_count <- sum(is.na(data))
message(sprintf("[split] NA count: %s", na_count))

suppressPackageStartupMessages({
  if (requireNamespace("dplyr", quietly = TRUE)) {
    data <- dplyr::as_tibble(data)
    data <- dplyr::mutate(data, dplyr::across(where(is.character), as.factor))
    data <- as.data.frame(data)
  } else {
    for (nm in names(data)) if (is.character(data[[nm]])) data[[nm]] <- as.factor(data[[nm]])
  }
})

X_full <- data[, setdiff(names(data), dependent_variable), drop = FALSE]
y_full <- data[, dependent_variable, drop = FALSE]
colname_y <- colnames(y_full)

numeric_columns <- c('age','creatinine_phosphokinase','ejection_fraction',
                     'platelets','serum_creatinine','serum_sodium','time')

USE_TIME_SPLIT <- TRUE

if (USE_TIME_SPLIT) {
  stopifnot(nrow(X_full) == nrow(y_full))
  total_num_samples <- nrow(X_full)
  
  p_train <- 0.70
  p_val   <- 0.15
  
  num_training_samples   <- max(1L, floor(p_train * total_num_samples))
  num_validation_samples <- max(1L, floor(p_val   * total_num_samples))
  num_test_samples       <- max(0L, total_num_samples - num_training_samples - num_validation_samples)
  
  train_indices      <- seq_len(num_training_samples)
  validation_indices <- if (num_validation_samples > 0L)
    seq(from = max(train_indices) + 1L, length.out = num_validation_samples)
  else integer()
  test_indices <- if (num_test_samples > 0L)
    seq(from = max(c(train_indices, validation_indices)) + 1L, length.out = num_test_samples)
  else integer()
  
  X_train_raw      <- X_full[train_indices,      , drop = FALSE]; y_train_raw      <- y_full[train_indices,      , drop = FALSE]
  X_validation_raw <- X_full[validation_indices, , drop = FALSE]; y_validation_raw <- y_full[validation_indices, , drop = FALSE]
  X_test_raw       <- X_full[test_indices,       , drop = FALSE]; y_test_raw       <- y_full[test_indices,       , drop = FALSE]
  
  cat(sprintf("[SPLIT chrono] train=%d val=%d test=%d\n",
              nrow(X_train_raw), nrow(X_validation_raw), nrow(X_test_raw)))
} else {
  stop("This Techila runner is locked to chrono split for parity.", call. = FALSE)  #$$$$$$$$$$$$$
}

X_train_scaled <- scale(X_train_raw)
center <- attr(X_train_scaled, "scaled:center")
scale_ <- attr(X_train_scaled, "scaled:scale")

X_validation_scaled <- scale(X_validation_raw, center = center, scale = scale_)
X_test_scaled       <- scale(X_test_raw,       center = center, scale = scale_)

max_val <- suppressWarnings(max(abs(X_train_scaled)))
if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1

X_train_scaled      <- X_train_scaled      / max_val
X_validation_scaled <- X_validation_scaled / max_val
X_test_scaled       <- X_test_scaled       / max_val

X_train <- as.matrix(X_train_scaled)
y_train <- as.matrix(y_train_raw)

X_validation <- as.matrix(X_validation_scaled)
y_validation <- as.matrix(y_validation_raw)

X_test <- as.matrix(X_test_scaled)
y_test <- as.matrix(y_test_raw)

colnames(y_train) <- colname_y
colnames(y_validation) <- colname_y
colnames(y_test) <- colname_y

Rdata  <- X_train
labels <- y_train

input_size  <- ncol(Rdata)
output_size <- 1L

## ============================================================
## SECTION: 4) RUN_DIR + output files (per run)                  #$$$$$$$$$$$$$
## ============================================================
ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

## Match the “always under reports with stamp” pattern            #$$$$$$$$$$$$$
REPORTS_DIR <- normalizePath(file.path(RUNNER_ROOT, "R", "reports"), winslash = "/", mustWork = FALSE)  #$$$$$$$$$$$$$
dir.create(REPORTS_DIR, recursive = TRUE, showWarnings = FALSE)                                         #$$$$$$$$$$$$$

RUN_DIR <- normalizePath(
  file.path(REPORTS_DIR, sprintf("Techila_MVP_run_artifacts_%s", ts_stamp)),                             #$$$$$$$$$$$$$
  winslash = "/",
  mustWork = FALSE
)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

agg_metrics_file_train <- file.path(RUN_DIR, "agg_metrics_train.rds")
agg_metrics_file_test  <- file.path(RUN_DIR, "agg_metrics_test.rds")

cat("[TECHILA] RUN_DIR = ", RUN_DIR, "\n", sep = "")

## ============================================================
## SECTION: 5) Helper: flatten + filter metrics                  #$$$$$$$$$$$$$
## ============================================================
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
    
    drop_details <- is_details & !whitelist_details
    drop <- is_custom_rel_err | is_grid_used | drop_details
    
    raw_flat <- raw_flat[!drop]
  }
  
  out_list <- if (length(raw_flat)) as.list(raw_flat) else list()
  num_coerced <- suppressWarnings(as.numeric(raw_flat))
  
  for (jj in seq_along(raw_flat)) {
    out_list[[names(raw_flat)[jj]]] <-
      if (!is.na(num_coerced[jj])) num_coerced[jj]
    else as.character(raw_flat[[jj]])
  }
  
  out_list
}

## ============================================================
## SECTION: 6) Register Techila foreach backend                  #$$$$$$$$$$$$$
## ============================================================
techila::registerDoTechila()
cat("[TECHILA] Backend registered: ", foreach::getDoParName(),
    " | workers=", foreach::getDoParWorkers(), "\n", sep = "")

## ============================================================
## SECTION: 7) foreach + Techila workers (FULL FIXED)            #$$$$$$$$$$$$$
## - Per-worker metadata env + eval-writer AGG files + read back #$$$$$$$$$$$$$
## ============================================================
seeds <- 111L  # set to vector if desired: c(111L, 222L, 333L)

res_list <- foreach::foreach(
  i = seq_along(seeds),
  .combine = "c",
  .options.files = files_to_source,
  .options.packages = pkgs_for_workers,
  .export = ls()
) %dopar% {
  
  s <- as.integer(seeds[i])
  set.seed(s)
  cat(sprintf("[WORKER %d] seed %d\n", i, s))
  
  ## --------------------- worker run dir (avoid collisions) ---------------------  #$$$$$$$$$$$$$
  WORKER_RUN_DIR <- normalizePath(                                                   #$$$$$$$$$$$$$
    file.path(RUN_DIR, sprintf("run_%03d_seed_%s", as.integer(i), as.integer(s))),    #$$$$$$$$$$$$$
    winslash = "/",
    mustWork = FALSE
  )
  dir.create(WORKER_RUN_DIR, recursive = TRUE, showWarnings = FALSE)                  #$$$$$$$$$$$$$
  
  ## --------------------- model construct ---------------------
  N_local <- if (!isTRUE(ML_NN)) {
    input_size + output_size
  } else {
    input_size + sum(hidden_sizes) + output_size
  }
  
  run_model <- DDESONN$new(
    num_networks    = max(1L, as.integer(num_networks)),
    input_size      = ncol(Rdata),
    hidden_sizes    = hidden_sizes,
    output_size     = 1L,
    N               = N_local,
    lambda          = lambda,
    ensemble_number = 0L,
    ensembles       = NULL,
    ML_NN           = ML_NN,
    activation_functions         = activation_functions,
    activation_functions_predict = activation_functions_predict,
    init_method     = init_method,
    custom_scale    = custom_scale
  )
  
  ## --------------------- train ---------------------
  model_results <- run_model$train(
    Rdata                       = Rdata,
    labels                      = labels,
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
    reg_type                    = NULL,
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
    train                       = isTRUE(train),
    grouped_metrics             = grouped_metrics,
    viewTables                  = viewTables,
    verbose                     = verbose
  )
  
  ## --------------------- train metrics (flatten/filter) ---------------------
  pm <- try(model_results$performance_relevance_data$performance_metric, silent = TRUE)
  rm <- try(model_results$performance_relevance_data$relevance_metric,   silent = TRUE)
  
  train_metrics_list <- flatten_and_filter_metrics(pm, rm)
  
  train_row <- data.frame(
    run_index = as.integer(i),
    seed      = as.integer(s),
    split     = "train",
    status    = if (!inherits(pm, "try-error")) "ok" else "fail",
    as.data.frame(train_metrics_list, check.names = TRUE),
    row.names = NULL
  )
  
  ## ============================================================
  ## SECTION: FIX — build metadata env EXACTLY like local         #$$$$$$$$$$$$$
  ## ============================================================
  MODEL_SLOT <- 1L                                                                     #$$$$$$$$$$$$$
  env_name   <- sprintf("Ensemble_Main_0_model_%d_metadata", as.integer(MODEL_SLOT))   #$$$$$$$$$$$$$
  
  slot_obj <- NULL
  if (!is.null(run_model$ensemble) && length(run_model$ensemble) >= as.integer(MODEL_SLOT)) {
    slot_obj <- run_model$ensemble[[as.integer(MODEL_SLOT)]]
  } else {
    slot_obj <- run_model
  }
  
  .safe_listify_record <- function(rec) {
    if (is.null(rec)) return(NULL)
    if (is.list(rec)) return(rec)
    list(rec)
  }
  
  .force_layer_list <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.list(x)) return(x)
    list(x)
  }
  
  W_best <- NULL
  B_best <- NULL
  
  if (!is.null(slot_obj$best_weights_record)) W_best <- slot_obj$best_weights_record
  if (!is.null(slot_obj$best_biases_record))  B_best <- slot_obj$best_biases_record
  
  if (is.null(W_best) && !is.null(slot_obj$weights_record)) W_best <- slot_obj$weights_record
  if (is.null(B_best) && !is.null(slot_obj$b_record))      B_best <- slot_obj$b_record
  if (is.null(B_best) && !is.null(slot_obj$biases_record)) B_best <- slot_obj$biases_record
  
  W_best <- .safe_listify_record(W_best)
  B_best <- .safe_listify_record(B_best)
  
  W_best <- .force_layer_list(W_best)
  B_best <- .force_layer_list(B_best)
  
  predictor_fn_safe <- local({                                                        #$$$$$$$$$$$$$
    W   <- W_best
    B   <- B_best
    AFp <- activation_functions_predict                                                #$$$$$$$$$$$$$
    function(X, ...) slot_obj$predict(
      X,
      weights = W,
      biases  = B,
      activation_functions_predict = AFp,
      ...
    )
  })
  
  md <- list(
    model_serial_num     = sprintf("0.main.%d", as.integer(MODEL_SLOT)),
    predictor            = slot_obj,
    predictor_fn         = predictor_fn_safe,                                          #$$$$$$$$$$$$$
    best_weights_record  = .force_layer_list(W_best),                                  #$$$$$$$$$$$$$
    best_biases_record   = .force_layer_list(B_best),                                  #$$$$$$$$$$$$$
    weights_record       = .force_layer_list(W_best),                                  #$$$$$$$$$$$$$
    biases_record        = .force_layer_list(B_best),                                  #$$$$$$$$$$$$$
    b_record             = .force_layer_list(B_best),                                  #$$$$$$$$$$$$$
    model                = list(
      best_weights_record = .force_layer_list(W_best),
      best_biases_record  = .force_layer_list(B_best)
    ),
    MODEL_SLOT           = as.integer(MODEL_SLOT),
    ML_NN                = isTRUE(ML_NN),
    feature_names        = colnames(X_test),
    X_train              = as.matrix(X_train),
    y_train              = as.matrix(y_train),
    X_validation         = as.matrix(X_validation),
    y_validation         = as.matrix(y_validation),
    X_test               = as.matrix(X_test),
    y_test               = as.matrix(y_test),
    CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
    threshold            = threshold,
    threshold_function   = threshold_function,
    numeric_columns      = numeric_columns
  )
  
  assign(env_name, md, envir = .GlobalEnv)
  
  ## ============================================================
  ## SECTION: FIX — eval-writer files + read-back (TEST)          #$$$$$$$$$$$$$
  ## ============================================================
  agg_pred_file_test_eval    <- file.path(WORKER_RUN_DIR, sprintf("SingleRun_Pretty_Test_Metrics_seed_%s.rds", s)) #$$$$$$$$$$$$$
  agg_metrics_file_test_eval <- file.path(WORKER_RUN_DIR, sprintf("SingleRun_Test_Metrics_seed_%s.rds",         s)) #$$$$$$$$$$$$$
  
  test_eval <- try(
    DDESONN_predict_eval(
      LOAD_FROM_RDS        = FALSE,
      ENV_META_NAME        = env_name,                                                  #$$$$$$$$$$$$$
      INPUT_SPLIT          = "test",
      CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
      RUN_INDEX            = as.integer(i),
      SEED                 = as.integer(s),
      OUTPUT_DIR           = WORKER_RUN_DIR,                                             #$$$$$$$$$$$$$
      OUT_DIR_ASSERT       = WORKER_RUN_DIR,                                             #$$$$$$$$$$$$$
      SAVE_METRICS_RDS     = TRUE,                                                       #$$$$$$$$$$$$$
      METRICS_PREFIX       = "metrics_test",
      AGG_PREDICTIONS_FILE = agg_pred_file_test_eval,                                    #$$$$$$$$$$$$$
      AGG_METRICS_FILE     = agg_metrics_file_test_eval,                                 #$$$$$$$$$$$$$
      MODEL_SLOT           = as.integer(MODEL_SLOT)
    ),
    silent = TRUE
  )
  
  test_df <- NULL
  if (file.exists(agg_metrics_file_test_eval)) {
    test_df <- try(readRDS(agg_metrics_file_test_eval), silent = TRUE)
    if (inherits(test_df, "try-error")) test_df <- NULL
  }
  
  if (!is.null(test_df) && is.data.frame(test_df) && NROW(test_df) >= 1L) {
    last <- test_df[NROW(test_df), , drop = FALSE]
    if (!("run_index" %in% names(last))) last$run_index <- as.integer(i)
    if (!("seed" %in% names(last)))      last$seed      <- as.integer(s)
    if (!("split" %in% names(last)))     last$split     <- "test"
    if (!("status" %in% names(last)))    last$status    <- "ok"
    test_row <- last
  } else {
    test_row <- data.frame(
      run_index = as.integer(i),
      seed      = as.integer(s),
      split     = "test",
      status    = if (inherits(test_eval, "try-error")) "predict_eval_failed" else "no-metrics-written",
      row.names = NULL
    )
  }
  
  list(train_rows = train_row, test_rows = test_row)
}

## ============================================================
## SECTION: 8) Aggregate and save results (master)               #$$$$$$$$$$$$$
## ============================================================
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
