#!/usr/bin/env Rscript

suppressPackageStartupMessages({ library(R6) })

## ============================================================
## SECTION: RUN ROOT (Techila-safe)                              
## ============================================================
source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "activation_functions.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "optimizers.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "update_weights_block.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "update_biases_block.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "performance_relevance_metrics.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "DDESONN.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "utils.R"),
                     winslash = "/", mustWork = TRUE))                               

source(normalizePath(file.path(dirname(sys.frames()[[1]]$ofile), "R", "reports", "evaluate_predictions_report_original.R"),
                     winslash = "/", mustWork = TRUE))

## ============================================================
## SECTION: Hyperparameters
## ============================================================
CLASSIFICATION_MODE <- "binary"
self_org <- FALSE
set.seed(111)
x <- 1
train <- TRUE
test <- TRUE
init_method <- "he"
optimizer <- "adagrad"
lookahead_step <- 5
batch_normalize_data <- TRUE
shuffle_bn <- FALSE
gamma_bn <- .6
beta_bn <- .6
epsilon_bn <- 1e-6
momentum_bn <- 0.9
is_training_bn <- TRUE
beta1 <- .9
beta2 <- 0.8

if (CLASSIFICATION_MODE == "binary") {
  init_method <- "he"
  optimizer <- "adagrad"
  lr <- .125
  lambda <- 0.00028
  num_epochs <- 3
  custom_scale <- 1.04349
} else if (CLASSIFICATION_MODE == "multiclass") {
  init_method <- "he"
  optimizer <- "adagrad"
  lr <- 0.22
  lambda <- 1e-4
  custom_scale <- 1.0
  num_epochs <- 1
} else if (CLASSIFICATION_MODE == "regression") {
  init_method <- "he"
  optimizer <- "adagrad"
  lr <- .121
  lambda <- 0.0003
  custom_scale <- .05
  num_epochs <- 130
  custom_scale <- 0.05
}

lr_decay_rate  <- 0.5
lr_decay_epoch <- 20
lr_min <- 1e-5
validation_metrics <- TRUE
best_weights_on_latest_weights_off <- TRUE

ML_NN <- TRUE
grouped_metrics <- FALSE
update_weights <- TRUE
update_biases <- TRUE

hidden_sizes <- c(64, 32)

if (CLASSIFICATION_MODE == "binary") {
  activation_functions <- list(relu, relu, sigmoid)
} else if (CLASSIFICATION_MODE == "multiclass") {
  activation_functions <- list(relu, relu, softmax)
} else if (CLASSIFICATION_MODE == "regression") {
  activation_functions <- list(relu, relu, identity)
}

activation_functions_predict <- activation_functions
epsilon <- 1e-7

if (CLASSIFICATION_MODE %in% c("binary", "multiclass")) {
  loss_type <- "CrossEntropy"
} else {
  loss_type <- "MSE"
}

dropout_rates <- list(0.10)
threshold_function <- tune_threshold_accuracy
threshold <- .5

do_ensemble         <- FALSE
num_networks        <- 1L
num_temp_iterations <- 0L

ensemble_number <- 0L
ensembles <- NULL

reg_type <- NULL
sample_weights <- NULL
preprocessScaledData <- FALSE

viewTables <- FALSE
verbose <- TRUE

## ============================================================
## SECTION: SONN / DDESONN PLOTS (flags)                         
## ============================================================
accuracy_plot     <- FALSE    
saturation_plot   <- FALSE    
max_weight_plot   <- FALSE    

performance_high_mean_plots <- FALSE  
performance_low_mean_plots  <- FALSE  
relevance_high_mean_plots   <- FALSE  
relevance_low_mean_plots    <- FALSE  

viewAllPlots <- FALSE                 

## ============================================================
## SECTION: RUN_DIR + Output files                               
## ============================================================
# FIX: never write artifacts next to script; ALWAYS under /R/reports with stamp          
SCRIPT_DIR  <- normalizePath(dirname(sys.frames()[[1]]$ofile), winslash = "/", mustWork = TRUE)   
REPORTS_DIR <- normalizePath(file.path(SCRIPT_DIR, "R", "reports"), winslash = "/", mustWork = FALSE) 
dir.create(REPORTS_DIR, recursive = TRUE, showWarnings = FALSE)                                   

ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")                                                   
RUN_DIR  <- normalizePath(                                                                        
  file.path(REPORTS_DIR, sprintf("Local_MVP_run_artifacts_%s", ts_stamp)),
  winslash = "/",
  mustWork = FALSE
)                                                                                                 
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)                                       

# FIX: if legacy ./run_artifacts exists next to script, migrate it into reports + delete it       
legacy_artifacts <- normalizePath(file.path(SCRIPT_DIR, "run_artifacts"), winslash="/", mustWork=FALSE) 
if (dir.exists(legacy_artifacts)) {                                                               
  legacy_files <- try(list.files(legacy_artifacts, all.files = TRUE, full.names = TRUE, no.. = TRUE), silent = TRUE) 
  has_any <- (!inherits(legacy_files, "try-error") && length(legacy_files) > 0L)                  
  
  if (isTRUE(has_any)) {                                                                          
    moved <- try(file.rename(legacy_artifacts, RUN_DIR), silent = TRUE)                            
    if (inherits(moved, "try-error") || !isTRUE(moved)) {                                         
      file.copy(from = legacy_files, to = RUN_DIR, recursive = TRUE, overwrite = TRUE)            
      unlink(legacy_artifacts, recursive = TRUE, force = TRUE)                                    
    }                                                                                             
  } else {                                                                                        
    unlink(legacy_artifacts, recursive = TRUE, force = TRUE)                                      
  }                                                                                               
}                                                                                                 

seed_value <- 111L                                                                                 
agg_metrics_file_train <- file.path(RUN_DIR, sprintf("agg_metrics_train_seed_%s.rds", seed_value)) 
agg_metrics_file_test  <- file.path(RUN_DIR, sprintf("agg_metrics_test_seed_%s.rds",  seed_value)) 

## ============================================================
## SECTION: FIX — eval-writer AGG files (do NOT collide w outputs) 
## ============================================================
# These are ONLY for DDESONN_predict_eval() to append into, mirroring single-run behavior.  
# They are separate from agg_metrics_file_* which you write at the end of this script.     
agg_pred_file_test_eval    <- file.path(RUN_DIR, sprintf("SingleRun_Pretty_Test_Metrics_seed_%s.rds", seed_value))  
agg_metrics_file_test_eval <- file.path(RUN_DIR, sprintf("SingleRun_Test_Metrics_seed_%s.rds",         seed_value))  

## ============================================================
## SECTION: FIX — ensure MODEL_SLOT always exists                
## ============================================================
if (!exists("MODEL_SLOT", envir = .GlobalEnv, inherits = TRUE) || is.null(get("MODEL_SLOT", envir = .GlobalEnv))) { 
  MODEL_SLOT <- 1L                                                                                 
} else {                                                                                            
  MODEL_SLOT <- as.integer(get("MODEL_SLOT", envir = .GlobalEnv))                                   
}                                                                                                   
assign("MODEL_SLOT", MODEL_SLOT, envir = .GlobalEnv)                                                 

## ============================================================
## SECTION: Load the dataset (Techila-safe path)                 
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

if (CLASSIFICATION_MODE == "binary") {
  data <- read.csv(csv_path, stringsAsFactors = FALSE)                                            
  dependent_variable <- "DEATH_EVENT"
} else {
  stop("This single runner is locked to binary HF for exact parity.", call. = FALSE)              
}

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

input_columns <- setdiff(colnames(data), dependent_variable)
Rdata  <- data[, input_columns, drop = FALSE]
labels <- data[, dependent_variable, drop = FALSE]

input_size  <- ncol(Rdata)
output_size <- 1L

reduce_data <- TRUE

X <- data
if (requireNamespace("dplyr", quietly = TRUE)) {
  X <- dplyr::select(X, -dplyr::all_of(dependent_variable))
  y <- dplyr::select(data, dplyr::all_of(dependent_variable))
} else {
  X <- X[, setdiff(names(X), dependent_variable), drop = FALSE]
  y <- data[, dependent_variable, drop = FALSE]
}

colname_y <- colnames(y)

if (CLASSIFICATION_MODE == "binary") {
  numeric_columns <- c('age','creatinine_phosphokinase','ejection_fraction',
                       'platelets','serum_creatinine','serum_sodium','time')
}

USE_TIME_SPLIT <- TRUE

if (USE_TIME_SPLIT) {
  stopifnot(nrow(X) == nrow(y))
  total_num_samples <- nrow(X)
  
  p_train <- 0.70
  p_val   <- 0.15
  
  num_training_samples   <- max(1L, floor(p_train * total_num_samples))
  num_validation_samples <- max(1L, floor(p_val   * total_num_samples))
  num_test_samples       <- max(0L, total_num_samples - num_training_samples - num_validation_samples)
  
  train_indices      <- seq_len(num_training_samples)
  validation_indices <- if (num_validation_samples > 0L)
    seq(from = max(train_indices) + 1L,
        length.out = num_validation_samples)
  else integer()
  test_indices       <- if (num_test_samples > 0L)
    seq(from = max(c(train_indices, validation_indices)) + 1L,
        length.out = num_test_samples)
  else integer()
  
  X_train      <- X[train_indices,      , drop = FALSE]; y_train      <- y[train_indices,      , drop = FALSE]
  X_validation <- X[validation_indices, , drop = FALSE]; y_validation <- y[validation_indices, , drop = FALSE]
  X_test       <- X[test_indices,       , drop = FALSE]; y_test       <- y[test_indices,       , drop = FALSE]
  
  cat(sprintf("[SPLIT chrono] train=%d val=%d test=%d\n",
              nrow(X_train), nrow(X_validation), nrow(X_test)))
} else {
  total_num_samples <- nrow(X)
  desired_val  <- 800L
  desired_test <- 800L
  
  num_validation_samples <- min(desired_val,  floor(total_num_samples / 3))
  num_test_samples       <- min(desired_test, floor((total_num_samples - num_validation_samples) / 2))
  num_training_samples   <- total_num_samples - num_validation_samples - num_test_samples
  
  indices <- sample.int(total_num_samples)
  train_indices      <- indices[seq_len(num_training_samples)]
  validation_indices <- indices[seq(from = num_training_samples + 1L,
                                    length.out = num_validation_samples)]
  test_indices       <- indices[seq(from = num_training_samples + num_validation_samples + 1L,
                                    length.out = num_test_samples)]
  
  X_train      <- X[train_indices,      , drop = FALSE]; y_train      <- y[train_indices,      , drop = FALSE]
  X_validation <- X[validation_indices, , drop = FALSE]; y_validation <- y[validation_indices, , drop = FALSE]
  X_test       <- X[test_indices,       , drop = FALSE]; y_test       <- y[test_indices,       , drop = FALSE]
  
  cat(sprintf("[SPLIT random] train=%d val=%d test=%d\n",
              nrow(X_train), nrow(X_validation), nrow(X_test)))
}

if (CLASSIFICATION_MODE == "binary") {
  
  X_train_scaled <- scale(X_train)
  center <- attr(X_train_scaled, "scaled:center")
  scale_ <- attr(X_train_scaled, "scaled:scale")
  
  X_validation_scaled <- scale(X_validation, center = center, scale = scale_)
  X_test_scaled       <- scale(X_test,       center = center, scale = scale_)
  
  max_val <- suppressWarnings(max(abs(X_train_scaled)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1
  
  X_train_scaled      <- X_train_scaled      / max_val
  X_validation_scaled <- X_validation_scaled / max_val
  X_test_scaled       <- X_test_scaled       / max_val
  
  scaledData <- TRUE
  
  if (isTRUE(scaledData)) {
    X <- as.matrix(X_train_scaled)
    y <- as.matrix(y_train)
    
    X_validation <- as.matrix(X_validation_scaled)
    y_validation <- as.matrix(y_validation)
    
    X_test <- as.matrix(X_test_scaled)
    y_test <- as.matrix(y_test)
  } else {
    X <- as.matrix(X_train)
    y <- as.matrix(y_train)
    
    X_validation <- as.matrix(X_validation)
    y_validation <- as.matrix(y_validation)
    
    X_test <- as.matrix(X_test)
    y_test <- as.matrix(y_test)
  }
  
  colnames(y) <- colname_y
}

Rdata  <- X
labels <- y

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
    is_details        <- grepl("(^|\\.)details(\\.|$)", nms)
    
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

cat("[LOCAL] constructing model...\n")

N <- if (!ML_NN) {
  input_size + output_size
} else {
  input_size + sum(hidden_sizes) + output_size
}

run_model <- DDESONN$new(
  num_networks    = max(1L, as.integer(num_networks)),
  input_size      = ncol(Rdata),
  hidden_sizes    = hidden_sizes,
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

if (length(run_model$ensemble)) {                                                     
  for (m in seq_along(run_model$ensemble)) {                                          
    run_model$ensemble[[m]]$PerEpochViewPlotsConfig <- list(                          
      accuracy_plot   = isTRUE(accuracy_plot),                                       
      saturation_plot = isTRUE(saturation_plot),                                     
      max_weight_plot = isTRUE(max_weight_plot),                                     
      viewAllPlots    = isTRUE(viewAllPlots),                                        
      verbose         = isTRUE(verbose)                                              
    )                                                                                
    run_model$ensemble[[m]]$FinalUpdatePerformanceandRelevanceViewPlotsConfig <- list( 
      performance_high_mean_plots = isTRUE(performance_high_mean_plots),             
      performance_low_mean_plots  = isTRUE(performance_low_mean_plots),              
      relevance_high_mean_plots   = isTRUE(relevance_high_mean_plots),               
      relevance_low_mean_plots    = isTRUE(relevance_low_mean_plots),                
      viewAllPlots                = isTRUE(viewAllPlots),                            
      verbose                     = isTRUE(verbose)                                  
    )                                                                                
  }                                                                                  
}                                                                                    

cat("[LOCAL] training...\n")

model_results <- run_model$train(
  Rdata                       = Rdata,
  labels                      = labels,
  X_train                     = X,
  y_train                     = y,
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
  train                       = train,
  grouped_metrics             = grouped_metrics,
  viewTables                  = viewTables,
  verbose                     = verbose
)

cat("[LOCAL] train() finished.\n")

pm <- try(model_results$performance_relevance_data$performance_metric, silent = TRUE)
rm <- try(model_results$performance_relevance_data$relevance_metric,   silent = TRUE)

train_metrics_list <- flatten_and_filter_metrics(pm, rm)

train_row <- data.frame(
  seed      = seed_value,
  split     = "train",
  status    = if (!inherits(pm, "try-error")) "ok" else "fail",
  as.data.frame(train_metrics_list, check.names = TRUE),
  row.names = NULL
)

## ============================================================
## SECTION: METADATA BUILD (report-safe)                         
## ============================================================
env_name   <- sprintf("Ensemble_Main_0_model_%d_metadata", as.integer(MODEL_SLOT))      

slot_obj <- NULL                                                                       
if (!is.null(run_model$ensemble) && length(run_model$ensemble) >= as.integer(MODEL_SLOT)) { 
  slot_obj <- run_model$ensemble[[as.integer(MODEL_SLOT)]]                              
} else {                                                                               
  slot_obj <- run_model                                                                
}                                                                                      

## ============================================================
## SECTION: FIX — force list-shaped records everywhere eval uses 
## ============================================================
.safe_listify_record <- function(rec) {                                                
  if (is.null(rec)) return(NULL)                                                       
  if (is.list(rec)) return(rec)                                                        
  list(rec)                                                                            
}                                                                                      

# HARD FORCE (assignment-point coercion guard)                                         
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

# DOUBLE HARDEN: force list-shape right before md assignment                             
W_best <- .force_layer_list(W_best)                                                    
B_best <- .force_layer_list(B_best)                                                    

## ============================================================
## SECTION: BUILD md (predict signature safe)                    
## ============================================================

# FIX: SONN$predict requires weights/biases -> always pass them
# FIX: also force activation_functions_predict so eval does NOT fall back to identity(NULL)  
predictor_fn_safe <- local({                                                           
  W   <- W_best                                                                        
  B   <- B_best                                                                        
  AFp <- activation_functions_predict                                                   
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
  
  predictor_fn         = predictor_fn_safe,                                              FIX
  
  best_weights_record  = .force_layer_list(W_best),                                      FIX
  best_biases_record   = .force_layer_list(B_best),                                      FIX
  weights_record       = .force_layer_list(W_best),                                      FIX
  biases_record        = .force_layer_list(B_best),                                      FIX
  b_record             = .force_layer_list(B_best),                                      FIX
  
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

# IMPORTANT: do NOT write back into predictor slots (R6 may coerce to matrix)            

assign(env_name, md, envir = .GlobalEnv)                                                

X_test <<- md$X_test                                                                    
y_test <<- md$y_test                                                                    

## ============================================================
## SECTION: DEBUG PROOF (must show TRUE)                         
## ============================================================
cat("[TEST DEBUG] meta$best_weights_record is.list: ", is.list(get(env_name, .GlobalEnv)$best_weights_record), "\n", sep="") 
cat("[TEST DEBUG] meta$model$best_weights_record is.list: ", is.list(get(env_name, .GlobalEnv)$model$best_weights_record), "\n", sep="") 
cat("[TEST DEBUG] predictor_fn exists: ", is.function(get(env_name, .GlobalEnv)$predictor_fn), "\n", sep="") 

## ============================================================
## SECTION: TEST DEBUG (named "test" so you can spot it)         
## ============================================================
cat("\n[TEST DEBUG] RUN_DIR: ", RUN_DIR, "\n", sep = "")                               
cat("[TEST DEBUG] ENV_META_NAME: ", env_name, "\n", sep="")                            
cat("[TEST DEBUG] exists(meta): ", exists(env_name, envir = .GlobalEnv), "\n", sep="") 
cat("[TEST DEBUG] meta keys:\n")                                                      
print(names(get(env_name, envir = .GlobalEnv)))                                       
cat("[TEST DEBUG] meta$model class: ", paste(class(get(env_name, envir=.GlobalEnv)$model), collapse=", "), "\n", sep="") 
cat("[TEST DEBUG] dims X_test / y_test:\n")                                           
print(dim(get(env_name, envir = .GlobalEnv)$X_test))                                  
print(dim(get(env_name, envir = .GlobalEnv)$y_test))                                  

cat("[LOCAL] running predict_eval...\n")

## ============================================================
## SECTION: FIX — TEST uses eval-writer + reads written metrics   
## ============================================================
test_eval <- try(
  DDESONN_predict_eval(
    LOAD_FROM_RDS        = FALSE,
    ENV_META_NAME        = env_name,                                                   
    INPUT_SPLIT          = "test",
    CLASSIFICATION_MODE  = CLASSIFICATION_MODE,
    RUN_INDEX            = 1L,
    SEED                 = seed_value,
    OUTPUT_DIR           = RUN_DIR,
    OUT_DIR_ASSERT       = RUN_DIR,
    SAVE_METRICS_RDS     = TRUE,                                                       
    METRICS_PREFIX       = "metrics_test",
    AGG_PREDICTIONS_FILE = agg_pred_file_test_eval,                                     
    AGG_METRICS_FILE     = agg_metrics_file_test_eval,                                  
    MODEL_SLOT           = as.integer(MODEL_SLOT)
  ),
  silent = TRUE
)

if (inherits(test_eval, "try-error")) {                                               
  cat("\n[TEST DEBUG] DDESONN_predict_eval FAILED\n")                                  
  print(attr(test_eval, "condition"))                                                  
  cat("\n")                                                                            
} else {                                                                              
  cat("\n[TEST DEBUG] DDESONN_predict_eval OK\n")                                      
  cat("[TEST DEBUG] names(test_eval):\n")                                              
  print(names(test_eval))                                                              
  cat("[TEST DEBUG] (note) metrics are read from AGG_METRICS_FILE, not return object.\n")  
  cat("\n")                                                                            
}                                                                                     

## ============================================================
## SECTION: TEST METRICS — read what predict_eval wrote          
## ============================================================
test_df <- NULL                                                                       
if (file.exists(agg_metrics_file_test_eval)) {                                         
  test_df <- try(readRDS(agg_metrics_file_test_eval), silent = TRUE)                   
  if (inherits(test_df, "try-error")) test_df <- NULL                                  
}                                                                                     

if (!is.null(test_df) && is.data.frame(test_df) && NROW(test_df) >= 1L) {             
  last <- test_df[NROW(test_df), , drop = FALSE]                                      
  if (!("seed" %in% names(last))) last$seed <- seed_value                             
  if (!("split" %in% names(last))) last$split <- "test"                               
  if (!("status" %in% names(last))) last$status <- "ok"                               
  test_row <- last                                                                    
} else {                                                                              
  test_row <- data.frame(                                                             
    seed      = seed_value,
    split     = "test",
    status    = "no-metrics-written",
    row.names = NULL
  )
}                                                                                     

if (!dir.exists(RUN_DIR)) stop("RUN_DIR disappeared before saveRDS(): ", RUN_DIR)
saveRDS(train_row, agg_metrics_file_train)
saveRDS(test_row,  agg_metrics_file_test)

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
