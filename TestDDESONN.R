# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Ensemble Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# 
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

source("DDESONN.R")
source("utils/utils.R")

# # Define parameters
## =========================
## Classification mode
## =========================
# CLASSIFICATION_MODE <- "multiclass"   # "binary" | "multiclass" | "regression"
CLASSIFICATION_MODE <- "binary"
# CLASSIFICATION_MODE <- "regression"
self_org <- FALSE
set.seed(111)
#number of seeds;if doing seed loop
x <- 1
train <- TRUE
test <- TRUE
init_method <- "he" #variance_scaling" #glorot_uniform" #"orthogonal" #"orthogonal" #lecun" #xavier"
optimizer <- "adagrad" #"lamb" #ftrl #nag #"sgd" #NULL "rmsprop" #adam #sgd_momentum #lookahead #adagrad
lookahead_step <- 5
batch_normalize_data <- TRUE
shuffle_bn <- FALSE
gamma_bn <- .6
beta_bn <- .6
epsilon_bn <- 1e-6  # Increase for numerical stability
momentum_bn <- 0.9 # Improved convergence
is_training_bn <- TRUE
beta1 <- .9 # Standard Adam value
beta2 <- 0.8 # Slightly lower for better adaptability

lr <- .125
lr_decay_rate  <- 0.5
lr_decay_epoch <- 20
lr_min <- 1e-5
lambda <- 0.00028
# lambda <- 0.00013
num_epochs <- 3
validation_metrics <- TRUE

custom_scale <- 1.04349


ML_NN <- TRUE

grouped_metrics <- FALSE

update_weights <- TRUE
update_biases <- TRUE

# hidden_sizes <- NULL
hidden_sizes <- c(64, 32)

# Activation functions applied in forward pass during prediction | predict(). # hidden layers + output layer
if (CLASSIFICATION_MODE == "binary") {
  activation_functions <- list(relu, relu, sigmoid)
  
} else if (CLASSIFICATION_MODE == "multiclass") {
  activation_functions <- list(relu, relu, softmax)
  
} else if (CLASSIFICATION_MODE == "regression") {
  activation_functions <- list(relu, relu, identity)
}


# Activation functions applied in forward pass during training | learn() # You can keep them the same as predict() or customize, e.g. list(relu, selu, sigmoid) 
activation_functions_predict <- activation_functions
epsilon <- 1e-7

if (CLASSIFICATION_MODE %in% c("binary", "multiclass")) {
  loss_type <- "CrossEntropy" #NULL #'MSE', 'MAE', 'CrossEntropy', or 'CategoricalCrossEntropy'
} else{
  loss_type <- "MSE"
}
ropout_rates <- list(0.10, 0.00) # NULL for output layer
dropout_rates <- list(0.10) # NULL for output layer

threshold_function <- tune_threshold_accuracy
threshold <- .5  # Classification threshold (not directly used in Random Forest)



num_layers <- length(hidden_sizes) + 1
output_size <- 1  # For binary classification

## Kaggle Data
runData <- FALSE
if(runData){
  library(reticulate)
  
  # Ensure we're using reticulate's default Python env (created above)
  # If you prefer, you can create/use your own venv and call use_virtualenv() instead.
  
  # 1) Install kagglehub into the active Python used by reticulate
  py_install("kagglehub", pip = TRUE)
  
  # 2) Download with kagglehub from Python, returning the path back to R
  res <- py_run_string("
import kagglehub
p = kagglehub.dataset_download('matiflatif/walmart-complete-stocks-dataweekly-updated')
print('Downloaded to:', p)
path = p
")
  src_path <- res$path
  
  # 3) Copy the downloaded dataset into your target folder
  dst_path <- "C:/Users/wfky1/Desktop/DDESONN/data"
  dir.create(dst_path, showWarnings = FALSE, recursive = TRUE)
  # Copy everything (preserves directories); overwrite existing
  file.copy(from = list.files(src_path, full.names = TRUE),
            to   = dst_path,
            recursive = TRUE, overwrite = TRUE)
  
  message("Copied dataset from:\n ", src_path, "\n→\n ", dst_path)
}


## -------------------------
## Load the dataset
## -------------------------
if (CLASSIFICATION_MODE == "binary") {
  data <- read.csv("data/heart_failure_clinical_records.csv")
  dependent_variable <- "DEATH_EVENT"
} else if (CLASSIFICATION_MODE == "multiclass") {
  data <- read.csv("data/train_multiclass_customer_segmentation.csv")
  dependent_variable <- "Segmentation"
} else if (CLASSIFICATION_MODE == "regression") {
  data <- read.csv("data/WMT_1970-10-01_2025-03-15.csv")
  # --- NEW: enforce time order + create future target (close[t+1]) ---
  stopifnot("date" %in% names(data))
  data <- data %>% arrange(date)
  data <- data %>% mutate(future_close = dplyr::lead(close, 1L))
  data <- data %>% filter(!is.na(future_close))
  
  # --- NEW: set dependent variable to future_close ---
  dependent_variable <- "future_close"
} else {
  stop("CLASSIFICATION_MODE must be 'binary' or 'multiclass'")
}


## Quick NA check
na_count <- sum(is.na(data))
message(sprintf("[split] NA count: %s", na_count))


# Assuming there are no missing values, or handle them if they exist
# Convert categorical variables to factors if any
data <- data %>% mutate(across(where(is.character), as.factor))



## -------------------------
## Features/labels (full)
## -------------------------
input_columns <- setdiff(colnames(data), dependent_variable)
Rdata  <- data[, input_columns, drop = FALSE]
labels <- data[, dependent_variable, drop = FALSE]  # keep as data.frame (1 col)

if (CLASSIFICATION_MODE %in% c("binary", "regression")) {
  input_size  <- ncol(Rdata)
  output_size <- 1L
}


reduce_data <- TRUE

if (CLASSIFICATION_MODE == "regression" && reduce_data) {
  stopifnot(is.character(dependent_variable), length(dependent_variable) == 1)
  stopifnot(dependent_variable %in% names(data))
  
  # Keep only raw OHLCV + date, plus the dependent variable (future_close)
  base_keep <- c("date", "open", "high", "low", "close", "volume")
  keep_cols <- unique(c(base_keep, dependent_variable))
  
  data_reduced <- data %>%
    dplyr::select(dplyr::any_of(keep_cols))
  
  # Split into X (features at time t) and y (close at t+1)
  X <- data_reduced %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y <- data_reduced[[dependent_variable]]
  
  # Safety checks (no leakage)
  if ("adj_close" %in% names(X)) {
    stop("Leak detected: 'adj_close' showed up in X unexpectedly.")
  }
  if (!is.null(y) && anyNA(y)) {
    warning("Target 'y' contains NA values.")
  }
  
  # Optional: make 'date' numeric if present
  if ("date" %in% names(X)) {
    d <- X[["date"]]
    if (inherits(d, "POSIXt")) {
      X[["date"]] <- as.numeric(as.Date(d))
    } else if (inherits(d, "Date")) {
      X[["date"]] <- as.numeric(d)
    } else {
      suppressWarnings({ parsed <- as.Date(d) })
      if (!all(is.na(parsed))) {
        X[["date"]] <- as.numeric(parsed)
      } # else leave as-is
    }
  }
}

# Keep your existing fallback exactly as written
if (CLASSIFICATION_MODE == "regression" && reduce_data) {
  X <- data_reduced %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y <- data_reduced %>% dplyr::select(dplyr::all_of(dependent_variable))
} else {
  X <- data %>% dplyr::select(-dplyr::all_of(dependent_variable))
  y <- data %>% dplyr::select(dplyr::all_of(dependent_variable))
}

colname_y <- colnames(y)

# ensure downstream uses lagged setup (place right after your X/y fallback)
# Rdata  <- X
# labels <- y

if (CLASSIFICATION_MODE == "binary") {
  numeric_columns <- c('age','creatinine_phosphokinase','ejection_fraction',
                       'platelets','serum_creatinine','serum_sodium','time')
} else if (CLASSIFICATION_MODE == "multiclass") {
  numeric_columns <- c("Age","Work_Experience","Family_Size")
} else if (CLASSIFICATION_MODE == "regression") {
  # Predicting future_close (t+1) from today's OHLCV (t)
  numeric_columns <- c("date","open","high","low","close","volume")
  
}





## -------------------------
## Split selector
## -------------------------
USE_TIME_SPLIT <- TRUE  # toggle here: TRUE=new chrono split, FALSE=old random split

if (USE_TIME_SPLIT) {
  ## -------------------------
  ## Time-ordered split
  ## -------------------------
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
  ## -------------------------
  ## Alternative random split (⚠ may cause data leakage)
  ## -------------------------
  
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



## IMPORTANT:
## When you train, pass X_train / y_train — NOT the full Rdata/labels.
## Likewise, for validation, forward X_validation and compare with y_validation.


# ------------------------------------------------------------
# BINARY PATH (untouched)  vs  MULTICLASS PATH (make numeric + labels)
# ------------------------------------------------------------
if (CLASSIFICATION_MODE == "binary") {
  
  # $$$$$$$$$$$$$ Feature scaling without leakage (standardization first)
  X_train_scaled <- scale(X_train)
  center <- attr(X_train_scaled, "scaled:center")
  scale_ <- attr(X_train_scaled, "scaled:scale")
  
  X_validation_scaled <- scale(X_validation, center = center, scale = scale_)
  X_test_scaled       <- scale(X_test,       center = center, scale = scale_)
  
  # $$$$$$$$$$$$$ Further rescale to prevent exploding activations (keep parity)
  max_val <- suppressWarnings(max(abs(X_train_scaled)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1
  
  X_train_scaled      <- X_train_scaled      / max_val
  X_validation_scaled <- X_validation_scaled / max_val
  X_test_scaled       <- X_test_scaled       / max_val
  
  # ==============================================================
  # Choose whether to use scaled or raw data for NN training
  # ==============================================================
  scaledData <- TRUE   # <<<<<< set to FALSE to use raw data
  
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
  
  # ----- diagnostics (binary has no one-hot by design) -----
  cat("=== Unscaled Rdata summary (X_train) ===\n")
  print(summary(as.vector(as.matrix(X_train))))
  cat("First 5 rows of unscaled X_train:\n")
  print(as.matrix(X_train)[1:5, 1:min(5, ncol(X_train)), drop = FALSE])
  
  cat("=== Scaled Rdata summary (X_train_scaled) ===\n")
  print(summary(as.vector(as.matrix(X_train_scaled))))
  cat("First 5 rows of scaled X_train_scaled:\n")
  print(as.matrix(X_train_scaled)[1:5, 1:min(5, ncol(X_train_scaled)), drop = FALSE])
  
  cat("Dimensions of scaled sets:\n")
  cat("Training:",   dim(X), "\n")
  cat("Validation:", dim(X_validation), "\n")
  cat("Test:",       dim(X_test), "\n")
  
  cat("Any NAs in scaled sets:\n")
  cat("Training:",   anyNA(X), "\n")
  cat("Validation:", anyNA(X_validation), "\n")
  cat("Test:",       anyNA(X_test), "\n")
  
} else if (CLASSIFICATION_MODE == "multiclass") {
  
  cat("\n==================== [MC] START ====================\n")
  
  # ---------- A) Row-ID setup so we can align y to X ----------
  # Ensure row names on split frames reflect original row indices
  if (is.null(rownames(X_train)))      rownames(X_train)      <- as.character(train_indices)
  if (is.null(rownames(X_validation))) rownames(X_validation) <- as.character(validation_indices)
  if (is.null(rownames(X_test)))       rownames(X_test)       <- as.character(test_indices)
  
  # ---------- A0) Numeric imputation with TRAIN medians (prevents NA drops downstream) ----------
  impute_with_train_median <- function(df_train, df_other) {
    num_cols <- names(df_train)[vapply(df_train, is.numeric, TRUE)]
    for (nm in num_cols) {
      med <- suppressWarnings(median(df_train[[nm]], na.rm = TRUE))
      if (!is.finite(med) || is.na(med)) med <- 0
      if (nm %in% names(df_train))   df_train[[nm]][is.na(df_train[[nm]])] <- med
      if (nm %in% names(df_other))   df_other[[nm]][is.na(df_other[[nm]])] <- med
    }
    list(train = df_train, other = df_other)
  }
  tmp <- impute_with_train_median(X_train, X_validation); X_train <- tmp$train; X_validation <- tmp$other
  tmp <- impute_with_train_median(X_train, X_test);       X_test  <- tmp$other
  
  # Quick predictor type scan
  pred_types <- vapply(X_train, function(col) {
    if (is.numeric(col)) "numeric"
    else if (is.factor(col)) "factor"
    else class(col)[1]
  }, character(1))
  n_num <- sum(pred_types == "numeric")
  n_fac <- sum(pred_types == "factor")
  cat("[mc] predictors summary: numeric =", n_num, " | factor =", n_fac, " | total =", length(pred_types), "\n")
  if (n_fac > 0) {
    cat("[mc] factor columns:\n"); print(names(which(pred_types == "factor")))
  }
  
  all_numeric <- all(pred_types == "numeric")
  cat("[mc] all_numeric predictors? ->", all_numeric, "\n")
  
  # ---------- B) Build X (features) ----------
  if (all_numeric) {
    cat("[mc] PATH A: numeric-only (no one-hot for X)\n")
    
    X_train_scaled <- scale(as.data.frame(X_train))
    center <- attr(X_train_scaled, "scaled:center")
    scale_ <- attr(X_train_scaled, "scaled:scale"); scale_[!is.finite(scale_) | scale_ == 0] <- 1
    
    X_validation_scaled <- sweep(sweep(as.matrix(as.data.frame(X_validation)), 2, center, "-"), 2, scale_, "/")
    X_test_scaled       <- sweep(sweep(as.matrix(as.data.frame(X_test)),       2, center, "-"), 2, scale_, "/")
    
    X            <- as.matrix(X_train_scaled)
    X_validation <- as.matrix(X_validation_scaled)
    X_test       <- as.matrix(X_test_scaled)
    
  } else {
    cat("[mc] PATH B: one-hot features with model.matrix() + consistent TRAIN terms + row alignment\n")
    
    # Make factor NAs explicit so rows are preserved as levels
    X_train_f      <- dplyr::mutate(X_train,      dplyr::across(where(is.factor), ~forcats::fct_explicit_na(., "(Missing)")))
    X_validation_f <- dplyr::mutate(X_validation, dplyr::across(where(is.factor), ~forcats::fct_explicit_na(., "(Missing)")))
    X_test_f       <- dplyr::mutate(X_test,       dplyr::across(where(is.factor), ~forcats::fct_explicit_na(., "(Missing)")))
    
    # Build design on TRAIN ONLY to lock columns
    mm_terms <- terms(~ . - 1, data = X_train_f)
    X_train_mm      <- model.matrix(mm_terms, data = X_train_f)
    X_validation_mm <- model.matrix(mm_terms, data = X_validation_f)
    X_test_mm       <- model.matrix(mm_terms, data = X_test_f)
    
    cat("[mc] dim(X_train_mm)=", paste(dim(X_train_mm), collapse="×"),
        " | dim(X_val_mm)=", paste(dim(X_validation_mm), collapse="×"),
        " | dim(X_test_mm)=", paste(dim(X_test_mm), collapse="×"), "\n")
    
    # Scale with train stats
    X_train_scaled <- scale(X_train_mm)
    center <- attr(X_train_scaled, "scaled:center")
    scale_ <- attr(X_train_scaled, "scaled:scale"); scale_[!is.finite(scale_) | scale_ == 0] <- 1
    
    X_validation_scaled <- sweep(sweep(X_validation_mm, 2, center, "-"), 2, scale_, "/")
    X_test_scaled       <- sweep(sweep(X_test_mm,       2, center, "-"), 2, scale_, "/")
    
    X            <- as.matrix(X_train_scaled)
    X_validation <- as.matrix(X_validation_scaled)
    X_test       <- as.matrix(X_test_scaled)
  }
  
  # Stabilize magnitude (parity with your binary path)
  max_val <- suppressWarnings(max(abs(X)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1
  X            <- X            / max_val
  X_validation <- X_validation / max_val
  X_test       <- X_test       / max_val
  
  cat("[mc] dim(X) train/val/test: ",
      paste(dim(X), collapse="×"), " / ",
      paste(dim(X_validation), collapse="×"), " / ",
      paste(dim(X_test), collapse="×"), "\n")
  
  # ---------- C) Align y to X rows (train/val/test) ----------
  pull_y_vec <- function(obj) {
    if (is.matrix(obj)) as.vector(obj[, 1, drop = TRUE])
    else if (is.data.frame(obj)) as.vector(obj[[1]])
    else as.vector(obj)
  }
  align_y_to_X <- function(X_mat, y_df, idx_vec) {
    kept_rn <- rownames(X_mat)
    if (is.null(kept_rn)) kept_rn <- as.character(idx_vec[seq_len(nrow(X_mat))])
    pos <- match(kept_rn, as.character(idx_vec))
    if (anyNA(pos)) stop("[mc][align] Could not map X rownames to original indices.")
    pull_y_vec(y_df)[pos]
  }
  
  y_vec_tr <- align_y_to_X(X,            y_train,      train_indices)
  y_vec_va <- align_y_to_X(X_validation, y_validation, validation_indices)
  y_vec_te <- align_y_to_X(X_test,       y_test,       test_indices)
  
  cat("[mc] lens y_vec (aligned): train/val/test = ",
      length(y_vec_tr), "/", length(y_vec_va), "/", length(y_vec_te), "\n")
  
  # ---------- D) One-hot labels (shared levels from full dataset) ----------
  y_full_vec  <- pull_y_vec(y)
  levels_y    <- levels(factor(y_full_vec))
  output_size <- length(levels_y)
  cat("[mc] class levels (", output_size, "): ", paste(levels_y, collapse=", "), "\n", sep="")
  
  to_one_hot <- function(v, lvls) {
    idx <- match(v, lvls)
    m <- matrix(0L, nrow = length(v), ncol = length(lvls))
    m[cbind(seq_along(idx), idx)] <- 1L
    colnames(m) <- lvls
    m
  }
  
  y_train_one_hot_aligned <- to_one_hot(y_vec_tr, levels_y)
  y_validation_one_hot    <- to_one_hot(y_vec_va, levels_y)
  y_test_one_hot          <- to_one_hot(y_vec_te, levels_y)
  
  # Back-compat alias if other code references y_train_one_hot
  y_train_one_hot <- y_train_one_hot_aligned
  
  cat("[mc] dim(y one-hot) train/val/test: ",
      paste(dim(y_train_one_hot_aligned), collapse="×"), " / ",
      paste(dim(y_validation_one_hot), collapse="×"), " / ",
      paste(dim(y_test_one_hot), collapse="×"), "\n")
  
  # Keep original y matrices too
  y            <- as.matrix(y_train); colnames(y) <- colname_y
  y_validation <- as.matrix(y_validation)
  y_test       <- as.matrix(y_test)
  
  # ---------- E) Single source of truth for training ----------
  Rdata       <- X                               # training features
  labels      <- y_train_one_hot_aligned         # training labels aligned to X
  input_size  <- ncol(Rdata)
  output_size <- ncol(labels)
  
  cat("[mc] FINAL dim(Rdata)=", paste(dim(Rdata), collapse="×"),
      " | dim(labels)=", paste(dim(labels), collapse="×"),
      " | input_size=", input_size, " | output_size=", output_size, "\n")
  
  if (nrow(Rdata) != nrow(labels)) {
    stop(sprintf("[mc][FATAL] Row mismatch persists: nrow(Rdata)=%d vs nrow(labels)=%d.\nCheck alignment prints above.",
                 nrow(Rdata), nrow(labels)))
  }
  
  cat("[mc] N =", N, "\n")
  
  cat("===================== [MC] END =====================\n\n")
} else if (CLASSIFICATION_MODE == "regression") {
  cat("\n==================== [REG] START ====================\n")
  
  # ---------- A) Optional: make date numeric if present ----------
  make_date_numeric <- function(df) {
    if (!"date" %in% names(df)) return(df)
    d <- df[["date"]]
    if (inherits(d, "POSIXt")) {
      df[["date"]] <- as.numeric(as.Date(d))       # days since epoch
    } else if (inherits(d, "Date")) {
      df[["date"]] <- as.numeric(d)                # days since epoch
    } else {
      suppressWarnings({ parsed <- as.Date(d) })
      if (all(is.na(parsed))) {
        warning("[reg] 'date' column could not be parsed; converting to NA (will be imputed).")
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
  
  # ---------- B) Numeric imputation with TRAIN medians (no leakage) ----------
  impute_with_train_median <- function(df_train, df_other) {
    num_cols <- names(df_train)[vapply(df_train, is.numeric, TRUE)]
    for (nm in num_cols) {
      med <- suppressWarnings(median(df_train[[nm]], na.rm = TRUE))
      if (!is.finite(med) || is.na(med)) med <- 0
      if (nm %in% names(df_train)) df_train[[nm]][is.na(df_train[[nm]])] <- med
      if (nm %in% names(df_other)) df_other[[nm]][is.na(df_other[[nm]])] <- med
    }
    list(train = df_train, other = df_other)
  }
  tmp <- impute_with_train_median(X_train, X_validation); X_train <- tmp$train; X_validation <- tmp$other
  tmp <- impute_with_train_median(X_train, X_test);       X_test  <- tmp$other
  
  # ---------- C) Feature scaling without leakage ----------
  X_train_df <- as.data.frame(X_train)
  X_val_df   <- as.data.frame(X_validation)
  X_test_df  <- as.data.frame(X_test)
  
  num_mask <- vapply(X_train_df, is.numeric, TRUE)
  if (!any(num_mask)) stop("[reg] No numeric predictors found after preprocessing.")
  
  X_train_num <- as.matrix(X_train_df[, num_mask, drop = FALSE])
  X_val_num   <- as.matrix(X_val_df[,   num_mask, drop = FALSE])
  X_test_num  <- as.matrix(X_test_df[,  num_mask, drop = FALSE])
  
  X_train_scaled <- scale(X_train_num)
  center <- attr(X_train_scaled, "scaled:center")
  scale_ <- attr(X_train_scaled, "scaled:scale"); scale_[!is.finite(scale_) | scale_ == 0] <- 1
  
  X_validation_scaled <- sweep(sweep(X_val_num,  2, center, "-"), 2, scale_, "/")
  X_test_scaled       <- sweep(sweep(X_test_num, 2, center, "-"), 2, scale_, "/")
  
  # ---------- D) Further rescale to prevent exploding activations ----------
  max_val <- suppressWarnings(max(abs(X_train_scaled)))
  if (!is.finite(max_val) || is.na(max_val) || max_val == 0) max_val <- 1
  
  # --- Save training-time preprocessing for predict() ---
  feature_names <- colnames(X_train_num)  # exact order used to train
  train_medians <- vapply(as.data.frame(X_train_df[, feature_names, drop = FALSE]),
                          function(col) suppressWarnings(median(col, na.rm = TRUE)), numeric(1))
  train_medians[!is.finite(train_medians)] <- 0
  
  # ---------- E) y handling (train-only): optional z-score ----------
  SCALE_Y_WITH_ZSCORE <- FALSE  # TRUE = train on z-scored y; FALSE = raw y
  
  `%||%` <- get0("%||%", inherits = TRUE, ifnotfound = function(x, y) if (is.null(x)) y else x)
  
  # 1) Robust y extraction (vector)
  y_vec <- if (is.matrix(y_train) || is.data.frame(y_train)) {
    as.numeric(y_train[, 1])
  } else {
    as.numeric(y_train)
  }
  stopifnot(length(y_vec) == NROW(X_train))
  
  # 2) Decide transform type (train-only)
  is_regression <- identical(tolower(CLASSIFICATION_MODE %||% "regression"), "regression")
  use_zscore    <- is_regression && is.numeric(y_vec) && isTRUE(SCALE_Y_WITH_ZSCORE)
  
  if (use_zscore) {
    y_center <- mean(y_vec, na.rm = TRUE)
    y_scale  <- stats::sd(y_vec, na.rm = TRUE)
    if (!is.finite(y_scale) || y_scale == 0) y_scale <- 1
    
    y_vec_scaled <- (y_vec - y_center) / y_scale
    
    target_transform <- list(
      type   = "zscore",
      params = list(center = y_center, scale = y_scale)
    )
    y_trained_scaled <- TRUE
  } else {
    # Train on raw y (predict-only does identity inverse)
    y_vec_scaled <- y_vec
    target_transform <- list(
      type   = "identity",
      params = list(center = 0, scale = 1)
    )
    y_trained_scaled <- FALSE
  }
  
  
  # 3) one-col numeric matrix for training
  y <- matrix(as.numeric(y_vec_scaled), ncol = 1L)
  storage.mode(y) <- "double"
  colnames(y) <- colname_y
  
  # ---------- F) SINGLE source of truth object (no meta writes) ----------
  # Build preprocessScaledData containing BOTH X and y handling so predict-only can invert correctly.
  center_vec <- setNames(as.numeric(center[feature_names]),        feature_names)
  scale_vec  <- setNames(as.numeric(scale_[feature_names]),        feature_names)
  med_vec    <- setNames(as.numeric(train_medians[feature_names]), feature_names)
  
  preprocessScaledData <- list(
    # X preprocess
    feature_names     = as.character(feature_names),
    center            = center_vec,
    scale             = scale_vec,
    max_val           = as.numeric(max_val),
    divide_by_max_val = TRUE,
    train_medians     = med_vec,
    date_policy       = "as.Date -> numeric days; char parsed via as.Date()",
    used_scaled_X     = TRUE,
    scaler            = "standardize+divide_by_max",
    imputer           = "train_median",
    input_size        = ncol(X_train_num),
    
    # y / target transform lives HERE
    target_transform  = target_transform,      # identity or zscore
    y_trained_scaled  = isTRUE(y_trained_scaled)
  )
  
  # Make available to outer scope (for your writer / store_metadata)
  assign("preprocessScaledData", preprocessScaledData, inherits = TRUE)
  assign("target_transform",     target_transform,     inherits = TRUE)
  
  # ---------- (Optional but recommended) mirror into meta BEFORE you write the RDS ----------
  # This does NOT write to disk here; it only ensures the in-memory `meta` will contain the fields
  # when your existing store_metadata() persists it.
  # ---------- Add model-critical configs into meta ----------
  if (exists("meta", inherits = TRUE)) {
    try({
      meta$preprocessScaledData <- preprocessScaledData
      meta$target_transform     <- target_transform
      meta$train_target_center  <- target_transform$params$center %||% NA_real_
      meta$train_target_scale   <- target_transform$params$scale  %||% NA_real_
      
      # store activations and dropout etc.
      meta$activation_functions <- activation_functions
      meta$dropout_rates        <- dropout_rates
      meta$hidden_sizes         <- self$hidden_sizes %||% meta$hidden_sizes
      meta$output_size          <- self$output_size %||% meta$output_size
      meta$ML_NN                <- self$ML_NN %||% meta$ML_NN
    }, silent = TRUE)
  }
  
  
  # ---------- (Helpful debug) ----------
  if (isTRUE(get0("DEBUG_PREDICT", inherits = TRUE, ifnotfound = TRUE))) {
    if (use_zscore) {
      cat(sprintf("[y-train] using zscore: center=%.6f scale=%.6f | y sd(raw)=%.6f sd(scaled)=%.6f\n",
                  y_center, y_scale, stats::sd(y_vec), stats::sd(y_vec_scaled)))
    } else {
      cat(sprintf("[y-train] using identity target transform | y sd=%.6f\n", stats::sd(y_vec)))
    }
  }
  
  # `y` is now ready for training; `preprocessScaledData`/`target_transform` are ready for store_metadata().
  
  
  
  # ---------- G) Feed scaled inputs to the NN (divide-by-max applied) ----------
  X_train_scaled      <- X_train_scaled      / max_val
  X_validation_scaled <- X_validation_scaled / max_val
  X_test_scaled       <- X_test_scaled       / max_val
  
  scaledData <- TRUE  # use scaled inputs by default
  if (isTRUE(scaledData)) {
    X <- as.matrix(X_train_scaled)
    # IMPORTANT: keep y computed above (possibly scaled) — do NOT overwrite with raw y_train
    X_validation <- as.matrix(X_validation_scaled)
    X_test       <- as.matrix(X_test_scaled)
  } else {
    X <- as.matrix(X_train_num)
    X_validation <- as.matrix(X_val_num)
    X_test       <- as.matrix(X_test_num)
  }
  
  # Ensure y is 1 column numeric
  if (ncol(y) != 1L) {
    y <- matrix(as.numeric(y[, 1]), ncol = 1L)
  } else {
    storage.mode(y) <- "double"
  }
  colnames(y) <- colname_y
  
  # ---------- H) Diagnostics ----------
  cat("=== [reg] Unscaled X_train (numeric subset) summary ===\n")
  print(summary(as.vector(X_train_num)))
  cat("First 5 rows of unscaled numeric X_train:\n")
  print(X_train_num[1:5, 1:min(5, ncol(X_train_num)), drop = FALSE])
  
  cat("=== [reg] Scaled X_train summary ===\n")
  print(summary(as.vector(X)))
  cat("First 5 rows of scaled X (train):\n")
  print(X[1:5, 1:min(5, ncol(X)), drop = FALSE])
  
  cat("[reg] Dimensions (train/val/test):\n")
  cat("X:",            paste(dim(X), collapse="×"),          "\n")
  cat("X_validation:", paste(dim(X_validation), collapse="×"), "\n")
  cat("X_test:",       paste(dim(X_test), collapse="×"),      "\n")
  cat("[reg] Any NAs?  train:", anyNA(X),
      "  val:", anyNA(X_validation),
      "  test:", anyNA(X_test), "\n")
  
  # ---------- I) Final wiring into trainer ----------
  Rdata       <- X
  labels      <- y
  input_size  <- ncol(Rdata)
  output_size <- 1L
  
  cat("[reg] FINAL dim(Rdata)=", paste(dim(Rdata), collapse="×"),
      " | dim(labels)=", paste(dim(labels), collapse="×"),
      " | input_size=", input_size, " | output_size=", output_size, "\n")
  
  if (nrow(Rdata) != nrow(labels)) {
    stop(sprintf("[reg][FATAL] Row mismatch: nrow(Rdata)=%d vs nrow(labels)=%d.", nrow(Rdata), nrow(labels)))
  }
  
  cat("[reg] N =", N, "\n")
  cat("==================== [REG] END ====================\n\n")
}

# ---------- Compute N now that sizes are final ----------
if (!ML_NN) {
  N <- input_size + output_size
} else {
  N <- input_size + sum(hidden_sizes) + output_size
}



if (CLASSIFICATION_MODE == "binary"){
  preprocessScaledData <- NULL
}else if (CLASSIFICATION_MODE == "multiclass") {
  preprocessScaledData <- NULL
}


if (CLASSIFICATION_MODE == "multiclass") {
  input_size  <- ncol(Rdata)                    # after model.matrix/processing
  output_size <- ncol(y_train_one_hot_aligned)  # number of classes (fixed name)
}


# ==============================================================
# Optional Random Forest-based feature selection (default OFF)
# ==============================================================

importanceFeaturesOnly <- FALSE   # default: don't filter features

if (isTRUE(importanceFeaturesOnly)) {
  library(randomForest)
  
  # --- Train RF on TRAIN split (X, y) ---
  rf_data <- as.data.frame(X)                         
  rf_data$DEATH_EVENT <- as.factor(as.vector(y[, 1])) # ensure 1D factor
  
  set.seed(42)
  rf_model <- randomForest(DEATH_EVENT ~ ., data = rf_data, importance = TRUE)
  
  # Compute feature importance and select features above mean
  importance_scores <- importance(rf_model, type = 2)[, 1]  # MeanDecreaseGini
  threshold <- mean(importance_scores)
  selected_features <- names(importance_scores[importance_scores > threshold])
  
  # Safety net if none pass the threshold
  if (length(selected_features) == 0L) {
    k <- min(10L, length(importance_scores))
    selected_features <- names(sort(importance_scores, decreasing = TRUE))[seq_len(k)]
  }
  
  # Helper: enforce same columns & order; add any missing as zeros
  ensure_feature_columns <- function(M, wanted) {
    M <- as.matrix(M)
    miss <- setdiff(wanted, colnames(M))
    if (length(miss)) {
      M <- cbind(M, matrix(0, nrow = nrow(M), ncol = length(miss),
                           dimnames = list(NULL, miss)))
    }
    M[, wanted, drop = FALSE]
  }
  
  # ---- Apply the filter to ALL splits (train/val/test) ----
  X            <- ensure_feature_columns(X,            selected_features)
  X_validation <- ensure_feature_columns(X_validation, selected_features)
  X_test       <- ensure_feature_columns(X_test,       selected_features)
  
  # Update input size for neural network initialization
  input_size <- ncol(X)
  
  # Keep numeric_columns in sync (if present)
  if (exists("numeric_columns")) {
    numeric_columns <- intersect(numeric_columns, selected_features)
  }
  
  # (optional) quick checks
  stopifnot(identical(colnames(X), colnames(X_validation)),
            identical(colnames(X), colnames(X_test)))
  cat(sprintf("[RF] kept %d features; input_size=%d\n",
              length(selected_features), input_size))
}

# ==============================================================
# Adaptive Sample Weights (default OFF)
# ==============================================================

sampleWeights <- FALSE   # <-- toggle this flag; default = FALSE

if (isTRUE(sampleWeights)) {
  # --- Assume you already have P (n×1 probs) and yi (length n labels in {0,1}) ---
  probs  <- as.numeric(P[, 1])
  labels <- as.numeric(yi)
  
  # Optional: build flags from in-memory features (use your own logic).
  # If you have the *unscaled* data frame used for this split (e.g., X_validation_raw),
  # compute flags on that; otherwise set them to 0s as a safe default.
  deceptive_flags <- rep(0L, length(labels))
  risky_flags     <- rep(0L, length(labels))
  
  # Example (only if you have the needed columns in a DF called X_raw with same row order):
  # q <- function(x, p) quantile(x, p, na.rm = TRUE)
  # deceptive_flags <- as.integer(
  #   X_raw$serum_creatinine < q(X_raw$serum_creatinine, 0.10) &
  #   X_raw$age               < q(X_raw$age,               0.15) &
  #   X_raw$creatinine_phosphokinase < q(X_raw$creatinine_phosphokinase, 0.20)
  # )
  # risky_flags <- as.integer( ...your risky rule here... )
  
  # Sanity
  stopifnot(length(probs) == length(labels),
            length(deceptive_flags) == length(labels),
            length(risky_flags) == length(labels))
  
  # Error magnitude
  errors <- abs(probs - labels)
  
  # Base weights
  base_weights <- rep(1, length(labels))
  base_weights[labels == 1] <- base_weights[labels == 1] * 2
  base_weights[labels == 1 & risky_flags == 1] <- base_weights[labels == 1 & risky_flags == 1] * log(20) * 4
  base_weights[labels == 1 & deceptive_flags == 1] <- base_weights[labels == 1 & deceptive_flags == 1] * 3
  
  # Blend with error + clip
  raw_weights <- base_weights * errors
  raw_weights <- pmin(pmax(raw_weights, 0.05), 23)
  
  # Final adaptive weights (normalized)
  sample_weights <- 0.6 * base_weights + 0.4 * raw_weights
  sample_weights <- sample_weights / mean(sample_weights)
  
  cat("Sample weights created. Mean =", sprintf("%.4f", mean(sample_weights)), "\n")
  
} else {
  sample_weights <- NULL
  cat("Sample weights disabled (sampleWeights=FALSE).\n")
}



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#    ██████  ██████  ███    ██ ████████ ██████   ██████  ██          ██████   █████  ███    ██ ███████ ██  $$$$$$$$$$$$$$
#  ██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██          ██   ██ ██   ██ ████   ██ ██      ██   $$$$$$$$$$$$$$
# ██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██          ██████  ███████ ██ ██  ██ █████   ██    $$$$$$$$$$$$$$
#██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██          ██      ██   ██ ██  ██ ██ ██      ██     $$$$$$$$$$$$$$
#██████  ██████  ██   ████    ██    ██   ██  ██████  ███████     ██      ██   ██ ██   ████ ███████ ███████ $$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

viewTables <- FALSE
Losses_At_Optimal_Epoch_filenumber <- 3
writeTofiles <- FALSE
#########################################################################################################################

#########################################################################################################################


hyperparameter_grid_setup <- TRUE
reg_type = "L1" #"Max_Norm" #"L2" #Max_Norm" #"Group_Lasso" #"L1_L2"

# input_size <- 13 # This should match the actual number of features in your data
# hidden_size <- 2


plot_robustness <- FALSE
predict_models <- FALSE
use_loaded_weights <- FALSE
saveToDisk <- FALSE

# === Step 1: Hyperparameter setup ===
hyperparameter_grid_setup <- FALSE  # Set to FALSE to run a single combo manually


## =========================
## DDESONN Runner – Modes
## =========================
## SCENARIO A: Single-run only (no ensemble, ONE model)
# do_ensemble         <- FALSE
# num_networks        <- 1L
# num_temp_iterations <- 0L   # ignored when do_ensemble = FALSE
#
## SCENARIO B: Single-run, MULTI-MODEL (no ensemble)
# do_ensemble         <- FALSE
# num_networks        <- 4L          # e.g., run 5 models in one DDESONN instance
# num_temp_iterations <- 0L
#
## SCENARIO C: Main ensemble only (no TEMP/prune-add)
# do_ensemble         <- TRUE
# num_networks        <- 5L          # example main size
# num_temp_iterations <- 0L
#
## SCENARIO D: Main + TEMP iterations (prune/add enabled)
do_ensemble         <- TRUE
num_networks        <- 3L          # example main size
num_temp_iterations <- 2L          # MAIN + 1 TEMP pass (set higher for more TEMP passes)
#
## You can set the above variables BEFORE sourcing this file. The defaults below are fallbacks.

## ====== GLOBALS ======
results   <- data.frame(lr = numeric(), lambda = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)

`%||%` <- function(a,b) if (is.null(a) || length(a)==0) b else a

# You can set these BEFORE sourcing the file. Defaults below are only fallbacks.
num_networks        <- get0("num_networks", ifnotfound = 1L)
num_temp_iterations <- get0("num_temp_iterations", ifnotfound = 0L)   # 0 = MAIN only (no TEMP)
do_ensemble         <- get0("do_ensemble", ifnotfound = FALSE)         # TRUE ⇒ run MAIN (+ TEMP if >0)

# firstRun is only used to build the MAIN holder in ensemble mode
firstRun <- TRUE

j <- 1L
ensembles <- list(main_ensemble = vector("list"), temp_ensemble = vector("list"))

metric_name <- "accuracy"
viewTables  <- FALSE

## ====== Control panel flags ======
viewAllPlots <- FALSE  # TRUE shows all plots regardless of individual flags
verbose      <- FALSE  # TRUE enables additional plot/debug output

# SONN plots
accuracy_plot     <- FALSE    # show training accuracy/loss
saturation_plot   <- FALSE   # show output saturation
max_weight_plot   <- FALSE    # show max weight magnitude

# DDESONN plots
performance_high_mean_plots <- FALSE
performance_low_mean_plots  <- FALSE
relevance_high_mean_plots   <- FALSE
relevance_low_mean_plots    <- FALSE



## —— Artifacts
ARTIFACTS_DIR       <- file.path(getwd(), "artifacts")

if(train) {
  
  ## =========================================================================================
  ## SINGLE-RUN MODE (no logs, no lineage, no temp/prune/add) — Scenario A & B
  ## Foldering: artifacts/SingleRuns/<timestamp>__m<num_networks>__wSeed|wNoSeed
  ## =========================================================================================
  ## =========================
  ## SINGLE-RUN MODE (no ensemble)
  ## =========================
  if (!isTRUE(do_ensemble)) {
    cat(sprintf(
      "Single-run mode → training %d model%s inside one DDESONN instance, skipping all ensemble/logging.\n",
      as.integer(num_networks), if (num_networks == 1L) "" else "s"
    ))
    
    ## --------------------- RUN DIR SETUP (single stamp) ---------------------
    seeds <- suppressWarnings(as.integer(if (exists("seeds", inherits = TRUE) && length(seeds) >= 1L) seeds else 1L))
    if (!length(seeds) || any(is.na(seeds))) seeds <- 1L
    
    ts_stamp  <- format(Sys.time(), "%Y%m%d_%H%M%S")
    seed_tag  <- if (length(seeds) > 1L) "wSeed" else "wNoSeed"
    run_stamp <- sprintf("%s__m%d__%s", ts_stamp, as.integer(num_networks), seed_tag)
    
    OUT_ROOT <- file.path("artifacts", "SingleRuns")
    RUN_DIR  <- normalizePath(file.path(OUT_ROOT, run_stamp), winslash = "/", mustWork = FALSE)
    cat(sprintf("[SINGLE] RUN_DIR = %s\n", RUN_DIR))
    
    MODELS_DIR_MAIN <- file.path(RUN_DIR, "models", "main")
    dir.create(MODELS_DIR_MAIN, recursive = TRUE, showWarnings = FALSE)
    
    # Filenames (inside RUN_DIR)
    s_chr <- as.character(length(seeds))
    agg_pred_file_test     <- file.path(RUN_DIR, sprintf("SingleRun_Pretty_Test_Metrics_%s_seeds_%s.rds", s_chr, ts_stamp))
    agg_metrics_file_test  <- file.path(RUN_DIR, sprintf("SingleRun_Test_Metrics_%s_seeds_%s.rds",          s_chr, ts_stamp))
    agg_metrics_file_train <- file.path(RUN_DIR, sprintf("SingleRun_Train_Acc_Val_Metrics_%s_seeds_%s.rds", s_chr, ts_stamp))
    
    main_model  <- NULL
    metrics_rows <- list()  # collect TRAIN metrics rows (seed × slot)
    
    for (i in seq_along(seeds)) {
      s <- seeds[i]
      set.seed(s)
      
      run_model <- DDESONN$new(
        num_networks    = max(1L, as.integer(num_networks)),
        input_size      = input_size,
        hidden_sizes    = hidden_sizes,
        output_size     = output_size,
        N               = N,
        lambda          = lambda,
        ensemble_number = 0L,
        ensembles       = NULL,
        ML_NN           = ML_NN,
        activation_functions=activation_functions,
        activation_functions_predict=activation_functions_predict,
        method          = init_method,
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
      
      model_results <<- run_model$train(
        Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
        lr_min=lr_min, ensemble_number=0L, num_epochs=num_epochs, self_org=self_org,
        threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns, CLASSIFICATION_MODE=CLASSIFICATION_MODE,
        activation_functions=activation_functions, activation_functions_predict=activation_functions_predict,
        dropout_rates=dropout_rates, optimizer=optimizer,
        beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
        batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
        epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
        shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights, preprocessScaledData=preprocessScaledData,
        X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
        train=train, grouped_metrics=grouped_metrics, viewTables=viewTables, verbose=verbose
      )
      
      ## === STAMP MAIN_0 METADATA with best_* including best_val_prediction_time ===
      best_train_acc_ret     <- try(model_results$predicted_outputAndTime$best_train_acc,           silent = TRUE); if (inherits(best_train_acc_ret, "try-error")) best_train_acc_ret <- NA_real_
      best_epoch_train_ret   <- try(model_results$predicted_outputAndTime$best_epoch_train,         silent = TRUE); if (inherits(best_epoch_train_ret, "try-error")) best_epoch_train_ret <- NA_integer_
      best_val_acc_ret       <- try(model_results$predicted_outputAndTime$best_val_acc,             silent = TRUE); if (inherits(best_val_acc_ret, "try-error")) best_val_acc_ret <- NA_real_
      best_val_epoch_ret     <- try(model_results$predicted_outputAndTime$best_val_epoch,           silent = TRUE); if (inherits(best_val_epoch_ret, "try-error")) best_val_epoch_ret <- NA_integer_
      best_val_pred_time_ret <- try(model_results$predicted_outputAndTime$best_val_prediction_time, silent = TRUE); if (inherits(best_val_pred_time_ret, "try-error")) best_val_pred_time_ret <- NA_real_
      
      for (k in seq_len(max(1L, as.integer(num_networks)))) {
        mvar <- sprintf("Ensemble_Main_0_model_%d_metadata", as.integer(k))
        if (!exists(mvar, envir = .GlobalEnv)) next
        md <- get(mvar, envir = .GlobalEnv)
        
        # select per-slot scalar without helpers (prefer k-th if vector)
        val <- suppressWarnings(as.numeric(md$best_train_acc)); md$best_train_acc <- if (length(val) >= k && is.finite(val[k])) val[k] else if (length(val) >= 1 && is.finite(val[1])) val[1] else best_train_acc_ret
        val <- suppressWarnings(as.numeric(md$best_epoch_train)); md$best_epoch_train <- as.integer(if (length(val) >= k && is.finite(val[k])) val[k] else if (length(val) >= 1 && is.finite(val[1])) val[1] else best_epoch_train_ret)
        val <- suppressWarnings(as.numeric(md$best_val_acc)); md$best_val_acc <- if (length(val) >= k && is.finite(val[k])) val[k] else if (length(val) >= 1 && is.finite(val[1])) val[1] else best_val_acc_ret
        val <- suppressWarnings(as.numeric(md$best_val_epoch)); md$best_val_epoch <- as.integer(if (length(val) >= k && is.finite(val[k])) val[k] else if (length(val) >= 1 && is.finite(val[1])) val[1] else best_val_epoch_ret)
        val <- suppressWarnings(as.numeric(md$best_val_prediction_time)); md$best_val_prediction_time <- if (length(val) >= k && is.finite(val[k])) val[k] else if (length(val) >= 1 && is.finite(val[1])) val[1] else best_val_pred_time_ret
        
        assign(mvar, md, envir = .GlobalEnv)
        cat(sprintf("[STAMPED][SINGLE MAIN_0] slot=%d best_val_acc=%s\n", k, as.character(md$best_val_acc)))
        
        # inside your loop over k just before saveRDS(md, ...)
        slot_obj <- run_model$ensemble[[k]]
        
        # 1) put the callable onto metadata (object with $predict)
        md$predictor <- slot_obj
        # (optional but common in your code paths)
        md$predictor_fn <- function(X, ...) slot_obj$predict(X, ...)
        
        # 2) keep feature names and splits so coerce/select_split never bails
        if (is.null(md$feature_names) || !length(md$feature_names)) md$feature_names <- colnames(X)
        md$X_train      <- X;              md$y_train      <- y
        md$X_validation <- X_validation;   md$y_validation <- y_validation
        md$X_test       <- get0("X_test", inherits = TRUE, ifnotfound = NULL)
        md$y_test       <- get0("y_test", inherits = TRUE, ifnotfound = NULL)
        
        assign(mvar, md, envir = .GlobalEnv)  # ensure ENV copy reflects the predictor
        saveRDS(md, file.path(MODELS_DIR_MAIN, sprintf("%s_%s_seed%s.rds", mvar, ts_stamp, s)))
        
        
        
        # save the stamped metadata for this seed/slot
        saveRDS(md, file.path(
          MODELS_DIR_MAIN,
          sprintf("%s_%s_seed%s.rds", mvar, ts_stamp, as.character(s))
        ))
      }
      
      ## Ensure the trained slot carries its metadata
      if (length(run_model$ensemble)) {
        for (k in seq_len(max(1L, as.integer(num_networks)))) {
          mvar <- sprintf("Ensemble_Main_0_model_%d_metadata", as.integer(k))
          if (exists(mvar, envir = .GlobalEnv)) {
            run_model$ensemble[[k]]$metadata <- get(mvar, envir = .GlobalEnv)
          }
        }
      }
      
      # ============================
      # TRAIN METRICS (per-slot): collect ALL metrics available at the SLOT.
      # Try multiple canonical locations on the slot (NO seed-level fallback).
      # ============================
      for (k in seq_len(max(1L, as.integer(num_networks)))) {
        pm_slot <- NULL
        rm_slot <- NULL
        
        # 1) Nested container
        tmp <- try(run_model$ensemble[[k]]$performance_relevance_data$performance_metric, silent = TRUE)
        if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) pm_slot <- tmp
        tmp <- try(run_model$ensemble[[k]]$performance_relevance_data$relevance_metric,   silent = TRUE)
        if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) rm_slot <- tmp
        
        # 2) Direct fields on slot
        if (is.null(pm_slot)) {
          tmp <- try(run_model$ensemble[[k]]$performance_metric, silent = TRUE)
          if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) pm_slot <- tmp
        }
        if (is.null(rm_slot)) {
          tmp <- try(run_model$ensemble[[k]]$relevance_metric, silent = TRUE)
          if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) rm_slot <- tmp
        }
        
        # 3) Generic $metrics container (some builds store here)
        if (is.null(pm_slot)) {
          tmp <- try(run_model$ensemble[[k]]$metrics$performance_metric, silent = TRUE)
          if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) pm_slot <- tmp
        }
        if (is.null(rm_slot)) {
          tmp <- try(run_model$ensemble[[k]]$metrics$relevance_metric, silent = TRUE)
          if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) rm_slot <- tmp
        }
        
        # 4) Metadata (last resort, still SLOT-scoped)
        if (is.null(pm_slot)) {
          tmp <- try(run_model$ensemble[[k]]$metadata$performance_metric, silent = TRUE)
          if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) pm_slot <- tmp
        }
        if (is.null(rm_slot)) {
          tmp <- try(run_model$ensemble[[k]]$metadata$relevance_metric, silent = TRUE)
          if (!inherits(tmp, "try-error") && is.list(tmp) && length(tmp)) rm_slot <- tmp
        }
        
        if (is.null(pm_slot) && is.null(rm_slot)) {
          row_df <- data.frame(run_index = i, seed = s, model_slot = k, stringsAsFactors = FALSE)
        } else {
          flat <- tryCatch(
            rapply(list(performance_metric = pm_slot, relevance_metric = rm_slot),
                   f = function(z) z, how = "unlist"),
            error = function(e) setNames(vector("list", 0L), character(0))
          )
          if (length(flat)) {
            L <- as.list(flat)
            flat <- flat[vapply(L, is.atomic, logical(1)) & lengths(L) == 1L]
          }
          nms <- names(flat)
          if (length(nms)) {
            drop <- grepl("custom_relative_error_binned", nms, fixed = TRUE) |
              grepl("grid_used",                    nms, fixed = TRUE) |
              grepl("(^|\\.)details(\\.|$)",        nms)
            keep <- !drop & !is.na(flat)
            flat <- flat[keep]; nms <- names(flat)
          }
          if (length(flat) == 0L) {
            row_df <- data.frame(run_index = i, seed = s, model_slot = k, stringsAsFactors = FALSE)
          } else {
            out <- setNames(vector("list", length(flat)), nms)
            num <- suppressWarnings(as.numeric(flat))
            for (jj in seq_along(flat)) out[[jj]] <- if (!is.na(num[jj])) num[jj] else as.character(flat[[jj]])
            row_df <- as.data.frame(out, check.names = TRUE, stringsAsFactors = FALSE)
            row_df <- cbind(data.frame(run_index = i, seed = s, model_slot = k, stringsAsFactors = FALSE), row_df)
          }
        }
        
        # Add best_* per-slot from metadata
        env_name <- sprintf("Ensemble_Main_0_model_%d_metadata", as.integer(k))
        md_k <- tryCatch(run_model$ensemble[[k]]$metadata, error = function(e) NULL)
        if (is.null(md_k) || !is.list(md_k)) {
          if (exists(env_name, envir = .GlobalEnv)) md_k <- get(env_name, envir = .GlobalEnv)
        }
        row_df$best_train_acc           <- tryCatch(suppressWarnings(as.numeric(md_k$best_train_acc))[1],           error = function(e) NA_real_)
        row_df$best_epoch_train         <- tryCatch(suppressWarnings(as.integer(md_k$best_epoch_train))[1],         error = function(e) NA_integer_)
        row_df$best_val_acc             <- tryCatch(suppressWarnings(as.numeric(md_k$best_val_acc))[1],             error = function(e) NA_real_)
        row_df$best_val_epoch           <- tryCatch(suppressWarnings(as.integer(md_k$best_val_epoch))[1],           error = function(e) NA_integer_)
        row_df$best_val_prediction_time <- tryCatch(suppressWarnings(as.numeric(md_k$best_val_prediction_time))[1], error = function(e) NA_real_)
        
        metrics_rows[[length(metrics_rows) + 1L]] <- row_df
      }
      
      # ============================
      # TEST EVAL (per-slot, per-seed)
      # ============================
      if (isTRUE(test)) {
        for (k in seq_len(max(1L, as.integer(num_networks)))) {
          env_name <- sprintf("Ensemble_Main_0_model_%d_metadata", as.integer(k))
          
          md_k <- tryCatch(run_model$ensemble[[k]]$metadata, error = function(e) NULL)
          if (is.null(md_k) || !is.list(md_k)) {
            if (exists(env_name, envir = .GlobalEnv)) {
              md_k <- get(env_name, envir = .GlobalEnv)
            }
          }
          if (is.null(md_k) || !is.list(md_k)) {
            warning(sprintf("[single-run][TEST] Missing metadata for slot %d (both model and ENV); skipping.", k))
            next
          }
          if (is.null(md_k$model_serial_num) || !nzchar(as.character(md_k$model_serial_num))) {
            md_k$model_serial_num <- sprintf("0.main.%d", as.integer(k))
          }
          assign(env_name, md_k, envir = .GlobalEnv)
          
          ret <- tryCatch(
            DDESONN_predict_eval(
              LOAD_FROM_RDS = FALSE,
              ENV_META_NAME = env_name,
              INPUT_SPLIT   = "test",
              CLASSIFICATION_MODE = CLASSIFICATION_MODE,
              RUN_INDEX = i,
              SEED      = s,
              OUTPUT_DIR = RUN_DIR,
              OUT_DIR_ASSERT = RUN_DIR,
              SAVE_METRICS_RDS = TRUE,
              METRICS_PREFIX   = "metrics_test",
              AGG_PREDICTIONS_FILE = agg_pred_file_test,
              AGG_METRICS_FILE     = agg_metrics_file_test,
              MODEL_SLOT           = k
            ),
            error = function(e) {
              message(sprintf("[single-run][TEST] seed=%s slot=%d predict_eval ERROR: %s", s, k, conditionMessage(e)))
              NULL
            }
          )
          
          ok <- !is.null(ret) && is.null(ret$problem_stage)
          
          if (!ok) {
            st <- if (is.null(ret)) "threw-error" else paste0("failed-stage:", ret$problem_stage)
            message(sprintf("[single-run][TEST] seed=%s slot=%d did NOT write (%s). stage_log=%s",
                            s, k, st, paste(ret$stage_log %||% character(), collapse=" > ")))
          } else {
            cat(sprintf("[single-run][TEST] seed=%s slot=%d wrote ✓\n", s, k))
          }
        
          
        
          
        }
        
        

        
        
        
      }
      
      # retain last model for container attach
      if (i == length(seeds)) main_model <- run_model
    } # end seeds loop
    
    # ============================
    # POST: bind all TRAIN rows (seed × slot) and normalize
    # ============================
    if (length(metrics_rows) == 0L) {
      df_tr <- data.frame()
    } else {
      df_tr <- metrics_rows[[1]]
      if (length(metrics_rows) > 1L) {
        for (ii in 2:length(metrics_rows)) {
          x <- df_tr
          y <- metrics_rows[[ii]]
          for (m in setdiff(names(y), names(x))) x[[m]] <- NA
          for (m in setdiff(names(x), names(y))) y[[m]] <- NA
          ord <- union(names(x), names(y))
          df_tr <- rbind(x[, ord, drop = FALSE], y[, ord, drop = FALSE])
        }
      }
    }
    
    # flatten prefixes so QE/TE/DB and others become plain columns
    colnames(df_tr) <- gsub("^(performance_metric|relevance_metric)\\.", "", colnames(df_tr))
    # drop any duplicate columns created by flattening (keep first)
    dups <- duplicated(colnames(df_tr))
    if (any(dups)) df_tr <- df_tr[, !dups, drop = FALSE]
    
    # numeric normalization (common)
    numeric_maybe <- c(
      "MSE","MAE","RMSE","R2","MAPE","SMAPE","WMAPE","MASE",
      "accuracy","precision","recall","f1","f1_score",
      "balanced_accuracy","specificity","sensitivity","auc","logloss","brier",
      "quantization_error","topographic_error","clustering_quality_db",
      "generalization_ability","speed","speed_learn1","speed_learn2","speed_learn3","speed_learn4",
      "memory_usage","robustness","hit_rate","ndcg","diversity","serendipity",
      "confusion_matrix.TP","confusion_matrix.FP","confusion_matrix.TN","confusion_matrix.FN",
      "accuracy_precision_recall_f1_tuned.accuracy",
      "accuracy_precision_recall_f1_tuned.precision",
      "accuracy_precision_recall_f1_tuned.recall",
      "accuracy_precision_recall_f1_tuned.f1",
      paste0("accuracy_precision_recall_f1_tuned.confusion_matrix.", c("TP","FP","TN","FN"))
    )
    present_num <- intersect(numeric_maybe, names(df_tr))
    for (nm in present_num) df_tr[[nm]] <- suppressWarnings(as.numeric(df_tr[[nm]]))
    
    # ensure types
    if ("run_index"  %in% names(df_tr)) df_tr$run_index  <- suppressWarnings(as.integer(df_tr$run_index))
    if ("seed"       %in% names(df_tr)) df_tr$seed       <- suppressWarnings(as.integer(df_tr$seed))
    if ("model_slot" %in% names(df_tr)) df_tr$model_slot <- suppressWarnings(as.integer(df_tr$model_slot))
    if ("split"      %in% names(df_tr)) df_tr$split      <- as.character(df_tr$split)
    
    out_path <- file.path(
      RUN_DIR,
      sprintf("SingleRun_Train_Acc_Val_Metrics_%s_seeds_%s.rds", s_chr, ts_stamp)
    )
    saveRDS(df_tr, out_path)
    cat("Saved TRAIN table to:", out_path, " | rows=", nrow(df_tr), " cols=", ncol(df_tr), "\n")
    if ("accuracy" %in% names(df_tr) && "model_slot" %in% names(df_tr)) {
      cat(sprintf("[TRAIN] accuracy by seed/slot: %s\n",
                  paste(sprintf("seed%s.slot%s=%.3f", df_tr$seed, df_tr$model_slot, df_tr$accuracy), collapse=" | ")))
    }
    
    # ============================
    # POST: Clean TEST agg file (unchanged behavior + flatten)
    # ============================
    if (file.exists(agg_metrics_file_test)) {
      df <- try(readRDS(agg_metrics_file_test), silent = TRUE)
      if (!inherits(df, "try-error") && is.data.frame(df) && NROW(df)) {
        if (!("accuracy"   %in% names(df))) df$accuracy   <- NA_real_
        if (!("precision"  %in% names(df))) df$precision  <- NA_real_
        if (!("recall"     %in% names(df))) df$recall     <- NA_real_
        if (!("f1"         %in% names(df))) df$f1         <- NA_real_
        if (!("f1_score"   %in% names(df))) df$f1_score   <- NA_real_
        
        tuned_acc  <- "accuracy_precision_recall_f1_tuned.accuracy"
        tuned_pr   <- "accuracy_precision_recall_f1_tuned.precision"
        tuned_rec  <- "accuracy_precision_recall_f1_tuned.recall"
        tuned_f1   <- "accuracy_precision_recall_f1_tuned.f1"
        
        if (tuned_acc %in% names(df)) {
          na_idx <- is.na(suppressWarnings(as.numeric(df$accuracy)))
          if (any(na_idx)) df$accuracy[na_idx] <- suppressWarnings(as.numeric(df[[tuned_acc]])[na_idx])
        }
        if (tuned_pr %in% names(df)) {
          na_idx <- is.na(suppressWarnings(as.numeric(df$precision)))
          if (any(na_idx)) df$precision[na_idx] <- suppressWarnings(as.numeric(df[[tuned_pr]])[na_idx])
        }
        if (tuned_rec %in% names(df)) {
          na_idx <- is.na(suppressWarnings(as.numeric(df$recall)))
          if (any(na_idx)) df$recall[na_idx] <- suppressWarnings(as.numeric(df[[tuned_rec]])[na_idx])
        }
        if (tuned_f1 %in% names(df)) {
          na_idx <- is.na(suppressWarnings(as.numeric(df$f1)))
          if (any(na_idx)) df$f1[na_idx] <- suppressWarnings(as.numeric(df[[tuned_f1]])[na_idx])
        }
        na_f1s <- is.na(suppressWarnings(as.numeric(df$f1_score)))
        if (any(na_f1s)) df$f1_score[na_f1s] <- suppressWarnings(as.numeric(df$f1))[na_f1s]
        
        if ("MODEL_SLOT" %in% names(df) && !("model_slot" %in% names(df))) {
          names(df)[names(df) == "MODEL_SLOT"] <- "model_slot"
        }
        if ("run_index"  %in% names(df)) df$run_index  <- suppressWarnings(as.integer(df$run_index))
        if ("seed"       %in% names(df)) df$seed       <- suppressWarnings(as.integer(df$seed))
        if ("model_slot" %in% names(df)) df$model_slot <- suppressWarnings(as.integer(df$model_slot))
        if ("split"      %in% names(df)) df$split      <- as.character(df$split)
        
        # flatten nested perf columns and dedupe
        colnames(df) <- gsub("^(performance_metric|relevance_metric)\\.", "", colnames(df))
        dups <- duplicated(colnames(df))
        if (any(dups)) df <- df[, !dups, drop = FALSE]
        
        saveRDS(df, agg_metrics_file_test)
      }
    }
    
    # attach last main model to container
    if (!is.null(main_model)) {
      main_model$ensemble_number <- 0L
      ensembles <- attach_run_to_container(ensembles, main_model)
      print_ensembles_summary(ensembles)
      
      if (exists("ensembles", inherits = TRUE) && is.list(ensembles)) {
        if (is.null(ensembles$main_ensemble)) ensembles$main_ensemble <- list()
        ensembles$main_ensemble[[1]] <- main_model
      }
      
      if (!is.null(main_model$performance_metric)) {
        cat("\nSingle run performance_metric (DDESONN-level):\n"); print(main_model$performance_metric)
      }
      if (!is.null(main_model$relevance_metric)) {
        cat("\nSingle run relevance_metric (DDESONN-level):\n"); print(main_model$relevance_metric)
      }
    }
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  else {
    ## =======================
    ## ENSEMBLE (multi-seed)
    ##   - Scenario C: num_temp_iterations == 0
    ##   - Scenario D: num_temp_iterations > 0 (prune/add)
    ##   Output: ONE ROW PER MODEL SLOT PER SEED
    ## =======================
    
    '%||%' <- get0("%||%", ifnotfound = function(x, y) if (is.null(x)) y else x)
    
    main_meta_var <- function(i) sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(i))
    temp_meta_var <- function(e,i) sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(e), as.integer(i))
    
    snapshot_main_serials_meta <- function() {
      vars <- grep("^Ensemble_Main_(0|1)_model_\\d+_metadata$", ls(.GlobalEnv), value = TRUE)
      if (!length(vars)) return(character())
      ord  <- suppressWarnings(as.integer(sub("^Ensemble_Main_(?:0|1)_model_(\\d+)_metadata$", "\\1", vars)))
      vars <- vars[order(ord)]
      vapply(vars, function(v) {
        md <- get(v, envir = .GlobalEnv)
        as.character(md$model_serial_num %||% NA_character_)
      }, character(1))
    }
    
    get_temp_serials_meta <- function(iter_j) {
      e <- iter_j + 1L
      vars <- grep(sprintf("^Ensemble_Temp_%d_model_\\d+_metadata$", e), ls(.GlobalEnv), value = TRUE)
      if (!length(vars)) return(character())
      ord <- suppressWarnings(as.integer(sub(sprintf("^Ensemble_Temp_%d_model_(\\d+)_metadata$", e), "\\1", vars)))
      vars <- vars[order(ord)]
      vapply(vars, function(v) {
        md <- get(v, envir = .GlobalEnv)
        s  <- md$model_serial_num
        if (!is.null(s) && nzchar(as.character(s))) as.character(s) else NA_character_
      }, character(1))
    }
    
    .metric_minimize <- function(metric) isTRUE(get0(paste0("MINIMIZE_", metric), ifnotfound = FALSE, inherits = TRUE))
    is_real_serial   <- function(s) is.character(s) && length(s) == 1L && nzchar(s) && !is.na(s)
    EMPTY_SLOT <- structure(list(.empty_slot = TRUE), class = "EMPTY_SLOT")
    
    .scalar_num <- function(x, idx = NA_integer_) {
      v <- suppressWarnings(as.numeric(x))
      if (!length(v)) return(NA_real_)
      if (is.finite(idx) && !is.na(idx) && idx >= 1L && idx <= length(v)) return(v[idx])
      v[1]
    }
    
    prune_network_from_ensemble <- function(ensembles, target_metric_name_worst) {
      minimize  <- .metric_minimize(target_metric_name_worst)
      main_sers <- snapshot_main_serials_meta()
      if (!length(main_sers)) return(NULL)
      
      get_metric_by_serial <- function(serial, metric_name) {
        vars <- grep("^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
                     ls(.GlobalEnv), value = TRUE)
        for (v in vars) {
          md <- get(v, envir = .GlobalEnv)
          if (identical(as.character(md$model_serial_num %||% NA_character_), as.character(serial))) {
            val <- tryCatch(md$performance_metric[[metric_name]], error = function(e) NULL)
            if (is.null(val)) val <- tryCatch(md$relevance_metric[[metric_name]], error = function(e) NULL)
            vn <- suppressWarnings(as.numeric(val))
            return(if (length(vn) && is.finite(vn[1])) vn[1] else NA_real_)
          }
        }
        NA_real_
      }
      
      main_vals <- vapply(main_sers, get_metric_by_serial, numeric(1), target_metric_name_worst)
      if (all(!is.finite(main_vals))) return(NULL)
      
      worst_idx  <- if (minimize) which.max(main_vals) else which.min(main_vals)
      worst_slot <- as.integer(worst_idx)
      if (!(length(ensembles$main_ensemble) >= 1L)) return(NULL)
      main_container <- ensembles$main_ensemble[[1]]
      if (is.null(main_container$ensemble) || !length(main_container$ensemble)) return(NULL)
      if (worst_slot < 1L || worst_slot > length(main_container$ensemble)) return(NULL)
      
      removed_model <- main_container$ensemble[[worst_slot]]
      main_container$ensemble[[worst_slot]] <- EMPTY_SLOT
      ensembles$main_ensemble[[1]] <- main_container
      
      list(
        removed_network   = removed_model,
        updated_ensembles = ensembles,
        worst_model_index = worst_slot,
        worst_slot        = worst_slot,
        worst_serial      = as.character(main_sers[worst_slot]),
        worst_value       = as.numeric(main_vals[worst_slot])
      )
    }
    
    add_network_to_ensemble <- function(ensembles, target_metric_name_best,
                                        removed_network, ensemble_number,
                                        worst_model_index, removed_serial, removed_value) {
      minimize     <- .metric_minimize(target_metric_name_best)
      temp_serials <- get_temp_serials_meta(ensemble_number)
      if (!length(temp_serials)) return(list(updated_ensembles=ensembles, worst_slot=as.integer(worst_model_index)))
      
      get_metric_by_serial <- function(serial, metric_name) {
        vars <- grep("^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
                     ls(.GlobalEnv), value = TRUE)
        for (v in vars) {
          md <- get(v, envir = .GlobalEnv)
          if (identical(as.character(md$model_serial_num %||% NA_character_), as.character(serial))) {
            val <- tryCatch(md$performance_metric[[metric_name]], error = function(e) NULL)
            if (is.null(val)) val <- tryCatch(md$relevance_metric[[metric_name]], error = function(e) NULL)
            vn <- suppressWarnings(as.numeric(val))
            return(if (length(vn) && is.finite(vn[1])) vn[1] else NA_real_)
          }
        }
        NA_real_
      }
      
      temp_vals <- vapply(temp_serials, get_metric_by_serial, numeric(1), target_metric_name_best)
      if (all(!is.finite(temp_vals))) return(list(updated_ensembles=ensembles, worst_slot=as.integer(worst_model_index)))
      
      best_idx    <- if (minimize) which.min(temp_vals) else which.max(temp_vals)
      best_serial <- as.character(temp_serials[best_idx])
      
      ## place candidate if truly better
      worst_slot <- as.integer(worst_model_index)
      if (!(length(ensembles$main_ensemble) >= 1L)) return(list(updated_ensembles=ensembles, worst_slot=worst_slot))
      main_container <- ensembles$main_ensemble[[1]]
      if (is.null(main_container$ensemble) || !length(main_container$ensemble)) return(list(updated_ensembles=ensembles, worst_slot=worst_slot))
      
      temp_parts       <- strsplit(best_serial, "\\.")[[1]]
      temp_model_index <- suppressWarnings(as.integer(temp_parts[3]))
      if (!is.finite(temp_model_index) || is.na(temp_model_index)) return(list(updated_ensembles=ensembles, worst_slot=worst_slot))
      
      if (!(length(ensembles$temp_ensemble) >= 1L) || is.null(ensembles$temp_ensemble[[1]]$ensemble)) return(list(updated_ensembles=ensembles, worst_slot=worst_slot))
      temp_container <- ensembles$temp_ensemble[[1]]
      if (temp_model_index < 1L || temp_model_index > length(temp_container$ensemble)) return(list(updated_ensembles=ensembles, worst_slot=worst_slot))
      
      candidate_model <- temp_container$ensemble[[temp_model_index]]
      main_container$ensemble[[worst_slot]] <- candidate_model
      ensembles$main_ensemble[[1]] <- main_container
      
      temp_e  <- suppressWarnings(as.integer(temp_parts[1]))
      tvar <- temp_meta_var(temp_e, temp_model_index)
      mvar <- main_meta_var(worst_slot)
      if (exists(tvar, envir = .GlobalEnv)) {
        tmd <- get(tvar, envir = .GlobalEnv)
        tmd$model_serial_num <- best_serial
        assign(mvar, tmd, envir = .GlobalEnv)
      }
      list(updated_ensembles=ensembles, worst_slot=worst_slot)
    }
    
    resolve_env_meta <- function(slot = 1L, prefer = c("main","temp"), temp_iter_fallback = 1L) {
      prefer <- match.arg(prefer)
      cand <- if (prefer == "main") {
        c(sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(slot)),
          sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(temp_iter_fallback), as.integer(slot)))
      } else {
        c(sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(temp_iter_fallback), as.integer(slot)),
          sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(slot)))
      }
      hit <- cand[vapply(cand, exists, logical(1), envir = .GlobalEnv, inherits = TRUE)]
      if (length(hit)) return(hit[[1L]])
      stop(sprintf("resolve_env_meta: No MAIN/TEMP metadata found for slot=%d (tried: %s)",
                   as.integer(slot), paste(cand, collapse = ", ")))
    }
    
    .fix_agg_layout_for_fuser <- function(
    path,
    run_index = NULL,
    seed = NULL,
    split = "test",
    classification_mode = "binary"   # prefer CLASSIFICATION_MODE only
    ) {
      if (!file.exists(path)) return(invisible(FALSE))
      df <- try(readRDS(path), silent = TRUE)
      if (inherits(df, "try-error") || !is.data.frame(df) || !NROW(df)) return(invisible(FALSE))
      
      # ---- coerce types safely ----
      as_int <- function(x) suppressWarnings(as.integer(x))
      as_num <- function(x) suppressWarnings(as.numeric(x))
      as_chr <- function(x) suppressWarnings(as.character(x))
      
      # Canonical ids (keep both lower + UPPER for compatibility)
      if (!"run_index" %in% names(df) && "RUN_INDEX" %in% names(df)) df$run_index <- as_int(df$RUN_INDEX)
      if (!"RUN_INDEX" %in% names(df) && "run_index" %in% names(df)) df$RUN_INDEX <- as_int(df$run_index)
      
      if (!"seed" %in% names(df) && "SEED" %in% names(df)) df$seed <- as_int(df$SEED)
      if (!"SEED" %in% names(df) && "seed" %in% names(df)) df$SEED <- as_int(df$seed)
      
      if (!"model_slot" %in% names(df) && "MODEL_SLOT" %in% names(df)) df$model_slot <- as_int(df$MODEL_SLOT)
      if (!"MODEL_SLOT" %in% names(df) && "model_slot" %in% names(df)) df$MODEL_SLOT <- as_int(df$model_slot)
      
      # Split (keep lowercase 'split'; also write legacy 'SPLIT')
      if (!"split" %in% names(df)) {
        if ("SPLIT" %in% names(df)) df$split <- tolower(as_chr(df$SPLIT)) else df$split <- tolower(split)
      } else {
        df$split <- tolower(as_chr(df$split))
      }
      df$SPLIT <- toupper(df$split)
      
      # Classification mode (keep CLASSIFICATION_MODE only; uppercase stored, lowercase used by code)
      if (!"CLASSIFICATION_MODE" %in% names(df)) {
        df$CLASSIFICATION_MODE <- toupper(classification_mode)
      } else {
        df$CLASSIFICATION_MODE <- toupper(as_chr(df$CLASSIFICATION_MODE))
      }
      
      # Prediction cols: we now standardize on y_prob (numeric), no y_pred_full synthesis
      if (!"y_prob" %in% names(df) && "y_pred" %in% names(df)) df$y_prob <- as_num(df$y_pred)
      if ("y_prob" %in% names(df)) df$y_prob <- as_num(df$y_prob)
      if ("y_true" %in% names(df)) df$y_true <- as_num(df$y_true)
      
      # (Optional) keep legacy y_pred as mirror of y_prob if some readers expect it
      if (!"y_pred" %in% names(df) && "y_prob" %in% names(df)) df$y_pred <- df$y_prob
      
      # Optional narrowing to avoid accidental empty subsets later
      if (!is.null(run_index) && "RUN_INDEX" %in% names(df)) {
        df <- df[df$RUN_INDEX == as_int(run_index), , drop = FALSE]
      }
      if (!is.null(seed) && "SEED" %in% names(df)) {
        df <- df[df$SEED == as_int(seed), , drop = FALSE]
      }
      
      saveRDS(df, path)
      invisible(TRUE)
    }
    
    ## ========= END FIX helpers =========
    
    ## -------------------------
    ## Seed loop
    ## -------------------------
    seeds <- 1:x
    per_slot_rows <- list()
    ts_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    
    ## Run folder
    OUT_ROOT <- file.path("artifacts", "EnsembleRuns")
    RUN_DIR  <- file.path(OUT_ROOT, ts_stamp)
    dir.create(file.path(RUN_DIR, "fused"), recursive = TRUE, showWarnings = FALSE)
    log_dir <- file.path(RUN_DIR, "logs")
    dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
    
    # models/main directory
    MODELS_DIR_MAIN <- file.path(RUN_DIR, "models", "main")
    dir.create(MODELS_DIR_MAIN, recursive = TRUE, showWarnings = FALSE)
    
    TARGET_METRIC <- get0("metric_name", ifnotfound = get0("TARGET_METRIC", ifnotfound = "accuracy", inherits = TRUE), inherits = TRUE)
    num_temp_iterations <- as.integer(num_temp_iterations %||% 0L)
    
    total_seeds_chr <- as.character(length(seeds))
    agg_pred_file    <- file.path(RUN_DIR, sprintf("agg_predictions_test__%s_seeds_%s.rds", total_seeds_chr, ts_stamp))
    agg_metrics_file <- file.path(RUN_DIR, sprintf("agg_metrics_test__%s_seeds_%s.rds",     total_seeds_chr, ts_stamp))
    
    ## Pre-create train/val metrics file in the run folder
    out_path_train <- file.path(RUN_DIR, sprintf("Ensembles_Train_Acc_Val_Metrics_%s_seeds_%s.rds",
                                                 length(seeds), ts_stamp))
    saveRDS(data.frame(), out_path_train)
    
    row_ptr <- 0L
    for (i in seq_along(seeds)) {
      s <- seeds[i]
      set.seed(s)
      cat(sprintf("[ENSEMBLE] Seed %d/%d\n", s, length(seeds)))
      
      ## reset metadata
      vars <- grep("^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
                   ls(.GlobalEnv), value = TRUE)
      if (length(vars)) rm(list = vars, envir = .GlobalEnv)
      if (exists("ensembles", inherits = TRUE)) try(rm(ensembles, inherits = TRUE), silent = TRUE)
      
      ## tables for logs
      ensembles <- list(
        main_ensemble = list(), temp_ensemble = list(),
        tables = list(
          main_log = data.frame(iteration=integer(), phase=character(), slot=integer(),
                                serial=character(), metric_name=character(),
                                metric_value=numeric(), message=character(),
                                timestamp=as.POSIXct(character()), stringsAsFactors=FALSE),
          movement_log = data.frame(iteration=integer(), phase=character(), slot=integer(),
                                    role=character(), serial=character(),
                                    metric_name=character(), metric_value=numeric(),
                                    message=character(), timestamp=as.POSIXct(character()),
                                    stringsAsFactors=FALSE),
          change_log = data.frame(iteration=integer(), role=character(), serial=character(),
                                  metric_name=character(), metric_value=numeric(),
                                  message=character(), timestamp=as.POSIXct(character()),
                                  stringsAsFactors=FALSE)
        )
      )
      
      ## MAIN
      main_model <- DDESONN$new(
        num_networks    = max(1L, as.integer(num_networks)),
        input_size      = input_size, hidden_sizes = hidden_sizes, output_size = output_size,
        N = N, lambda = lambda, ensemble_number = 1L, ensembles = ensembles,
        ML_NN = ML_NN, activation_functions=activation_functions, activation_functions_predict=activation_functions_predict, init_method = init_method, custom_scale = custom_scale
      )
      
      model_results_main <<- main_model$train(
        Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
        lr_min=lr_min, ensemble_number=1L, num_epochs=num_epochs, self_org=self_org,
        threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns, CLASSIFICATION_MODE=CLASSIFICATION_MODE,
        activation_functions=activation_functions, activation_functions_predict=activation_functions_predict,
        dropout_rates=dropout_rates, optimizer=optimizer,
        beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
        batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
        epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
        shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights, preprocessScaledData=preprocessScaledData,
        X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
        train=train, grouped_metrics=grouped_metrics, viewTables=viewTables, verbose=verbose
      )
      ensembles$main_ensemble[[1]] <- main_model
      
      ## STAMP MAIN META
      best_train_acc_ret     <- try(model_results_main$predicted_outputAndTime$best_train_acc,           silent=TRUE); if (inherits(best_train_acc_ret,"try-error")) best_train_acc_ret <- NA_real_
      best_epoch_train_ret   <- try(model_results_main$predicted_outputAndTime$best_epoch_train,         silent=TRUE); if (inherits(best_epoch_train_ret,"try-error")) best_epoch_train_ret <- NA_integer_
      best_val_acc_ret       <- try(model_results_main$predicted_outputAndTime$best_val_acc,             silent=TRUE); if (inherits(best_val_acc_ret,"try-error")) best_val_acc_ret <- NA_real_
      best_val_epoch_ret     <- try(model_results_main$predicted_outputAndTime$best_val_epoch,           silent=TRUE); if (inherits(best_val_epoch_ret,"try-error")) best_val_epoch_ret <- NA_integer_
      best_val_pred_time_ret <- try(model_results_main$predicted_outputAndTime$best_val_prediction_time, silent=TRUE); if (inherits(best_val_pred_time_ret,"try-error")) best_val_pred_time_ret <- NA_real_
      
      K <- max(1L, as.integer(num_networks))
      for (k in seq_len(K)) {
        mvar <- main_meta_var(k)
        if (!exists(mvar, envir = .GlobalEnv)) next
        md <- get(mvar, envir = .GlobalEnv)
        
        md$best_train_acc           <- .scalar_num(md$best_train_acc           %||% best_train_acc_ret,     idx = k)
        md$best_epoch_train         <- as.integer(.scalar_num(md$best_epoch_train %||% best_epoch_train_ret, idx = k))
        md$best_val_acc             <- .scalar_num(md$best_val_acc             %||% best_val_acc_ret,       idx = k)
        md$best_val_epoch           <- as.integer(.scalar_num(md$best_val_epoch %||% best_val_epoch_ret,    idx = k))
        md$best_val_prediction_time <- .scalar_num(md$best_val_prediction_time %||% best_val_pred_time_ret, idx = k)
        
        ## [FIX] Attach callable predictor + context (mirrors single-run)
        slot_obj <- try(main_model$ensemble[[k]], silent = TRUE)
        if (!inherits(slot_obj, "try-error") && !is.null(slot_obj)) {
          md$predictor    <- slot_obj
          md$predictor_fn <- function(X, ...) slot_obj$predict(X, ...)
          if (is.null(md$feature_names) || !length(md$feature_names)) md$feature_names <- colnames(X)
          md$X_train      <- X;              md$y_train      <- y
          md$X_validation <- X_validation;   md$y_validation <- y_validation
          md$X_test       <- get0("X_test", inherits = TRUE, ifnotfound = NULL)
          md$y_test       <- get0("y_test", inherits = TRUE, ifnotfound = NULL)
        }
        
        assign(mvar, md, envir = .GlobalEnv)
        cat(sprintf("[STAMPED][MAIN] slot=%d best_val_acc=%s\n", k, as.character(md$best_val_acc)))
        
        # persist MAIN metadata (seed-tagged)
        saveRDS(md, file.path(MODELS_DIR_MAIN, sprintf("%s_%s_seed%s.rds", mvar, ts_stamp, as.character(s))))
      }
      
      ## === MAIN snapshot for Scenario C (no prune/add) → persist ONLY main_log ===
      if (num_temp_iterations == 0L) {
        ts_iter <- Sys.time()
        main_only <- snapshot_main_serials_meta()
        if (length(main_only)) {
          get_metric_by_serial_local <- function(serial, metric_name) {
            vars <- grep("^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
                         ls(.GlobalEnv), value = TRUE)
            for (v in vars) {
              md <- get(v, envir = .GlobalEnv)
              if (identical(as.character(md$model_serial_num %||% NA_character_), as.character(serial))) {
                val <- tryCatch(md$performance_metric[[metric_name]], error=function(e) NULL)
                if (is.null(val)) val <- tryCatch(md$relevance_metric[[metric_name]], error=function(e) NULL)
                vn <- suppressWarnings(as.numeric(val))
                return(if (length(vn) && is.finite(vn[1])) vn[1] else NA_real_)
              }
            }
            NA_real_
          }
          vals_only <- vapply(main_only, get_metric_by_serial_local, numeric(1), TARGET_METRIC)
          rows_only <- data.frame(
            iteration=NA_integer_, phase="main_only", slot=seq_along(main_only),
            serial=as.character(main_only), metric_name=TARGET_METRIC,
            metric_value=as.numeric(vals_only), message="", timestamp=ts_iter,
            stringsAsFactors=FALSE
          )
          ensembles$tables$main_log <- rbind(ensembles$tables$main_log, rows_only)
          ml_path_rds <- file.path(log_dir, sprintf("main_log_run%03d_seed%s_%s.rds", i, s, ts_stamp))
          saveRDS(ensembles$tables$main_log, ml_path_rds)
          cat("[LOGS] main_log (Scenario C) saved → ", ml_path_rds, "\n", sep = "")
        }
      }
      
      ## Scenario C test + fusion
      if (num_temp_iterations == 0L && isTRUE(test)) {
        for (k in seq_len(K)) {
          ENV_META_NAME <- resolve_env_meta(k, "main", 1L)
          DDESONN_predict_eval(
            LOAD_FROM_RDS = FALSE, ENV_META_NAME = ENV_META_NAME, INPUT_SPLIT = "test",
            CLASSIFICATION_MODE = CLASSIFICATION_MODE, RUN_INDEX = i, SEED = s,
            OUTPUT_DIR = RUN_DIR, SAVE_METRICS_RDS = TRUE,      # <— [FIX-OPTIONAL] match single-run
            METRICS_PREFIX = "metrics_test",
            AGG_PREDICTIONS_FILE = agg_pred_file, AGG_METRICS_FILE = agg_metrics_file,
            MODEL_SLOT = k
          )
        }
        ## >>> FIX: normalize agg file so fuser never sees "no entries"
        .fix_agg_layout_for_fuser(agg_pred_file, run_index = i, seed = s, split = "test", classification_mode = CLASSIFICATION_MODE)
        
        ## >>> NEW: post-filter ensemble TEST agg metrics
        if (file.exists(agg_metrics_file)) {
          df <- try(readRDS(agg_metrics_file), silent = TRUE)
          if (!inherits(df, "try-error") && is.data.frame(df) && NROW(df)) {
            nms <- names(df)
            
            id_cols <- c("run_index","seed","model_slot","MODEL_SLOT","split","mode","K","ts","ts_stamp")
            flat_names <- c(
              "quantization_error","topographic_error","clustering_quality_db",
              "MSE","MAE","RMSE","R2","MAPE","SMAPE","WMAPE","MASE",
              "accuracy","precision","recall","f1","f1_score","balanced_accuracy",
              "specificity","sensitivity","auc","logloss","brier",
              "confusion_matrix.TP","confusion_matrix.FP","confusion_matrix.TN","confusion_matrix.FN",
              "generalization_ability","speed","speed_learn1","speed_learn2",
              "memory_usage","robustness","hit_rate","ndcg","diversity","serendipity"
            )
            keep_regexes <- c(
              "^accuracy_precision_recall_f1_tuned\\.(accuracy|precision|recall|f1)$",
              "^accuracy_precision_recall_f1_tuned\\.confusion_matrix\\.(TP|FP|TN|FN)$",
              "^(performance_metric|relevance_metric)\\.(accuracy|precision|recall|f1|f1_score|auc|balanced_accuracy|specificity|sensitivity|logloss|brier|MSE|MAE|RMSE|R2|MAPE|SMAPE|WMAPE|MASE|hit_rate|ndcg|diversity|serendipity)$",
              "^(performance_metric|relevance_metric)\\.(quantization_error|topographic_error|clustering_quality_db|generalization_ability|speed|speed_learn1|speed_learn2|memory_usage|robustness)$"
            )
            
            drop_mask <- grepl("^(roc_.*_points|pr_.*_points|calibration_bins|lift_table|gain_table)$", nms, perl = TRUE)
            is_scalar_col <- vapply(df, function(x) !is.list(x), logical(1))
            
            keep_mask <-
              (nms %in% id_cols) |
              (nms %in% flat_names) |
              Reduce(`|`, lapply(keep_regexes, function(rx) grepl(rx, nms, perl = TRUE)))
            
            keep_nms <- nms[ keep_mask & !drop_mask & is_scalar_col ]
            df <- df[, intersect(keep_nms, names(df)), drop = FALSE]
            
            if (!"run_index"  %in% names(df) && "RUN_INDEX"  %in% names(df)) df$run_index  <- suppressWarnings(as.integer(df$RUN_INDEX))
            if (!"seed"       %in% names(df) && "SEED"       %in% names(df)) df$seed       <- suppressWarnings(as.integer(df$SEED))
            if (!"model_slot" %in% names(df) && "MODEL_SLOT" %in% names(df)) df$model_slot <- suppressWarnings(as.integer(df$MODEL_SLOT))
            if ("model_slot" %in% names(df)) df$model_slot <- suppressWarnings(as.integer(df$model_slot))
            if ("split"      %in% names(df)) df$split      <- as.character(df$split)
            
            names(df) <- sub("^(performance_metric|relevance_metric)\\.", "", names(df))
            
            if (!("accuracy" %in% names(df)))  df$accuracy  <- NA_real_
            if (!("precision" %in% names(df))) df$precision <- NA_real_
            if (!("recall" %in% names(df)))    df$recall    <- NA_real_
            if (!("f1" %in% names(df)))        df$f1        <- NA_real_
            if (!("f1_score" %in% names(df)))  df$f1_score  <- NA_real_
            
            if ("accuracy_precision_recall_f1_tuned.accuracy" %in% names(df)) {
              na_idx <- is.na(suppressWarnings(as.numeric(df$accuracy)))
              if (any(na_idx)) df$accuracy[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.accuracy"]]))[na_idx]
            }
            if ("accuracy_precision_recall_f1_tuned.precision" %in% names(df)) {
              na_idx <- is.na(suppressWarnings(as.numeric(df$precision)))
              if (any(na_idx)) df$precision[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.precision"]]))[na_idx]
            }
            if ("accuracy_precision_recall_f1_tuned.recall" %in% names(df)) {
              na_idx <- is.na(suppressWarnings(as.numeric(df$recall)))
              if (any(na_idx)) df$recall[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.recall"]]))[na_idx]
            }
            if ("accuracy_precision_recall_f1_tuned.f1" %in% names(df)) {
              na_idx <- is.na(suppressWarnings(as.numeric(df$f1)))
              if (any(na_idx)) df$f1[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.f1"]]))[na_idx]
            }
            na_f1s <- is.na(suppressWarnings(as.numeric(df$f1_score)))
            if (any(na_f1s)) df$f1_score[na_f1s] <- suppressWarnings(as.numeric(df$f1))[na_f1s]
            
            has_tp <- "confusion_matrix.TP" %in% names(df)
            has_fp <- "confusion_matrix.FP" %in% names(df)
            has_fn <- "confusion_matrix.FN" %in% names(df)
            
            if (has_tp && has_fn) {
              TP <- suppressWarnings(as.numeric(df[["confusion_matrix.TP"]]))
              FN <- suppressWarnings(as.numeric(df[["confusion_matrix.FN"]]))
              rec_na <- is.na(suppressWarnings(as.numeric(df$recall)))
              if (any(rec_na)) {
                denom <- TP + FN
                rec_calc <- ifelse(denom > 0, TP / denom, NA_real_)
                df$recall[rec_na] <- rec_calc[rec_na]
              }
            }
            
            if (any(is.na(suppressWarnings(as.numeric(df$f1))))) {
              prec_vec <- suppressWarnings(as.numeric(df$precision))
              if (all(is.na(prec_vec)) && ("accuracy_precision_recall_f1_tuned.precision" %in% names(df))) {
                prec_vec <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.precision"]]))
                na_prec <- is.na(suppressWarnings(as.numeric(df$precision)))
                if (any(na_prec)) df$precision[na_prec] <- prec_vec[na_prec]
              }
              rec_vec <- suppressWarnings(as.numeric(df$recall))
              f1_calc <- ifelse((prec_vec + rec_vec) > 0, 2 * prec_vec * rec_vec / (prec_vec + rec_vec), NA_real_)
              na_f1 <- is.na(suppressWarnings(as.numeric(df$f1)))
              if (any(na_f1)) df$f1[na_f1] <- f1_calc[na_f1]
              na_f1s <- is.na(suppressWarnings(as.numeric(df$f1_score)))
              if (any(na_f1s)) df$f1_score[na_f1s] <- df$f1[na_f1s]
            }
            
            id_order <- c("run_index","seed","model_slot","split","mode","K","ts","ts_stamp")
            metric_order <- c(
              "quantization_error","topographic_error","clustering_quality_db",
              "MSE","MAE","RMSE","R2","MAPE","SMAPE","WMAPE","MASE",
              "accuracy","precision","recall","f1","f1_score","balanced_accuracy",
              "specificity","sensitivity","auc","logloss","brier",
              "accuracy_precision_recall_f1_tuned.accuracy",
              "accuracy_precision_recall_f1_tuned.precision",
              "accuracy_precision_recall_f1_tuned.recall",
              "accuracy_precision_recall_f1_tuned.f1",
              "accuracy_precision_recall_f1_tuned.confusion_matrix.TP",
              "accuracy_precision_recall_f1_tuned.confusion_matrix.FP",
              "accuracy_precision_recall_f1_tuned.confusion_matrix.TN",
              "accuracy_precision_recall_f1_tuned.confusion_matrix.FN",
              "confusion_matrix.TP","confusion_matrix.FP","confusion_matrix.TN","confusion_matrix.FN",
              "generalization_ability","speed","speed_learn1","speed_learn2",
              "memory_usage","robustness","hit_rate","ndcg","diversity","serendipity"
            )
            
            prefer <- c(id_order, metric_order)
            have_pref <- intersect(prefer, names(df))
            rest <- setdiff(names(df), have_pref)
            df <- df[, c(have_pref, rest), drop = FALSE]
            
            saveRDS(df, agg_metrics_file)
          }
        }
        
        if (num_networks > 1L) {
          yi <- get0("y_test", inherits=TRUE, ifnotfound=NULL); stopifnot(!is.null(yi))
          fused <- DDESONN_fuse_from_agg(
            AGG_PREDICTIONS_FILE = agg_pred_file, RUN_INDEX = i, SEED = s,
            y_true = yi, methods = c("avg","wavg","vote_soft","vote_hard"),
            weight_column = "tuned_f1", use_tuned_threshold_for_vote = TRUE,
            default_threshold = 0.5, classification_mode = CLASSIFICATION_MODE
          )
          fused_path <- file.path(RUN_DIR, "fused", sprintf("fused_run%03d_seed%s_%s.rds", i, s, ts_stamp))
          saveRDS(fused, fused_path)
          cat("[SAVE] fused → ", fused_path, "\n", sep = "")
        }
      }
      
      ## Scenario D prune/add with logs
      if (num_temp_iterations > 0L) {
        for (j in seq_len(num_temp_iterations)) {
          ensembles$temp_ensemble <- vector("list", 1L)
          temp_model <<- DDESONN$new(
            num_networks=max(1L, as.integer(num_networks)), input_size=input_size,
            hidden_sizes=hidden_sizes, output_size=output_size, N=N, lambda=lambda,
            ensemble_number = j + 1L, ensembles = ensembles, ML_NN=ML_NN, activation_functions=activation_functions, activation_functions_predict=activation_functions_predict, method=init_method, custom_scale=custom_scale
          )
          ensembles$temp_ensemble[[1]] <- temp_model
          
          model_results_temp <<- temp_model$train(
            Rdata=X, labels=y, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch,
            lr_min=lr_min, ensemble_number=j+1L, num_epochs=num_epochs, self_org=self_org,
            threshold=threshold, reg_type=reg_type, numeric_columns=numeric_columns, CLASSIFICATION_MODE=CLASSIFICATION_MODE,
            activation_functions=activation_functions, activation_functions_predict=activation_functions_predict,
            dropout_rates=dropout_rates, optimizer=optimizer,
            beta1=beta1, beta2=beta2, epsilon=epsilon, lookahead_step=lookahead_step,
            batch_normalize_data=batch_normalize_data, gamma_bn=gamma_bn, beta_bn=beta_bn,
            epsilon_bn=epsilon_bn, momentum_bn=momentum_bn, is_training_bn=is_training_bn,
            shuffle_bn=shuffle_bn, loss_type=loss_type, sample_weights=sample_weights, preprocessScaledData=preprocessScaledData,
            X_validation=X_validation, y_validation=y_validation, validation_metrics=validation_metrics, threshold_function=threshold_function, ML_NN=ML_NN,
            train=train, grouped_metrics=grouped_metrics, viewTables=viewTables, verbose=verbose
          )
          
          #  models/temp_eXX directory for this iteration
          MODELS_DIR_TEMP <- file.path(RUN_DIR, "models", sprintf("temp_e%02d", j + 1L))
          dir.create(MODELS_DIR_TEMP, recursive = TRUE, showWarnings = FALSE)
          
          ## MAIN snapshot BEFORE prune/add → main_log
          ts_iter <- Sys.time()
          main_before <- snapshot_main_serials_meta()
          if (length(main_before)) {
            get_metric_by_serial_local <- function(serial, metric_name) {
              vars <- grep("^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
                           ls(.GlobalEnv), value = TRUE)
              for (v in vars) {
                md <- get(v, envir = .GlobalEnv)
                if (identical(as.character(md$model_serial_num %||% NA_character_), as.character(serial))) {
                  val <- tryCatch(md$performance_metric[[metric_name]], error=function(e) NULL)
                  if (is.null(val)) val <- tryCatch(md$relevance_metric[[metric_name]], error=function(e) NULL)
                  vn <- suppressWarnings(as.numeric(val))
                  return(if (length(vn) && is.finite(vn[1])) vn[1] else NA_real_)
                }
              }
              NA_real_
            }
            vals_before <- vapply(main_before, get_metric_by_serial_local, numeric(1), TARGET_METRIC)
            rows_before <- data.frame(
              iteration=j, phase="main_before", slot=seq_along(main_before),
              serial=as.character(main_before), metric_name=TARGET_METRIC,
              metric_value=as.numeric(vals_before), message="", timestamp=ts_iter,
              stringsAsFactors=FALSE
            )
            ensembles$tables$main_log <- rbind(ensembles$tables$main_log, rows_before)
            ml_path_rds <- file.path(log_dir, sprintf("main_log_run%03d_seed%s_%s.rds", i, s, ts_stamp))
            saveRDS(ensembles$tables$main_log, ml_path_rds)
            cat("[LOGS] main_log (before) saved → ", ml_path_rds, "\n", sep = "")
          }
          
          ## STAMP TEMP META
          best_train_acc_tmp     <- try(model_results_temp$predicted_outputAndTime$best_train_acc,           silent=TRUE); if (inherits(best_train_acc_tmp,"try-error")) best_train_acc_tmp <- NA_real_
          best_epoch_train_tmp   <- try(model_results_temp$predicted_outputAndTime$best_epoch_train,         silent=TRUE); if (inherits(best_epoch_train_tmp,"try-error")) best_epoch_train_tmp <- NA_integer_
          best_val_acc_tmp       <- try(model_results_temp$predicted_outputAndTime$best_val_acc,             silent=TRUE); if (inherits(best_val_acc_tmp,"try-error")) best_val_acc_tmp <- NA_real_
          best_val_epoch_tmp     <- try(model_results_temp$predicted_outputAndTime$best_val_epoch,           silent=TRUE); if (inherits(best_val_epoch_tmp,"try-error")) best_val_epoch_tmp <- NA_integer_
          best_val_pred_time_tmp <- try(model_results_temp$predicted_outputAndTime$best_val_prediction_time, silent=TRUE); if (inherits(best_val_pred_time_tmp,"try-error")) best_val_pred_time_tmp <- NA_real_

          for (k in seq_len(K)) {
            tvar <- temp_meta_var(j + 1L, k)
            if (!exists(tvar, envir = .GlobalEnv)) next
            tmd <- get(tvar, envir = .GlobalEnv)
            tmd$best_train_acc           <- .scalar_num(tmd$best_train_acc           %||% best_train_acc_tmp,     idx = k)
            tmd$best_epoch_train         <- as.integer(.scalar_num(tmd$best_epoch_train %||% best_epoch_train_tmp, idx = k))
            tmd$best_val_acc             <- .scalar_num(tmd$best_val_acc             %||% best_val_acc_tmp,       idx = k)
            tmd$best_val_epoch           <- as.integer(.scalar_num(tmd$best_val_epoch %||% best_val_epoch_tmp,    idx = k))
            tmd$best_val_prediction_time <- .scalar_num(tmd$best_val_prediction_time %||% best_val_pred_time_tmp, idx = k)
            assign(tvar, tmd, envir = .GlobalEnv)
            cat(sprintf("[STAMPED][TEMP e=%d] slot=%d best_val_acc=%s\n", j+1L, k, as.character(tmd$best_val_acc)))
            
            saveRDS(tmd, file.path( MODELS_DIR_TEMP, sprintf("%s_%s_seed%s.rds", tvar, ts_stamp, as.character(s))))
          }
          
          pruned <- prune_network_from_ensemble(
            ensembles,
            get0("metric_name", ifnotfound = get0("TARGET_METRIC", ifnotfound = "accuracy", inherits = TRUE), inherits = TRUE)
          )
          if (!is.null(pruned)) {
            added <- add_network_to_ensemble(
              ensembles               = pruned$updated_ensembles,
              target_metric_name_best = get0("metric_name", ifnotfound = get0("TARGET_METRIC", ifnotfound = "accuracy", inherits = TRUE), inherits = TRUE),
              removed_network         = pruned$removed_network,
              ensemble_number         = j,
              worst_model_index       = pruned$worst_model_index,
              removed_serial          = pruned$worst_serial,
              removed_value           = pruned$worst_value
            )
            ensembles <- added$updated_ensembles
            
            ## [FIX] After any replace, refresh MAIN predictors/context (hooks can change)
            for (kk in seq_len(K)) {
              mvar <- main_meta_var(kk)
              if (!exists(mvar, envir = .GlobalEnv)) next
              md   <- get(mvar, envir = .GlobalEnv)
              slot_obj <- try(ensembles$main_ensemble[[1]]$ensemble[[kk]], silent = TRUE)
              if (!inherits(slot_obj, "try-error") && !is.null(slot_obj)) {
                md$predictor    <- slot_obj
                md$predictor_fn <- function(X, ...) slot_obj$predict(X, ...)
                if (is.null(md$feature_names) || !length(md$feature_names)) md$feature_names <- colnames(X)
                md$X_train      <- X;              md$y_train      <- y
                md$X_validation <- X_validation;   md$y_validation <- y_validation
                md$X_test       <- get0("X_test", inherits = TRUE, ifnotfound = NULL)
                md$y_test       <- get0("y_test", inherits = TRUE, ifnotfound = NULL)
                assign(mvar, md, envir = .GlobalEnv)
                saveRDS(md, file.path(MODELS_DIR_MAIN, sprintf("%s_%s_seed%s.rds", mvar, ts_stamp, as.character(s))))
              }
            }
            
            ## movement/change logs
            ts_iter <- Sys.time()
            if (is_real_serial(pruned$worst_serial)) {
              rrow <- data.frame(
                iteration=j, phase="removed", slot=pruned$worst_model_index, role="removed",
                serial=pruned$worst_serial, metric_name=TARGET_METRIC,
                metric_value=as.numeric(pruned$worst_value),
                message=if (!is.null(added$worst_slot)) sprintf("%s replaced", pruned$worst_serial) else "removed (no replacement)",
                timestamp=ts_iter, stringsAsFactors=FALSE
              )
              ensembles$tables$movement_log <- rbind(ensembles$tables$movement_log, rrow)
            }
            if (!is.null(added$worst_slot)) {
              arow <- data.frame(
                iteration=j, phase="added", slot=added$worst_slot, role="added",
                serial=get0(main_meta_var(added$worst_slot), ifnotfound=list(model_serial_num=NA), inherits=TRUE)$model_serial_num %||% NA_character_,
                metric_name=TARGET_METRIC, metric_value=NA_real_,
                message="candidate moved into main", timestamp=ts_iter, stringsAsFactors=FALSE
              )
              ensembles$tables$movement_log <- rbind(ensembles$tables$movement_log, arow)
            }
            mv_path_rds <- file.path(log_dir, sprintf("movement_log_run%03d_seed%s_%s.rds", i, s, ts_stamp))
            saveRDS(ensembles$tables$movement_log, mv_path_rds)
            cat("[LOGS] movement_log saved → ", mv_path_rds, "\n", sep = "")
            
            ch_path_rds <- file.path(log_dir, sprintf("change_log_run%03d_seed%s_%s.rds", i, s, ts_stamp))
            saveRDS(ensembles$tables$change_log, ch_path_rds)
            cat("[LOGS] change_log saved → ", ch_path_rds, "\n", sep = "")
          }
          
          ## MAIN snapshot AFTER prune/add → main_log
          ts_iter <- Sys.time()
          main_after <- snapshot_main_serials_meta()
          if (length(main_after)) {
            get_metric_by_serial_local <- function(serial, metric_name) {
              vars <- grep("^(Ensemble_Main_(0|1)_model_\\d+_metadata|Ensemble_Temp_\\d+_model_\\d+_metadata)$",
                           ls(.GlobalEnv), value = TRUE)
              for (v in vars) {
                md <- get(v, envir = .GlobalEnv)
                if (identical(as.character(md$model_serial_num %||% NA_character_), as.character(serial))) {
                  val <- tryCatch(md$performance_metric[[metric_name]], error=function(e) NULL)
                  if (is.null(val)) val <- tryCatch(md$relevance_metric[[metric_name]], error=function(e) NULL)
                  vn <- suppressWarnings(as.numeric(val))
                  return(if (length(vn) && is.finite(vn[1])) vn[1] else NA_real_)
                }
              }
              NA_real_
            }
            vals_after <- vapply(main_after, get_metric_by_serial_local, numeric(1), TARGET_METRIC)
            rows_after <- data.frame(
              iteration=j, phase="main_after", slot=seq_along(main_after),
              serial=as.character(main_after), metric_name=TARGET_METRIC,
              metric_value=as.numeric(vals_after), message="", timestamp=ts_iter,
              stringsAsFactors=FALSE
            )
            ensembles$tables$main_log <- rbind(ensembles$tables$main_log, rows_after)
            ml_path_rds <- file.path(log_dir, sprintf("main_log_run%03d_seed%s_%s.rds", i, s, ts_stamp))
            saveRDS(ensembles$tables$main_log, ml_path_rds)
            cat("[LOGS] main_log (after) saved → ", ml_path_rds, "\n", sep = "")
          }
        } ## end temp iterations
        
        ## Scenario D test + fusion
        if (isTRUE(test)) {
          for (k in seq_len(K)) {
            ENV_META_NAME <- resolve_env_meta(k, "main", num_temp_iterations)
            DDESONN_predict_eval(
              LOAD_FROM_RDS = FALSE, ENV_META_NAME = ENV_META_NAME, INPUT_SPLIT = "test",
              CLASSIFICATION_MODE = CLASSIFICATION_MODE, RUN_INDEX = i, SEED = s,
              OUTPUT_DIR = RUN_DIR, SAVE_METRICS_RDS = TRUE,   # <— [FIX-OPTIONAL] per-slot files
              METRICS_PREFIX = "metrics_test",
              AGG_PREDICTIONS_FILE = agg_pred_file, AGG_METRICS_FILE = agg_metrics_file,
              MODEL_SLOT = k
            )
          }
          ## >>> FIX: normalize agg file so fuser never sees "no entries"
          .fix_agg_layout_for_fuser(agg_pred_file, run_index = i, seed = s, split = "test", classification_mode = CLASSIFICATION_MODE)
          
          ## >>> NEW: post-filter ensemble TEST agg metrics (same as Scenario C)
          if (file.exists(agg_metrics_file)) {
            df <- try(readRDS(agg_metrics_file), silent = TRUE)
            if (!inherits(df, "try-error") && is.data.frame(df) && NROW(df)) {
              nms <- names(df)
              
              id_cols <- c("run_index","seed","model_slot","MODEL_SLOT","split","mode","K","ts","ts_stamp")
              flat_names <- c(
                "quantization_error","topographic_error","clustering_quality_db",
                "MSE","MAE","RMSE","R2","MAPE","SMAPE","WMAPE","MASE",
                "accuracy","precision","recall","f1","f1_score","balanced_accuracy",
                "specificity","sensitivity","auc","logloss","brier",
                "confusion_matrix.TP","confusion_matrix.FP","confusion_matrix.TN","confusion_matrix.FN",
                "generalization_ability","speed","speed_learn1","speed_learn2",
                "memory_usage","robustness","hit_rate","ndcg","diversity","serendipity"
              )
              keep_regexes <- c(
                "^accuracy_precision_recall_f1_tuned\\.(accuracy|precision|recall|f1)$",
                "^accuracy_precision_recall_f1_tuned\\.confusion_matrix\\.(TP|FP|TN|FN)$",
                "^(performance_metric|relevance_metric)\\.(accuracy|precision|recall|f1|f1_score|auc|balanced_accuracy|specificity|sensitivity|logloss|brier|MSE|MAE|RMSE|R2|MAPE|SMAPE|WMAPE|MASE|hit_rate|ndcg|diversity|serendipity)$",
                "^(performance_metric|relevance_metric)\\.(quantization_error|topographic_error|clustering_quality_db|generalization_ability|speed|speed_learn1|speed_learn2|memory_usage|robustness)$"
              )
              
              drop_mask <- grepl("^(roc_.*_points|pr_.*_points|calibration_bins|lift_table|gain_table)$", nms, perl = TRUE)
              is_scalar_col <- vapply(df, function(x) !is.list(x), logical(1))
              
              keep_mask <-
                (nms %in% id_cols) |
                (nms %in% flat_names) |
                Reduce(`|`, lapply(keep_regexes, function(rx) grepl(rx, nms, perl = TRUE)))
              
              keep_nms <- nms[ keep_mask & !drop_mask & is_scalar_col ]
              df <- df[, intersect(keep_nms, names(df)), drop = FALSE]
              
              if (!"run_index"  %in% names(df) && "RUN_INDEX"  %in% names(df)) df$run_index  <- suppressWarnings(as.integer(df$RUN_INDEX))
              if (!"seed"       %in% names(df) && "SEED"       %in% names(df)) df$seed       <- suppressWarnings(as.integer(df$SEED))
              if (!"model_slot" %in% names(df) && "MODEL_SLOT" %in% names(df)) df$model_slot <- suppressWarnings(as.integer(df$MODEL_SLOT))
              if ("model_slot" %in% names(df)) df$model_slot <- suppressWarnings(as.integer(df$model_slot))
              if ("split"      %in% names(df)) df$split      <- as.character(df$split)
              
              names(df) <- sub("^(performance_metric|relevance_metric)\\.", "", names(df))
              
              if (!("accuracy" %in% names(df)))  df$accuracy  <- NA_real_
              if (!("precision" %in% names(df))) df$precision <- NA_real_
              if (!("recall" %in% names(df)))    df$recall    <- NA_real_
              if (!("f1" %in% names(df)))        df$f1        <- NA_real_
              if (!("f1_score" %in% names(df)))  df$f1_score  <- NA_real_
              
              if ("accuracy_precision_recall_f1_tuned.accuracy" %in% names(df)) {
                na_idx <- is.na(suppressWarnings(as.numeric(df$accuracy)))
                if (any(na_idx)) df$accuracy[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.accuracy"]]))[na_idx]
              }
              if ("accuracy_precision_recall_f1_tuned.precision" %in% names(df)) {
                na_idx <- is.na(suppressWarnings(as.numeric(df$precision)))
                if (any(na_idx)) df$precision[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.precision"]]))[na_idx]
              }
              if ("accuracy_precision_recall_f1_tuned.recall" %in% names(df)) {
                na_idx <- is.na(suppressWarnings(as.numeric(df$recall)))
                if (any(na_idx)) df$recall[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.recall"]]))[na_idx]
              }
              if ("accuracy_precision_recall_f1_tuned.f1" %in% names(df)) {
                na_idx <- is.na(suppressWarnings(as.numeric(df$f1)))
                if (any(na_idx)) df$f1[na_idx] <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.f1"]]))[na_idx]
              }
              na_f1s <- is.na(suppressWarnings(as.numeric(df$f1_score)))
              if (any(na_f1s)) df$f1_score[na_f1s] <- suppressWarnings(as.numeric(df$f1))[na_f1s]
              
              has_tp <- "confusion_matrix.TP" %in% names(df)
              has_fp <- "confusion_matrix.FP" %in% names(df)
              has_fn <- "confusion_matrix.FN" %in% names(df)
              
              if (has_tp && has_fn) {
                TP <- suppressWarnings(as.numeric(df[["confusion_matrix.TP"]]))
                FN <- suppressWarnings(as.numeric(df[["confusion_matrix.FN"]]))
                rec_na <- is.na(suppressWarnings(as.numeric(df$recall)))
                if (any(rec_na)) {
                  denom <- TP + FN
                  rec_calc <- ifelse(denom > 0, TP / denom, NA_real_)
                  df$recall[rec_na] <- rec_calc[rec_na]
                }
              }
              
              if (any(is.na(suppressWarnings(as.numeric(df$f1))))) {
                prec_vec <- suppressWarnings(as.numeric(df$precision))
                if (all(is.na(prec_vec)) && ("accuracy_precision_recall_f1_tuned.precision" %in% names(df))) {
                  prec_vec <- suppressWarnings(as.numeric(df[["accuracy_precision_recall_f1_tuned.precision"]]))
                  na_prec <- is.na(suppressWarnings(as.numeric(df$precision)))
                  if (any(na_prec)) df$precision[na_prec] <- prec_vec[na_prec]
                }
                rec_vec <- suppressWarnings(as.numeric(df$recall))
                f1_calc <- ifelse((prec_vec + rec_vec) > 0, 2 * prec_vec * rec_vec / (prec_vec + rec_vec), NA_real_)
                na_f1 <- is.na(suppressWarnings(as.numeric(df$f1)))
                if (any(na_f1)) df$f1[na_f1] <- f1_calc[na_f1]
                na_f1s <- is.na(suppressWarnings(as.numeric(df$f1_score)))
                if (any(na_f1s)) df$f1_score[na_f1s] <- df$f1[na_f1s]
              }
              
              id_order <- c("run_index","seed","model_slot","split","mode","K","ts","ts_stamp")
              metric_order <- c(
                "quantization_error","topographic_error","clustering_quality_db",
                "MSE","MAE","RMSE","R2","MAPE","SMAPE","WMAPE","MASE",
                "accuracy","precision","recall","f1","f1_score","balanced_accuracy",
                "specificity","sensitivity","auc","logloss","brier",
                "accuracy_precision_recall_f1_tuned.accuracy",
                "accuracy_precision_recall_f1_tuned.precision",
                "accuracy_precision_recall_f1_tuned.recall",
                "accuracy_precision_recall_f1_tuned.f1",
                "accuracy_precision_recall_f1_tuned.confusion_matrix.TP",
                "accuracy_precision_recall_f1_tuned.confusion_matrix.FP",
                "accuracy_precision_recall_f1_tuned.confusion_matrix.TN",
                "accuracy_precision_recall_f1_tuned.confusion_matrix.FN",
                "confusion_matrix.TP","confusion_matrix.FP","confusion_matrix.TN","confusion_matrix.FN",
                "generalization_ability","speed","speed_learn1","speed_learn2",
                "memory_usage","robustness","hit_rate","ndcg","diversity","serendipity"
              )
              
              prefer <- c(id_order, metric_order)
              have_pref <- intersect(prefer, names(df))
              rest <- setdiff(names(df), have_pref)
              df <- df[, c(have_pref, rest), drop = FALSE]
              
              saveRDS(df, agg_metrics_file)
            }
          }
          
          if (num_networks > 1L) {
            yi <- get0("y_test", inherits=TRUE, ifnotfound=NULL); stopifnot(!is.null(yi))
            fused <- DDESONN_fuse_from_agg(
              AGG_PREDICTIONS_FILE = agg_pred_file, RUN_INDEX = i, SEED = s,
              y_true = yi, methods = c("avg","wavg","vote_soft","vote_hard"),
              weight_column = "tuned_f1", use_tuned_threshold_for_vote = TRUE,
              default_threshold = 0.5, classification_mode = CLASSIFICATION_MODE
            )
            fused_path <- file.path(RUN_DIR, "fused",
                                    sprintf("fused_run%03d_seed%s_%s.rds", i, s, ts_stamp))
            saveRDS(fused, fused_path)
            cat("[SAVE] fused → ", fused_path, "\n", sep = "")
          }
        }
      }
      
      ## ==========================
      ## Per-SLOT rows for this seed
      ## ==========================
      for (k in seq_len(K)) {
        mvar <- main_meta_var(k)
        if (!exists(mvar, envir = .GlobalEnv)) next
        md <- get(mvar, envir = .GlobalEnv)
        
        flat <- tryCatch(
          rapply(list(performance_metric = md$performance_metric,
                      relevance_metric   = md$relevance_metric),
                 f = function(z) z, how = "unlist"),
          error = function(e) setNames(vector("list", 0L), character(0))
        )
        
        if (length(flat)) {
          L <- as.list(flat)
          flat <- flat[vapply(L, is.atomic, logical(1)) & lengths(L) == 1L]
        }
        nms <- names(flat)
        if (length(nms)) {
          drop <- grepl("custom_relative_error_binned", nms, fixed = TRUE) |
            grepl("grid_used", nms, fixed = TRUE) |
            grepl("(^|\\.)details(\\.|$)", nms)
          keep <- !drop                    # <-- keep NA values; do NOT filter them out
          flat <- flat[keep]; nms <- names(flat)
        }
        
        if (length(flat) == 0L) {
          row_df <- data.frame(run_index = i, seed = s, MODEL_SLOT = k, stringsAsFactors = FALSE)
        } else {
          out <- setNames(vector("list", length(flat)), nms)
          num <- suppressWarnings(as.numeric(flat))
          for (jj in seq_along(flat)) out[[jj]] <- if (!is.na(num[jj])) num[jj] else as.character(flat[[jj]])
          row_df <- as.data.frame(out, check.names = TRUE, stringsAsFactors = FALSE)
          row_df <- cbind(data.frame(run_index = i, seed = s, MODEL_SLOT = k, stringsAsFactors = FALSE), row_df)
        }
        
        row_df$serial     <- as.character(md$model_serial_num %||% NA_character_)
        row_df$model_name <- md$model_name %||% NA_character_
        
        get_num <- function(x) suppressWarnings(as.numeric(x))
        if (!("accuracy" %in% names(row_df)) || !is.finite(get_num(row_df$accuracy))) {
          if ("accuracy_precision_recall_f1_tuned.accuracy" %in% names(row_df)) {
            row_df$accuracy <- get_num(row_df[["accuracy_precision_recall_f1_tuned.accuracy"]])
          }
        }
        
        have_cm <- all(c("confusion_matrix.TP","confusion_matrix.FP","confusion_matrix.TN","confusion_matrix.FN") %in% names(row_df))
        if (!have_cm) {
          map_tuned_cm <- function(src_name, dst_name) {
            if (src_name %in% names(row_df) && !(dst_name %in% names(row_df))) {
              row_df[[dst_name]] <<- get_num(row_df[[src_name]])
            }
          }
          map_tuned_cm("accuracy_precision_recall_f1_tuned.confusion_matrix.TP", "confusion_matrix.TP")
          map_tuned_cm("accuracy_precision_recall_f1_tuned.confusion_matrix.FP", "confusion_matrix.FP")
          map_tuned_cm("accuracy_precision_recall_f1_tuned.confusion_matrix.TN", "confusion_matrix.TN")
          map_tuned_cm("accuracy_precision_recall_f1_tuned.confusion_matrix.FN", "confusion_matrix.FN")
        }
        
        row_df$best_train_acc           <- .scalar_num(md$best_train_acc)
        row_df$best_epoch_train         <- as.integer(.scalar_num(md$best_epoch_train))
        row_df$best_val_acc             <- .scalar_num(md$best_val_acc)
        row_df$best_val_epoch           <- as.integer(.scalar_num(md$best_val_epoch))
        row_df$best_val_prediction_time <- .scalar_num(md$best_val_prediction_time)
        
        row_ptr <- row_ptr + 1L
        per_slot_rows[[row_ptr]] <- row_df
      }
      
    } ## seeds
    
    ## ---- Aggregate: ONE ROW PER MODEL SLOT PER SEED ----
    if (length(per_slot_rows) == 0L) {
      results_table <- data.frame()
    } else {
      results_table <- per_slot_rows[[1]]
      if (length(per_slot_rows) > 1L) {
        for (k in 2:length(per_slot_rows)) {
          x <- results_table; y <- per_slot_rows[[k]]
          for (m in setdiff(names(y), names(x))) x[[m]] <- NA
          for (m in setdiff(names(x), names(y))) y[[m]] <- NA
          ord <- union(names(x), names(y))
          results_table <- rbind(x[, ord, drop = FALSE], y[, ord, drop = FALSE])
        }
      }
    }
    
    colnames(results_table) <- sub("^(performance_metric|relevance_metric)\\.", "", colnames(results_table))
    
    saveRDS(results_table, out_path_train)
    cat("Saved ENSEMBLE per-slot TRAIN metrics table to:", out_path_train,
        " | rows=", nrow(results_table), " cols=", ncol(results_table), "\n")
    
    ## Reindex AGG files
    for (agg_path in c(agg_metrics_file, agg_pred_file)) {
      if (!file.exists(agg_path)) next
      df <- try(readRDS(agg_path), silent = TRUE); if (inherits(df, "try-error")) next
      if (!is.data.frame(df) || !NROW(df)) next
      order_keys <- snapshot_main_serials_meta(); if (!length(order_keys)) next
      if (!("serial" %in% names(df))) df$serial <- NA_character_
      need <- is.na(df$serial) | !nzchar(df$serial)
      if (any(need)) {
        slot_col <- if ("MODEL_SLOT" %in% names(df)) "MODEL_SLOT" else if ("model" %in% names(df)) "model" else NA_character_
        if (!is.na(slot_col) && length(order_keys)) {
          idx <- which(need)
          slot_vals <- suppressWarnings(as.integer(df[[slot_col]]))
          ok  <- idx[slot_vals[idx] >= 1L & slot_vals[idx] <= length(order_keys)]
          if (length(ok)) df$serial[ok] <- order_keys[slot_vals[ok]]
        }
      }
      n <- NROW(df)
      ord_idx  <- match(df$serial, order_keys)
      tie_seed <- if ("SEED" %in% names(df)) df$SEED else rep(999999L, n)
      tie_slot <- if ("MODEL_SLOT" %in% names(df)) df$MODEL_SLOT else if ("model" %in% names(df)) suppressWarnings(as.integer(df$model)) else seq_len(n)
      o <- try(order(ord_idx, tie_seed, tie_slot, na.last = TRUE), silent = TRUE)
      if (inherits(o, "try-error") || length(o) != n) o <- seq_len(n)
      df <- df[o, , drop = FALSE]; df$run_index <- seq_len(nrow(df))
      saveRDS(df, agg_path)
      cat("Reindexed (by serial) AGG file:", agg_path, " | rows=", nrow(df), "\n")
    }
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
}





















saveToDisk <- FALSE

if (saveToDisk) {
  # ---- ensure output dir ----
  out_dir <- "artifacts_runs"
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  # ---- locate main ensemble (required) ----
  if (!exists("ensembles", inherits = TRUE) || is.null(ensembles$main_ensemble)) {
    stop("[saveToDisk] ensembles$main_ensemble is missing or NULL; cannot save.")
  }
  main_ensemble_obj <- ensembles$main_ensemble
  
  # ---- collect temp ensembles (optional) ----
  temp_list <- list()
  
  # 1) If your code maintains a list like ensembles$temp_ensembles, grab it
  if (!is.null(ensembles$temp_ensembles) && length(ensembles$temp_ensembles) > 0) {
    temp_list <- ensembles$temp_ensembles
  }
  
  # 2) Also sweep the global env for any temp_ensemble_<n> variables and add them
  temp_sym_names <- ls(envir = .GlobalEnv, pattern = "^temp_ensemble_\\d+$")
  if (length(temp_sym_names)) {
    for (nm in temp_sym_names) {
      obj <- get(nm, envir = .GlobalEnv)
      # avoid duplicate names from ensembles$temp_ensembles
      if (is.null(names(temp_list)) || !(nm %in% names(temp_list))) {
        temp_list[[nm]] <- obj
      }
    }
  }
  
  # ---- bundle payload ----
  ensemble_results <- list(
    main_ensemble  = main_ensemble_obj,
    temp_ensembles = temp_list  # may be an empty list, which is fine
  )
  
  # ---- filename logic (no overwrite) ----
  base_file_name <- file.path(out_dir, "ensemble_results")
  
  generate_new_file_name <- function(base_name) {
    i <- 1L
    candidate <- paste0(base_name, "_", i, ".rds")
    while (file.exists(candidate)) {
      i <- i + 1L
      candidate <- paste0(base_name, "_", i, ".rds")
    }
    candidate
  }
  
  file_name <- if (file.exists(paste0(base_file_name, ".rds"))) {
    generate_new_file_name(base_file_name)
  } else {
    paste0(base_file_name, ".rds")
  }
  
  # ---- save ----
  saveRDS(ensemble_results, file_name)
  cat("Data saved to file:", file_name, "\n")
  cat(sprintf("[saveToDisk] Included %d temp ensemble(s).\n", length(temp_list)))
}