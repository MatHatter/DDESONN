## ----setup, include=FALSE-----------------------------------------------------
is_check_env <- nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_"))
.vig_tmp_root <- file.path(tempdir(), "ddesonn-vig-s1-ensemble")
dir.create(.vig_tmp_root, recursive = TRUE, showWarnings = FALSE)
options(DDESONN_OUTPUT_ROOT = .vig_tmp_root)
Sys.setenv(DDESONN_ARTIFACTS_ROOT = .vig_tmp_root)
# ============================================================
# FILE: vignettes/plot-controls_scenario1_ensemble-runs_scenarioC-D.Rmd
# FULL FIXED — Scenario 1 — Ensemble Runs: Scenario C & D
# ============================================================

knitr::opts_chunk$set(
  echo = TRUE,
  message = TRUE,
  warning = FALSE,
  fig.width = 8,
  fig.height = 5,
  fig.retina = 2,                                                                   
  fig.align = "center",
  out.width = "900px",
  fig.path = file.path(.vig_tmp_root, "figures", ""),
  cache.path = file.path(.vig_tmp_root, "cache", "")
)

if (!requireNamespace("DDESONN", quietly = TRUE)) {
  stop("DDESONN must be installed to build this vignette.", call. = FALSE)
}

library(DDESONN)

.fig_root <- knitr::opts_chunk$get("fig.path")
if (!nzchar(.fig_root)) .fig_root <- file.path(tempdir(), "figure", "")
dir.create(.fig_root, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# SECTION: RUN FLAGS (Scenario 1 — Ensemble)                    
# - Default: run both
# ============================================================
RUN_SCENARIO_C_S1 <- TRUE   
RUN_SCENARIO_D_S1 <- FALSE   

# ============================================================
#  FIX: DO NOT put DDESONN outputs under vignette _files/
# - Windows build copy fails due to very deep paths under *_files/figure-html/...
# - Use a SHORT tempdir() root for artifacts/plots instead.
# ============================================================
.dd_out_root <- file.path(.vig_tmp_root, "DDESONN_vignette_plot_controls_s1_ensemble_CD")  
outC <- file.path(.dd_out_root, "DDESONN_plots_scenarioC_s1")                         
outD <- file.path(.dd_out_root, "DDESONN_plots_scenarioD_s1")                         

if (!is_check_env) {
  dir.create(outC, recursive = TRUE, showWarnings = FALSE)
  dir.create(outD, recursive = TRUE, showWarnings = FALSE)
}

include_saved_plots <- function(output_root, header) {
  plot_dir <- ddesonn_plots_dir(output_root)
  if (!dir.exists(plot_dir)) {
    return(knitr::asis_output(paste0("\n\nNo plots at: ", plot_dir, "\n\n")))
  }
  pngs <- list.files(plot_dir, pattern = "\\.png$", recursive = TRUE, full.names = TRUE)
  if (!length(pngs)) {
    return(knitr::asis_output(paste0("\n\nNo PNGs under: ", plot_dir, "\n\n")))
  }
  out <- c(paste0("\n\n## ", header, "\n\n"))
  for (p in sort(pngs)) {
    uri <- knitr::image_uri(p)  # NOTE: this embeds base64 (your original behavior)
    out <- c(out, paste0('<img src="', uri,
                         '" style="width:900px; max-width:100%; height:auto;" />\n\n'))
  }
  knitr::asis_output(paste(out, collapse = ""))
}

# =======================
# DATA SPLIT (robust like @examples)                            
# =======================
set.seed(111)

# ------------------------------------------------------------
# 1) Locate package extdata folder (robust across check/install)  
# ------------------------------------------------------------
ext_dir <- system.file("extdata", package = "DDESONN")
if (!nzchar(ext_dir)) {
  stop("Could not find DDESONN extdata folder. Is the package installed?",
       call. = FALSE)
}

# ------------------------------------------------------------
# 1b) Find CSVs (recursive + check-dir edge cases)               
# ------------------------------------------------------------
csvs <- list.files(
  ext_dir,
  pattern = "\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)

# Defensive fallback for rare nested layouts                     
if (!length(csvs)) {                                             
  ext_dir2 <- file.path(ext_dir, "inst", "extdata")               
  if (dir.exists(ext_dir2)) {                                    
    csvs <- list.files(
      ext_dir2,
      pattern = "\\.csv$",
      full.names = TRUE,
      recursive = TRUE
    )
  }
}

# If no data, skip vignette runs (do NOT error)                  
DATA_OK <- TRUE                                                  
if (!length(csvs)) {                                             
  DATA_OK <- FALSE                                               
  message(sprintf(
    "No .csv files found under: %s — vignette will skip runs.",
    ext_dir
  ))
}                                                                

if (isTRUE(DATA_OK)) {                                           

  hf_path <- file.path(ext_dir, "heart_failure_clinical_records.csv")  
  data_path <- if (file.exists(hf_path)) hf_path else csvs[[1]]        

  cat("[extdata] using:", data_path, "\n")                             

  # ------------------------------------------------------------
  # 2) Load data
  # ------------------------------------------------------------
  df <- read.csv(data_path)

  # Prefer DEATH_EVENT if present; otherwise infer a binary target
  target_col <- if ("DEATH_EVENT" %in% names(df)) {               
    "DEATH_EVENT"                                                
  } else {                                                       
    cand <- names(df)[vapply(df, function(col) {
      v <- suppressWarnings(as.numeric(col))
      if (all(is.na(v))) return(FALSE)
      u <- unique(v[is.finite(v)])
      length(u) <= 2 && all(sort(u) %in% c(0, 1))
    }, logical(1))]
    if (!length(cand)) {
      stop(
        "Could not infer a binary target column. ",
        "Provide a binary CSV in extdata or rename target to DEATH_EVENT.",
        call. = FALSE
      )
    }
    cand[[1]]
  }

  cat("[data] target_col =", target_col, "\n")                   

  # ------------------------------------------------------------
  # 3) Build X and y
  # ------------------------------------------------------------
  y_all <- matrix(as.integer(df[[target_col]]), ncol = 1)         
  x_df <- df[, setdiff(names(df), target_col), drop = FALSE]      
  x_all <- as.matrix(x_df)                                        
  storage.mode(x_all) <- "double"                                 

  # ------------------------------------------------------------
  # 4) Split 70 / 15 / 15
  # ------------------------------------------------------------
  n <- nrow(x_all)                                                
  idx <- sample.int(n)                                            

  n_train <- floor(0.70 * n)                                      
  n_valid <- floor(0.15 * n)                                      

  i_tr <- idx[1:n_train]                                          
  i_va <- idx[(n_train + 1):(n_train + n_valid)]                  
  i_te <- idx[(n_train + n_valid + 1):n]                          

  x_train <- x_all[i_tr, , drop = FALSE]                          
  y_train <- y_all[i_tr, , drop = FALSE]                          

  x_valid <- x_all[i_va, , drop = FALSE]                          
  y_valid <- y_all[i_va, , drop = FALSE]                          

  x_test  <- x_all[i_te, , drop = FALSE]                          
  y_test  <- y_all[i_te, , drop = FALSE]                          

  cat(sprintf("[split] train=%d valid=%d test=%d\n",
              nrow(x_train), nrow(x_valid), nrow(x_test)))        

  # ------------------------------------------------------------
  # 5) Scale using TRAIN stats only (no leakage)                  
  # ------------------------------------------------------------
  x_train_s <- scale(x_train)                                     
  ctr <- attr(x_train_s, "scaled:center")                         
  scl <- attr(x_train_s, "scaled:scale")                          
  scl[!is.finite(scl) | scl == 0] <- 1                             

  x_valid_s <- sweep(sweep(x_valid, 2, ctr, "-"), 2, scl, "/")     
  x_test_s  <- sweep(sweep(x_test,  2, ctr, "-"), 2, scl, "/")     

  mx <- suppressWarnings(max(abs(x_train_s)))                      
  if (!is.finite(mx) || mx == 0) mx <- 1                           

  x_train <- x_train_s / mx                                        
  x_valid <- x_valid_s / mx                                        
  x_test  <- x_test_s  / mx                                        

} # end DATA_OK                                                   





## ----scenarioC_s1, results="asis", echo=FALSE, eval=RUN_SCENARIO_C_S1---------
options(DDESONN_OUTPUT_ROOT = if (is_check_env) .vig_tmp_root else outC)
Sys.setenv(DDESONN_ARTIFACTS_ROOT = if (is_check_env) .vig_tmp_root else outC)

res_scenarioC_s1 <- ddesonn_run(
  x = x_train,
  y = y_train,
  classification_mode = "binary",
  hidden_sizes = c(64, 32),

  # ----------------------------------------------------------
  # ENSEMBLE CONFIG
  # ----------------------------------------------------------
  do_ensemble   = TRUE,
  num_networks  = 2L,
  seeds         = 1:1,

  validation = list(x = x_valid, y = y_valid),
  test       = list(x = x_test,  y = y_test),

  # ----------------------------------------------------------
  # TRAINING OVERRIDES (Scenario 1 knobs)
  # ----------------------------------------------------------
  training_overrides = list(
    init_method = "he",
    optimizer   = "adagrad",
    lr          = 0.125,
    lambda      = 0.00028,

    activation_functions = list(relu, relu, sigmoid),
    dropout_rates        = list(0.10),
    loss_type            = "CrossEntropy",

    validation_metrics     = TRUE,
    num_epochs             = 3,
    final_summary_decimals = 6L,

    # ========================================================
    # SECTION: output_controls
    # ========================================================
    viewTables = FALSE,
    debug      = FALSE,
    verboseLow = FALSE,
    browse     = FALSE,

    # ========================================================
    # SECTION: per_epoch_plots
    # ========================================================
    per_epoch_plots = list(
      saveEnabled = !is_check_env,
      loss_curve  = TRUE,
      probe_plots = TRUE,
      verbose     = FALSE
    ),

    # ========================================================
    # SECTION: performance_relevance (Scenario 1)
    # - bridged internally to plot_controls$performance_relevance
    # ========================================================
    performance_relevance = list(
      saveEnabled  = !is_check_env,
      viewAllPlots = FALSE,

      performance_high_mean_plots = TRUE,
      performance_low_mean_plots  = TRUE,
      relevance_high_mean_plots   = TRUE,
      relevance_low_mean_plots    = TRUE,

      verbose = FALSE
    ),

    # ========================================================
    # SECTION: evaluate_predictions_report_plots
    # ========================================================
    evaluate_predictions_report_plots = list(
      accuracy_plot      = TRUE,
      accuracy_plot_mode = "both",
      plot_roc           = TRUE,
      plot_pr            = TRUE,
      show_auprc         = TRUE,
      viewAllPlots       = FALSE,
      verbose            = FALSE
    )
  )
)

# ============================================================
# INCLUDE SAVED PLOTS (interactive / RStudio use)
# ============================================================
if (exists("include_saved_plots")) {
  if (!is_check_env) include_saved_plots(outC, "Scenario C — Ensemble (2 networks, 360 epochs)")
}


## ----scenarioD_s1, results="asis", echo=FALSE, eval=RUN_SCENARIO_D_S1---------
# options(DDESONN_OUTPUT_ROOT = if (is_check_env) .vig_tmp_root else outD)
# Sys.setenv(DDESONN_ARTIFACTS_ROOT = if (is_check_env) .vig_tmp_root else outD)
# 
# res_scenarioD_s1 <- ddesonn_run(
#   x = x_train,
#   y = y_train,
#   classification_mode = "binary",
#   hidden_sizes = c(64, 32),
# 
#   # ----------------------------------------------------------
#   # ENSEMBLE CONFIG
#   # ----------------------------------------------------------
#   do_ensemble   = TRUE,
#   num_networks  = 2L,
#   num_temp_iterations = 1L,
#   seeds         = 1:1,
# 
#   validation = list(x = x_valid, y = y_valid),
#   test       = list(x = x_test,  y = y_test),
# 
#   # ----------------------------------------------------------
#   # TRAINING OVERRIDES (Scenario 1 knobs)
#   # ----------------------------------------------------------
#   training_overrides = list(
#     init_method = "he",
#     optimizer   = "adagrad",
#     lr          = 0.125,
#     lambda      = 0.00028,
# 
#     activation_functions = list(relu, relu, sigmoid),
#     dropout_rates        = list(0.10),
#     loss_type            = "CrossEntropy",
# 
#     validation_metrics     = TRUE,
#     num_epochs             = 3,
#     final_summary_decimals = 6L,
# 
#     # ========================================================
#     # SECTION: per_epoch_plots
#     # ========================================================
#     per_epoch_plots = list(
#       saveEnabled = !is_check_env,
#       loss_curve  = TRUE,
#       probe_plots = TRUE,
#       verbose     = TRUE
#     ),
# 
#     # ========================================================
#     # SECTION: performance_relevance (Scenario 1)
#     # - bridged internally to plot_controls$performance_relevance
#     # ========================================================
#     performance_relevance = list(
#       saveEnabled  = !is_check_env,
#       viewAllPlots = FALSE,
# 
#       performance_high_mean_plots = TRUE,
#       performance_low_mean_plots  = TRUE,
#       relevance_high_mean_plots   = TRUE,
#       relevance_low_mean_plots    = TRUE,
# 
#       verbose = TRUE
#     ),
# 
#     # ========================================================
#     # SECTION: evaluate_predictions_report_plots
#     # ========================================================
#     evaluate_predictions_report_plots = list(
#       accuracy_plot      = TRUE,
#       accuracy_plot_mode = "both",
#       plot_roc           = TRUE,
#       plot_pr            = TRUE,
#       show_auprc         = TRUE,
#       viewAllPlots       = FALSE,
#       verbose            = TRUE
#     )
#   )
# )
# 
# if (!is_check_env) include_saved_plots(
#   outD,
#   "Plot Controls: Scenario 1 — Ensemble Runs — Scenario D"
# )
# 

