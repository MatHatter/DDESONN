# =====================================================================
# artifacts/PredictOnly/LoadandPredict.R
# Minimal predict-only runner with optional overwrite of agg files
# =====================================================================

if (!exists("DDESONN_predict_eval") || !is.function(DDESONN_predict_eval)) {
  stop("DDESONN_predict_eval() is not available in scope.")
}

LoadandPredict <- function(
    source = c("SingleRuns","EnsembleRuns","env"),
    folder = NULL,                       # subfolder under artifacts/<source>/..., or NULL → most recent
    env_meta_name = NULL,                # used only when source="env"
    predict_split = "test",              # "test" | "validation" | "train"
    CLASSIFICATION_MODE = "BINARY",
    run_index = 1L,
    seed_val  = 1L,                      # fallback if filename lacks _seedN
    run_dir_name = format(Sys.time(), "%Y%m%d_%H%M%S_predict"),
    selection = c("first","all"),
    overwrite = FALSE                    # <-- clear old agg files in output dir
) {
  source    <- match.arg(source)
  selection <- match.arg(selection)
  
  # Output dir + aggregate files
  base_dir <- file.path("artifacts", "PredictOnly", run_dir_name)
  if (!dir.exists(base_dir)) dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
  
  agg_predictions_file <- file.path(base_dir, paste0("agg_predictions_", predict_split, ".rds"))
  agg_metrics_file     <- file.path(base_dir, paste0("agg_metrics_",     predict_split, ".rds"))
  
  # Optional: start clean to avoid appending to old logs
  if (isTRUE(overwrite)) {
    if (file.exists(agg_predictions_file)) unlink(agg_predictions_file, force = TRUE)
    if (file.exists(agg_metrics_file))     unlink(agg_metrics_file,     force = TRUE)
  }
  
  # ENV path (no disk scan)
  if (identical(source, "env")) {
    if (is.null(env_meta_name) || !nzchar(env_meta_name)) {
      candidates <- c(
        sprintf("Ensemble_Main_0_model_%d_metadata", 1:64),
        as.vector(outer(1:8, 1:64, function(e,k) sprintf("Ensemble_Temp_%d_model_%d_metadata", e, k)))
      )
      hits <- candidates[sapply(candidates, function(nm) exists(nm, envir = .GlobalEnv))]
      if (!length(hits)) stop("No canonical metadata object found in .GlobalEnv.")
      env_meta_name <- hits[1]
    } else if (!exists(env_meta_name, envir = .GlobalEnv)) {
      stop(sprintf("env_meta_name not found in .GlobalEnv: %s", env_meta_name))
    }
    
    slot <- suppressWarnings(as.integer(sub("^.*_model_([0-9]+)_metadata.*$", "\\1", env_meta_name)))
    if (!is.finite(slot)) slot <- 1L
    
    DDESONN_predict_eval(
      LOAD_FROM_RDS = FALSE,
      ENV_META_NAME = env_meta_name,
      INPUT_SPLIT   = predict_split,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      RUN_INDEX = run_index,
      SEED      = seed_val,
      OUTPUT_DIR = base_dir,
      SAVE_METRICS_RDS = TRUE,
      METRICS_PREFIX   = sprintf("metrics_%s", predict_split),
      AGG_PREDICTIONS_FILE = agg_predictions_file,
      AGG_METRICS_FILE     = agg_metrics_file,
      MODEL_SLOT           = slot
    )
    
    cat(sprintf("[LoadandPredict:env] wrote (seed=%s slot=%d)\n", as.character(seed_val), slot))
    return(invisible(list(run_dir = base_dir, agg_metrics = agg_metrics_file)))
  }
  
  # Folder path (SingleRuns / EnsembleRuns)
  family_root <- file.path("artifacts", source)
  if (!dir.exists(family_root)) stop(sprintf("Family root does not exist: %s", family_root))
  
  run_root <- if (!is.null(folder) && nzchar(folder)) {
    rr <- file.path(family_root, folder); if (!dir.exists(rr)) stop("Requested folder not found: ", rr); rr
  } else {
    kids <- list.dirs(family_root, full.names = TRUE, recursive = FALSE)
    if (!length(kids)) stop("No subfolders found in: ", family_root)
    kids[order(file.info(kids)$mtime, decreasing = TRUE)][1L]
  }
  cat(sprintf("[LoadandPredict:%s] Using folder: %s\n", source, run_root))
  
  models_root <- file.path(run_root, "models")
  if (!dir.exists(models_root)) stop("Expected models/ not found under: ", run_root)
  
  # Gather candidate metadata files
  candidates <- character(0)
  if (identical(source, "SingleRuns")) {
    candidates <- list.files(models_root, pattern = "metadata.*\\.rds$", full.names = TRUE, recursive = FALSE)
  } else {
    main_dir <- file.path(models_root, "main")
    if (dir.exists(main_dir)) {
      candidates <- c(candidates, list.files(main_dir, pattern = "metadata.*\\.rds$", full.names = TRUE, recursive = FALSE))
    }
    temp_dirs <- list.dirs(models_root, full.names = TRUE, recursive = FALSE)
    temp_dirs <- temp_dirs[grepl("temp_e\\d{2}$", temp_dirs)]
    for (td in temp_dirs) {
      candidates <- c(candidates, list.files(td, pattern = "metadata.*\\.rds$", full.names = TRUE, recursive = FALSE))
    }
  }
  if (!length(candidates)) stop("No metadata .rds found.")
  
  candidates <- sort(candidates)
  chosen <- if (identical(selection, "first")) candidates[1L] else candidates
  cat("[LoadandPredict:", source, "] selected file(s):\n  - ",
      paste(chosen, collapse = "\n  - "), "\n", sep = "")
  
  # Helper: parse seed from filename; fallback to seed_val
  .parse_seed_from_name <- function(fname, default_seed = 1L) {
    m <- regexec("_seed([0-9]+)\\b", fname)
    r <- regmatches(fname, m)
    if (length(r) && length(r[[1]]) == 2L) as.integer(r[[1]][2]) else default_seed
  }
  
  n_ok <- 0L
  for (p in chosen) {
    b <- basename(p)
    
    # Canonical var name (…_metadata), drop trailing timestamp/seed chunks
    base <- sub("\\.rds$", "", b)
    base <- sub("_(\\d{8}_\\d{6})$", "", base)
    base <- sub("_seed\\d+$", "", base)
    base <- sub("(_.*)?$", "", sub("(.*_metadata).*", "\\1", base))
    
    slot <- suppressWarnings(as.integer(sub("^.*_model_([0-9]+)_metadata.*$", "\\1", base)))
    if (!is.finite(slot)) slot <- 1L
    seed_from_file <- .parse_seed_from_name(b, default_seed = seed_val)
    
    meta_obj <- readRDS(p)
    assign(base, meta_obj, envir = .GlobalEnv)
    
    DDESONN_predict_eval(
      LOAD_FROM_RDS = FALSE,
      ENV_META_NAME = base,
      INPUT_SPLIT   = predict_split,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      RUN_INDEX = run_index,
      SEED      = seed_from_file,
      OUTPUT_DIR = base_dir,
      SAVE_METRICS_RDS = TRUE,
      METRICS_PREFIX   = sprintf("metrics_%s", predict_split),
      AGG_PREDICTIONS_FILE = agg_predictions_file,
      AGG_METRICS_FILE     = agg_metrics_file,
      MODEL_SLOT           = slot
    )
    
    n_ok <- n_ok + 1L
    cat(sprintf("[LoadandPredict:%s] wrote (seed=%s slot=%d file=%s)\n",
                source, as.character(seed_from_file), slot, b))
  }
  
  invisible(list(
    run_dir         = base_dir,
    agg_predictions = agg_predictions_file,
    agg_metrics     = agg_metrics_file,
    n_models_scored = n_ok
  ))
}

# ---- Example (fresh, 4 rows expected) ----
LoadandPredict(
  source="EnsembleRuns",
  folder="20250928_102703",   # or NULL for latest
  predict_split="test",
  CLASSIFICATION_MODE="BINARY",
  run_index=1L,
  run_dir_name="predict_from_latest_ensemble_fresh",  # <-- new dir
  selection="all",
  overwrite=TRUE                                     # <-- start clean
)
