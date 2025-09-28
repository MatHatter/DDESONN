# =====================================================================
# artifacts/PredictOnly/LoadandPredict.R
#
# Predict-only runner that:
# - Leaves your DDESONN_predict_eval() unchanged.
# - Loads metas from artifacts or env.
# - Ensures a DDESONN$predict() is available; if not, defines a minimal
#   forward-pass shim using weights/biases/activations from meta.
# - Writes fresh agg files: agg_metrics_<split>.rds, agg_predictions_<split>.rds
# =====================================================================

LoadandPredict <- function(
    source               = c("env","EnsembleRuns"),
    folder               = NULL,         # for source="EnsembleRuns": subfolder under artifacts/EnsembleRuns; NULL → latest
    env_meta_name_base   = NULL,         # REQUIRED only when source="env"; ignored for EnsembleRuns
    seeds                = c(1L, 2L),
    slots                = c(1L, 2L),
    predict_split        = c("test","validation","train"),
    CLASSIFICATION_MODE  = c("binary","multiclass","regression"),
    run_index            = 1L,
    output_dir_base      = "artifacts/PredictOnly",
    run_dir_name         = format(Sys.time(), "%Y%m%d_%H%M%S_predict"),
    overwrite            = TRUE
) {
  # ---------- normalize args ----------
  source              <- match.arg(source)
  predict_split       <- match.arg(predict_split)
  CLASSIFICATION_MODE <- match.arg(CLASSIFICATION_MODE)
  run_index           <- as.integer(run_index)
  seeds               <- as.integer(seeds)
  slots               <- as.integer(slots)
  
  if (source == "env" && (is.null(env_meta_name_base) || !nzchar(env_meta_name_base))) {
    stop("env_meta_name_base is required when source='env'.")
  }
  
  # ---------- tiny utils ----------
  `%||%` <- function(x,y) if (is.null(x)) y else x
  .as_num_vec <- function(v){
    if (is.matrix(v))  { if (ncol(v)>1L) stop("labels matrix has >1 col"); v <- v[,1] }
    if (is.data.frame(v)) { if (ncol(v)!=1L) stop("labels df has !=1 col"); v <- v[[1]] }
    if (is.list(v)) v <- vapply(v, function(z){ if (is.list(z)) z <- unlist(z, use.names=FALSE); suppressWarnings(as.numeric(if (length(z)) z[1] else NA)) }, numeric(1))
    if (is.factor(v)) v <- as.character(v)
    if (is.logical(v)) v <- as.integer(v)
    suppressWarnings(as.numeric(v))
  }
  .latest_subdir <- function(root) {
    if (!dir.exists(root)) return(NULL)
    kids <- list.dirs(root, full.names = TRUE, recursive = FALSE)
    if (!length(kids)) return(NULL)
    info <- file.info(kids)
    kids[order(info$mtime, decreasing = TRUE)][1L]
  }
  .safe_name <- function(...) {
    raw <- paste0(unlist(list(...)), collapse = "_")
    nm  <- gsub("[^A-Za-z0-9_]", "_", raw)
    sub("^([0-9])", "_\\1", nm)
  }
  
  # ---------- ensure DDESONN$predict exists (shim if missing) ----------
  .ensure_predictor <- function() {
    if (exists("DDESONN", envir=.GlobalEnv, inherits=FALSE)) return(invisible(TRUE))
    
    # Try to source a real implementation from common paths
    candidates <- c(
      "R/DDESONN.R","R/DESONN.R","R/desonn.R","DDESONN.R","DESONN.R",
      "src/DDESONN.R","inst/DDESONN.R"
    )
    for (p in candidates) {
      if (file.exists(p)) {
        try(source(p, local=FALSE), silent=TRUE)
        if (exists("DDESONN", envir=.GlobalEnv, inherits=FALSE)) return(invisible(TRUE))
      }
    }
    
    # Define a minimal forward-pass shim so .safe_run_predict can proceed
    # It expects: DDESONN$predict(X, meta, model_index, ML_NN=TRUE, ...)
    sigmoid <- function(z) 1/(1+exp(-z))
    tanh_act <- function(z) tanh(z)
    relu <- function(z) { z[z<0] <- 0; z }
    leaky_relu <- function(z, a=0.01) { z[z<0] <- a*z[z<0]; z }
    softmax <- function(z) {
      z <- as.matrix(z); mx <- apply(z,1,max); ex <- exp(z - mx); sm <- rowSums(ex); ex/sm
    }
    linear <- function(z) z
    
    apply_act <- function(Z, name) {
      if (is.null(name)) return(Z)
      n <- tolower(as.character(name))
      if (n %in% c("sigmoid","logistic")) return(sigmoid(Z))
      if (n %in% c("tanh"))               return(tanh_act(Z))
      if (n %in% c("relu"))               return(relu(Z))
      if (n %in% c("lrelu","leaky_relu")) return(leaky_relu(Z))
      if (n %in% c("softmax"))            return(softmax(Z))
      if (n %in% c("linear","identity"))  return(linear(Z))
      # default safe
      return(Z)
    }
    
    add_bias <- function(Z, b) {
      if (is.null(b)) return(Z)
      b <- as.numeric(b)
      Z + matrix(rep(b, each=nrow(Z)), nrow=nrow(Z), ncol=length(b), byrow=FALSE)
    }
    
    forward_from_meta <- function(X, meta, model_index=1L) {
      # choose weights/biases: prefer best_* if available
      W <- meta$best_weights_record %||% meta$weights
      B <- meta$best_biases_record   %||% meta$biases
      if (is.null(W) || is.null(B)) stop("[shim] No weights/biases in meta")
      
      acts <- meta$activation_functions_predict %||% meta$activation_functions
      if (is.list(acts) && !is.null(acts[[model_index]])) acts <- acts[[model_index]]
      
      # Some metas store per-layer in lists indexed by layer
      # Normalize to a simple list of matrices
      if (is.list(W) && !is.matrix(W[[1]])) {
        WL <- W
      } else {
        WL <- W
      }
      BL <- B
      
      H <- as.matrix(X)
      L <- length(WL)
      for (l in seq_len(L)) {
        Wl <- WL[[l]]
        bl <- BL[[l]]
        Z  <- H %*% Wl
        Z  <- add_bias(Z, bl)
        act_name <- if (is.null(acts)) NULL else acts[[l]] %||% acts[l] %||% acts
        # No dropout during predict
        H <- apply_act(Z, act_name)
      }
      list(predicted_output = as.matrix(H))
    }
    
    DDESONN <<- list(
      predict = function(X, meta, model_index=1L, ML_NN=TRUE, ...) {
        forward_from_meta(X, meta, model_index)
      }
    )
    invisible(TRUE)
  }
  
  .ensure_predictor()
  
  # ---------- ENV meta resolver ----------
  .resolve_meta_env <- function(base, slot, seed) {
    tries <- unique(c(
      base,
      paste0(base, "_model_", as.integer(slot), "_metadata"),
      paste0(base, "_seed", as.character(seed))
    ))
    for (nm in tries) if (exists(nm, envir=.GlobalEnv, inherits=FALSE)) return(get(nm, envir=.GlobalEnv))
    
    syms <- ls(.GlobalEnv)
    hits <- syms[grepl(base, syms, fixed=TRUE)]
    if (!length(hits)) stop("No ENV meta found matching: ", base)
    slot_pat <- paste0("_model_", as.integer(slot), "_")
    seed_pat <- paste0("_seed",  as.character(seed))
    prefer <- hits[grepl(slot_pat, hits, fixed=TRUE) & grepl(seed_pat, hits, fixed=TRUE)]
    if (!length(prefer)) prefer <- hits[grepl(slot_pat, hits, fixed=TRUE)]
    nm <- (if (length(prefer)) prefer else hits)[1]
    get(nm, envir=.GlobalEnv)
  }
  
  # ---------- Artifacts meta resolver ----------
  .resolve_meta_artifacts <- function(root_folder, slot, seed, base=NULL) {
    if (!dir.exists(root_folder)) stop("Artifacts folder not found: ", root_folder)
    files <- list.files(root_folder, pattern="\\.[Rr][Dd][Ss]$", full.names=TRUE, recursive=TRUE)
    if (!length(files)) stop("No .rds files in ", root_folder)
    
    bn <- basename(files)
    hit <- grepl("metadata", bn, ignore.case=TRUE) &
      grepl(paste0("_model_", as.integer(slot), "_"), bn, fixed=TRUE) &
      grepl(paste0("_seed",  as.character(seed)),     bn, fixed=TRUE)
    if (!is.null(base) && nzchar(base)) hit <- hit & grepl(base, bn, fixed=TRUE)
    
    if (!any(hit)) stop("No matching metadata RDS for slot=", slot, ", seed=", seed, " in ", root_folder)
    
    cand <- files[hit]
    info <- file.info(cand)
    file <- cand[order(info$mtime, decreasing=TRUE)][1L]
    meta <- readRDS(file)
    
    sym  <- .safe_name("LP_META", "slot", slot, "seed", seed)
    assign(sym, meta, envir=.GlobalEnv)
    
    list(meta_sym = sym, path = file)
  }
  
  # ---------- split accessor ----------
  .select_split <- function(meta, split){
    sl <- tolower(split)
    if (sl=="test")            { list(X=meta$X_test,       y=meta$y_test,       split="test") }
    else if (sl=="validation") { list(X=meta$X_validation, y=meta$y_validation, split="validation") }
    else if (sl=="train")      { list(X=meta$X %||% meta$X_train, y=meta$y %||% meta$y_train, split="train") }
    else stop("bad split")
  }
  
  # ---------- prepare output dir ----------
  out_dir <- file.path(output_dir_base, run_dir_name)
  dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
  agg_metrics_path <- file.path(out_dir, sprintf("agg_metrics_%s.rds", predict_split))
  agg_preds_path   <- file.path(out_dir, sprintf("agg_predictions_%s.rds", predict_split))
  if (overwrite) {
    suppressWarnings(try(unlink(agg_metrics_path, force=TRUE), silent=TRUE))
    suppressWarnings(try(unlink(agg_preds_path,   force=TRUE), silent=TRUE))
  }
  
  # ---------- resolve artifacts folder ----------
  artifacts_root <- NULL
  if (source == "EnsembleRuns") {
    root <- file.path("artifacts","EnsembleRuns")
    if (!dir.exists(root)) stop("Expected artifacts root does not exist: ", root)
    artifacts_root <- if (is.null(folder)) .latest_subdir(root) else file.path(root, folder)
    if (is.null(artifacts_root) || !dir.exists(artifacts_root)) {
      stop("Could not resolve artifacts folder under: ", normalizePath(root, winslash="/", mustWork=FALSE))
    }
  }
  
  # ---------- accumulators ----------
  metrics_rows <- list()
  preds_rows   <- list()
  mr_idx <- 0L
  pr_idx <- 0L
  
  message(sprintf("Predicting split='%s' | mode='%s' | run_index=%d", predict_split, CLASSIFICATION_MODE, run_index))
  
  # ---------- core loop ----------
  for (seed in seeds) {
    for (slot in slots) {
      
      if (source == "env") {
        meta <- .resolve_meta_env(env_meta_name_base, slot, seed)
        env_token_for_eval <- env_meta_name_base
      } else {
        art <- .resolve_meta_artifacts(artifacts_root, slot, seed, base = env_meta_name_base)
        meta <- get(art$meta_sym, envir=.GlobalEnv)
        env_token_for_eval <- art$meta_sym
      }
      
      ss <- .select_split(meta, predict_split)
      if (is.null(ss$X) || is.null(ss$y)) {
        warning(sprintf("Split '%s' not present (seed=%d, slot=%d) — skipping", predict_split, seed, slot))
        next
      }
      
      # Run your existing evaluator (unchanged). We keep LOAD_FROM_RDS=FALSE to stay in the fast path.
      res <- DDESONN_predict_eval(
        LOAD_FROM_RDS         = FALSE,
        ENV_META_NAME         = env_token_for_eval,
        INPUT_SPLIT           = predict_split,
        CLASSIFICATION_MODE   = CLASSIFICATION_MODE,
        RUN_INDEX             = run_index,
        SEED                  = seed,
        OUTPUT_DIR            = out_dir,
        SAVE_METRICS_RDS      = FALSE,
        METRICS_PREFIX        = sprintf("metrics_%s", predict_split),
        AGG_PREDICTIONS_FILE  = NULL,
        AGG_METRICS_FILE      = NULL,
        MODEL_SLOT            = slot,
        DEBUG                 = TRUE,
        OUT_DIR_ASSERT        = out_dir
      )
      
      # ---- metrics row
      mr <- res$metrics_row
      if (is.null(mr) || !is.data.frame(mr) || !nrow(mr)) {
        warning(sprintf("No metrics_row for seed=%d slot=%d — skipping", seed, slot))
      } else {
        mr$run_index <- as.integer(run_index)
        mr$seed      <- as.integer(seed)
        mr$model_slot<- as.integer(slot)
        mr$split     <- tolower(predict_split)
        mr$CLASSIFICATION_MODE <- tolower(CLASSIFICATION_MODE)
        mr_idx <- mr_idx + 1L
        metrics_rows[[mr_idx]] <- mr
      }
      
      # ---- predictions row(s)
      P <- res$probs
      if (is.null(P) || !is.matrix(P) || !nrow(P)) {
        warning(sprintf("Empty probs for seed=%d slot=%d — skipping preds", seed, slot))
      } else {
        y_true <- .as_num_vec(ss$y)
        if (length(y_true) != nrow(P)) {
          nmin <- min(length(y_true), nrow(P))
          y_true <- y_true[seq_len(nmin)]; P <- P[seq_len(nmin), , drop=FALSE]
        }
        if (CLASSIFICATION_MODE == "binary") {
          thr <- suppressWarnings(as.numeric(res$results_compact$tuned_threshold))
          if (!is.finite(thr)) thr <- 0.5
          y_prob <- as.numeric(P[,1]); y_pred <- as.integer(y_prob >= thr)
        } else if (CLASSIFICATION_MODE == "multiclass") {
          y_prob <- apply(P, 1, max); y_pred <- max.col(P, ties.method = "first")
        } else {
          y_prob <- as.numeric(P[,1]); y_pred <- y_prob
        }
        n <- length(y_true)
        if (n > 0L) {
          pr <- data.frame(
            run_index   = rep.int(as.integer(run_index), n),
            seed        = rep.int(as.integer(seed), n),
            model_slot  = rep.int(as.integer(slot), n),
            y_true      = as.numeric(y_true),
            y_prob      = as.numeric(y_prob),
            y_pred      = as.numeric(y_pred),
            split       = rep.int(tolower(predict_split), n),
            CLASSIFICATION_MODE = rep.int(tolower(CLASSIFICATION_MODE), n),
            stringsAsFactors = FALSE, check.names = TRUE
          )
          pr_idx <- pr_idx + 1L
          preds_rows[[pr_idx]] <- pr
        }
      }
      
      message(sprintf(" ✓ seed=%d | slot=%d", seed, slot))
    }
  }
  
  # ---------- bind + write ----------
  agg_metrics     <- if (length(metrics_rows)) do.call(rbind, metrics_rows) else data.frame()
  agg_predictions <- if (length(preds_rows))   do.call(rbind, preds_rows)   else data.frame()
  saveRDS(agg_metrics,     agg_metrics_path)
  saveRDS(agg_predictions, agg_preds_path)
  
  message("LoadandPredict complete.")
  list(
    output_dir            = normalizePath(out_dir, winslash="/", mustWork=FALSE),
    agg_metrics_file      = normalizePath(agg_metrics_path, winslash="/", mustWork=FALSE),
    agg_predictions_file  = normalizePath(agg_preds_path, winslash="/", mustWork=FALSE),
    agg_metrics_preview   = utils::head(agg_metrics),
    agg_predictions_rows  = nrow(agg_predictions)
  )
}

# =====================================================================
# EXAMPLES
# =====================================================================

# Example 1: ENV metas (no disk)
# ex1 <- LoadandPredict(
#   source="env",
#   env_meta_name_base="Ensemble_Main_0_model_1_metadata",
#   seeds=c(1,2),
#   slots=c(1,2),
#   predict_split="test",
#   CLASSIFICATION_MODE="binary",
#   run_index=1,
#   output_dir_base="artifacts/PredictOnly",
#   run_dir_name="predict_env_all",
#   overwrite=TRUE
# )
# print(readRDS(ex1$agg_metrics_file))
# print(head(readRDS(ex1$agg_predictions_file)))

# Example 2: EnsembleRuns (from disk)
ex2 <- LoadandPredict(
  source="EnsembleRuns",
  folder=NULL,   # latest under artifacts/EnsembleRuns
  seeds=c(1,2),
  slots=c(1,2),
  predict_split="test",
  CLASSIFICATION_MODE="binary",
  run_index=1,
  output_dir_base="artifacts/PredictOnly",
  run_dir_name="predict_from_latest_ensemble_all",
  overwrite=TRUE
)
print(readRDS(ex2$agg_metrics_file))
print(head(readRDS(ex2$agg_predictions_file)))
