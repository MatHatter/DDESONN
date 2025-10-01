# =====================================================================
# artifacts/PredictOnly/LoadandPredict.R
# Predict-only runner (keeps your DDESONN_predict_eval() intact)
# - EnsembleRuns: models/<ensemble_model_subdir> (default "main")
# - SingleRuns  : models/
# - Robust coercion for X, weights, biases (nested lists OK)
# - Activations can be strings OR functions (closures)
# - No <<-
# - No external helpers: fully self-contained (no .resolve_* / .find_*)
# =====================================================================

suppressWarnings(try(source("utils/utils.R"), silent = TRUE))
suppressWarnings(try(source("performance_relevance_metrics.R"), silent = TRUE))

LoadandPredict <- function(
    source                = c("EnsembleRuns","SingleRuns","env"),
    folder                = NULL,               # subfolder under artifacts/<source>; NULL → latest
    seeds                 = c(1L,2L),
    slots                 = c(1L,2L,3L),
    predict_split         = c("test","validation","train"),
    CLASSIFICATION_MODE   = c("binary","multiclass","regression"),
    run_index             = 1L,
    output_dir_base       = "artifacts/PredictOnly",
    run_dir_name          = format(Sys.time(), "%Y%m%d_%H%M%S_predict"),
    overwrite             = TRUE,
    ensemble_model_subdir = "main",
    env_meta_name_base    = NULL
){
  source              <- match.arg(source)
  predict_split       <- match.arg(predict_split)
  CLASSIFICATION_MODE <- match.arg(CLASSIFICATION_MODE)
  run_index           <- as.integer(run_index)
  seeds               <- as.integer(seeds)
  slots               <- as.integer(slots)
  
  `%||%` <- function(x,y) if (is.null(x)) y else x
  
  # ---------- Small utils (local, no external helpers) ----------
  .as_num_vec <- function(v){
    if (is.matrix(v))       { if (ncol(v)>1L) stop("labels matrix >1 col"); v <- v[,1] }
    if (is.data.frame(v))   { if (ncol(v)!=1L) stop("labels df !=1 col");   v <- v[[1]] }
    if (is.list(v))         v <- vapply(v, function(z){ if (is.list(z)) z <- unlist(z, use.names=FALSE); suppressWarnings(as.numeric(if (length(z)) z[1] else NA)) }, numeric(1))
    if (is.factor(v))       v <- as.character(v)
    if (is.logical(v))      v <- as.integer(v)
    suppressWarnings(as.numeric(v))
  }
  .latest_subdir <- function(root){
    if (!dir.exists(root)) return(NULL)
    kids <- list.dirs(root, full.names=TRUE, recursive=FALSE)
    if (!length(kids)) return(NULL)
    info <- file.info(kids)
    kids[order(info$mtime, decreasing=TRUE)][1L]
  }
  .safe_name <- function(...) {
    raw <- paste0(unlist(list(...)), collapse = "_")
    nm  <- gsub("[^A-Za-z0-9_]", "_", raw)
    sub("^([0-9])", "_\\1", nm)
  }
  
  # Self-contained "find meta" copied-style search used by SingleRuns, adapted to Ensembles too
  .resolve_run_root <- function(source, folder) {
    root_base <- file.path("artifacts", source)
    if (!dir.exists(root_base)) stop(sprintf("Artifacts base not found: %s", root_base))
    if (is.null(folder)) {
      rd <- .latest_subdir(root_base)
      if (is.null(rd)) stop(sprintf("No dated runs under: %s", root_base))
      return(rd)
    } else {
      rd <- file.path(root_base, folder)
      if (!dir.exists(rd)) stop(sprintf("Run folder not found: %s", rd))
      return(rd)
    }
  }
  
  .models_dir_for <- function(run_root, source, ensemble_model_subdir) {
    if (identical(source, "SingleRuns")) {
      file.path(run_root, "models")
    } else if (identical(source, "EnsembleRuns")) {
      file.path(run_root, "models", ensemble_model_subdir %||% "main")
    } else {
      NULL
    }
  }
  
  # Search for an RDS metadata file that contains both model_<slot> and seed<seed> in the name, preferring newest
  .find_meta_file_local <- function(models_dir, slot, seed) {
    if (is.null(models_dir) || !dir.exists(models_dir)) return(NA_character_)
    # Be permissive but specific: require "metadata" and "model_<slot>" and "seed<seed>"
    patt <- "metadata.*\\.rds$"
    cand <- list.files(models_dir, pattern = patt, recursive = TRUE, full.names = TRUE)
    if (!length(cand)) return(NA_character_)
    bnames <- basename(cand)
    want <- grepl(sprintf("model_%d", as.integer(slot)), bnames, ignore.case = TRUE) &
      grepl(sprintf("seed\\s*%d", as.integer(seed)), bnames, ignore.case = TRUE)
    cand <- cand[want]
    if (!length(cand)) {
      # fallback: if seed not in filename (some SingleRuns variants), keep only model_<slot>
      cand2 <- list.files(models_dir, pattern = sprintf("model_%d.*metadata.*\\.rds$", as.integer(slot)),
                          recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
      if (!length(cand2)) return(NA_character_)
      cand <- cand2
    }
    info <- file.info(cand)
    cand[order(info$mtime, decreasing = TRUE)][1L]
  }
  
  .resolve_meta_env <- function(env_base, slot, seed) {
    # Try exact: <env_base>_slot_<slot>_seed_<seed>
    nm1 <- sprintf("%s_slot_%d_seed_%d", env_base, as.integer(slot), as.integer(seed))
    if (exists(nm1, envir = .GlobalEnv, inherits = FALSE)) return(get(nm1, envir = .GlobalEnv))
    # Try provided base directly (points to a list-of-lists)
    if (exists(env_base, envir = .GlobalEnv, inherits = FALSE)) {
      obj <- get(env_base, envir = .GlobalEnv)
      if (is.list(obj)) return(obj)
    }
    stop(sprintf("ENV meta not found for base='%s' (slot=%d seed=%d).", env_base, slot, seed))
  }
  
  # ---------- install robust predictor shim into DDESONN_predict_eval env ----------
  .ensure_predictor_in_eval_env <- function(){
    if (!exists("DDESONN_predict_eval", inherits=TRUE) || !is.function(DDESONN_predict_eval)) {
      stop("DDESONN_predict_eval() not found.")
    }
    dse_env <- environment(DDESONN_predict_eval)
    
    # activations
    sigmoid    <- function(z){ z <- as.matrix(z); 1/(1+exp(-z)) }
    tanh_act   <- function(z) tanh(z)
    relu       <- function(z){ z <- as.matrix(z); z[z<0] <- 0; z }
    leaky_relu <- function(z,a=0.01){ z <- as.matrix(z); z[z<0] <- a*z[z<0]; z }
    softmax    <- function(z){ z <- as.matrix(z); mx <- apply(z,1,max); ex <- exp(z-mx); ex/rowSums(ex) }
    linear     <- function(z){ as.matrix(z) }
    
    .apply_act <- function(Z, a){
      if (is.null(a)) return(as.matrix(Z))
      if (is.function(a)) return(a(Z))
      if (is.symbol(a))   a <- as.character(a)
      if (is.language(a)) a <- as.character(a)[1]
      if (length(a)>1)    a <- a[1]
      n <- tolower(as.character(a))
      if (n %in% c("sigmoid","logistic")) return(sigmoid(Z))
      if (n %in% c("tanh"))               return(tanh_act(Z))
      if (n %in% c("relu"))               return(relu(Z))
      if (n %in% c("lrelu","leaky_relu")) return(leaky_relu(Z))
      if (n %in% c("softmax"))            return(softmax(Z))
      if (n %in% c("linear","identity"))  return(linear(Z))
      as.matrix(Z)
    }
    
    add_bias <- function(Z, b){
      if (is.null(b)) return(as.matrix(Z))
      b <- as.numeric(b)
      Z <- as.matrix(Z)
      if (length(b) == 1L) return(Z + b)
      if (length(b) == ncol(Z)) return(Z + matrix(rep(b, each=nrow(Z)), nrow=nrow(Z), ncol=ncol(Z), byrow=FALSE))
      stop(sprintf("Bias length (%d) incompatible with ncol(Z) (%d)", length(b), ncol(Z)))
    }
    
    .to_num_mat <- function(X){
      if (is.list(X) && !is.data.frame(X)) {
        for (k in c("X","x","data","Data","features","input","inputs","mat","matrix","M"))
          if (!is.null(X[[k]])) return(.to_num_mat(X[[k]]))
        lens <- lengths(X)
        if (length(lens) && length(unique(lens))==1 && unique(lens)>0 && all(sapply(X, function(e) is.numeric(e) || is.integer(e)))) {
          M <- do.call(rbind, X); M <- as.matrix(M); storage.mode(M) <- "double"; return(M)
        }
        X <- tryCatch(as.data.frame(X, stringsAsFactors=FALSE), error=function(e) NULL)
        if (is.null(X)) stop("Unsupported list structure for X")
      }
      if (inherits(X,"tbl_df")) X <- as.data.frame(X)
      if (is.data.frame(X)) { M <- data.matrix(X); storage.mode(M) <- "double"; return(M) }
      if (is.matrix(X))     { storage.mode(X) <- "double"; return(X) }
      if (is.vector(X))     { M <- matrix(as.numeric(X), ncol=length(X)); storage.mode(M) <- "double"; return(M) }
      stop(sprintf("Unsupported X type: %s", paste(class(X), collapse=",")))
    }
    
    .first_num_mat <- function(obj){
      if (is.matrix(obj))     { storage.mode(obj)<-"double"; return(obj) }
      if (is.data.frame(obj)) { M <- data.matrix(obj); storage.mode(M)<-"double"; return(M) }
      if (is.numeric(obj) && is.vector(obj)) { M <- matrix(obj, ncol=length(obj)); storage.mode(M)<-"double"; return(M) }
      if (is.list(obj)) {
        for (nm in c("W","w","weights","Weight","weight","theta"))
          if (!is.null(obj[[nm]])) { r <- .first_num_mat(obj[[nm]]); if (!is.null(r)) return(r) }
        if (length(obj)>=1L) {
          r1 <- try(.first_num_mat(obj[[1]]), silent=TRUE)
          if (!inherits(r1,"try-error") && !is.null(r1)) return(r1)
          for (k in seq_along(obj)) {
            r <- try(.first_num_mat(obj[[k]]), silent=TRUE)
            if (!inherits(r,"try-error") && !is.null(r)) return(r)
          }
        }
      }
      NULL
    }
    
    .first_num_vec <- function(obj){
      if (is.null(obj)) return(NULL)
      if (is.numeric(obj) && is.vector(obj)) return(as.numeric(obj))
      if (is.matrix(obj) && (nrow(obj)==1L || ncol(obj)==1L)) return(as.numeric(obj))
      if (is.data.frame(obj) && ncol(obj)==1L) return(as.numeric(obj[[1]]))
      if (is.list(obj)) {
        for (nm in c("b","bias","biases","B","beta")) if (!is.null(obj[[nm]])) return(.first_num_vec(obj[[nm]]))
        if (length(obj)>=1L) {
          v1 <- try(.first_num_vec(obj[[1]]), silent=TRUE)
          if (!inherits(v1,"try-error") && !is.null(v1)) return(v1)
          for (k in seq_along(obj)) {
            v <- try(.first_num_vec(obj[[k]]), silent=TRUE)
            if (!inherits(v,"try-error") && !is.null(v)) return(v)
          }
        }
      }
      NULL
    }
    
    .maybe_pick_network <- function(obj, model_index=1L){
      if (is.list(obj) && length(obj)>=1L && is.list(obj[[1]]) && !is.matrix(obj[[1]])) {
        idx <- max(1L, min(model_index, length(obj)))
        return(obj[[idx]])
      }
      obj
    }
    
    .get_act_for_layer <- function(acts, l){
      if (is.null(acts)) return(NULL)
      if (is.function(acts)) return(acts)                 # single fn for all layers
      a <- NULL
      if (is.list(acts)) {
        if (!is.null(acts[[l]])) a <- acts[[l]]
        else if (!is.null(acts[l])) a <- acts[[l]]
      }
      if (is.null(a)) a <- acts
      a
    }
    
    .forward_from_meta <- function(X, meta, model_index=1L){
      W <- meta$best_weights_record %||% meta$weights
      B <- meta$best_biases_record   %||% meta$biases
      if (is.null(W) || is.null(B)) stop("[shim] No weights/biases in meta")
      
      W <- .maybe_pick_network(W, model_index=model_index)
      B <- .maybe_pick_network(B, model_index=model_index)
      
      acts <- meta$activation_functions_predict %||% meta$activation_functions
      if (is.list(acts) && length(acts)>=1L && is.list(acts[[1]])) {
        acts <- acts[[ max(1L, min(model_index, length(acts))) ]]
      }
      
      H  <- .to_num_mat(X)
      L  <- length(W)
      for (l in seq_len(L)) {
        Wl_raw <- W[[l]]
        Bl_raw <- if (l <= length(B)) B[[l]] else NULL
        
        Wl <- .first_num_mat(Wl_raw)
        if (is.null(Wl)) stop(sprintf("[shim] Could not extract numeric weight matrix for layer %d", l))
        Z  <- as.matrix(H) %*% as.matrix(Wl)
        
        bl <- .first_num_vec(Bl_raw)
        Z  <- add_bias(Z, bl)
        
        a  <- .get_act_for_layer(acts, l)
        H  <- .apply_act(Z, a)
      }
      as.matrix(H)
    }
    
    DDESONN_fn <- function(X, meta, model_index=1L, ML_NN=TRUE, ...){
      list(predicted_output = .forward_from_meta(X, meta, model_index))
    }
    
    assign("DDESONN",         DDESONN_fn, envir=dse_env)
    assign("DDESONN_predict", DDESONN_fn, envir=dse_env)
    
    .run_predict_core <- function(predictor, X, meta, model_index=1L, ...){
      out <- NULL
      if (is.function(predictor)) {
        out <- predictor(X, meta, model_index=model_index, ...)
      } else if (is.list(predictor) && is.function(predictor$predict)) {
        out <- predictor$predict(X, meta, model_index=model_index, ...)
      } else stop(".run_predict: predictor is neither function nor list$predict")
      
      if (is.matrix(out)) return(list(predicted_output = out))
      if (is.list(out) && !is.null(out$predicted_output)) return(out)
      if (is.vector(out)) return(list(predicted_output = as.matrix(out)))
      stop(".run_predict: unexpected predictor return")
    }
    
    assign(".run_predict",
           function(...){
             dots <- list(...); nm <- names(dots); n <- length(dots)
             is_pred <- function(o) is.function(o) || (is.list(o) && is.function(o$predict))
             predictor <- NULL; X <- NULL; meta <- NULL; model_index <- 1L
             
             if (!is.null(nm) && "predictor" %in% nm) { predictor <- dots$predictor; dots$predictor <- NULL }
             else if (n>=1 && is_pred(dots[[1]]))     { predictor <- dots[[1]]; dots <- dots[-1] }
             else predictor <- get("DDESONN", envir=dse_env)
             
             nm <- names(dots)
             if (!is.null(nm) && "X" %in% nm)           { X <- dots$X;    dots$X <- NULL }
             if (!is.null(nm) && "meta" %in% nm)        { meta <- dots$meta; dots$meta <- NULL }
             if (!is.null(nm) && "model_index" %in% nm) { model_index <- dots$model_index; dots$model_index <- NULL }
             if (is.null(X)   && length(dots)>=1) { X <- dots[[1]];   dots <- dots[-1] }
             if (is.null(meta)&& length(dots)>=1) { meta <- dots[[1]]; dots <- dots[-1] }
             if (length(dots)>=1 && is.numeric(dots[[1]])) { model_index <- as.integer(dots[[1]]) }
             
             .run_predict_core(predictor, X, meta, model_index=model_index)
           },
           envir=dse_env
    )
    invisible(TRUE)
  }
  
  # Patch loader used by DDESONN_predict_eval so it can read either ENV or a direct file path
  .patch_eval_loader <- function(){
    if (!exists("DDESONN_predict_eval", inherits=TRUE) || !is.function(DDESONN_predict_eval)) return(invisible(FALSE))
    dse_env <- environment(DDESONN_predict_eval)
    assign("load_meta",
           function(LOAD_FROM_RDS=FALSE, ENV_META_NAME=NULL, ...){
             if (isTRUE(LOAD_FROM_RDS)) {
               if (!is.null(ENV_META_NAME) && file.exists(ENV_META_NAME)) return(readRDS(ENV_META_NAME))
               stop("load_meta(LOAD_FROM_RDS=TRUE): file not found / not provided")
             }
             if (!is.null(ENV_META_NAME) && exists(ENV_META_NAME, envir=.GlobalEnv, inherits=FALSE))
               return(get(ENV_META_NAME, envir=.GlobalEnv))
             if (!is.null(ENV_META_NAME) && file.exists(ENV_META_NAME))
               return(readRDS(ENV_META_NAME))
             stop("load_meta: cannot resolve meta (ENV_META_NAME not in env and not a file).")
           },
           envir=dse_env
    )
    invisible(TRUE)
  }
  
  # Minimal schema aligner (light touch) so EnsembleRuns metrics match SingleRuns shape
  .normalize_metrics_schema_minimal <- function(df, predict_split, CLASSIFICATION_MODE, force_run_index = TRUE) {
    if (!is.data.frame(df) || !NROW(df)) return(df)
    as_int <- function(x) suppressWarnings(as.integer(x))
    as_num <- function(x) suppressWarnings(as.numeric(x))
    as_chr <- function(x) suppressWarnings(as.character(x))
    
    if (!"run_index"  %in% names(df) && "RUN_INDEX"  %in% names(df)) df$run_index  <- as_int(df$RUN_INDEX)
    if (!"seed"       %in% names(df) && "SEED"       %in% names(df)) df$seed       <- as_int(df$SEED)
    if (!"model_slot" %in% names(df) && "MODEL_SLOT" %in% names(df)) df$model_slot <- as_int(df$MODEL_SLOT)
    
    if ("split" %in% names(df)) df$split <- tolower(as_chr(df$split))
    if ("SPLIT" %in% names(df)) df$SPLIT <- toupper(as_chr(df$SPLIT))
    if ("CLASSIFICATION_MODE" %in% names(df)) df$CLASSIFICATION_MODE <- tolower(as_chr(df$CLASSIFICATION_MODE))
    
    if (isTRUE(force_run_index)) {
      if ("run_index" %in% names(df)) {
        df$run_index <- 1L
      } else if ("RUN_INDEX" %in% names(df)) {
        df$RUN_INDEX <- 1L
      }
    }
    
    tune_map <- list(
      accuracy  = "accuracy_precision_recall_f1_tuned.accuracy",
      precision = "accuracy_precision_recall_f1_tuned.precision",
      recall    = "accuracy_precision_recall_f1_tuned.recall",
      f1        = "accuracy_precision_recall_f1_tuned.f1"
    )
    for (k in names(tune_map)) {
      src <- tune_map[[k]]
      if ((k %in% names(df)) && (src %in% names(df))) {
        na_idx <- is.na(suppressWarnings(as_num(df[[k]])))
        if (any(na_idx)) df[[k]][na_idx] <- as_num(df[[src]])[na_idx]
      }
    }
    if ("f1_score" %in% names(df) && "f1" %in% names(df)) {
      na_idx <- is.na(suppressWarnings(as_num(df$f1_score)))
      if (any(na_idx)) df$f1_score[na_idx] <- suppressWarnings(as_num(df$f1))[na_idx]
    }
    
    if (all(c("confusion_matrix.TP","confusion_matrix.FN") %in% names(df)) && "recall" %in% names(df)) {
      rec_na <- is.na(suppressWarnings(as_num(df$recall)))
      if (any(rec_na)) {
        TP <- as_num(df[["confusion_matrix.TP"]]); FN <- as_num(df[["confusion_matrix.FN"]])
        rec_calc <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
        df$recall[rec_na] <- rec_calc[rec_na]
      }
    }
    if (("f1" %in% names(df)) && ("precision" %in% names(df)) && ("recall" %in% names(df))) {
      f1_na <- is.na(suppressWarnings(as_num(df$f1)))
      if (any(f1_na)) {
        P <- suppressWarnings(as_num(df$precision)); R <- suppressWarnings(as_num(df$recall))
        f1_calc <- ifelse((P + R) > 0, 2 * P * R / (P + R), NA_real_)
        df$f1[f1_na] <- f1_calc[f1_na]
      }
    }
    df
  }
  
  # Ensure predictor + loader are present
  .ensure_predictor_in_eval_env()
  .patch_eval_loader()
  
  # -------- Determine run roots and models dir (SingleRuns logic copied to EnsembleRuns) --------
  run_root <- if (!identical(source, "env")) .resolve_run_root(source, folder) else NULL
  models_dir <- if (!identical(source, "env")) .models_dir_for(run_root, source, ensemble_model_subdir) else NULL
  
  # Output dir must be defined BEFORE we call predict_eval (fixing earlier out_dir timing)
  out_dir <- file.path(output_dir_base, run_dir_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (!is.null(run_root)) {
    message(sprintf("[LP-DBG %s] SOURCE=%s | RUN_ROOT=%s",
                    format(Sys.time(), "%H:%M:%S"), source, normalizePath(run_root, winslash="/", mustWork=FALSE)))
    if (!is.null(models_dir)) {
      message(sprintf("[LP-DBG %s] MODELS_DIR=%s", format(Sys.time(), "%H:%M:%S"),
                      normalizePath(models_dir, winslash="/", mustWork=FALSE)))
    }
  }
  
  message(sprintf("Predicting split='%s' | mode='%s' | run_index=%d", predict_split, CLASSIFICATION_MODE, run_index))
  
  metrics_rows <- list(); preds_rows <- list(); mr_idx <- 0L; pr_idx <- 0L
  requested_grid <- list()
  .collect_rg <- function(seed, slot, file, status){
    requested_grid[[length(requested_grid)+1L]] <<- data.frame(
      seed=seed, slot=slot, found_file=ifelse(is.na(file), NA_character_, file),
      status=status, stringsAsFactors=FALSE
    )
  }
  
  .extract_probs <- function(res){
    if (is.null(res)) return(NULL)
    for (nm in c("probs","probabilities","preds","predicted_output","Y_hat","y_hat")) {
      if (!is.null(res[[nm]])) {
        M <- res[[nm]]
        if (is.vector(M))     M <- matrix(as.numeric(M), ncol=1L)
        if (is.data.frame(M)) M <- as.matrix(M)
        if (is.matrix(M))     return(M)
      }
    }
    NULL
  }
  
  processed_any <- FALSE
  
  for (seed in seeds) for (slot in slots) {
    meta <- NULL; meta_sym <- NULL; meta_path <- NA_character_
    
    if (identical(source, "env")) {
      if (is.null(env_meta_name_base)) stop("env_meta_name_base is required when source='env'")
      meta <- .resolve_meta_env(env_meta_name_base, slot, seed)
      meta_sym <- .safe_name("LP_META", "slot", slot, "seed", seed)
      assign(meta_sym, meta, envir=.GlobalEnv)
      meta_path <- sprintf("ENV:%s", meta_sym)
    } else {
      meta_path <- .find_meta_file_local(models_dir, slot, seed)
      if (!is.na(meta_path) && nzchar(meta_path)) {
        meta <- readRDS(meta_path)
        meta_sym <- .safe_name("LP_META", "slot", slot, "seed", seed)
        assign(meta_sym, meta, envir=.GlobalEnv)
        message(sprintf("[LP] Using %s = %s ", meta_sym, meta_path))
      } else {
        message(sprintf("[LP] Skip: seed=%d slot=%d | no metadata under %s", seed, slot, models_dir))
        .collect_rg(seed, slot, NA_character_, "skipped_missing_metadata"); next
      }
    }
    
    # Pull split data (mirror SingleRuns flow)
    sl <- tolower(predict_split)
    X  <- if (sl=="test") meta$X_test else if (sl=="validation") meta$X_validation else (meta$X %||% meta$X_train)
    y  <- if (sl=="test") meta$y_test else if (sl=="validation") meta$y_validation else (meta$y %||% meta$y_train)
    if (is.null(X) || is.null(y)) {
      warning(sprintf("Split '%s' not present (seed=%d, slot=%d) — skipping", predict_split, seed, slot))
      .collect_rg(seed, slot, meta_path, "skipped_missing_split"); next
    }
    
    message(sprintf("[DSE-DBG %s]  OUTPUT_DIR=%s ", format(Sys.time(), "%H:%M:%S"),
                    normalizePath(out_dir, winslash="/", mustWork=FALSE)))
    message(sprintf("[DSE-DBG %s]  CFG split=%s mode=%s run=%d seed=%d slot=%d ",
                    format(Sys.time(), "%H:%M:%S"), predict_split, CLASSIFICATION_MODE, run_index, seed, slot))
    
    res <- tryCatch(
      DDESONN_predict_eval(
        LOAD_FROM_RDS         = FALSE,
        ENV_META_NAME         = meta_sym,
        INPUT_SPLIT           = predict_split,
        CLASSIFICATION_MODE   = CLASSIFICATION_MODE,
        RUN_INDEX             = run_index,
        SEED                  = seed,
        MODEL_SLOT            = slot,
        OUTPUT_DIR            = out_dir,
        SAVE_METRICS_RDS      = FALSE,
        METRICS_PREFIX        = sprintf("metrics_%s", predict_split),
        AGG_PREDICTIONS_FILE  = NULL,
        AGG_METRICS_FILE      = NULL,
        DEBUG                 = TRUE,
        OUT_DIR_ASSERT        = out_dir
      ),
      error=function(e){ warning(sprintf("[DSE-DBG %s]  [FAIL] eval error: %s", format(Sys.time(), "%H:%M:%S"), conditionMessage(e))); NULL }
    )
    
    got_outputs <- FALSE
    if (!is.null(res)) {
      mr <- res$metrics_row %||% res$metrics %||% res$metrics_row_compact
      if (!is.null(mr) && is.data.frame(mr) && nrow(mr)) {
        mr$run_index           <- as.integer(run_index)
        mr$seed                <- as.integer(seed)
        mr$model_slot          <- as.integer(slot)
        mr$split               <- tolower(predict_split)
        mr$CLASSIFICATION_MODE <- tolower(CLASSIFICATION_MODE)
        metrics_rows[[ (mr_idx <- mr_idx + 1L) ]] <- mr
        got_outputs <- TRUE
      }
      
      P <- .extract_probs(res)
      if (!is.null(P) && is.matrix(P) && nrow(P) > 0L) {
        y_true <- .as_num_vec(y)
        if (length(y_true) != nrow(P)) {
          nmin <- min(length(y_true), nrow(P))
          y_true <- y_true[seq_len(nmin)]; P <- P[seq_len(nmin), , drop=FALSE]
        }
        if (CLASSIFICATION_MODE == "binary") {
          thr <- suppressWarnings(as.numeric(res$results_compact$tuned_threshold)) %||%
            suppressWarnings(as.numeric(res$tuned_threshold))
          if (!is.finite(thr)) thr <- 0.5
          y_prob <- as.numeric(P[,1]); y_pred <- as.integer(y_prob >= thr)
        } else if (CLASSIFICATION_MODE == "multiclass") {
          y_prob <- apply(P, 1, max); y_pred <- max.col(P, ties.method = "first")
        } else {
          y_prob <- as.numeric(P[,1]); y_pred <- y_prob
        }
        n <- length(y_true)
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
        preds_rows[[ (pr_idx <- pr_idx + 1L) ]] <- pr
        got_outputs <- TRUE
      }
    }
    
    .collect_rg(seed, slot, meta_path, if (got_outputs) "ok" else "failed_eval_no_outputs")
    processed_any <- processed_any || got_outputs
    message(sprintf(" ✓ seed=%d | slot=%d", seed, slot))
  }
  
  if (!processed_any) warning("No (seed, slot) pairs were processed. Check available models in: ",
                              if (is.null(models_dir)) "<env>" else models_dir)
  
  # ---------- outputs ----------
  agg_metrics_path <- file.path(out_dir, sprintf("agg_metrics_%s.rds", predict_split))
  agg_preds_path   <- file.path(out_dir, sprintf("agg_predictions_%s.rds", predict_split))
  if (isTRUE(overwrite)) {
    suppressWarnings(try(unlink(agg_metrics_path, force=TRUE), silent=TRUE))
    suppressWarnings(try(unlink(agg_preds_path,   force=TRUE), silent=TRUE))
  }
  
  agg_metrics     <- if (length(metrics_rows)) do.call(rbind, metrics_rows) else data.frame()
  agg_predictions <- if (length(preds_rows))   do.call(rbind, preds_rows)   else data.frame()
  
  # Match SingleRuns schema when reading EnsembleRuns
  if (identical(source, "EnsembleRuns")) {
    agg_metrics <- .normalize_metrics_schema_minimal(
      agg_metrics,
      predict_split = predict_split,
      CLASSIFICATION_MODE = CLASSIFICATION_MODE,
      force_run_index = TRUE
    )
  }
  
  saveRDS(agg_metrics,     agg_metrics_path)
  saveRDS(agg_predictions, agg_preds_path)
  if (!file.exists(agg_metrics_path)) stop("Internal: metrics file not written")
  if (!file.exists(agg_preds_path))   stop("Internal: predictions file not written")
  
  message("LoadandPredict complete.")
  
  requested_grid_df <- if (length(requested_grid)) do.call(rbind, requested_grid) else
    data.frame(seed=integer(), slot=integer(), found_file=character(), status=character(), stringsAsFactors=FALSE)
  
  list(
    output_dir            = out_dir,
    agg_metrics_file      = agg_metrics_path,
    agg_predictions_file  = agg_preds_path,
    agg_metrics_preview   = utils::head(agg_metrics),
    agg_predictions_rows  = nrow(agg_predictions),
    requested_grid        = requested_grid_df
  )
}

# -------------------------------
# Examples (commented)
# -------------------------------
# # EnsembleRuns — models/main (default)
ex1 <- LoadandPredict(
  source="SingleRuns", folder=NULL, seeds=c(1), slots=1:4,
  predict_split="test", CLASSIFICATION_MODE="binary", run_index=1,
  output_dir_base="artifacts/PredictOnly", run_dir_name="predict_from_latest_ensemble_main",
  overwrite=TRUE, ensemble_model_subdir="main"
)
print(readRDS(ex1$agg_metrics_file)); head(readRDS(ex1$agg_predictions_file)); ex1$requested_grid

# # SingleRuns — models/
# ex3 <- LoadandPredict(
#   source="SingleRuns", folder=NULL, seeds=1, slots=1:3,
#   predict_split="test", CLASSIFICATION_MODE="binary", run_index=1,
#   output_dir_base="artifacts/PredictOnly", run_dir_name="predict_from_latest_single_all",
#   overwrite=TRUE
# )
# print(readRDS(ex3$agg_metrics_file)); head(readRDS(ex3$agg_predictions_file)); ex3$requested_grid
