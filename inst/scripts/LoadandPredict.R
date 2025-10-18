# =====================================================================
# artifacts/PredictOnly/LoadandPredict.R
# Predict-only runner (keeps your DDESONN_predict_eval() intact)
#
# TWO EXPLICIT FLOWS (exactly as requested):
#   1) FLOW="TestDDESONN"
#        - METADATA: disk only (run_root/models/.../metadata_*.rds)
#        - DATA X/y: meta-embedded or run_root/datasets/ (no ENV)
#   2) FLOW="API"
#        - METADATA: ENV only (env_meta_name_base). If ENV holds only weights/biases
#          and not a full meta, we assemble a minimal meta with best_*_record.
#        - DATA X/y: ENV first, then run_root/datasets/ (no meta-embedded by default)
#
# NOTES
# - EnsembleRuns: models/<ensemble_model_subdir> (default "main")
# - SingleRuns  : models/
# - Activations can be strings OR functions (closures) — sanitized to names for safety
# - No <<-
# - Rich DEBUG prints with [LP-DBG]/[SAFE-DBG]/[DSE-DBG].
# =====================================================================

suppressWarnings(try(source("R/utils.R"), silent = TRUE))
suppressWarnings(try(source("R/performance_relevance_metrics.R"), silent = TRUE))

# -----------------------
# Schema guard (minimal)
# -----------------------
if (!exists(".normalize_metrics_schema_minimal", mode="function")) {
  .normalize_metrics_schema_minimal <- function(df, predict_split, CLASSIFICATION_MODE, force_run_index=TRUE) {
    if (is.null(df) || !nrow(df)) return(data.frame())
    if (force_run_index && !"run_index" %in% names(df)) df$run_index <- 1L
    df
  }
}

# -----------------------------------------------------------
# Robust best_* records extractor (prefers user's helper)
# -----------------------------------------------------------
.get_best_records <- function(meta, ML_NN, model_index=1L) {
  `%OR%` <- function(a,b) if (!is.null(a)) a else b
  # Prefer user's helper if present
  if (exists("extract_best_records", inherits = TRUE)) {
    rec <- try(extract_best_records(meta, ML_NN = ML_NN, model_index = model_index), silent = TRUE)
    if (!inherits(rec, "try-error") && !is.null(rec)) {
      W <- rec$weights %OR% rec$W %OR% rec$wrec %OR% rec$best_weights_record
      B <- rec$biases  %OR% rec$B %OR% rec$brec %OR% rec$best_biases_record
      if (!is.null(W) && !is.null(B)) return(list(weights = W, biases = B))
    }
  }
  grab <- function(x) if (!is.null(x)) x else NULL
  pick_idx <- function(obj, i) {
    if (is.null(obj)) return(NULL)
    if (is.list(obj) && length(obj) >= i && !is.null(obj[[i]])) return(obj[[i]])
    obj
  }
  srcs_w <- list(
    grab(meta$best_model_metadata$best_weights_record),
    grab(meta$best_weights_record),
    grab(meta$weights_record),
    grab(meta$model$best_weights_record),
    grab(meta$model$weights_record),
    grab(meta$predictor$metadata$best_model_metadata$best_weights_record),
    grab(meta$predictor$metadata$best_weights_record),
    grab(meta$predictor$metadata$weights_record),
    grab(meta$W), grab(meta$weights),
    grab(meta$model$W), grab(meta$model$weights)
  )
  srcs_b <- list(
    grab(meta$best_model_metadata$best_biases_record),
    grab(meta$best_biases_record),
    grab(meta$biases_record),
    grab(meta$model$best_biases_record),
    grab(meta$model$biases_record),
    grab(meta$predictor$metadata$best_model_metadata$best_biases_record),
    grab(meta$predictor$metadata$best_biases_record),
    grab(meta$predictor$metadata$biases_record),
    grab(meta$B), grab(meta$biases),
    grab(meta$model$B), grab(meta$model$biases)
  )
  W <- NULL; B <- NULL
  for (c in srcs_w) { if (!is.null(c)) { W <- pick_idx(c, model_index); if (!is.null(W)) break } }
  for (c in srcs_b) { if (!is.null(c)) { B <- pick_idx(c, model_index); if (!is.null(B)) break } }
  if (is.null(W) || is.null(B)) return(NULL)
  
  # Try user's normalize_records, but fall back cleanly on error (lists-of-layers are OK)
  if (exists("normalize_records", inherits = TRUE)) {
    out <- try(normalize_records(wrec = W, brec = B, ML_NN = ML_NN), silent = TRUE)
    if (!inherits(out, "try-error") && !is.null(out)) return(out)
    message(sprintf("[SAFE-DBG %s] normalize_records failed; using raw records (lists-of-layers).",
                    format(Sys.time(), "%H:%M:%S")))
  }
  list(weights = W, biases = B)
}

# -----------------------
# Small local helpers
# -----------------------
`%||%` <- function(x,y) if (is.null(x)) y else x

.as_num_vec <- function(v){
  if (is.matrix(v))       { if (ncol(v)>1L) stop("labels matrix >1 col"); v <- v[,1] }
  if (is.data.frame(v))   { if (ncol(v)!=1L) stop("labels df !=1 col");   v <- v[[1]] }
  if (is.list(v))         v <- vapply(v, function(z){
    if (is.list(z)) z <- unlist(z, use.names=FALSE)
    suppressWarnings(as.numeric(if (length(z)) z[1] else NA))
  }, numeric(1))
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
.shape_of <- function(x) {
  if (is.null(x)) return("NULL")
  if (is.matrix(x)) return(paste0(dim(x), collapse="x"))
  if (is.list(x))  return(paste0("list[", length(x), "]"))
  class(x)[1]
}
.sanitize_acts_to_names <- function(acts){
  pick_name <- function(a){
    if (is.function(a)) {
      nm <- attr(a, "name")
      if (is.null(nm)) return("linear")
      return(tolower(as.character(nm)))
    }
    tolower(as.character(a))
  }
  if (is.null(acts)) return(NULL)
  if (is.list(acts)) return(vapply(acts, pick_name, character(1)))
  pick_name(acts)
}

# -----------------------
# Split loaders (by flow)
# -----------------------
.load_split_from_env <- function(env_data_prefix, split){
  if (is.null(env_data_prefix)) return(list(X=NULL, y=NULL))
  split <- tolower(split)
  x1 <- sprintf("%s_X_%s", env_data_prefix, split)
  y1 <- sprintf("%s_y_%s", env_data_prefix, split)
  x2 <- sprintf("%s_%s_X", env_data_prefix, split)
  y2 <- sprintf("%s_%s_y", env_data_prefix, split)
  pick <- function(nms){
    for (nm in nms) if (exists(nm, envir=.GlobalEnv, inherits=FALSE)) {
      val <- get(nm, envir=.GlobalEnv)
      message(sprintf("[LP-DBG %s]  ENV pick %s: %s (class=%s)",
                      format(Sys.time(), "%H:%M:%S"), split, nm, paste(class(val), collapse=",")))
      return(val)
    }
    NULL
  }
  list(X = pick(c(x1,x2)), y = pick(c(y1,y2)))
}

.load_split_meta_or_disk <- function(run_root, split, meta) {
  # Used for TestDDESONN flow (meta-embedded or datasets dir; no ENV).
  split <- tolower(split)
  x_keys <- switch(split,
                   "test"       = c("X_test", "x_test"),
                   "validation" = c("X_validation", "x_validation", "X_valid", "x_valid"),
                   "train"      = c("X_train", "x_train", "X"),
                   character(0))
  y_keys <- switch(split,
                   "test"       = c("y_test", "labels_test"),
                   "validation" = c("y_validation", "labels_validation", "y_valid"),
                   "train"      = c("y_train", "labels_train", "y"),
                   character(0))
  pick_first_nonnull <- function(obj, keys) { for (k in keys) if (!is.null(obj[[k]])) return(obj[[k]]); NULL }
  
  # 1) Embedded in meta (classic TestDDESONN)
  X <- pick_first_nonnull(meta, x_keys)
  Y <- pick_first_nonnull(meta, y_keys)
  if (!is.null(X) && !is.null(Y)) {
    attr(X, "LP_SOURCE") <- "meta(TestDDESONN.R)"
    message(sprintf("[LP-DBG %s]  META split '%s' found in meta (embedded)",
                    format(Sys.time(), "%H:%M:%S"), split))
    return(list(X=X, y=Y))
  }
  
  # 2) datasets/ under run_root
  datasets_dir <- NULL
  cand_meta_ds <- pick_first_nonnull(meta, c("datasets_dir", "DATASETS_DIR"))
  if (is.character(cand_meta_ds) && dir.exists(cand_meta_ds)) datasets_dir <- cand_meta_ds
  if (is.null(datasets_dir) && !is.null(run_root)) {
    cand <- file.path(run_root, "datasets")
    if (dir.exists(cand)) datasets_dir <- cand
  }
  if (!is.null(datasets_dir)) {
    message(sprintf("[LP-DBG %s]  DISK datasets dir: %s", format(Sys.time(), "%H:%M:%S"),
                    normalizePath(datasets_dir, winslash="/", mustWork=FALSE)))
    x_paths <- c(
      file.path(datasets_dir, sprintf("X_%s.rds", split)),
      file.path(datasets_dir, sprintf("X-%s.rds", split)),
      file.path(datasets_dir, sprintf("X.%s.rds", split))
    )
    y_paths <- c(
      file.path(datasets_dir, sprintf("y_%s.rds", split)),
      file.path(datasets_dir, sprintf("y-%s.rds", split)),
      file.path(datasets_dir, sprintf("y.%s.rds", split))
    )
    if (split == "validation") {
      x_paths <- c(x_paths, file.path(datasets_dir, "X_valid.rds"))
      y_paths <- c(y_paths, file.path(datasets_dir, "y_valid.rds"))
    }
    if (split == "test") {
      x_paths <- c(x_paths, file.path(datasets_dir, "test_X.rds"))
      y_paths <- c(y_paths, file.path(datasets_dir, "test_y.rds"))
    }
    x_hit <- x_paths[file.exists(x_paths)][1]
    y_hit <- y_paths[file.exists(y_paths)][1]
    if (length(x_hit) && length(y_hit) && nzchar(x_hit) && nzchar(y_hit)) {
      message(sprintf("[LP-DBG %s]  DISK hits: X=%s | y=%s",
                      format(Sys.time(), "%H:%M:%S"),
                      normalizePath(x_hit, winslash="/", mustWork=FALSE),
                      normalizePath(y_hit, winslash="/", mustWork=FALSE)))
      X <- readRDS(x_hit); Y <- readRDS(y_hit)
      attr(X, "LP_SOURCE") <- "disk(API/Test)"
      return(list(X=X, y=Y))
    } else {
      message(sprintf("[LP-DBG %s]  DISK miss for split '%s' (checked %d X, %d y candidates)",
                      format(Sys.time(), "%H:%M:%S"), split, length(x_paths), length(y_paths)))
    }
  }
  
  message(sprintf("[LP-DBG %s]  split '%s' not found in meta or disk", format(Sys.time(), "%H:%M:%S"), split))
  list(X=NULL, y=NULL)
}

# -----------------------
# Run roots & model dirs (used only by TestDDESONN flow)
# -----------------------
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
  } else NULL
}
.find_meta_file_local <- function(models_dir, slot, seed) {
  if (is.null(models_dir) || !dir.exists(models_dir)) return(NA_character_)
  patt <- "metadata.*\\.rds$"
  cand <- list.files(models_dir, pattern = patt, recursive = TRUE, full.names = TRUE)
  if (!length(cand)) return(NA_character_)
  bnames <- basename(cand)
  want <- grepl(sprintf("model_%d", as.integer(slot)), bnames, ignore.case = TRUE) &
    grepl(sprintf("seed\\s*%d", as.integer(seed)), bnames, ignore.case = TRUE)
  cand <- cand[want]
  if (!length(cand)) {
    cand2 <- list.files(models_dir,
                        pattern = sprintf("model_%d.*metadata.*\\.rds$", as.integer(slot)),
                        recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
    if (!length(cand2)) return(NA_character_)
    cand <- cand2
  }
  info <- file.info(cand)
  cand[order(info$mtime, decreasing = TRUE)][1L]
}

# -----------------------
# ENV meta resolver (API flow)
# IMPORTANT: In API flow there is NO disk metadata. We resolve:
#  - a direct ENV meta list; OR
#  - any <base>_model_*_metadata; OR
#  - minimal meta assembled from ENV weights/biases buckets.
# -----------------------
.resolve_meta_env <- function(env_base, slot, seed) {
  if (is.null(env_base) || !nzchar(env_base))
    stop("ENV meta not found: env_meta_name_base is NULL/empty")
  
  exists_local <- function(nm) exists(nm, envir=.GlobalEnv, inherits=FALSE)
  get_local    <- function(nm) get(nm, envir=.GlobalEnv)
  
  # 0) If caller gives the FULL object name, just use it (most robust).
  if (exists_local(env_base)) {
    obj <- get_local(env_base)
    message(sprintf("[LP-DBG %s]  ENV meta (exact) hit: %s", format(Sys.time(), "%H:%M:%S"), env_base))
    return(obj)
  }
  
  # 1) Try common direct candidates that include slot/seed
  direct_cands <- c(
    sprintf("%s_slot_%d_seed_%d", env_base, as.integer(slot), as.integer(seed)),
    sprintf("%s_model_%d_metadata", env_base, as.integer(slot)),
    sprintf("%s_%d_metadata", env_base, as.integer(slot)),
    sprintf("%s_model_%d", env_base, as.integer(slot))
  )
  for (nm in direct_cands) {
    if (exists_local(nm)) {
      message(sprintf("[LP-DBG %s]  ENV meta hit: %s", format(Sys.time(), "%H:%M:%S"), nm))
      return(get_local(nm))
    }
  }
  
  # 2) Flexible fallback: accept ANY metadata for this base (e.g., _model_1_metadata)
  all_objs <- ls(envir=.GlobalEnv, all.names=TRUE)
  base_rgx <- paste0("^", gsub("([\\W])","\\\\\\1", env_base))
  meta_hits <- grep(paste0(base_rgx, ".*(_model_\\d+)?_metadata$"), all_objs, value=TRUE, ignore.case=TRUE)
  if (!length(meta_hits)) {
    meta_hits <- grep(paste0(base_rgx, ".*_model_\\d+$"), all_objs, value=TRUE, ignore.case=TRUE)
  }
  if (length(meta_hits)) {
    pick <- meta_hits[order(nchar(meta_hits), meta_hits)][1]
    message(sprintf("[LP-DBG %s]  ENV meta fallback hit: %s", format(Sys.time(), "%H:%M:%S"), pick))
    return(get(pick, envir=.GlobalEnv))
  }
  
  # 3) Assemble minimal meta from ENV records (weights/biases)
  cand_weights <- c(
    sprintf("%s_weights_seed%d", env_base, as.integer(seed)),
    sprintf("%s_best_weights_seed%d", env_base, as.integer(seed)),
    sprintf("%s_weights", env_base),
    sprintf("%s_best_weights", env_base)
  )
  cand_biases <- c(
    sprintf("%s_biases_seed%d", env_base, as.integer(seed)),
    sprintf("%s_best_biases_seed%d", env_base, as.integer(seed)),
    sprintf("%s_biases", env_base),
    sprintf("%s_best_biases", env_base)
  )
  w_obj <- NULL; b_obj <- NULL
  for (nm in cand_weights) if (exists_local(nm)) { w_obj <- get_local(nm); break }
  for (nm in cand_biases) if (exists_local(nm)) { b_obj <- get_local(nm); break }
  if (!is.null(w_obj) && !is.null(b_obj)) {
    message(sprintf("[LP-DBG %s]  ENV records resolved → minimal meta(best_*_record) assembled", format(Sys.time(), "%H:%M:%S")))
    return(list(best_weights_record = w_obj, best_biases_record = b_obj))
  }
  
  # 4) Last try: bucket lists like "<base>_seed<seed>" or "<base>_model_seed<seed>" or just "<base>"
  for (nm in c(sprintf("%s_seed%d", env_base, as.integer(seed)),
               sprintf("%s_model_seed%d", env_base, as.integer(seed)),
               env_base)) {
    if (exists_local(nm)) {
      bucket <- get_local(nm)
      if (is.list(bucket)) {
        W <- bucket$best_weights_record %||% bucket$weights_record %||% bucket$W %||% bucket$weights
        B <- bucket$best_biases_record  %||% bucket$biases_record  %||% bucket$B %||% bucket$biases
        if (!is.null(W) && !is.null(B)) {
          message(sprintf("[LP-DBG %s]  ENV bucket → minimal meta assembled (W,B).", format(Sys.time(), "%H:%M:%S")))
          return(list(best_weights_record=W, best_biases_record=B))
        }
      }
    }
  }
  
  base_like <- grep(base_rgx, all_objs, value=TRUE)
  stop(sprintf(
    "ENV meta not found for base='%s' (slot=%d seed=%d).\nENV has %d objects matching base:\n- %s",
    env_base, slot, seed, length(base_like), paste(base_like, collapse="\n- ")
  ))
}

# -----------------------------------------------------------
# Predictor shims (forward from best_* if available)
# -----------------------------------------------------------
.ensure_predictor_in_eval_env <- function(){
  if (!exists("DDESONN_predict_eval", inherits=TRUE) || !is.function(DDESONN_predict_eval)) {
    stop("DDESONN_predict_eval() not found.")
  }
  dse_env <- environment(DDESONN_predict_eval)
  message(sprintf("[SAFE-DBG %s] install predictor shims into DDESONN_predict_eval env", format(Sys.time(), "%H:%M:%S")))
  
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
    b <- as.numeric(b); Z <- as.matrix(Z)
    if (length(b) == 1L) return(Z + b)
    if (length(b) == ncol(Z)) return(Z + matrix(rep(b, each=nrow(Z)),
                                                nrow=nrow(Z), ncol=ncol(Z), byrow=FALSE))
    stop(sprintf("Bias length (%d) incompatible with ncol(Z) (%d)", length(b), ncol(Z)))
  }
  .to_num_mat <- function(X){
    if (is.list(X) && !is.data.frame(X)) {
      for (k in c("X","x","data","Data","features","input","inputs","mat","matrix","M"))
        if (!is.null(X[[k]])) return(.to_num_mat(X[[k]]))
      lens <- lengths(X)
      if (length(lens) && length(unique(lens))==1 && unique(lens)>0 &&
          all(sapply(X, function(e) is.numeric(e) || is.integer(e)))) {
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
      idx <- max(1L, min(model_index, length(obj))); return(obj[[idx]])
    }
    obj
  }
  .get_act_for_layer <- function(acts, l){
    if (is.null(acts)) return(NULL)
    if (is.function(acts)) return(acts)
    a <- NULL
    if (is.list(acts)) { if (!is.null(acts[[l]])) a <- acts[[l]] else if (!is.null(acts[l])) a <- acts[[l]] }
    if (is.null(a)) a <- acts
    a
  }
  
  .forward_from_meta <- function(X, meta, model_index=1L){
    `%OR%` <- function(a,b) if (!is.null(a)) a else b
    rec <- .get_best_records(meta, ML_NN = TRUE, model_index = model_index)
    if (is.null(rec)) stop("[shim] could not resolve best_* records from meta")
    W <- rec$W %OR% rec$weights %OR% rec$wrec %OR% rec$best_weights_record
    B <- rec$B %OR% rec$biases  %OR% rec$brec %OR% rec$best_biases_record
    acts <- meta$activation_functions_predict %OR% meta$activation_functions
    W <- .maybe_pick_network(W, model_index=model_index)
    B <- .maybe_pick_network(B, model_index=model_index)
    H  <- .to_num_mat(X)
    L  <- length(W)
    for (l in seq_len(L)) {
      Wl <- .first_num_mat(W[[l]]); if (is.null(Wl)) stop(sprintf("[shim] bad W at layer %d", l))
      Z  <- as.matrix(H) %*% as.matrix(Wl)
      bl <- .first_num_vec(if (l <= length(B)) B[[l]] else NULL)
      Z  <- add_bias(Z, bl)
      H  <- .apply_act(Z, .get_act_for_layer(acts, l))
    }
    as.matrix(H)
  }
  
  DDESONN_fn <- function(X, meta, model_index=1L, ML_NN=TRUE, ...){
    list(predicted_output = .forward_from_meta(X, meta, model_index))
  }
  assign("DDESONN",         DDESONN_fn, envir=dse_env)
  assign("DDESONN_predict", DDESONN_fn, envir=dse_env)
  
  # load_meta shim: ENV or file (file path used only by TestDDESONN flow)
  assign("load_meta",
         function(LOAD_FROM_RDS=FALSE, ENV_META_NAME=NULL, ...){
           if (LOAD_FROM_RDS) {
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

# ===========================================================
# MAIN
# ===========================================================
LoadandPredict <- function(
    FLOW                  = c("TestDDESONN","API"),
    # TestDDESONN flow uses these for disk meta
    source                = c("EnsembleRuns","SingleRuns","env"),
    folder                = NULL,
    ensemble_model_subdir = "main",
    
    # Shared controls
    seeds                 = c(1L,2L),
    slots                 = c(1L,2L,3L),
    predict_split         = c("test","validation","train"),
    CLASSIFICATION_MODE   = c("binary","multiclass","regression"),
    run_index             = 1L,
    output_dir_base       = "artifacts/PredictOnly",
    run_dir_name          = "predict_flow",
    overwrite             = TRUE,
    
    # API flow only
    env_meta_name_base    = NULL,   # required for FLOW="API" (ENV-only meta)
    env_data_prefix       = NULL,   # X/y in ENV
    
    # Direct matrices override (either flow)
    X_override            = NULL,
    y_override            = NULL
){
  VERBOSE <- TRUE
  vcat <- function(...) if (VERBOSE) message(sprintf(...))
  
  FLOW                 <- match.arg(FLOW)
  source               <- match.arg(source)
  predict_split        <- match.arg(predict_split)
  CLASSIFICATION_MODE  <- match.arg(CLASSIFICATION_MODE)
  run_index            <- as.integer(run_index)
  seeds                <- as.integer(seeds)
  slots                <- as.integer(slots)
  
  .ensure_predictor_in_eval_env()
  
  # Resolve run roots only for TestDDESONN flow
  run_root  <- NULL
  models_dir<- NULL
  if (identical(FLOW, "TestDDESONN")) {
    run_root   <- .resolve_run_root(source, folder)
    models_dir <- .models_dir_for(run_root, source, ensemble_model_subdir)
    vcat("[LP-DBG %s] FLOW=TestDDESONN | SOURCE=%s | RUN_ROOT=%s",
         format(Sys.time(), "%H:%M:%S"), source, normalizePath(run_root, winslash="/", mustWork=FALSE))
    if (!is.null(models_dir)) {
      vcat("[LP-DBG %s] MODELS_DIR=%s", format(Sys.time(), "%H:%M:%S"),
           normalizePath(models_dir, winslash="/", mustWork=FALSE))
    }
  } else {
    # API flow: explicitly avoid touching disk metadata
    if (is.null(env_meta_name_base))
      stop("FLOW='API' requires env_meta_name_base (metadata in ENV).")
    vcat("[LP-DBG %s] FLOW=API | ENV_META_BASE=%s", format(Sys.time(), "%H:%M:%S"), env_meta_name_base)
  }
  
  # Output dir
  out_dir <- file.path(output_dir_base, run_dir_name)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  vcat("Predicting split='%s' | mode='%s' | run_index=%d | flow=%s", predict_split, CLASSIFICATION_MODE, run_index, FLOW)
  
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
    vcat("[LP-DBG %s] -------- run_index=%d | seed=%d | slot=%d --------",
         format(Sys.time(), "%H:%M:%S"), run_index, seed, slot)
    
    meta <- NULL; meta_sym <- NULL; meta_path <- NA_character_
    
    if (identical(FLOW, "TestDDESONN")) {
      # Disk RDS metadata ONLY
      meta_path <- .find_meta_file_local(models_dir, slot, seed)
      if (!is.na(meta_path) && nzchar(meta_path)) {
        vcat("[LP-DBG %s] meta path hit: %s", format(Sys.time(), "%H:%M:%S"),
             normalizePath(meta_path, winslash="/", mustWork=FALSE))
        meta <- readRDS(meta_path)
        meta_sym <- .safe_name("LP_META", "slot", slot, "seed", seed)
        assign(meta_sym, meta, envir=.GlobalEnv)
        vcat("[LP] Using %s = %s ", meta_sym, meta_path)
      } else {
        vcat("[LP] Skip: seed=%d slot=%d | no metadata under %s", seed, slot, models_dir)
        .collect_rg(seed, slot, NA_character_, "skipped_missing_metadata")
        next
      }
    } else {
      # API flow: ENV metadata ONLY (or assemble minimal meta from ENV weights/biases)
      meta <- .resolve_meta_env(env_meta_name_base, slot, seed)
      meta_sym <- .safe_name("LP_META", "slot", slot, "seed", seed)
      
      # --- sanitize activations to avoid closure→character coercion in downstream logs ---
      acts_pred <- meta$activation_functions_predict %||% meta$activation_functions
      meta$activation_functions_predict <- .sanitize_acts_to_names(acts_pred)
      meta$activation_functions         <- meta$activation_functions_predict
      
      assign(meta_sym, meta, envir=.GlobalEnv)
      meta_path <- sprintf("ENV:%s", meta_sym)
    }
    
    # Split selection per flow
    sl <- tolower(predict_split)
    
    X <- NULL; y <- NULL; src_used <- NA_character_
    if (!is.null(X_override) && !is.null(y_override)) {
      X <- X_override; y <- y_override; src_used <- "override"
      vcat("[LP-DBG %s]  split '%s' resolved from: override (X_override/y_override)", format(Sys.time(), "%H:%M:%S"), sl)
    } else if (identical(FLOW, "TestDDESONN")) {
      # meta-embedded OR disk datasets (no env) — classic test runner behavior
      sp <- .load_split_meta_or_disk(run_root, sl, meta)
      X <- sp$X; y <- sp$y; src_used <- if (!is.null(X) && !is.null(y)) attr(X, "LP_SOURCE") %||% "disk(Test)" else NA_character_
    } else {
      # API: ENV first, then datasets dir (no meta-embedded by default)
      sp_env <- .load_split_from_env(env_data_prefix, sl)
      if (!is.null(sp_env$X) && !is.null(sp_env$y)) {
        X <- sp_env$X; y <- sp_env$y; src_used <- "env(API)"
      } else {
        # optional disk fallback if source/folder provided
        run_root_api <- try(.resolve_run_root(source, folder), silent = TRUE)
        if (inherits(run_root_api, "try-error")) run_root_api <- NULL
        sp_ds <- .load_split_meta_or_disk(run_root_api, sl, list())  # empty meta to skip meta-embedded
        X <- sp_ds$X; y <- sp_ds$y; src_used <- if (!is.null(X) && !is.null(y)) "disk(API)" else NA_character_
      }
    }
    
    if (is.null(X) || is.null(y)) {
      stop(sprintf("No data found for split '%s' under FLOW='%s'.", sl, FLOW))
    }
    
    ## --- y normalization (avoid one-hot pitfalls in multiclass) ---
    if (tolower(CLASSIFICATION_MODE) == "multiclass") {
      if (is.matrix(y) && ncol(y) > 1L) {
        # one-hot -> class ids (1..K)
        y <- as.integer(max.col(y, ties.method = "first"))
      } else if (is.factor(y)) {
        y <- as.integer(y)
      } else if (is.data.frame(y) && ncol(y) == 1L) {
        y <- as.numeric(y[[1]])
      } else if (is.matrix(y) && ncol(y) == 1L) {
        y <- as.numeric(y[,1])
      } else {
        y <- as.numeric(y)
      }
    } else {
      # binary or regression: keep existing behavior
      if (is.data.frame(y) && ncol(y) == 1L) y <- as.numeric(y[[1]])
      if (is.matrix(y) && ncol(y) == 1L)     y <- as.numeric(y[,1])
    }
    
    # expose X/y to meta for eval fns that expect meta$X_*/y_*
    meta[[switch(sl,"test"="X_test","validation"="X_validation","train"="X_train")]] <- X
    meta[[switch(sl,"test"="y_test","validation"="y_validation","train"="y_train")]] <- y
    
    # Best-records elevation; if present we null predictors to force forward-from-records
    ML_NN <- isTRUE(meta$ML_NN) || identical(meta$network_type, "ML_NN")
    best  <- .get_best_records(meta, ML_NN = ML_NN, model_index = slot)
    if (!is.null(best)) {
      meta$best_weights_record <- best$weights %||% best$W %||% best$wrec %||% meta$best_weights_record
      meta$best_biases_record  <- best$biases  %||% best$B %||% best$brec %||% meta$best_biases_record
      meta$predictor_fn    <- NULL
      meta$predictor       <- NULL
      meta$predictor_class <- NULL
    }
    assign(meta_sym, meta, envir=.GlobalEnv)
    
    # If no records at all, skip; (keeps flows deterministic)
    if (is.null(meta$best_weights_record) || is.null(meta$best_biases_record)) {
      vcat("[LP-DBG %s]  best_* records unresolved for seed=%d slot=%d — skipping pair.",
           format(Sys.time(), "%H:%M:%S"), seed, slot)
      .collect_rg(seed, slot, meta_path, "skipped_missing_best_records")
      next
    }
    
    vcat("[SAFE-DBG %s] X dims=%d x %d | y len=%d | source=%s",
         format(Sys.time(), "%H:%M:%S"),
         nrow(as.matrix(X)), ncol(as.matrix(X)), length(as.vector(y)), src_used)
    
    vcat("[DSE-DBG %s]  OUTPUT_DIR=%s ", format(Sys.time(), "%H:%M:%S"),
         normalizePath(out_dir, winslash="/", mustWork=FALSE))
    vcat("[DSE-DBG %s]  CFG split=%s mode=%s run=%d seed=%d slot=%d ",
         format(Sys.time(), "%H:%M:%S"), predict_split, CLASSIFICATION_MODE, run_index, seed, slot)
    
    res <- tryCatch(
      DDESONN_predict_eval(
        LOAD_FROM_RDS         = identical(FLOW, "TestDDESONN"), # only TRUE for TestDDESONN if you use file path inside
        ENV_META_NAME         = meta_sym,         # points to ENV meta (real or minimal)
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
      error=function(e){
        warning(sprintf("[DSE-DBG %s]  [FAIL] predict/eval error: %s",
                        format(Sys.time(), "%H:%M:%S"), conditionMessage(e))); NULL
      }
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
        vcat("[LP-DBG %s]  metrics captured: %d rows", format(Sys.time(), "%H:%M:%S"), nrow(mr))
        got_outputs <- TRUE
      }
      
      y_true <- if (tolower(CLASSIFICATION_MODE) == "multiclass" && is.matrix(y) && ncol(y) > 1L) {
        as.integer(max.col(y, ties.method = "first"))
      } else {
        .as_num_vec(y)
      }
      
      if (!is.null(P) && is.matrix(P) && nrow(P) > 0L) {
        y_true <- .as_num_vec(y)
        if (length(y_true) != nrow(P)) {
          nmin <- min(length(y_true), nrow(P))
          y_true <- y_true[seq_len(nmin)]; P <- P[seq_len(nmin), , drop=FALSE]
          vcat("[LP-DBG %s]  aligned lengths: y_true=%d, P=%d", format(Sys.time(), "%H:%M:%S"), length(y_true), nrow(P))
        }
        if (CLASSIFICATION_MODE == "binary") {
          thr <- suppressWarnings(as.numeric(res$results_compact$tuned_threshold)) %||%
            suppressWarnings(as.numeric(res$tuned_threshold))
          if (!is.finite(thr)) thr <- 0.5
          y_prob <- as.numeric(P[,1]); y_pred <- as.integer(y_prob >= thr)
          vcat("[LP-DBG %s]  binary preds: n=%d | thr=%.4f", format(Sys.time(), "%H:%M:%S"), length(y_prob), thr)
        } else if (CLASSIFICATION_MODE == "multiclass") {
          y_prob <- apply(P, 1, max); y_pred <- max.col(P, ties.method = "first")
          vcat("[LP-DBG %s]  multiclass preds: n=%d | k=%d", format(Sys.time(), "%H:%M:%S"), length(y_prob), ncol(P))
        } else {
          y_prob <- as.numeric(P[,1]); y_pred <- y_prob
          vcat("[LP-DBG %s]  regression preds: n=%d", format(Sys.time(), "%H:%M:%S"), length(y_prob))
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
        vcat("[LP-DBG %s]  predictions captured: %d rows", format(Sys.time(), "%H:%M:%S"), nrow(pr))
        got_outputs <- TRUE
      } else {
        vcat("[LP-DBG %s]  no prediction matrix returned", format(Sys.time(), "%H:%M:%S"))
      }
    } else {
      vcat("[LP-DBG %s]  eval returned NULL", format(Sys.time(), "%H:%M:%S"))
    }
    
    .collect_rg(seed, slot, meta_path, if (got_outputs) "ok" else "failed_eval_no_outputs")
    processed_any <- processed_any || got_outputs
    vcat(" ✓ seed=%d | slot=%d", seed, slot)
  }
  
  if (!processed_any) warning("No (seed, slot) pairs were processed. Flow=", FLOW,
                              if (identical(FLOW,"TestDDESONN")) paste0("; check models in: ", if (is.null(models_dir)) "<none>" else models_dir) else "")
  
  # ---------- outputs ----------
  agg_metrics_path <- file.path(out_dir, sprintf("agg_metrics_%s.rds", predict_split))
  agg_preds_path   <- file.path(out_dir, sprintf("agg_predictions_%s.rds", predict_split))
  if (overwrite) {
    suppressWarnings(try(unlink(agg_metrics_path, force=TRUE), silent=TRUE))
    suppressWarnings(try(unlink(agg_preds_path,   force=TRUE), silent=TRUE))
  }
  
  agg_metrics     <- if (length(metrics_rows)) do.call(rbind, metrics_rows) else data.frame()
  agg_predictions <- if (length(preds_rows))   do.call(rbind, preds_rows)   else data.frame()
  
  if (identical(FLOW, "TestDDESONN")) {
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
  
  vcat("[LP-DBG %s] agg_metrics_file=%s", format(Sys.time(), "%H:%M:%S"),
       normalizePath(agg_metrics_path, winslash="/", mustWork=FALSE))
  vcat("[LP-DBG %s] agg_predictions_file=%s", format(Sys.time(), "%H:%M:%S"),
       normalizePath(agg_preds_path, winslash="/", mustWork=FALSE))
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
# Examples (disabled)
# -------------------------------

# 1) TestDDESONN flow (disk metadata; meta/disk X/y; no ENV)
# ex_test <- LoadandPredict(
#   FLOW="TestDDESONN",
#   source="EnsembleRuns", folder=NULL, seeds=c(1), slots=1:5,
#   predict_split="test", CLASSIFICATION_MODE="binary", run_index=1,
#   output_dir_base="artifacts/PredictOnly", run_dir_name="predict_test_flow",
#   overwrite=TRUE, ensemble_model_subdir="main"
# )

# 2) API flow (ENV metadata + ENV/disk X/y; no meta-embedded)
ex_api <- LoadandPredict(
  FLOW="API",
  env_meta_name_base="Ensemble_Main_0_model_1_metadata",  # exact ENV object name or a base like "Ensemble_Main_0_model"
  env_data_prefix="LP_DATA",                               # e.g., LP_DATA_X_test / LP_DATA_y_test
  seeds=c(1), slots=1:5,
  predict_split="test", CLASSIFICATION_MODE="binary", run_index=1,
  output_dir_base="artifacts/PredictOnly", run_dir_name="predict_api_flow",
  overwrite=TRUE
)
