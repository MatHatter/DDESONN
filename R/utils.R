# ===============================================================
# DeepDynamic -- DDESONN
# Deep Dynamic Experimental Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2026 Mathew William Armitage Fok
#
# Released under the MIT License. See the file LICENSE.
# ===============================================================

`%||%` <- function(a,b) if (is.null(a) || length(a)==0) b else a  #$$$$$$$$$$$$$


# ============================================================  #$$$$$$$$$$$$$
# Console / debug / table helpers (verbosity control)           #$$$$$$$$$$$$$
# ============================================================   #$$$$$$$$$$$$$
ddesonn_console_log <- function(msg, level = c("important", "info"), verbose = NULL, verboseLow = NULL, ...) {  #$$$$$$$$$$$$$
  level <- match.arg(level)  #$$$$$$$$$$$$$
  if (is.null(verbose)) verbose <- FALSE  #$$$$$$$$$$$$$
  if (is.null(verboseLow)) verboseLow <- FALSE  #$$$$$$$$$$$$$
  verbose <- isTRUE(verbose)  #$$$$$$$$$$$$$
  verboseLow <- isTRUE(verboseLow)  #$$$$$$$$$$$$$
  msg <- paste0(msg, collapse = "")  #$$$$$$$$$$$$$
  if (identical(level, "important")) {  #$$$$$$$$$$$$$
    if (verboseLow) cat(msg, "\n", sep = "")  #$$$$$$$$$$$$$
  } else {  #$$$$$$$$$$$$$
    if (verbose) cat(msg, "\n", sep = "")  #$$$$$$$$$$$$$
  }  #$$$$$$$$$$$$$
  invisible(NULL)  #$$$$$$$$$$$$$
}  #$$$$$$$$$$$$$

ddesonn_debug <- function(msg, debug = FALSE) {  #$$$$$$$$$$$$$
  if (is.null(debug)) {  #$$$$$$$$$$$$$
    debug <- getOption("DDESONN.debug", FALSE)  #$$$$$$$$$$$$$
  }  #$$$$$$$$$$$$$
  debug <- isTRUE(debug)  #$$$$$$$$$$$$$
  if (isTRUE(debug)) {  #$$$$$$$$$$$$$
    msg <- paste0(msg, collapse = "")  #$$$$$$$$$$$$$
    cat(msg, "\n", sep = "")  #$$$$$$$$$$$$$
  }  #$$$$$$$$$$$$$
  invisible(NULL)  #$$$$$$$$$$$$$
}  #$$$$$$$$$$$$$

ddesonn_viewTables <- function(x, title = NULL, ...) {  #$$$$$$$$$$$$$
  if (!is.null(title)) message(title)  #$$$$$$$$$$$$$

  # Single authoritative renderer for table-like output.
  # viewTables=TRUE should always produce polished tabular formatting.
  if (is.data.frame(x) || is.matrix(x)) {
    if (!requireNamespace("knitr", quietly = TRUE)) {
      stop(
        "`viewTables = TRUE` requires the 'knitr' package for polished table rendering.",
        call. = FALSE
      )
    }

    fmt <- if (isTRUE(getOption("knitr.in.progress")) && isTRUE(knitr::is_html_output())) {
      "html"
    } else if (isTRUE(getOption("knitr.in.progress")) && isTRUE(knitr::is_latex_output())) {
      "latex"
    } else {
      "pipe"
    }

    tbl <- knitr::kable(x, format = fmt, ...)
    print(tbl)
    return(invisible(x))
  }

  # Non-tabular objects are printed as-is.
  print(x, ...)
  invisible(x)  #$$$$$$$$$$$$$
}  #$$$$$$$$$$$$$



# ============================================================  #$$$$$$$$$$$$$
# SECTION: Boxplot theme + title size consistency notes         #$$$$$$$$$$$$$
# ============================================================   #$$$$$$$$$$$$$
# Why identical code can yield different ggplot2 title sizes:    #$$$$$$$$$$$$$
# - theme_set()/theme_update() or global theme options earlier   #$$$$$$$$$$$$$
# - base_size differences in theme_minimal(base_size=...)        #$$$$$$$$$$$$$
# - plot.title inherits size from current theme text             #$$$$$$$$$$$$$
# - device scaling/DPI differences (ggsave/png/cairo/ragg)       #$$$$$$$$$$$$$
# - knitr/RMarkdown scaling (fig.retina/out.width/CSS)           #$$$$$$$$$$$$$
# - facet/strip text scaling interactions (perceived size)       #$$$$$$$$$$$$$
# - theme order: last-added theme overrides earlier choices      #$$$$$$$$$$$$$
# This helper sets explicit base_size + title size to enforce    #$$$$$$$$$$$$$
# consistent boxplot titles across runs/environments.            #$$$$$$$$$$$$$
ddesonn_boxplot_theme <- function(base_size = 12, title_size = 14) {  #$$$$$$$$$$$$$
  ggplot2::theme_minimal(base_size = base_size) +  #$$$$$$$$$$$$$
    ggplot2::theme(  #$$$$$$$$$$$$$
      text = ggplot2::element_text(size = base_size),  #$$$$$$$$$$$$$
      plot.title = ggplot2::element_text(hjust = 0.5, size = title_size, lineheight = 1.05, margin = ggplot2::margin(b = 8)),  #$$$$$$$$$$$$$
      plot.margin = ggplot2::margin(t = 12, r = 8, b = 8, l = 8)  #$$$$$$$$$$$$$
    )  #$$$$$$$$$$$$$
}  #$$$$$$$$$$$$$

ddesonn_wrap_plot_title <- function(title, width = 55L) {  #$$$$$$$$$$$$$
  if (is.null(title) || !length(title) || !nzchar(title[1])) return(title)  #$$$$$$$$$$$$$
  paste(strwrap(as.character(title[1]), width = as.integer(width)), collapse = "\n")  #$$$$$$$$$$$$$
}  #$$$$$$$$$$$$$


# ============================================================
# Activation Normalization Utility
# Converts strings or mixed specs into callable functions
# ============================================================

# ------------------------------------------------------------
# 1) Activation Normalizer (drop-in)
# ------------------------------------------------------------
.ddesonn_normalize_activations <- function(acts, L) {
  stopifnot(is.numeric(L), length(L) == 1L, L >= 1L, is.finite(L))
  L <- as.integer(L)
  
  resolve_one <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.function(x)) {
      if (is.null(attr(x, "name"))) attr(x, "name") <- "unnamed_function"
      return(x)
    }
    if (is.character(x)) {
      key <- tolower(x[1]); key <- gsub("[- ]", "_", key, perl = TRUE)
      if (key %in% c("linear","none")) key <- "identity"
      if (key == "logistic") key <- "sigmoid"
      fn <- switch(key,
                   "relu"=relu, "sigmoid"=sigmoid, "softmax"=softmax, "tanh"=tanh, "identity"=identity,
                   "leaky_relu"=leaky_relu, "elu"=elu, "swish"=swish, "gelu"=gelu, "selu"=selu, "mish"=mish,
                   "hard_sigmoid"=if (exists("hard_sigmoid","function")) hard_sigmoid else NULL,
                   "softplus"    =if (exists("softplus",    "function")) softplus     else NULL,
                   "prelu"       =if (exists("prelu",       "function")) prelu        else NULL,
                   "bent_identity"=if (exists("bent_identity","function")) bent_identity else NULL,
                   "maxout"      =if (exists("maxout",      "function")) maxout       else NULL,
                   NULL
      )
      if (is.null(fn)) stop(sprintf("Unsupported activation: '%s'", x[1]), call. = FALSE)
      if (is.null(attr(fn, "name"))) attr(fn, "name") <- key
      return(fn)
    }
    stop(sprintf("Unsupported activation spec type: %s", class(x)[1]), call. = FALSE)
  }
  
  # convert input spec to list, recycle last to length L, resolve each
  elems <- if (is.list(acts)) {
    acts
  } else if (is.function(acts) || is.null(acts) || (is.character(acts) && length(acts) == 1L)) {
    list(acts)
  } else if (is.character(acts) && length(acts) > 1L) {
    as.list(acts)
  } else {
    stop(sprintf("activation_functions must be NULL | function | string | character vector | list; got %s",
                 class(acts)[1]), call. = FALSE)
  }
  
  if (length(elems) < L)      elems <- c(elems, rep(list(elems[[length(elems)]]), L - length(elems)))
  else if (length(elems) > L) elems <- elems[seq_len(L)]
  
  lapply(elems, resolve_one)
}


# -----------------------------------------------------------------
# Multiclass label normalizer (keeps scopes local; no globals/<<-)
# -----------------------------------------------------------------
.normalize_mc_targets <- function(labels, P, meta = NULL) {
  # Ensure prediction matrix has usable column names
  if (is.null(colnames(P)) || !ncol(P)) {
    cls <- NULL
    if (!is.null(meta) && !is.null(meta$y_train)) {
      yy <- meta$y_train
      if (is.factor(yy)) {
        cls <- levels(yy)
      } else if (is.character(yy)) {
        cls <- sort(unique(yy))
      } else if (is.matrix(yy) && ncol(yy) > 1L && !is.null(colnames(yy))) {
        cls <- colnames(yy)
      }
    }
    if (is.null(cls) || !length(cls)) {
      cls <- sprintf("class_%02d", seq_len(ncol(P)))
    }
    colnames(P) <- cls[seq_len(ncol(P))]
  }
  
  cls <- colnames(P)
  
  # Coerce labels into factor over cls
  if (is.matrix(labels) && ncol(labels) > 1L) {
    L <- labels
    if (is.null(colnames(L))) {
      colnames(L) <- sprintf("class_%02d", seq_len(ncol(L)))
    }
    keep <- intersect(cls, colnames(L))
    if (!length(keep)) {
      stop("[metrics_align] label matrix has no overlapping columns with predictions.")
    }
    L <- L[, keep, drop = FALSE]
    miss <- setdiff(cls, colnames(L))
    if (length(miss)) {
      L <- cbind(L, matrix(0, nrow = nrow(L), ncol = length(miss),
                           dimnames = list(NULL, miss)))
    }
    L <- L[, cls, drop = FALSE]
    y_idx <- max.col(L, ties.method = "first")
  } else {
    v <- labels
    if (is.factor(v)) {
      v <- as.character(v)
    } else if (is.numeric(v)) {
      v <- as.character(v)
    }
    v <- as.character(v)
    ok <- v %in% cls & !is.na(v)
    if (!any(ok)) {
      stop(sprintf("[metrics_align] all %d labels are unknown. First few labels: %s | classes: %s",
                   length(v), paste(utils::head(unique(v), 6), collapse = ","),
                   paste(cls, collapse = ",")))
    }
    dropped <- sum(!ok)
    if (dropped) {
      message(sprintf("[metrics_align] dropping %d/%d rows with unknown/NA labels. Examples dropped: %s",
                      dropped, length(v),
                      paste(utils::head(unique(v[!ok]), 6), collapse = ", ")))
    }
    v <- v[ok]
    P <- P[ok, , drop = FALSE]
    y_idx <- match(v, cls)
  }
  
  list(
    P = P,
    y_idx = y_idx,
    classes = cls
  )
}





probe_preds_vs_labels <- function(preds, labs, tag = "GENERIC", save_global = FALSE,
                                  verbose = verbose) {
  r2_val <- tryCatch({
    ss_tot <- sum((labs - mean(labs))^2, na.rm = TRUE)
    ss_res <- sum((labs - preds)^2, na.rm = TRUE)
    1 - ss_res / ss_tot
  }, error = function(e) NA_real_)
  
  if (verbose) {
    message(sprintf(
      "[PROBE-R2 %s] preds min=%.4f mean=%.4f max=%.4f",
      tag, min(preds), mean(preds), max(preds)
    ))
    message(sprintf(
      "[PROBE-R2 %s] labs  min=%.4f mean=%.4f max=%.4f",
      tag, min(labs), mean(labs), max(labs)
    ))
    message(sprintf("[PROBE-R2 %s] R^2=%.6f (n=%d)",
                    tag, r2_val, length(preds)))
  }
  
  if (save_global) {
    dbg <- list(
      tag   = tag,
      preds = list(min = min(preds), mean = mean(preds), max = max(preds)),
      labs  = list(min = min(labs),  mean = mean(labs),  max = max(labs)),
      r2    = r2_val,
      n     = length(preds)
    )
    
    # Separate globals for train vs predict
    if (grepl("^TRAIN", tag)) {
      assign("probe_last_train", dbg, envir = .ddesonn_state)
    } else if (grepl("^PREDICT", tag)) {
      assign("probe_last_predict", dbg, envir = .ddesonn_state)
    } else {
      assign("probe_last_dbg", dbg, envir = .ddesonn_state)
    }
  }
}

probe_last_layer <- function(weights, biases, y, tag = "GENERIC", save_global = TRUE,
                             verbose = verbose) {
  W_last <- weights[[length(weights)]]
  b_last <- biases[[length(biases)]]
  
  stats <- list(
    tag = tag,
    W_last = list(
      dims  = dim(W_last),
      mean  = mean(W_last),
      sd    = sd(W_last),
      min   = min(W_last),
      max   = max(W_last)
    ),
    b_last = list(
      len   = length(b_last),
      mean  = mean(b_last),
      sd    = sd(b_last),
      min   = min(b_last),
      max   = max(b_last),
      head  = head(b_last, 10L)
    ),
    y = list(
      n     = length(y),
      mean  = mean(y),
      sd    = sd(y),
      min   = min(y),
      max   = max(y),
      head  = head(y, 10L)
    )
  )
  
  if (verbose) {
    message(sprintf(
      "[LASTLAYER %s] W dims=%s | mean=%.6f sd=%.6f range=[%.3f, %.3f]",
      tag, paste(dim(W_last), collapse = "x"),
      stats$W_last$mean, stats$W_last$sd,
      stats$W_last$min, stats$W_last$max
    ))
    message(sprintf(
      "[LASTLAYER %s] b len=%d | mean=%.6f range=[%.3f, %.3f]",
      tag, stats$b_last$len,
      stats$b_last$mean, stats$b_last$min, stats$b_last$max
    ))
    message(sprintf(
      "[LASTLAYER %s] y n=%d | mean=%.6f sd=%.6f range=[%.3f, %.3f]",
      tag, stats$y$n,
      stats$y$mean, stats$y$sd,
      stats$y$min, stats$y$max
    ))
  }
  
  if (save_global) {
    # store in global env
    assign(paste0("probe_last_layer_", tag), stats, envir = .ddesonn_state)
    
    # ===== Artifacts snapshot saver ===== 
    artifacts_dir <- ddesonn_artifacts_root(get0("output_root", inherits = TRUE, ifnotfound = NULL)) 
    fname <- file.path(artifacts_dir, sprintf("probe_last_layer_%s_%s.rds", 
                                              tag, format(Sys.time(), "%Y%m%d_%H%M%S"))) 
    saveRDS(stats, fname) 
    message("[LASTLAYER] Snapshot saved to: ", fname) 
  }
  
  invisible(stats)
}



## ===== Shared plot filename helper =====
# Builds a filename prefixer for a specific context
# utils_plots.R

make_fname_prefix <- function(do_ensemble,
                              num_networks = NULL,
                              total_models = NULL,
                              ensemble_number = NULL,
                              model_index = NULL,
                              who) {
  if (missing(who) || !nzchar(who)) stop("'who' must be 'SONN' or 'DDESONN'")
  who <- toupper(who)
  
  if (is.null(total_models))
    total_models <- if (!is.null(num_networks)) num_networks else get0("num_networks", ifnotfound = 1L)
  
  as_int_or_na <- function(x) {
    if (is.null(x) || length(x) == 0 || is.na(x)) return(NA_integer_)
    as.integer(x)
  }
  
  ens <- as_int_or_na(ensemble_number)
  mod <- as_int_or_na(model_index)
  tot <- as_int_or_na(total_models); if (is.na(tot)) tot <- 1L
  
  if (isTRUE(do_ensemble)) {
    prefixer <- function(base_name) {
      paste0(
        "DDESONN", if (!is.na(ens)) paste0("_", ens),    # omit if NA
        "_SONN",  if (!is.na(mod)) paste0("_", mod),
        "_", base_name
      )
    }
    attr(prefixer, "ensemble_number") <- ens
    attr(prefixer, "model_index") <- mod
    return(prefixer)
  }
  
  if (!is.na(tot) && tot > 1L) {
    if (who == "SONN") {
      prefixer <- function(base_name) {
        prefix <- if (!is.na(mod)) sprintf("SONN_%dof%d_", mod, tot) else sprintf("SONN_of%d_", tot)
        paste0(prefix, base_name)
      }
      attr(prefixer, "ensemble_number") <- ens
      attr(prefixer, "model_index") <- mod
      return(prefixer)
    } else if (who == "DDESONN") {
      prefixer <- function(base_name) paste0(sprintf("SONN_%d-%d_", 1L, tot), base_name)
      attr(prefixer, "ensemble_number") <- ens
      attr(prefixer, "model_index") <- mod
      return(prefixer)
    } else {
      stop("invalid 'who'")
    }
  }
  
  # single-model case
  prefixer <- function(base_name) {
    paste0("SONN", if (!is.na(mod)) paste0("_", mod), "_", base_name)
  }
  attr(prefixer, "ensemble_number") <- ens
  attr(prefixer, "model_index") <- mod
  prefixer
}

# ---- helper for
#  prepare_disk_only
# ----- tiny helper: cleanly stop script right here -----
.hard_stop <- function(msg = "[prepare_disk_only] Done; stopping script.") {
  cat(msg, "\n")
  if (interactive()) {
    # In RStudio: stop evaluating this script, but do NOT kill the session
    stop(invisible(structure(list(message = msg),
                             class = c("simpleError","error","condition"))),
         call. = FALSE)
  } else {
    # From Rscript/terminal: exit process
    quit(save = "no")
  }
}

#function used in Optimzers.R
lookahead_update <- function(params, grads_list, lr, beta1, beta2, epsilon, lookahead_step, base_optimizer, epoch, lambda) {
  updated_params_list <- list()
  
  cat(">> Lookahead optimizer running\n")
  
  # grads_list is just the matrix for this layer
  grad_matrix <- if (is.list(grads_list)) grads_list[[1]] else grads_list
  
  if (is.null(grad_matrix)) stop("Missing gradient matrix")
  
  #Don't double-index
  param_list <- params
  
  param <- param_list$param
  m <- param_list$m
  v <- param_list$v
  r <- param_list$r
  slow_param <- param_list$slow_weights
  lookahead_counter <- param_list$lookahead_counter
  lookahead_step_layer <- param_list$lookahead_step
  
  if (is.null(lookahead_counter)) {
    lookahead_counter <- 0
    cat("Initialized lookahead_counter = 0\n")
  }
  
  if (is.null(lookahead_step_layer)) {
    lookahead_step_layer <- lookahead_step
  }
  
  if (base_optimizer == "adam_update") {
    m <- beta1 * m + (1 - beta1) * grad_matrix
    v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
    
    m_hat <- m / (1 - beta1^epoch)
    v_hat <- v / (1 - beta2^epoch)
    
    update <- lr * m_hat / (sqrt(v_hat) + epsilon)
    param <- param - update
    
  } else if (base_optimizer == "lamb_update") {
    m <- beta1 * m + (1 - beta1) * grad_matrix
    v <- beta2 * v + (1 - beta2) * (grad_matrix^2)
    
    m_hat <- m / (1 - beta1^epoch)
    v_hat <- v / (1 - beta2^epoch)
    
    r1 <- sqrt(sum(param^2))
    r2 <- sqrt(sum((m_hat / (sqrt(v_hat) + epsilon))^2))
    ratio <- ifelse(r1 == 0 | r2 == 0, 1, r1 / r2)
    
    update <- lr * ratio * m_hat / (sqrt(v_hat) + epsilon)
    param <- param - update
    
  } else {
    stop("Unsupported base optimizer in lookahead_update()")
  }
  
  lookahead_counter <- lookahead_counter + 1
  if (lookahead_counter >= lookahead_step_layer) {
    cat(">> Lookahead sync\n")
    slow_param <- param
    lookahead_counter <- 0
  }
  
  updated_params_list <- list(
    param = param,
    m = m,
    v = v,
    r = r,
    slow_weights = slow_param,
    lookahead_counter = lookahead_counter,
    lookahead_step = lookahead_step_layer,
    weights_update = update
  )
  
  return(updated_params_list)
}

#train_with_l2_regulatization

.extract_vec <- function(x) {
  if (is.data.frame(x)) return(x[[1]])
  if (is.matrix(x) || is.array(x)) {
    if (ncol(x) == 1L) return(as.vector(x[,1]))
    return(as.vector(x[,1]))
  }
  x
}

.align_len <- function(v, n) {
  if (length(v) > n) v[seq_len(n)]
  else if (length(v) < n) c(v, rep(NA, n - length(v)))
  else v
}

.build_targets <- function(labels, n, K, CLASSIFICATION_MODE, debug = FALSE) {
  lv <- .align_len(.extract_vec(labels), n)
  dbg <- isTRUE(debug)
  if (dbg) cat("[dbg] targets: CLASSIFICATION_MODE =", CLASSIFICATION_MODE, "\n")
  if (dbg) cat("[dbg] targets: labels class =", paste(class(labels), collapse=","), " | lv length =", length(lv), "\n")
  
  if (identical(CLASSIFICATION_MODE, "multiclass")) {
    stopifnot(K >= 2)
    if (is.matrix(labels) && nrow(labels) >= n && ncol(labels) == K &&
        all(labels[seq_len(n), , drop=FALSE] %in% c(0,1))) {
      Y <- matrix(as.numeric(labels[seq_len(n), , drop=FALSE]), n, K)
      y_idx <- max.col(Y, ties.method = "first")
      if (dbg) cat("[dbg] targets: using provided one-hot (nxK)\n")
      return(list(Y=Y, y_idx=y_idx))
    } else {
      f <- if (is.factor(lv)) lv else factor(lv)
      idx <- as.integer(f)
      L <- nlevels(f)
      if (L > K) {
        if (dbg) cat(sprintf("[dbg] targets: L=%d > K=%d, truncating indices > K to K\n", L, K))
        idx[idx > K] <- K
      }
      if (dbg) cat("[dbg] targets: factor levels L =", L, " | head idx =", paste(utils::head(idx,6), collapse=", "), "\n")
      # use your global one_hot_from_ids
      return(list(Y=one_hot_from_ids(idx, K, N=n), y_idx=idx))
    }
    
  } else if (identical(CLASSIFICATION_MODE, "binary")) {
    stopifnot(K == 1)
    if (is.factor(lv)) {
      y <- as.integer(lv) - 1L
    } else {
      y <- suppressWarnings(as.numeric(lv))
      if (all(is.na(y))) { f <- factor(lv); y <- as.integer(f) - 1L }
    }
    y[is.na(y)] <- 0L
    y <- pmin(pmax(as.numeric(y), 0), 1)
    if (dbg) cat("[dbg] targets: binary y summary -> mean=", mean(y), " | sum=", sum(y), "\n")
    return(list(y=y))
    
  } else if (identical(CLASSIFICATION_MODE, "regression")) {
    if (dbg) cat("[dbg] targets: regression path | n=", n, " K=", K, "\n")
    
    if (is.matrix(labels) || is.data.frame(labels)) {
      Ytmp <- suppressWarnings(matrix(as.numeric(as.matrix(labels)),
                                      nrow = nrow(as.matrix(labels)),
                                      ncol = ncol(as.matrix(labels))))
    } else {
      Ytmp <- suppressWarnings(matrix(as.numeric(lv), nrow = length(lv), ncol = 1L))
    }
    
    if (all(is.na(Ytmp))) {
      if (dbg) cat("[dbg] targets: all NA after coercion; filling zeros\n")
      Ytmp <- matrix(0, nrow = max(1L, nrow(Ytmp)), ncol = max(1L, ncol(Ytmp)))
    }
    
    # conform rows
    if (nrow(Ytmp) > n) {
      Ytmp <- Ytmp[seq_len(n), , drop = FALSE]
    } else if (nrow(Ytmp) < n) {
      add <- n - nrow(Ytmp)
      last_row <- if (nrow(Ytmp) >= 1) Ytmp[nrow(Ytmp), , drop = FALSE] else matrix(0, 1, max(1L, ncol(Ytmp)))
      Ytmp <- rbind(Ytmp, do.call(rbind, replicate(add, last_row, simplify = FALSE)))
    }
    # conform cols
    if (ncol(Ytmp) > K) {
      Ytmp <- Ytmp[, seq_len(K), drop = FALSE]
    } else if (ncol(Ytmp) < K) {
      if (ncol(Ytmp) == 0L) Ytmp <- matrix(0, nrow = nrow(Ytmp), ncol = 1L)
      col_map <- rep(seq_len(ncol(Ytmp)), length.out = K)
      Ytmp <- Ytmp[, col_map, drop = FALSE]
    }
    
    storage.mode(Ytmp) <- "double"
    if (dbg) cat("[dbg] targets: regression Y dim ->", paste(dim(Ytmp), collapse="x"),
                 " | summary mean=", mean(Ytmp), "\n")
    
    # provide both Y and y for downstream compatibility
    return(list(Y = Ytmp, y = as.numeric(Ytmp[,1])))
    
  } else {
    stop("Unknown CLASSIFICATION_MODE. Use 'multiclass', 'binary', or 'regression'.")
  }
}

.bce_loss <- function(p, y, eps=1e-12) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  y <- pmin(pmax(as.numeric(y), 0), 1)
  val <- mean(-(y*log(p) + (1-y)*log(1-p)))
  if (!is.finite(val)) val <- NA_real_
  val
}

.ce_loss_multiclass <- function(P, Y, eps=1e-12) {
  P <- pmin(pmax(as.matrix(P), eps), 1 - eps)  # nxK
  if (is.vector(Y)) Y <- one_hot_from_ids(as.integer(Y), K=ncol(P), N=nrow(P))
  # --- numeric guards (multiclass) ---
  if (!is.matrix(P)) P <- as.matrix(P)
  P[!is.finite(P)] <- 0
  P <- pmin(pmax(P, 1e-12), 1 - 1e-12)
  
  rs <- rowSums(P)
  bad <- !is.finite(rs) | rs <= 0
  if (any(bad)) {
    K <- ncol(P)
    P[bad, ] <- 1 / K
  }
  P <- P / rowSums(P)
  
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  val <- mean(-rowSums(Y * log(P)))
  
  if (!is.finite(val)) val <- NA_real_
  val
}


# =======================
# PREDICT-ONLY HELPERS (for !train)
# =======================

# Needed: convert best_*_record to clean {weights,biases} for SL/ML prediction
if (!exists("normalize_records", inherits = TRUE)) {
  normalize_records <- function(wrec, brec, ML_NN) {
    if (ML_NN) {
      stopifnot(is.list(wrec), is.list(brec))
      list(
        weights = lapply(wrec, function(w) as.matrix(w)),
        biases  = lapply(brec, function(b) as.numeric(if (is.matrix(b)) b else b))
      )
    } else {
      list(
        weights = as.matrix(if (is.list(wrec)) wrec[[1]] else wrec),
        biases  = as.numeric(if (is.list(brec)) brec[[1]] else brec)
      )
    }
  }
}

# Needed: pulls best_*_record (supports nested best_model_metadata)
# Robust extractor: tries nested best_*, then common alternatives; never stop() here
extract_best_records <- function(meta, ML_NN, model_index = 1L) {
  grab <- function(x) if (!is.null(x)) x else NULL
  # 1) preferred nested best_* location
  srcs_w <- list(
    grab(meta$best_model_metadata$best_weights_record),
    grab(meta$best_weights_record),
    grab(meta$weights_record),
    grab(meta$model$best_weights_record),
    grab(meta$model$weights_record),
    grab(meta$W), grab(meta$weights), grab(meta$model$W), grab(meta$model$weights)
  )
  srcs_b <- list(
    grab(meta$best_model_metadata$best_biases_record),
    grab(meta$best_biases_record),
    grab(meta$biases_record),
    grab(meta$model$best_biases_record),
    grab(meta$model$biases_record),
    grab(meta$B), grab(meta$biases), grab(meta$model$B), grab(meta$model$biases)
  )
  pick_idx <- function(obj, i) {
    if (is.null(obj)) return(NULL)
    if (is.list(obj) && length(obj) >= i && !is.null(obj[[i]])) return(obj[[i]])
    obj
  }
  W <- NULL; B <- NULL
  for (c in srcs_w) { if (!is.null(c)) { W <- pick_idx(c, model_index); if (!is.null(W)) break } }
  for (c in srcs_b) { if (!is.null(c)) { B <- pick_idx(c, model_index); if (!is.null(B)) break } }
  if (is.null(W) || is.null(B)) return(NULL)
  
  # If you have normalize_records() in scope, use it to coerce ML_NN shapes
  if (exists("normalize_records", inherits = TRUE)) {
    return(normalize_records(wrec = W, brec = B, ML_NN = ML_NN))
  }
  # Otherwise return a simple list that our caller understands
  list(weights = W, biases = B)
}


# ===== helpers used by predict-only flow =====
.get_in <- function(x, path) {
  cur <- x
  for (p in path) {
    if (is.null(cur)) return(NULL)
    if (is.list(cur) && !is.null(cur[[p]])) cur <- cur[[p]] else return(NULL)
  }
  cur
}

.choose_X_from_meta <- function(meta) {
  cands <- list(
    c("datasets","X_validation"), c("datasets","X_val"), c("datasets","X_test"),
    c("X_validation"), c("X_val"), c("X_test"),
    c("datasets","X_validation_scaled"), c("X_validation_scaled"),
    c("datasets","X_train"), c("X")
  )
  for (cand in cands) {
    v <- .get_in(meta, cand); if (!is.null(v)) return(list(X=v, tag=paste(cand, collapse="/")))
  }
  NULL
}

.choose_y_from_meta <- function(meta) {
  cands <- list(
    c("datasets","y_validation"), c("datasets","y_val"), c("datasets","y_test"),
    c("y_validation"), c("y_val"), c("y_test"),
    c("datasets","y_train"), c("y")
  )
  for (cand in cands) {
    v <- .get_in(meta, cand); if (!is.null(v)) return(list(y=v, tag=paste(cand, collapse="/")))
  }
  NULL
}

.normalize_y <- function(y) {
  if (is.null(y)) return(NULL)
  if (is.list(y) && length(y) == 1L) y <- y[[1]]
  if (is.data.frame(y)) y <- y[[1]]
  if (is.matrix(y) && ncol(y) == 1L) y <- y[,1]
  if (is.factor(y)) y <- as.integer(y) - 1L
  as.numeric(y)
}
# helpers for reg test_metrics.rds to be saved properly after diff
ddesonn_to_log_return <- function(v) {
  if (is.null(v)) return(v)
  if (is.list(v) && !is.data.frame(v) && length(v) == 1L) v <- v[[1L]]
  if (is.data.frame(v)) v <- v[[1L]]
  if (is.matrix(v)) v <- v[, 1L]
  vv <- suppressWarnings(as.numeric(v))
  if (!length(vv)) return(numeric(0))
  clean <- vv
  clean[!is.finite(clean) | clean <= 0] <- NA_real_
  clean <- ifelse(is.na(clean), NA_real_, pmax(clean, 1e-12))
  diffs <- diff(log(clean))
  c(NA_real_, diffs)
}

ddesonn_drop_first_row <- function(obj) {
  if (is.null(obj)) return(obj)
  nr <- NROW(obj)
  if (nr <= 0) return(obj)
  if (is.matrix(obj)) {
    if (nr <= 1L) return(obj[0, , drop = FALSE])
    return(obj[-1, , drop = FALSE])
  }
  if (is.data.frame(obj)) {
    if (nr <= 1L) return(obj[0, , drop = FALSE])
    return(obj[-1, , drop = FALSE])
  }
  if (is.list(obj) && !is.data.frame(obj)) {
    return(lapply(obj, ddesonn_drop_first_row))
  }
  len <- length(obj)
  if (len <= 1L) {
    return(obj[0])
  }
  obj[-1L]
}

ddesonn_resolve_reg_target_mode <- function(meta = NULL, default = "price") {
  modes <- c(
    tryCatch(get0("REG_TARGET_MODE", inherits = TRUE), error = function(e) NULL),
    tryCatch(get0("reg_target_mode", inherits = TRUE), error = function(e) NULL),
    tryCatch(meta$reg_target_mode, error = function(e) NULL),
    tryCatch(meta$preprocessScaledData$reg_target_mode, error = function(e) NULL)
  )
  for (cand in modes) {
    if (is.null(cand)) next
    val <- tolower(as.character(cand[[1L]]))
    if (val %in% c("price", "return_log")) return(val)
  }
  tolower(as.character(default %||% "price"))
}
# ----------------------------------------------------------------#

# ------------------------------------------------------------------
# Shared numeric coercion helpers
# ------------------------------------------------------------------

coerce_to_numeric_matrix <- function(x, allow_model_matrix = TRUE) {
  if (is.null(x)) {
    return(matrix(numeric(0), nrow = 0, ncol = 0))
  }
  
  if (is.data.frame(x)) {
    if (!ncol(x)) {
      return(matrix(numeric(0), nrow = nrow(x), ncol = 0))
    }
    if (allow_model_matrix) {
      x <- stats::model.matrix(~ . - 1, data = x)
    } else {
      x[] <- lapply(x, function(col) {
        if (is.numeric(col)) col else as.numeric(as.factor(col))
      })
      x <- as.matrix(x)
    }
  } else if (is.factor(x) || is.character(x)) {
    x <- matrix(as.numeric(as.factor(x)), ncol = 1L)
  } else if (is.atomic(x) && !is.matrix(x)) {
    x <- matrix(x, ncol = 1L)
  } else {
    x <- as.matrix(x)
    if (!is.numeric(x)) {
      x <- apply(x, 2L, function(col) {
        if (is.numeric(col)) col else as.numeric(as.factor(col))
      })
      x <- as.matrix(x)
    }
  }
  
  storage.mode(x) <- "double"
  x
}

safe_one_hot_matrix <- function(idx, K) {
  idx <- suppressWarnings(as.integer(idx))
  if (!length(idx)) {
    return(matrix(0, nrow = 0, ncol = K))
  }
  
  idx[is.na(idx)] <- 1L
  idx[idx < 1L] <- 1L
  idx[idx > K]  <- K
  
  M <- one_hot_from_ids(idx, K, N = length(idx), strict = FALSE)
  storage.mode(M) <- "double"
  M
}

.align_by_names_safe <- function(Xi, Xref) {
  Xi <- as.matrix(Xi)
  if (is.null(Xref)) return(Xi)
  Xref <- as.matrix(Xref)
  if (is.null(colnames(Xi)) || is.null(colnames(Xref))) return(Xi)
  keep <- intersect(colnames(Xref), colnames(Xi))
  if (!length(keep)) return(Xi)
  as.matrix(Xi[, keep, drop = FALSE])
}

.apply_scaling_if_any <- function(X, meta) {
  pp <- meta$preprocessScaledData %||% meta$preprocess %||% meta$scaler
  X <- as.matrix(X); storage.mode(X) <- "double"
  if (is.null(pp)) {
    assign("LAST_APPLIED_X", X, .ddesonn_state)
    return(X)
  }
  # Column order + add missing with 0
  exp_names <- pp$feature_names %||% colnames(X)
  miss <- setdiff(exp_names, colnames(X))
  if (length(miss)) {
    X <- cbind(X, matrix(0, nrow=nrow(X), ncol=length(miss),
                         dimnames=list(NULL, miss)))
  }
  X <- X[, exp_names, drop=FALSE]
  
  # Impute with train medians (no leakage)
  if (!is.null(pp$train_medians)) {
    for (nm in intersect(names(pp$train_medians), colnames(X))) {
      idx <- is.na(X[, nm]); if (any(idx)) X[idx, nm] <- pp$train_medians[[nm]]
    }
  }
  
  # Z-score with train center/scale
  if (!is.null(pp$center) && !is.null(pp$scale)) {
    sc <- pp$scale; sc[!is.finite(sc) | sc==0] <- 1
    X <- sweep(sweep(X, 2, pp$center, "-"), 2, sc, "/")
  }
  
  # Optional extra compression -- disabled unless explicitly TRUE and >1
  if (isTRUE(pp$divide_by_max_val)) {
    mv <- as.numeric(pp$max_val %||% 1)
    if (is.finite(mv) && mv > 1) X <- X / mv
  }
  
  assign("LAST_APPLIED_X", X, .ddesonn_state)  # lets you inspect later
  X
}


# ---- Target transform helpers ----------------------------------------------
.get_target_transform <- function(meta) {
  meta$target_transform %||%
    (tryCatch(meta$preprocessScaledData$target_transform, error=function(e) NULL)) %||%
    (tryCatch(meta$preprocess$target_transform,         error=function(e) NULL))
}

._invert_target <- function(pred, tt, DEBUG=FALSE) {
  if (is.null(tt)) return(pred)
  type <- tolower(tt$type %||% "identity")
  par  <- tt$params %||% list()
  v <- as.numeric(pred[,1])
  
  if (type == "identity") {
    # do nothing
  } else if (type == "standardize") {
    mu <- par$y_mean; sdv <- par$y_sd
    if (is.finite(mu) && is.finite(sdv) && sdv > 0) v <- v * sdv + mu
    if (DEBUG) cat("[ASPM-DBG] inverse standardize applied\n")
  } else if (type == "minmax") {
    ymin <- par$y_min; ymax <- par$y_max
    if (is.finite(ymin) && is.finite(ymax)) v <- v * (ymax - ymin) + ymin
    if (DEBUG) cat("[ASPM-DBG] inverse minmax applied\n")
  } else if (type == "log") {
    base <- par$base %||% exp(1)
    v <- if (identical(base, 10)) 10^v else if (identical(base, 2)) 2^v else exp(v)
    if (!is.null(par$shift)) v <- v + par$shift
    if (DEBUG) cat("[ASPM-DBG] inverse log applied\n")
  } else if (type == "boxcox") {
    lambda <- par$lambda
    if (is.finite(lambda)) {
      if (abs(lambda) < 1e-8) v <- exp(v) else v <- (lambda*v + 1)^(1/lambda)
      if (!is.null(par$shift)) v <- v + par$shift
      if (DEBUG) cat("[ASPM-DBG] inverse boxcox applied\n")
    }
  } else if (type == "affine") {
    a <- par$a %||% 0; b <- par$b %||% 1
    v <- a + b * v
    if (DEBUG) cat("[ASPM-DBG] inverse affine applied\n")
  } else {
    if (DEBUG) cat("[ASPM-DBG] unknown target_transform; left as-is\n")
  }
  
  pred[,1] <- v
  pred
}

# ---- minimal predict-time prep using saved train scalers ----
prep_predict_X <- function(X_new, meta) {
  pp <- meta$preprocessScaledData
  stopifnot(!is.null(pp))
  
  # 1) same date handling as train
  if ("date" %in% names(X_new)) {
    d <- X_new[["date"]]
    if (inherits(d, "POSIXt"))      X_new[["date"]] <- as.numeric(as.Date(d))
    else if (inherits(d, "Date"))   X_new[["date"]] <- as.numeric(d)
    else {
      parsed <- suppressWarnings(as.Date(d))
      X_new[["date"]] <- if (all(is.na(parsed))) NA_real_ else as.numeric(parsed)
    }
  }
  
  # 2) align columns to training order
  fn <- pp$feature_names
  miss <- setdiff(fn, names(X_new))
  if (length(miss)) X_new[miss] <- NA_real_
  X_new <- X_new[, fn, drop = FALSE]
  
  # 3) numeric + TRAIN-median impute (no leakage)
  Xn <- as.data.frame(X_new)
  for (j in seq_along(fn)) {
    v <- suppressWarnings(as.numeric(Xn[[j]]))
    v[!is.finite(v)] <- NA_real_
    v[is.na(v)] <- pp$train_medians[[j]]
    Xn[[j]] <- v
  }
  Xn <- as.matrix(Xn); storage.mode(Xn) <- "double"
  
  # 4) apply TRAIN center/scale once (+ max_val if used in train)
  Xs <- sweep(sweep(Xn, 2, pp$center, "-"), 2, pp$scale, "/")
  mv <- pp$max_val %||% 1
  if (is.finite(mv) && mv != 0) Xs <- Xs / mv
  Xs
}


## =========================
## DEBUG TOGGLES (set TRUE)
## =========================
DEBUG_MODE_HELPER <- TRUE
DEBUG_ASPM        <- TRUE   # .as_pred_matrix()
DEBUG_SAFERUN     <- TRUE   # .safe_run_predict()
DEBUG_RUNPRED     <- TRUE   # .run_predict() shim

## ------------------------------------------------------------------------
## Null-coalescing helper
## ------------------------------------------------------------------------
`%||%` <- function(x, y) if (is.null(x)) y else x

## ------------------------------------------------------------------------
## Debug utilities (lightweight, no deps)
## ------------------------------------------------------------------------
if (!exists("LAST_DEBUG", inherits = TRUE)) {
  LAST_DEBUG <- new.env(parent = emptyenv())
}

# --- Drop-in replacement: robust, no integer overflow warnings ---
.hash_vec <- function(x) {
  if (requireNamespace("digest", quietly = TRUE)) {
    return(substr(digest::digest(x, algo = "xxhash64", serialize = TRUE), 1, 16))
  }
  nx <- suppressWarnings(as.numeric(x))
  if (!length(nx)) return("len0")
  fin <- is.finite(nx)
  if (!any(fin)) {
    s_len <- length(nx); s_sum <- 0; s_mean <- 0; s_sd <- 0
  } else {
    nx <- nx[fin]
    s_len <- length(nx); s_sum <- sum(nx); s_mean <- mean(nx); s_sd <- stats::sd(nx)
  }
  v <- abs(c(s_len, s_sum, s_mean, s_sd)) + c(1, 2, 3, 4)
  v[!is.finite(v)] <- 0
  v <- floor(v * 1e6)
  MOD <- (2^31 - 1)
  v <- as.double(v %% MOD)
  iv <- as.integer(v)
  paste(sprintf("%08x", iv), collapse = "")
}

.peek_num <- function(x, k = 6) {
  v <- tryCatch(as.numeric(x), error = function(e) numeric())
  if (!length(v)) return("")
  paste(sprintf("%.6f", utils::head(v, k)), collapse = ", ")
}

.summarize_matrix <- function(M) {
  nr <- NROW(M); nc <- NCOL(M)
  rng <- tryCatch(range(M, finite = TRUE), error=function(e) c(NA_real_, NA_real_))
  sprintf("dims=%sx%s | mean=%.6f sd=%.6f min=%.6f p50=%.6f max=%.6f",
          nr, nc, mean(M), sd(as.vector(M)), rng[1], stats::median(as.vector(M)), rng[2])
}

.in_range01 <- function(M) {
  rng <- suppressWarnings(range(M, finite = TRUE))
  isTRUE(is.finite(rng[1]) && is.finite(rng[2]) && rng[1] >= 0 && rng[2] <= 1)
}

## ------------------------------------------------------------------------
## Mode helper (global -> meta -> default) with optional tracing
## ------------------------------------------------------------------------
.get_mode <- function(meta) {
  g <- get0("CLASSIFICATION_MODE", inherits = TRUE, ifnotfound = NULL)
  m <- tryCatch(meta$CLASSIFICATION_MODE, error = function(e) NULL)
  final <- tolower(g %||% m %||% "regression")
  if (isTRUE(get0("DEBUG_MODE_HELPER", inherits = TRUE, ifnotfound = FALSE))) {
    cat(sprintf(
      "[MODE-DBG %s] resolved mode='%s' | global=%s | meta=%s | default=regression\n",
      format(Sys.time(), "%H:%M:%S"),
      final,
      if (is.null(g)) "NULL" else as.character(g),
      if (is.null(m)) "NULL" else as.character(m)
    ))
  }
  final
}

## ------------------------------------------------------------------------
## Output normalization (mode-aware; add unscale for regression)
## ------------------------------------------------------------------------
.as_pred_matrix <- function(pred, mode = NULL, meta = NULL,
                            DEBUG = get0("DEBUG_ASPM", inherits = TRUE, ifnotfound = FALSE)) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  stamp <- format(Sys.time(), "%H:%M:%S")
  if (isTRUE(DEBUG)) {
    cat(sprintf("[ASPM-DBG %s] entry: class=%s len=%s\n",
                stamp, paste0(class(pred), collapse=","), length(pred)))
    if (is.list(pred)) cat("[ASPM-DBG] list names: ", paste(names(pred), collapse=", "), "\n")
  }
  
  # --- unwrap list containers (outer and inner) ---
  if (is.list(pred) && "predicted_output" %in% names(pred)) {
    pred <- pred$predicted_output
  }
  if (is.list(pred) && length(pred) == 1L) {
    pred <- pred[[1L]]
  }
  
  # --- normalize types ---
  if (is.null(pred) || length(pred) == 0L) {
    if (isTRUE(DEBUG)) cat("[ASPM-DBG] empty -> returning 0x1 matrix\n")
    return(matrix(numeric(0), nrow = 0, ncol = 1))
  }
  if (is.data.frame(pred)) pred <- as.matrix(pred)
  if (is.vector(pred))     pred <- matrix(as.numeric(pred), ncol = 1L)
  if (is.list(pred))       pred <- matrix(as.numeric(unlist(pred)), ncol = 1L)
  
  pred <- as.matrix(pred)
  storage.mode(pred) <- "double"
  
  # --- resolve mode ---
  resolved_mode <- tolower(
    mode %||%
      get0("CLASSIFICATION_MODE", inherits = TRUE, ifnotfound = NULL) %||%
      (tryCatch(meta$CLASSIFICATION_MODE, error = function(e) NULL)) %||%
      "regression"
  )
  
  if (isTRUE(DEBUG)) {
    cat(sprintf("[ASPM-DBG %s] mode=%s | BEFORE squash: %s | hash=%s | head=[%s]\n",
                stamp, resolved_mode, .summarize_matrix(pred), .hash_vec(pred), .peek_num(pred)))
    cat(sprintf("[ASPM-DBG] in[0,1]? %s\n", .in_range01(pred)))
  }
  assign("LAST_ASPM_IN", pred, envir = LAST_DEBUG)
  
  # --- apply mode-specific transforms ---
  if (identical(resolved_mode, "binary")) {
    if (!.in_range01(pred)) {
      if (isTRUE(DEBUG)) cat("[ASPM-DBG] applying sigmoid (binary)\n")
      pred <- 1 / (1 + exp(-pred))
    }
  } else if (identical(resolved_mode, "multiclass")) {
    if (NCOL(pred) > 1 && !.in_range01(pred)) {
      if (isTRUE(DEBUG)) cat("[ASPM-DBG] applying softmax (multiclass)\n")
      mx <- apply(pred, 1, max)
      ex <- exp(pred - mx)
      sm <- rowSums(ex)
      pred <- ex / sm
    }
  } else {
    if (isTRUE(DEBUG)) cat("[ASPM-DBG] regression mode: no squashing applied\n")
    tt <- try(.get_target_transform(meta), silent = TRUE)
    if (!inherits(tt, "try-error") && !is.null(tt)) {
      pred <- tryCatch(
        ._invert_target(pred, tt, DEBUG = isTRUE(DEBUG)),
        error = function(e) {
          if (isTRUE(DEBUG)) cat("[ASPM-DBG] invert skipped: ", conditionMessage(e), "\n")
          pred
        }
      )
    }
  }
  
  if (isTRUE(DEBUG)) {
    cat(sprintf("[ASPM-DBG %s] mode=%s | AFTER squash/unscale: %s | hash=%s | head=[%s]\n",
                stamp, resolved_mode, .summarize_matrix(pred), .hash_vec(pred), .peek_num(pred)))
  }
  assign("LAST_ASPM_OUT", pred, envir = LAST_DEBUG)
  pred
}


.is_linear_verbose <- function(af, PROBE = TRUE) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  .now <- function() format(Sys.time(), "%H:%M:%S")
  .attr_name <- function(f) tolower(trimws(attr(f, "name") %||% ""))
  .is_identity_fn <- function(f) {
    if (!is.function(f)) return(FALSE)
    nm <- .attr_name(f)
    if (nm %in% c("identity","linear","id")) return(TRUE)
    if (identical(f, base::identity)) return(TRUE)
    # try to match a global 'identity' if user defined one
    if (exists("identity", inherits = TRUE)) {
      g <- get("identity", inherits = TRUE)
      if (is.function(g) && identical(f, g)) return(TRUE)
    }
    if (!PROBE) return(FALSE)
    v <- c(-2,-1,0,1,2)
    ok <- tryCatch({
      out <- f(v)
      is.numeric(out) && length(out) == length(v) && max(abs(out - v)) <= 1e-12
    }, error = function(e) {
      cat(sprintf("[AF-TRACE %s] probe error: %s\n", .now(), conditionMessage(e)))
      FALSE
    })
    ok
  }
  
  cat(sprintf("\n[AF-TRACE %s] activation_functions candidate:\n", .now()))
  if (is.null(af)) { cat("  <NULL>\n"); return(FALSE) }
  if (!is.list(af)) af <- as.list(af)
  
  for (i in seq_along(af)) {
    ai     <- af[[i]]
    is_fun <- is.function(ai)
    nm     <- if (is_fun) .attr_name(ai) else NA_character_
    cls    <- paste0(class(ai), collapse = ",")
    # short deparse/description
    desc <- tryCatch({
      paste(utils::capture.output(str(ai, give.attr = TRUE, vec.len = 6L)), collapse = " ")
    }, error = function(e) paste0("<str error: ", conditionMessage(e), ">"))
    
    cat(sprintf("  [%02d] class=%s | is.function=%s | name=%s\n      desc=%s\n",
                i, cls, is_fun, if (is.na(nm) || nm == "") "<NULL>" else nm, desc))
  }
  
  last <- af[[length(af)]]
  last_is_linear <-
    (is.character(last) && tolower(trimws(last)) %in% c("linear","identity","id")) ||
    (is.function(last)  && .is_identity_fn(last))
  
  # extra diags for last
  if (is.function(last)) {
    nm  <- .attr_name(last)
    sig <- tryCatch(paste(deparse(last, nlines = 1L), collapse = ""), error = function(e) "<deparse error>")
    cat(sprintf("  [HEAD %s] name=%s | signature=%s | linear?=%s\n",
                .now(), if (nm=="") "<NULL>" else nm, sig, last_is_linear))
  } else {
    cat(sprintf("  [HEAD %s] last is %s | value=%s | linear?=%s\n",
                .now(), paste0(class(last), collapse=","), as.character(last)[1], last_is_linear))
  }
  
  cat(sprintf("  -> last_is_linear=%s\n\n", last_is_linear))
  last_is_linear
}





## ------------------------------------------------------------------------
## Safe wrapper
## ------------------------------------------------------------------------
# ===========================================
# .safe_run_predict (passes verbose/debug through) -- accepts function, list$predict, or R6/env$predict
# .safe_run_predict -- accepts function, list$predict, or R6/env$predict.
# Passes weights/biases/activation_functions correctly (esp. for predictor_fn).
.safe_run_predict <- function(
    X, meta,
    model_index         = 1L,
    ML_NN               = NULL,
    CLASSIFICATION_MODE = NULL,   # <- fixed (no self-reference)
    ...,
    verbose = get0("VERBOSE_SAFERUN", inherits = TRUE, ifnotfound = FALSE),
    debug   = get0("DEBUG_SAFERUN",   inherits = TRUE, ifnotfound = FALSE),
    DEBUG   = get0("DEBUG_SAFERUN",   inherits = TRUE, ifnotfound = FALSE)
) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  vrb  <- isTRUE(verbose)
  dbg  <- isTRUE(DEBUG) || isTRUE(debug)
  stamp <- format(Sys.time(), "%H:%M:%S")
  
  # Resolve meta if a name was passed
  if (is.character(meta)) meta <- get(meta, envir = .ddesonn_state, inherits = TRUE)
  
  # Pull predictor variants from metadata
  predictor    <- tryCatch(meta$predictor,    error = function(e) NULL)
  predictor_fn <- tryCatch(meta$predictor_fn, error = function(e) NULL)
  
  if (vrb || dbg) {
    cat(sprintf("[SAFE-IN] meta_has_predictor=%s | predictor_class=%s | is_function=%s\n",
                as.character(!is.null(predictor)),
                paste(class(predictor), collapse = ","),
                as.character(is.function(predictor))))
  }
  
  if (dbg) {
    cat(sprintf("[SAFE-DBG %s] enter .safe_run_predict | X dims=%d x %d\n",
                stamp, NROW(X), NCOL(X)))
    suppressWarnings({
      xm  <- try(mean(as.numeric(X)), silent = TRUE)
      xs  <- try(sd(as.numeric(X)),   silent = TRUE)
      xmn <- try(min(as.numeric(X)),  silent = TRUE)
      xmx <- try(max(as.numeric(X)),  silent = TRUE)
      if (!inherits(xm, "try-error")) {
        cat(sprintf("[SAFE-DBG %s] X summary: mean=%.6f sd=%.6f min=%.6f max=%.6f\n",
                    stamp, as.numeric(xm), as.numeric(xs), as.numeric(xmn), as.numeric(xmx)))
      }
    })
  }
  
  # Mode hint (for logs/consistency)
  mode_hint <- tolower(as.character(
    CLASSIFICATION_MODE %||%
      get0("CLASSIFICATION_MODE", inherits = TRUE, ifnotfound = NULL) %||%
      meta$CLASSIFICATION_MODE %||% NA
  ))
  if (!mode_hint %in% c("binary","multiclass","regression")) mode_hint <- NULL
  if (dbg) cat(sprintf("[SAFE-DBG %s] mode_hint=%s\n", stamp, as.character(mode_hint)))
  
  # Choose callable and kind
  call_pred <- NULL
  pred_kind <- "none"
  if (is.function(predictor_fn)) {
    call_pred <- predictor_fn; pred_kind <- "predictor_fn"
  } else if (is.function(predictor)) {
    call_pred <- predictor;    pred_kind <- "function"
  } else if ((is.list(predictor) || is.environment(predictor)) && is.function(predictor$predict)) {
    call_pred <- predictor$predict; pred_kind <- "object$predict"
  }
  if (is.null(call_pred)) stop(".safe_run_predict: metadata missing predictor/predictor_fn")
  
  if (dbg) cat(sprintf("[SAFE-DBG %s] using meta predictor callable (%s)\n", stamp, pred_kind))
  
  # Introspect signature to name the data arg; still force-feed weights for predictor_fn
  fmls   <- tryCatch(names(formals(call_pred)), error = function(e) character(0))
  x_name <- if ("Rdata" %in% fmls) "Rdata" else if ("X" %in% fmls) "X" else if (length(fmls)) fmls[1] else "Rdata"
  args   <- list(); args[[x_name]] <- X
  
  # Best weights/biases (supports nested best_model_metadata)
  rec <- extract_best_records(meta, ML_NN = ML_NN, model_index = model_index)
  
  # Activation functions, if present
  af <- meta$activation_functions %||%
    (meta$model$activation_functions %||%
       (meta$preprocessScaledData$activation_functions %||% NULL))
  
  if (identical(pred_kind, "predictor_fn")) {
    # Wrapper uses ..., so ALWAYS pass these through
    args$weights <- rec$weights
    args$biases  <- rec$biases
    if (!is.null(af)) args$activation_functions <- af
    args$verbose <- vrb
    args$debug   <- dbg
  } else {
    # Respect explicit formals; don't pass unknown args to strict signatures
    if ("weights" %in% fmls) args$weights <- rec$weights
    if ("biases"  %in% fmls) args$biases  <- rec$biases
    if ("activation_functions" %in% fmls) {
      if (is.null(af)) stop(".safe_run_predict: activation_functions not found in metadata.")
      args$activation_functions <- af
    }
    if ("verbose" %in% fmls) args$verbose <- vrb
    if ("debug"   %in% fmls) args$debug   <- dbg
  }
  
  # Call and normalize to a numeric matrix
  out <- do.call(call_pred, args)
  raw <- if (is.list(out) && "predicted_output" %in% names(out)) out$predicted_output else out
  if (is.data.frame(raw)) raw <- as.matrix(raw)
  if (is.vector(raw))     raw <- matrix(as.numeric(raw), ncol = 1L)
  if (!is.matrix(raw))    stop(".safe_run_predict: predictor returned unsupported type")
  storage.mode(raw) <- "double"
  if (nrow(raw) == 0L)    stop(".safe_run_predict: predictor produced zero rows")
  
  if (dbg) {
    suppressWarnings({
      mu  <- try(mean(raw), silent = TRUE)
      sdv <- try(sd(as.vector(raw)), silent = TRUE)
      cat(sprintf("[SAFE-DBG %s] meta predictor result dims=%d x %d | mean=%s | sd=%s\n",
                  stamp, nrow(raw), ncol(raw),
                  if (!inherits(mu, "try-error")) sprintf("%.6f", as.numeric(mu)) else "NA",
                  if (!inherits(sdv, "try-error")) sprintf("%.6f", as.numeric(sdv)) else "NA"))
      cat("\n")
    })
  }
  
  raw
}



## ------------------------------------------------------------------------
## Predict shim (stateless, uses extract_best_records) -- MODE-AWARE
## ------------------------------------------------------------------------
## ------------------------------------------------------------------------
## Predict shim (stateless, uses extract_best_records) -- MODE-AWARE
## ------------------------------------------------------------------------
if (!exists(".run_predict", inherits = TRUE)) {
  .run_predict <- function(
    X, meta,
    model_index   = 1L,
    ML_NN         = NULL,
    expected_mode = NULL,
    ...,
    verbose = get0("VERBOSE_RUNPRED", inherits = TRUE, ifnotfound = FALSE),
    debug   = get0("DEBUG_RUNPRED",   inherits = TRUE, ifnotfound = FALSE)
  ) {
    `%||%` <- function(x, y) if (is.null(x)) y else x
    near <- function(a, b, tol = 1e-12) all(is.finite(a)) && all(is.finite(b)) && max(abs(a - b)) <= tol
    
    if (is.null(meta)) stop(".run_predict: 'meta' is NULL")
    X <- as.matrix(X); storage.mode(X) <- "double"
    if (nrow(X) == 0L) stop(".run_predict: X has zero rows")
    
    vrb <- isTRUE(verbose)
    dbg <- isTRUE(debug)
    stamp <- format(Sys.time(), "%H:%M:%S")
    
    ## ---- Resolve expected mode ----
    if (is.null(expected_mode) || !nzchar(expected_mode)) {
      expected_mode <- tolower(get0("CLASSIFICATION_MODE", inherits = TRUE,
                                    ifnotfound = meta$CLASSIFICATION_MODE %||% "regression"))
    } else {
      expected_mode <- tolower(expected_mode)
    }
    if (!expected_mode %in% c("binary","multiclass","regression")) expected_mode <- "regression"
    if (dbg) cat(sprintf("[MODE-DBG %s] expected_mode='%s'\n", stamp, expected_mode))
    
    ## ---- Extract best weights/biases ----
    rec <- extract_best_records(meta, ML_NN = ML_NN, model_index = model_index)
    if (is.null(rec$weights) || !length(rec$weights) || is.null(rec$biases) || !length(rec$biases)) {
      stop("[RUNPRED-ERR] Missing weights/biases in metadata extract_best_records(meta, ...).")
    }
    
    if (dbg) {
      wdims <- tryCatch(dim(rec$weights[[1]]), error = function(e) NULL)
      cat("[RUNPRED-DBG] have weights/biases: ",
          paste0(length(rec$weights), "W/", length(rec$biases), "b"),
          " | W1 dims=", if (is.null(wdims)) "NA" else paste(wdims, collapse = "x"), "\n", sep = "")
    }
    
    ## ---- Model config ----
    input_size   <- ncol(X)
    hidden_sizes <- meta$hidden_sizes %||% meta$model$hidden_sizes
    output_size  <- as.integer(meta$output_size %||% 1L)
    num_networks <- as.integer(meta$num_networks %||% length(meta$best_weights_record) %||% 1L)
    N            <- as.integer(meta$N %||% nrow(X))
    lambda       <- as.numeric(meta$lambda %||% 0)
    init_method  <- meta$method %||% "xavier"
    custom_scale <- meta$custom_scale %||% NULL
    
    if (dbg) cat("[RUNPRED-DBG] meta names:", paste(names(meta), collapse=","), "\n")
    activation_functions <- meta$activation_functions %||%
      (meta$model$activation_functions %||% (meta$preprocessScaledData$activation_functions %||% NULL))
    if (is.null(activation_functions)) {
      stop("[RUNPRED-ERR] activation_functions not found in meta.")
    }
    
    ML_NN <- isTRUE(meta$ML_NN) || isTRUE(ML_NN)
    
    # Last activation quick probe (debug-only)
    last_af <- tryCatch(activation_functions[[length(activation_functions)]], error = function(e) NULL)
    last_nm <- tolower(tryCatch(attr(last_af, "name"), error = function(e) NULL) %||% "")
    is_linear <- FALSE
    if (is.function(last_af)) {
      v <- c(-2,-1,0,1,2)
      outp <- try(last_af(v), silent = TRUE)
      if (!inherits(outp, "try-error") && is.numeric(outp) && length(outp) == length(v)) {
        is_linear <- dplyr::near(as.numeric(outp), v, tol = 1e-12) || identical(last_af, base::identity) || 
          last_nm %in% c("identity","linear","id")
      }
    }
    if (dbg) {
      cat(sprintf("[HEAD-DBG %s] last_activation='%s' | last_is_linear=%s | mode=%s\n",
                  stamp, if (nzchar(last_nm)) last_nm else "<unknown>", is_linear, expected_mode))
      if (!is_linear && expected_mode != "regression") {
        cat("[ACT-DBG] Non-linear head is correct for classification -- no regression flattening concern.\n")
      }
    }
    
    ## ---- Build a single-SONN wrapper and predict ----
    main_model <- DDESONN$new(
      num_networks    = num_networks,
      input_size      = input_size,
      hidden_sizes    = hidden_sizes,
      output_size     = output_size,
      N               = N,
      lambda          = lambda,
      ensemble_number = 1L,
      ensembles       = NULL,
      ML_NN           = ML_NN,
      method          = init_method,
      custom_scale    = custom_scale
    )
    sonn_idx  <- min(model_index, length(main_model$ensemble))
    model_obj <- main_model$ensemble[[sonn_idx]]
    
    call_args <- list(
      Rdata   = X,
      weights = rec$weights,
      biases  = rec$biases,
      activation_functions = activation_functions,
      verbose = vrb,
      debug   = dbg
    )
    
    out <- withCallingHandlers(
      do.call(model_obj$predict, call_args),
      warning = function(w) {
        msg <- conditionMessage(w)
        if (grepl("\\[ACT-DBG\\].*Last activation is NOT linear", msg)) {
          if (identical(expected_mode, "regression")) {
            return(invokeRestart("muffleWarning"))
          } else {
            if (dbg) message(sprintf("[ACT-DBG] Non-linear last activation during predict; expected for '%s'. Silencing.", expected_mode))
            invokeRestart("muffleWarning"); return(invisible())
          }
        }
      }
    )
    
    ## ---- Normalize output ----
    pred <- out$predicted_output %||% out
    if (is.list(pred) && length(pred) == 1L && !is.matrix(pred)) pred <- pred[[1]]
    if (is.data.frame(pred)) pred <- as.matrix(pred)
    if (is.vector(pred))     pred <- matrix(as.numeric(pred), ncol = 1L)
    pred <- as.matrix(pred); storage.mode(pred) <- "double"
    
    if (dbg) cat(sprintf("[RUNPRED-DBG %s] raw model out: %dx%d\n", stamp, nrow(pred), ncol(pred)))
    if (nrow(pred) == 0L) stop("[RUNPRED-ERR] model returned zero rows")
    
    # optional capture hook (silent)
    try({
      env <- get("LAST_DEBUG", inherits = TRUE)
      assign("LAST_RUNPRED_OUT", pred, envir = env)
    }, silent = TRUE)
    
    list(predicted_output = pred)
  }
}



##helpers for
## if (!isTRUE(do_ensemble)) else {} in TestDDESONN.R

## ====== HELPERS (needed in both modes) ======
is_real_serial <- function(x) is.character(x) && length(x) == 1 && !is.na(x) && nzchar(x)
.metric_minimize <- function(m) grepl("mse|mae|rmse|error|loss|quantization_error|topographic_error", tolower(m))

main_meta_var  <- function(i) sprintf("Ensemble_Main_1_model_%d_metadata", as.integer(i))
temp_meta_var  <- function(e,i) sprintf("Ensemble_Temp_%d_model_%d_metadata", as.integer(e), as.integer(i))

.resolve_metric_from_pm <- function(pm, metric_name) {
  if (is.null(pm)) return(NA_real_)
  if (is.list(pm) || is.environment(pm)) {
    val <- pm[[metric_name]]; if (!is.null(val)) return(as.numeric(val)[1])
    nm <- names(pm)
    if (!is.null(nm)) {
      hit <- which(tolower(nm) == tolower(metric_name))
      if (length(hit)) return(as.numeric(pm[[ nm[hit[1]] ]])[1])
    }
  }
  if (is.atomic(pm) && !is.null(names(pm))) {
    nm <- names(pm)
    if (metric_name %in% nm) return(as.numeric(pm[[metric_name]])[1])
    hit <- which(tolower(nm) == tolower(metric_name))
    if (length(hit)) return(as.numeric(pm[[ hit[1] ]])[1])
  }
  if (is.data.frame(pm)) {
    if (metric_name %in% names(pm)) return(as.numeric(pm[[metric_name]][1]))
    hit <- which(tolower(names(pm)) == tolower(metric_name))
    if (length(hit)) return(as.numeric(pm[[ hit[1] ]][1]))
    cn <- tolower(names(pm))
    if (all(c("metric","value") %in% cn)) {
      midx <- which(cn == "metric")[1]; vidx <- which(cn == "value")[1]
      rows <- which(tolower(pm[[midx]]) == tolower(metric_name))
      if (length(rows)) return(as.numeric(pm[[vidx]][ rows[1] ]))
    }
  }
  NA_real_
}

serial_to_meta_name <- function(serial) {
  if (!is_real_serial(serial)) return(NA_character_)
  p <- strsplit(serial, "\\.")[[1]]
  if (length(p) < 3) return(NA_character_)
  e <- suppressWarnings(as.integer(p[1])); i <- suppressWarnings(as.integer(p[3]))
  if (is.na(e) || is.na(i)) return(NA_character_)
  if (e == 1) sprintf("Ensemble_Main_%d_model_%d_metadata", e, i)
  else        sprintf("Ensemble_Temp_%d_model_%d_metadata", e, i)
}

get_metric_by_serial <- function(serial, metric_name) {
  var <- serial_to_meta_name(serial)
  if (nzchar(var) && exists(var, envir = .ddesonn_state)) {
    md <- get(var, envir = .ddesonn_state)
    return(.resolve_metric_from_pm(md$performance_metric, metric_name))
  }
  NA_real_
}

.collect_vals <- function(serials, metric_name) {
  if (!length(serials)) return(data.frame(serial = character(), value = numeric()))
  data.frame(
    serial = as.character(serials),
    value  = vapply(serials, get_metric_by_serial, numeric(1), metric_name),
    stringsAsFactors = FALSE
  )
}

##helpers for 
## =========================================================================================
## SINGLE-RUN MODE (no logs, no lineage, no temp/prune/add) -- covers Scenario A & Scenario B
## =========================================================================================
## if (!isTRUE(do_ensemble)) in TestDDESONN.R

`%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a

# Role helper: 0/1 => main, 2+ => temp
ensemble_role <- function(ensemble_number) {
  if (is.na(ensemble_number) || ensemble_number <= 1L) "main" else "temp"
}

# Attach a DDESONN run into the top-level container in a consistent way
attach_run_to_container <- function(ensembles_container, DDESONN_run) {
  stopifnot(is.list(ensembles_container))
  role <- ensemble_role((DDESONN_run$ensemble_number %||% 0L))
  if (role == "main") {
    ensembles_container$main_ensemble <- ensembles_container$main_ensemble %||% list()
    # single-run: place at [[1]]; real ensembles can append
    if (length(ensembles_container$main_ensemble) == 0L) {
      ensembles_container$main_ensemble[[1]] <- DDESONN_run
    } else {
      ensembles_container$main_ensemble[[length(ensembles_container$main_ensemble) + 1L]] <- DDESONN_run
    }
  } else {
    ensembles_container$temp_ensemble <- ensembles_container$temp_ensemble %||% list()
    ensembles_container$temp_ensemble[[length(ensembles_container$temp_ensemble) + 1L]] <- DDESONN_run
  }
  ensembles_container
}

# Get models inside a DDESONN run (SONN list) safely
get_models <- function(DDESONN_run) {
  x <- tryCatch(DDESONN_run$ensemble, error = function(...) NULL)
  if (is.list(x)) x else list()
}

# Pretty + explained summary (works for E=0 single-run and real ensembles)
print_ensembles_summary <- function(ensembles_container,
                                    explain = TRUE,
                                    show_models = TRUE,
                                    max_models = 5L) {
  `%||%` <- function(a, b) if (is.null(a) || !length(a)) b else a
  role_of <- function(e) if (is.na(e) || e <= 1L) "main" else "temp"
  
  me <- ensembles_container$main_ensemble %||% list()
  te <- ensembles_container$temp_ensemble %||% list()
  
  cat("=== ENSEMBLES SUMMARY ===\n")
  
  if (isTRUE(explain)) {
    cat(
      "Legend:\n",
      "  * E = ensemble_number label (E=0 denotes single-run labeling).\n",
      "  * R lists are 1-based, so the single run lives at main_ensemble[[1]].\n",
      "  * Role: 'main' when E in {0,1}; 'temp' when E >= 2.\n",
      "  * 'models' = SONN models inside a DDESONN run.\n\n", sep = ""
    )
  }
  
  # ---- Main runs ----
  cat(sprintf("Main ensembles (runs): %d\n", length(me)))
  for (i in seq_along(me)) {
    run  <- me[[i]]
    e    <- as.integer(run$ensemble_number %||% (i - 1L))
    mods <- tryCatch(run$ensemble, error = function(...) NULL)
    if (!is.list(mods)) mods <- list()
    
    label <- if (e == 0L) "single-run" else "main"
    cat(sprintf("  E=%d (%s) at main_ensemble[[%d]]: %d model(s)\n", e, label, i, length(mods)))
    
    if (isTRUE(show_models) && length(mods)) {
      upto <- min(length(mods), as.integer(max_models))
      for (m in seq_len(upto)) {
        mdl <- mods[[m]]
        hs  <- tryCatch(mdl$hidden_sizes, error = function(...) NULL)
        nl  <- tryCatch(mdl$num_layers,    error = function(...) NULL)
        act <- tryCatch({
          aa <- mdl$activation_functions
          if (is.list(aa)) {
            vapply(aa, function(f) attr(f, "name") %||% "?", character(1))
          } else {
            NULL
          }
        }, error = function(...) NULL)
        
        cat(sprintf("    M=%d | layers=%s | hidden=%s",
                    m,
                    if (length(nl)) paste0(nl, collapse = ",") else "?",
                    if (length(hs)) paste0(hs, collapse = ",") else "?"))
        if (length(act)) cat(sprintf(" | activations=%s", paste0(act, collapse = ",")))
        cat("\n")
      }
      if (length(mods) > upto) cat(sprintf("    ... (%d more models not shown)\n", length(mods) - upto))
    }
  }
  
  # ---- Temp runs ----
  cat(sprintf("Temp ensembles (runs): %d\n", length(te)))
  for (i in seq_along(te)) {
    run  <- te[[i]]
    e    <- as.integer(run$ensemble_number %||% (i + 1L))
    mods <- tryCatch(run$ensemble, error = function(...) NULL)
    if (!is.list(mods)) mods <- list()
    cat(sprintf("  E=%d (temp) at temp_ensemble[[%d]]: %d model(s)\n", e, i, length(mods)))
  }
  
  invisible(NULL)
}



## helper for prune and add
EMPTY_SLOT <- structure(list(.empty_slot = TRUE), class = "EMPTY_SLOT")
is_empty_slot <- function(x) is.list(x) && inherits(x, "EMPTY_SLOT")







tune_threshold_accuracy <- function(predicted_output, labels,
                                    metric = c("accuracy", "f1", "precision", "recall",
                                               "macro_f1", "macro_precision", "macro_recall"),
                                    threshold_grid = seq(0.05, 0.95, by = 0.01),
                                    verbose = FALSE) {
  grid <- threshold_grid
  metric <- match.arg(metric)
  
  # --- Sanitize 'grid' (avoid passing function/env/list/NULL and clamp to (0,1)) ---
  if (missing(grid) || is.null(grid) || is.function(grid) || is.environment(grid) || is.list(grid)) {
    grid <- seq(0.05, 0.95, by = 0.01)
  } else {
    grid <- tryCatch(as.numeric(unlist(grid, use.names = FALSE)), error = function(e) numeric(0))
    grid <- grid[is.finite(grid)]
    grid <- grid[grid > 0 & grid < 1]     # thresholds must be in (0,1)
    grid <- sort(unique(grid))
    if (length(grid) == 0L) grid <- seq(0.05, 0.95, by = 0.01)
  }
  
  # --- Coerce to numeric matrices and align columns ---
  P <- as.matrix(predicted_output); storage.mode(P) <- "double"
  L <- as.matrix(labels);           storage.mode(L) <- "double"
  
  nL <- ncol(L); nP <- ncol(P); K <- max(nL, nP)
  if (K < 1L) K <- 1L
  
  if (ncol(P) < K) {
    total_needed <- nrow(L) * K
    replicated <- rep(as.vector(P), length.out = total_needed)
    P <- matrix(replicated, nrow = nrow(L), ncol = K, byrow = FALSE)
  } else if (ncol(P) > K) {
    P <- P[, 1:K, drop = FALSE]
  }
  
  # =========================
  # Binary
  # =========================
  if (K == 1L) {
    y_true <- as.numeric(L[, 1])
    y_true01 <- as.integer(ifelse(y_true > 0, 1L, 0L))
    
    best_t <- NA_real_; best_val <- -Inf; best_pred <- NULL
    for (t in grid) {
      y_pred <- as.integer(P[, 1] >= t)
      TP <- sum(y_pred == 1 & y_true01 == 1)
      FP <- sum(y_pred == 1 & y_true01 == 0)
      FN <- sum(y_pred == 0 & y_true01 == 1)
      prec <- TP / (TP + FP + 1e-8)
      rec  <- TP / (TP + FN + 1e-8)
      
      val <- switch(metric,
                    accuracy  = mean(y_pred == y_true01),
                    f1        = 2 * prec * rec / (prec + rec + 1e-8),
                    precision = prec,
                    recall    = rec
      )
      if (val > best_val) { best_val <- val; best_t <- t; best_pred <- y_pred }
    }
    tuned_acc <- mean(best_pred == y_true01)
    if (verbose) cat(sprintf("[Binary] best_t=%.3f | tuned_%s=%.6f\n", best_t, metric, best_val))
    return(list(
      thresholds     = best_t,
      y_pred_class   = best_pred,
      tuned_score    = as.numeric(best_val),
      tuned_accuracy = as.numeric(tuned_acc),
      metric_used    = metric
    ))
  }
  
  # =========================
  # Multiclass
  # =========================
  if (ncol(L) > 1L) {
    y_true_ids <- max.col(L, ties.method = "first")
  } else {
    cls <- as.integer(L[, 1]); if (min(cls, na.rm = TRUE) == 0L) cls <- cls + 1L
    cls[cls < 1L] <- 1L
    cls[cls > K] <- K
    y_true_ids <- cls
  }
  
  thr <- numeric(K)
  for (k in seq_len(K)) {
    y_true01 <- as.integer(y_true_ids == k)
    best_tk <- NA_real_; best_valk <- -Inf
    for (t in grid) {
      y_pred01 <- as.integer(P[, k] >= t)
      TP <- sum(y_pred01 == 1 & y_true01 == 1)
      FP <- sum(y_pred01 == 1 & y_true01 == 0)
      FN <- sum(y_pred01 == 0 & y_true01 == 1)
      prec <- TP / (TP + FP + 1e-8)
      rec  <- TP / (TP + FN + 1e-8)
      
      val <- switch(metric,
                    macro_f1        = 2 * prec * rec / (prec + rec + 1e-8),
                    macro_precision = prec,
                    macro_recall    = rec,
                    accuracy        = mean(y_pred01 == y_true01)
      )
      if (val > best_valk) { best_valk <- val; best_tk <- t }
    }
    thr[k] <- best_tk
  }
  
  # Apply thresholds with masked argmax + fallback
  masked <- P
  for (k in seq_len(K)) masked[, k] <- ifelse(P[, k] >= thr[k], P[, k], -Inf)
  y_pred_ids <- max.col(masked, ties.method = "first")
  all_neg_inf <- !is.finite(apply(masked, 1, max))
  if (any(all_neg_inf)) {
    y_pred_ids[all_neg_inf] <- max.col(P[all_neg_inf, , drop = FALSE], ties.method = "first")
  }
  
  # Evaluate tuned metrics
  tuned_acc <- mean(y_pred_ids == y_true_ids)
  
  # If optimizing a macro metric, report its value from the confusion matrix
  tuned_score <- tuned_acc
  if (metric %in% c("macro_f1", "macro_precision", "macro_recall")) {
    tab <- table(factor(y_true_ids, levels = 1:K), factor(y_pred_ids, levels = 1:K))
    TPk <- diag(tab)
    FPk <- colSums(tab) - TPk
    FNk <- rowSums(tab) - TPk
    Prec_k <- ifelse((TPk + FPk) > 0, TPk / (TPk + FPk), 0)
    Rec_k  <- ifelse((TPk + FNk) > 0, TPk / (TPk + FNk), 0)
    if (metric == "macro_precision") tuned_score <- mean(Prec_k)
    if (metric == "macro_recall")    tuned_score <- mean(Rec_k)
    if (metric == "macro_f1") {
      F1_k <- ifelse((Prec_k + Rec_k) > 0, 2 * Prec_k * Rec_k / (Prec_k + Rec_k), 0)
      tuned_score <- mean(F1_k)
    }
  }
  
  if (verbose) {
    cat(sprintf("[Multiclass] tuned_acc=%.6f | metric=%s\n", tuned_acc, metric))
    cat(" thresholds: ", paste0(sprintf("%.3f", thr), collapse = ", "), "\n")
  }
  return(list(
    thresholds     = thr,
    y_pred_class   = y_pred_ids,
    tuned_score    = as.numeric(tuned_score),
    tuned_accuracy = as.numeric(tuned_acc),
    metric_used    = metric
  ))
}



evaluate_classification_metrics <- function(preds, labels) {
  labels <- as.vector(labels)
  preds <- as.vector(preds)
  
  TP <- sum(preds == 1 & labels == 1)
  FP <- sum(preds == 1 & labels == 0)
  FN <- sum(preds == 0 & labels == 1)
  
  precision <- TP / (TP + FP + 1e-8)
  recall <- TP / (TP + FN + 1e-8)
  F1 <- 2 * precision * recall / (precision + recall + 1e-8)
  
  return(list(precision = precision, recall = recall, F1 = F1))
}

##helpers for calculate_performance_grouped 

aggregate_predictions <- function(predicted_output_list,
                                  method = c("mean","median","vote"),
                                  weights = NULL) {
  method <- match.arg(method)
  mats <- lapply(predicted_output_list, function(x) { x <- as.matrix(x); storage.mode(x) <- "double"; x })
  
  N <- nrow(mats[[1]]); K <- ncol(mats[[1]]); M <- length(mats)
  stopifnot(all(vapply(mats, nrow, 1L) == N),
            all(vapply(mats, ncol, 1L) == K))
  
  if (is.null(weights)) weights <- rep(1/M, M) else {
    stopifnot(length(weights) == M); weights <- weights / sum(weights)
  }
  
  if (method == "median") {
    arr <- simplify2array(mats)        # N x K x M
    return(apply(arr, c(1,2), stats::median, na.rm = TRUE))
  }
  
  # mean / vote -> elementwise weighted mean, keeps N x K
  X <- do.call(cbind, lapply(mats, function(m) as.vector(m)))  # (N*K) x M
  out <- as.numeric(X %*% weights)                             # (N*K)
  matrix(out, nrow = N, ncol = K)
}

# Full replacement (compact + multiclass-safe)
# Full replacement -- compact, NA-safe, binary/multiclass
pick_representative_sonn <- function(SONN_list, predicted_output_list, labels) {
  to_class <- function(y) {
    if (is.matrix(y)) {
      # Argmax per row; keep NA if an entire row is NA
      apply(y, 1L, function(r) if (all(is.na(r))) NA_integer_ else which.max(r))
    } else if (is.factor(y)) {
      as.integer(y)
    } else if (is.character(y)) {
      as.integer(factor(y))
    } else {
      as.integer(y)
    }
  }
  
  f1_macro <- function(P, Y) {
    yt <- to_class(Y)
    yp <- if (is.matrix(P) && ncol(P) > 1L) {
      apply(P, 1L, function(r) if (all(is.na(r))) NA_integer_ else which.max(r))
    } else {
      p1 <- if (is.matrix(P)) P[, 1L, drop = TRUE] else P
      as.integer(as.numeric(p1) >= 0.5)
    }
    n <- min(length(yt), length(yp)); if (!n) return(0)
    yt <- yt[seq_len(n)]; yp <- yp[seq_len(n)]
    cls <- sort(unique(c(yt, yp))); if (!length(cls)) return(0)
    
    f1s <- sapply(cls, function(k) {
      tp <- sum(yp == k & yt == k, na.rm = TRUE)
      fp <- sum(yp == k & yt != k, na.rm = TRUE)
      fn <- sum(yp != k & yt == k, na.rm = TRUE)
      if (tp == 0 && fp == 0 && fn == 0) return(0)
      pr <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
      rc <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
      if ((pr + rc) == 0) 0 else 2 * pr * rc / (pr + rc)
    })
    mean(f1s, na.rm = TRUE)
  }
  
  if (!length(predicted_output_list)) return(SONN_list[[1L]])
  scores <- vapply(seq_along(predicted_output_list),
                   function(i) f1_macro(predicted_output_list[[i]], labels),
                   numeric(1))
  SONN_list[[which.max(scores)]]
}


# -------------------------------
# Helpers (keep top-level in file)
# -------------------------------
# ---- Safe, name-aligning flattener for per-model metrics ---------------------
# metric_list: list of length M; each element like list(metrics=<named list>, names=<char>)
# run_id:      character/list of length M with model labels
flatten_metrics_to_df <- function(metric_list, run_id) {
  if (is.null(metric_list) || length(metric_list) == 0) return(NULL)
  
  # Build one wide row per model
  rows <- lapply(seq_along(metric_list), function(i) {
    mi <- metric_list[[i]]
    if (is.null(mi) || is.null(mi$metrics)) return(NULL)
    
    # Flatten nested lists to atomic named vector
    flat <- tryCatch(
      unlist(mi$metrics, recursive = TRUE, use.names = TRUE),
      error = function(e) NULL
    )
    if (is.null(flat) || length(flat) == 0) return(NULL)
    
    # Coerce to one-row data.frame
    df <- as.data.frame(as.list(flat), stringsAsFactors = FALSE)
    
    # Ensure syntactically valid, unique names
    names(df) <- make.names(names(df), unique = TRUE)
    
    # Attach model label
    df$Model_Name <- if (!is.null(run_id) && length(run_id) >= i) run_id[[i]] else paste0("Model_", i)
    
    df
  })
  
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(NULL)
  
  # Align columns across all rows
  all_names <- unique(unlist(lapply(rows, names), use.names = FALSE))
  
  rows <- lapply(rows, function(df) {
    missing <- setdiff(all_names, names(df))
    if (length(missing)) {
      # fill missing metrics with NA (numeric); keep Model_Name as character
      for (nm in missing) df[[nm]] <- NA_real_
    }
    # Order columns consistently
    df <- df[all_names]
    df
  })
  
  # Bind safely (names now match); avoid base rbind name-matching issues
  wide <- do.call(rbind, rows)
  rownames(wide) <- NULL
  
  # Long tidy form (Model_Name + Metric + Value)
  metric_cols <- setdiff(names(wide), "Model_Name")
  
  # Use tidyr if available; otherwise base reshape
  if (requireNamespace("tidyr", quietly = TRUE)) {
    long <- tidyr::pivot_longer(
      wide,
      cols = dplyr::all_of(metric_cols),
      names_to = "Metric",
      values_to = "Value"
    )
  } else {
    # Base fallback
    long <- stats::reshape(
      wide,
      varying = metric_cols,
      v.names = "Value",
      timevar = "Metric",
      times = metric_cols,
      idvar = "Model_Name",
      direction = "long"
    )
    long <- long[ , c("Model_Name", "Metric", "Value")]
  }
  
  # Best-effort numeric coercion for Value
  suppressWarnings(long$Value <- as.numeric(long$Value))
  
  long
}

summarize_grouped <- function(long_df) {
  if (is.null(long_df) || !nrow(long_df)) return(NULL)
  
  keep <- with(long_df, tapply(Value, Metric, function(v) any(is.finite(v))))
  keep_metrics <- names(keep)[keep]
  df <- long_df[long_df$Metric %in% keep_metrics, c("Model_Name","Metric","Value"), drop = FALSE]
  if (!nrow(df)) return(NULL)
  
  stats_mat <- do.call(rbind, lapply(split(df$Value, df$Metric), function(v) {
    v <- v[is.finite(v)]
    c(mean = mean(v, na.rm = TRUE),
      median = stats::median(v, na.rm = TRUE),
      n = length(v))
  }))
  data.frame(Metric = rownames(stats_mat), stats_mat, row.names = NULL, check.names = FALSE)
}

##helpers for performance and relevance metric fun.

# Sanitize a numeric threshold grid
sanitize_grid <- function(grid, default = seq(0.05, 0.95, by = 0.01)) {
  if (is.null(grid) || is.function(grid) || is.environment(grid) || is.list(grid)) {
    return(default)
  }
  g <- suppressWarnings(as.numeric(grid))
  g <- unique(g[is.finite(g)])
  if (!length(g)) default else sort(g)
}

# Collapse label matrix to a 0/1 vector when possible
# - 1 col: >0 -> 1 else 0
# - 2 col: argmax -> {0,1}
# - >=3 col: return NULL (treat as multiclass upstream)
labels_to_binary_vec <- function(L) {
  if (is.null(ncol(L)) || ncol(L) == 1L) {
    return(ifelse(as.vector(L) > 0, 1L, 0L))
  }
  if (ncol(L) == 2L) {
    return(max.col(L, ties.method = "first") - 1L)  # 0/1
  }
  NULL
}

# Extract a single positive-class probability vector from predictions
# - 1 col: treat as P(pos)
# - 2 col: use column 2 as P(pos) (common {neg,pos})
# - >=3 col: return NULL (multiclass)
preds_to_pos_prob <- function(P) {
  if (is.null(ncol(P)) || ncol(P) == 1L) {
    p <- as.vector(P)
  } else if (ncol(P) == 2L) {
    p <- as.vector(P[, 2, drop = TRUE])
  } else {
    return(NULL)
  }
  p[!is.finite(p)] <- 0
  p <- pmin(pmax(p, 0), 1)
  p
}

# Decide if the task is binary based on labels (preferred) or predictions
infer_is_binary <- function(L, P) {
  Lb <- labels_to_binary_vec(L)
  if (!is.null(Lb)) {
    uniqL <- unique(na.omit(Lb))
    return(list(is_binary = length(uniqL) <= 2 && all(uniqL %in% c(0,1)), Lb = Lb))
  }
  # If labels inconclusive (>=3 cols), use predictions shape as hint
  list(is_binary = (!is.null(P) && ncol(P) <= 2L), Lb = NULL)
}

# Build one-hot matrix (N x K) from class ids (1..K)
one_hot_from_ids <- function(ids, K, N = NULL, strict = TRUE) {
  # rows = N or length(ids) by default
  if (is.null(N)) N <- length(ids)
  
  # initialize integer matrix
  M <- matrix(0L, nrow = N, ncol = K)
  
  # finiteness check on numeric-ish values (handles NA/NaN/Inf)
  ids_num <- suppressWarnings(as.numeric(ids))
  ok_finite <- is.finite(ids_num)
  
  # optional strict check: used ids must be whole numbers (e.g., 3.0 is fine, 3.2 is not)
  if (strict) {
    non_whole <- ok_finite & (abs(ids_num - round(ids_num)) > .Machine$double.eps^0.5)
    if (any(non_whole, na.rm = TRUE)) {
      stop("one_hot_from_ids: non-integer class ids detected among finite values.")
    }
  }
  
  # coerce to integer indices (after checks)
  ids_int <- suppressWarnings(as.integer(round(ids_num)))
  
  # valid positions: finite, in range [1, K]
  ok <- ok_finite & !is.na(ids_int) & ids_int >= 1L & ids_int <= K
  
  if (any(ok)) {
    M[cbind(seq_len(N)[ok], ids_int[ok])] <- 1L
  }
  M
}


# Safe call to a user-provided metrics helper (optional)
safe_eval_metrics <- function(y_pred_class, y_true01, verbose = FALSE) {
  out <- NULL
  try({
    out <- evaluate_classification_metrics(y_pred_class, y_true01)
  }, silent = !verbose)
  out
}


# =========================
# Metric helpers (general)
# =========================

.metric_minimize <- get0(".metric_minimize", ifnotfound = function(metric_name) {
  m <- tolower(metric_name)
  
  # metrics that should be minimized
  minimize_patterns <- "(loss|mse|mae|rmse|logloss|cross.?entropy|error|nll)"
  if (grepl(minimize_patterns, m)) return(TRUE)
  
  # explicit maximize overrides (always higher is better)
  maximize_set <- c("accuracy", "precision", "recall", "f1", "ndcg", "clustering_quality_db")
  if (m %in% maximize_set) return(FALSE)
  
  # default: maximize
  FALSE
})


# Robust metric fetcher: searches performance_* and relevance_* (incl. nested)
.get_metric_from_meta <- function(meta, metric_name) {
  if (is.null(meta)) return(NA_real_)
  # --- normalizer helpers ---
  .norm <- function(x) tolower(gsub("[^a-z0-9]+", "", trimws(as.character(x))))
  .is_num <- function(x) is.numeric(x) && length(x) == 1L && is.finite(x)
  
  # --- gather candidate maps (shallow + useful nested) ---
  maps <- list()
  if (!is.null(meta$performance_metric))    maps <- c(maps, list(performance_metric = meta$performance_metric))
  if (!is.null(meta$performance_metrics))   maps <- c(maps, list(performance_metrics = meta$performance_metrics))
  if (!is.null(meta$metrics))               maps <- c(maps, list(metrics = meta$metrics))
  if (!is.null(meta$relevance_metric))      maps <- c(maps, list(relevance_metric = meta$relevance_metric))
  if (!is.null(meta$relevance_metrics))     maps <- c(maps, list(relevance_metrics = meta$relevance_metrics))
  
  # specific nested commonly used in your structure
  acc_tuned <- tryCatch(meta$performance_metric$accuracy_tuned, error = function(e) NULL)
  if (!is.null(acc_tuned)) {
    maps <- c(maps, list(accuracy_tuned = acc_tuned))
    if (!is.null(acc_tuned$metrics)) maps <- c(maps, list(accuracy_tuned_metrics = acc_tuned$metrics))
  }
  
  if (!length(maps)) return(NA_real_)
  
  # --- flatten maps into a single name -> value registry (depth <= 2-3) ---
  key_norm   <- character()
  key_raw    <- character()
  val_store  <- list()
  
  .collect <- function(x, prefix = "") {
    if (is.list(x)) {
      for (nm in names(x)) {
        child <- x[[nm]]
        pfx <- if (nzchar(prefix)) paste0(prefix, ".", nm) else nm
        if (is.list(child)) {
          # one more level
          .collect(child, pfx)
        } else {
          # atomic leaf
          key_norm <<- c(key_norm, .norm(nm))
          key_raw  <<- c(key_raw,  pfx)
          val_store[[length(val_store) + 1L]] <<- child
        }
      }
    } else {
      # unnamed atomic (rare)
      key_norm <<- c(key_norm, .norm(prefix))
      key_raw  <<- c(key_raw, prefix)
      val_store[[length(val_store) + 1L]] <<- x
    }
  }
  
  for (m in maps) .collect(m)
  
  # coerce to numeric where possible
  vals_num <- suppressWarnings(as.numeric(unlist(val_store, use.names = FALSE)))
  # keep only those that are actually numeric scalars
  ok_num <- is.finite(vals_num)
  key_norm <- key_norm[ok_num]
  key_raw  <- key_raw[ok_num]
  vals_num <- vals_num[ok_num]
  if (!length(vals_num)) return(NA_real_)
  
  # --- aliasing for common names / variants ---
  req <- .norm(metric_name)
  alias <- list(
    "accuracy"  = c("accuracy", "accuracypercent", "acc"),
    "precision" = c("precision", "prec"),
    "recall"    = c("recall", "tpr", "sensitivity"),
    "f1"        = c("f1", "f1score", "f1_macro", "macrof1", "f1macro"),
    "macro_f1"  = c("macrof1", "f1macro", "f1_macro"),
    "micro_f1"  = c("microf1", "f1micro", "f1_micro"),
    "ndcg"      = c("ndcg", "ndcg@5", "ndcg@10", "ndcg@k"),
    "mse"       = c("mse"),
    "mae"       = c("mae"),
    "rmse"      = c("rmse"),
    "top1"      = c("top1", "top_1", "top-1")
  )
  cand <- unique(c(alias[[req]] %||% character(0), req))
  
  # 1) exact (normalized) match
  hit <- which(key_norm %in% cand)
  # 2) fallback: contains match
  if (!length(hit)) {
    hit <- unlist(lapply(cand, function(k) which(grepl(k, key_norm, fixed = TRUE))))
    hit <- unique(hit)
  }
  if (!length(hit)) return(NA_real_)
  
  # choose first finite numeric
  vals <- vals_num[hit]
  idx  <- which(is.finite(vals))[1L]
  if (is.na(idx)) return(NA_real_)
  vals[idx]
}


# utils.R
# -------------------------------------------------------
# Best-model finder by TARGET_METRIC (general, any kind)
# Depends on: bm_list_all(), bm_select_exact(), .metric_minimize(), .get_metric_from_meta()
# -------------------------------------------------------
find_best_model <- function(target_metric_name_best,
                            kind_filter  = c("Main","Temp"),
                            ens_filter   = NULL,
                            model_filter = NULL,
                            dir = .BM_DIR) {
  minimize <- .metric_minimize(target_metric_name_best)
  
  df <- bm_list_all(dir)
  if (!nrow(df)) {
    cat("\n==== FIND_BEST_MODEL ====\nNo candidates in env/RDS.\n")
    return(list(best_row=NULL, meta=NULL, tbl=data.frame(), minimize=minimize))
  }
  
  if (length(kind_filter))    df <- df[df$kind  %in% kind_filter, , drop = FALSE]
  if (!is.null(ens_filter))   df <- df[df$ens   %in% ens_filter,  , drop = FALSE]
  if (!is.null(model_filter)) df <- df[df$model %in% model_filter,, drop = FALSE]
  if (!nrow(df)) {
    cat("\n==== FIND_BEST_MODEL ====\nNo candidates after filters.\n")
    return(list(best_row=NULL, meta=NULL, tbl=df, minimize=minimize))
  }
  
  df$metric_value <- vapply(seq_len(nrow(df)), function(i) {
    meta_i <- tryCatch(bm_select_exact(df$kind[i], df$ens[i], df$model[i], dir = dir), error = function(e) NULL)
    if (is.null(meta_i)) return(NA_real_)
    .get_metric_from_meta(meta_i, target_metric_name_best)
  }, numeric(1))
  
  cat("\n==== FIND_BEST_MODEL (ANY KIND) ====\n")
  ok <- is.finite(df$metric_value)
  if (!any(ok)) {
    print(df[, c("name","kind","ens","model","source")], row.names = FALSE)
    cat("All metric values are NA/Inf for", target_metric_name_best, "\n")
    return(list(best_row=NULL, meta=NULL, tbl=df, minimize=minimize))
  }
  
  df_ok <- df[ok, , drop = FALSE]
  ord   <- if (minimize) order(df_ok$metric_value) else order(df_ok$metric_value, decreasing = TRUE)
  df_ok <- df_ok[ord, , drop = FALSE]
  
  top  <- df_ok[1, , drop = FALSE]
  meta <- bm_select_exact(top$kind, top$ens, top$model, dir = dir)
  
  print(head(df_ok[, c("name","kind","ens","model","metric_value")], 10), row.names = FALSE)
  cat(sprintf("-> Selected: %s | %s=%.6f (%s better)\n",
              top$name, target_metric_name_best, top$metric_value,
              if (minimize) "lower" else "higher"))
  
  list(best_row = top, meta = meta, tbl = df_ok, minimize = minimize)
}

#used to calc error in predict() in legacy, but now use helper and used in train()

# if (!is.null(predicted_outputAndTime$predicted_output_l2)) {
#   
#   all_predicted_outputs[[i]]       <- predicted_outputAndTime$predicted_output_l2$predicted_output
#   all_prediction_times[[i]]        <- predicted_outputAndTime$train_reg_prediction_time
#   all_errors[[i]]                  <- compute_error(predicted_outputAndTime$predicted_output_l2$predicted_output, y, CLASSIFICATION_MODE)


compute_error <- function(
    predicted_output,
    labels,
    CLASSIFICATION_MODE = c("binary","multiclass","regression")
) {
  CLASSIFICATION_MODE <- match.arg(CLASSIFICATION_MODE)
  if (is.null(labels)) return(NULL)
  
  # --- normalize predictions to a matrix ---
  P <- as.matrix(predicted_output)
  n <- nrow(P); k <- ncol(P)
  
  # --- helpers ---
  as_mat <- function(x) {
    if (is.data.frame(x)) return(as.matrix(x))
    if (is.vector(x))     return(matrix(x, ncol = 1L))
    as.matrix(x)
  }
  align_rows <- function(M, rows) {
    if (nrow(M) == rows) return(M)
    if (nrow(M) > rows)  return(M[seq_len(rows), , drop = FALSE])
    pad <- matrix(rep(M[nrow(M), , drop = FALSE], length.out = (rows - nrow(M)) * ncol(M)),
                  nrow = rows - nrow(M), byrow = TRUE)
    rbind(M, pad)
  }
  pad_or_trim_cols <- function(M, cols, pad = 0) {
    if (ncol(M) == cols) return(M)
    if (ncol(M) > cols)  return(M[, seq_len(cols), drop = FALSE])
    cbind(M, matrix(pad, nrow = nrow(M), ncol = cols - ncol(M)))
  }
  
  # =========================
  # BINARY CLASSIFICATION
  # =========================
  if (CLASSIFICATION_MODE == "binary") {
    pred <- if (k == 1L) P else matrix(P[, k], ncol = 1L)  # take last col if logits/probs provided
    L <- labels
    
    if (is.factor(L)) {
      L <- as.integer(L) - 1L
    } else if (is.character(L)) {
      lu <- sort(unique(L))
      L  <- as.integer(factor(L, levels = lu)) - 1L
    } else {
      L <- suppressWarnings(as.numeric(L))
      if (all(is.na(L))) {
        lu <- sort(unique(as.character(labels)))
        L  <- as.integer(factor(as.character(labels), levels = lu)) - 1L
      }
    }
    
    L <- as_mat(L)
    if (ncol(L) > 1L) L <- matrix(L[, ncol(L)], ncol = 1L)
    L <- align_rows(L, n)
    L <- pad_or_trim_cols(L, 1L, pad = 0)
    
    return(abs(L - pred))
  }
  
  # =========================
  # MULTICLASS CLASSIFICATION
  # =========================
  if (CLASSIFICATION_MODE == "multiclass") {
    L <- labels
    if (is.data.frame(L)) L <- as.matrix(L)
    
    # If labels are already one-hot / probability matrix -> align & diff
    if (!is.null(dim(L)) && ncol(L) > 1L) {
      Y <- align_rows(as.matrix(L), n)
      Y <- pad_or_trim_cols(Y, k, pad = 0)
      return(abs(Y - P))
    }
    
    # Otherwise, treat labels as class IDs / names
    class_order <- if (!is.null(colnames(P))) colnames(P) else sort(unique(as.character(L)))
    
    if (is.factor(L)) {
      L_idx <- as.integer(factor(L, levels = class_order))
    } else if (is.character(L)) {
      L_idx <- as.integer(factor(L, levels = class_order))
    } else {
      L_num <- suppressWarnings(as.numeric(L))
      if (!all(is.na(L_num))) {
        if (min(L_num, na.rm = TRUE) == 0 && max(L_num, na.rm = TRUE) <= (k - 1)) {
          L_idx <- as.integer(L_num + 1L)
        } else {
          L_idx <- as.integer(L_num)
        }
      } else {
        L_idx <- as.integer(factor(as.character(L), levels = class_order))
      }
    }
    
    if (anyNA(L_idx)) L_idx[is.na(L_idx)] <- 1L
    L_idx <- rep(L_idx, length.out = n)
    
    Y <- matrix(0, nrow = n, ncol = k)
    Y[cbind(seq_len(n), pmax(1L, pmin(k, L_idx)))] <- 1L
    
    return(abs(Y - P))
  }
  
  # =========================
  # REGRESSION
  # =========================
  # Goal: return absolute error |Y - P| with shape n x k.
  # - Accepts labels as vector/matrix/data.frame.
  # - Coerces to numeric, aligns rows, and matches columns:
  #   * If labels have 1 col and k > 1 -> replicate that column across k.
  #   * If labels have >1 col -> trim or pad with zeros to match k.
  L <- labels
  
  # Coerce to numeric matrix safely
  to_numeric_matrix <- function(x) {
    M <- as_mat(x)
    # Try numeric coercion while preserving shape
    suppressWarnings(storage.mode(M) <- "double")
    # If still all NA (e.g., character), try a character->numeric pass
    if (all(is.na(M))) {
      M_chr <- as_mat(as.character(x))
      suppressWarnings(storage.mode(M_chr) <- "double")
      M <- M_chr
    }
    M
  }
  
  Y <- to_numeric_matrix(L)
  if (is.null(dim(Y))) Y <- matrix(Y, ncol = 1L)
  
  # Align rows
  Y <- align_rows(Y, n)
  
  # Align columns to predictions
  if (ncol(Y) == 1L && k > 1L) {
    Y <- matrix(rep(Y[, 1L], times = k), nrow = n, ncol = k)
  } else {
    Y <- pad_or_trim_cols(Y, k, pad = 0)
  }
  
  return(abs(Y - P))
}

optimizers_log_update <- function(
    optimizer, epoch, layer, target,
    grads_matrix,            # gradient(s) passed in
    P_before = NULL,         # param before (weights or biases)
    P_after  = NULL,         # param after  (weights or biases)
    update_applied = NULL,   # the actual update step applied (same shape as P)
    verbose = verbose
) {
  verbose <- isTRUE(verbose %||% getOption("DDESONN.verbose", FALSE))  #$$$$$$$$$$$$$
  
  # ---- helpers ----
  .as_num <- function(x) {
    if (is.list(x)) x <- x[[1]]
    if (is.null(x)) return(numeric())
    as.numeric(x)
  }
  .stats <- function(x) {
    v <- suppressWarnings(.as_num(x))
    if (!length(v) || all(!is.finite(v))) return("min=NA mean=NA max=NA")
    sprintf("min=%.3g mean=%.3g max=%.3g", min(v, na.rm = TRUE),
            mean(v, na.rm = TRUE), max(v, na.rm = TRUE))
  }
  .shape <- function(x) {
    if (is.list(x)) x <- x[[1]]
    if (is.null(x)) return("EMPTY")
    d <- dim(x)
    if (is.null(d)) sprintf("len=%d", length(x)) else sprintf("%dx%d", d[1], d[2])
  }
  
  # ---- build message ----
  grads_msg  <- sprintf("grads(%s):[%s]",  .shape(grads_matrix),  .stats(grads_matrix))
  pre_msg    <- if (!is.null(P_before)) sprintf(" | %s_pre(%s):[%s]",  target, .shape(P_before), .stats(P_before)) else ""
  post_msg   <- if (!is.null(P_after))  sprintf(" | %s_post(%s):[%s]", target, .shape(P_after),  .stats(P_after))  else ""
  update_msg <- if (!is.null(update_applied)) sprintf(" | update(%s):[%s]", .shape(update_applied), .stats(update_applied)) else ""
  
  if (isTRUE(verbose)) {  #$$$$$$$$$$$$$
    cat(sprintf("[OPT=%s][E%d][L%d][%s] %s%s%s%s\n",  #$$$$$$$$$$$$$
                toupper(optimizer), epoch, layer, target,  #$$$$$$$$$$$$$
                grads_msg, pre_msg, post_msg, update_msg))  #$$$$$$$$$$$$$
  }  #$$$$$$$$$$$$$
  
}

coerce_pred_schema <- function(df) {
  stopifnot(is.data.frame(df))
  expected <- c("run_index","seed","model_slot","y_true","y_prob","y_pred")
  
  if (!"run_index"  %in% names(df)) df$run_index  <- NA_integer_
  if (!"seed"       %in% names(df)) df$seed       <- NA_integer_
  if (!"model_slot" %in% names(df)) df$model_slot <- NA_integer_
  if (!"y_true"     %in% names(df)) df$y_true     <- NA_real_
  
  if (!"y_prob" %in% names(df) && "y_pred" %in% names(df)) {
    df$y_prob <- suppressWarnings(as.numeric(df$y_pred))
  } else if (!"y_prob" %in% names(df)) {
    df$y_prob <- NA_real_
  }
  if (!"y_pred" %in% names(df) && "y_prob" %in% names(df)) {
    df$y_pred <- suppressWarnings(as.numeric(df$y_prob))
  } else if (!"y_pred" %in% names(df)) {
    df$y_pred <- NA_real_
  }
  
  df$run_index  <- suppressWarnings(as.integer(df$run_index))
  df$seed       <- suppressWarnings(as.integer(df$seed))
  df$model_slot <- suppressWarnings(as.integer(df$model_slot))
  df$y_true     <- suppressWarnings(as.numeric(df$y_true))
  df$y_prob     <- suppressWarnings(as.numeric(df$y_prob))
  df$y_pred     <- suppressWarnings(as.numeric(df$y_pred))
  
  # NON-DESTRUCTIVE: expected first, keep all extras afterward
  present <- expected[expected %in% names(df)]
  extras  <- setdiff(names(df), present)
  df[, c(present, extras), drop = FALSE]
}


#################################################################################################
# MAIN TEST PREDICT-ONLY FUNCTION
#################################################################################################
DDESONN_predict_eval <- function(
    LOAD_FROM_RDS = FALSE,
    ENV_META_NAME = "Ensemble_Main_0_model_1_metadata",
    INPUT_SPLIT   = "test",
    CLASSIFICATION_MODE,
    RUN_INDEX,
    SEED,
    OUTPUT_DIR = NULL, 
    SAVE_METRICS_RDS = TRUE,
    METRICS_PREFIX   = "metrics_test",
    AGG_PREDICTIONS_FILE = NULL,
    AGG_METRICS_FILE     = NULL,
    MODEL_SLOT           = 1L,
    DEBUG                = TRUE,
    OUT_DIR_ASSERT       = NULL
) {
  ## =========================
  ## tiny internals (debuggy)
  ## =========================
  dcat <- function(..., .force = FALSE) if (isTRUE(DEBUG) || isTRUE(.force))
    cat(sprintf("[DSE-DBG %s] ", format(Sys.time(), "%H:%M:%S")), paste0(..., collapse=""), "\n")
  
  `%||%` <- function(x,y) if (is.null(x)) y else x
  
  r6 <- function(x){
    if (is.null(x)) return(NA_real_)
    if (is.list(x)) x <- unlist(x, use.names=FALSE)
    suppressWarnings({
      xn <- as.numeric(x[1])
      if (!is.finite(xn)) return(NA_real_)
      round(xn, 6)
    })
  }
  
  ## type-aware head/skim that won't crash on non-numeric matrices
  dbg_head <- function(x, n=3, tag="(obj)") {
    cls <- paste(class(x), collapse=",")
    typ <- typeof(x)
    dcat(sprintf("%s class=%s type=%s", tag, cls, typ))
    if (is.matrix(x)) {
      nr <- nrow(x); nc <- ncol(x)
      if (is.numeric(x)) {
        sm_min <- suppressWarnings(min(x, na.rm=TRUE))
        sm_mn  <- suppressWarnings(mean(x, na.rm=TRUE))
        sm_sd  <- suppressWarnings(stats::sd(as.numeric(x), na.rm=TRUE))
        sm_max <- suppressWarnings(max(x, na.rm=TRUE))
        dcat(sprintf("%s dims=%d x %d | summary: min=%.6f mean=%.6f sd=%.6f max=%.6f",
                     tag, nr, nc, sm_min, sm_mn, sm_sd, sm_max))
      } else {
        # preview first column values instead of numeric summary
        ex <- tryCatch(utils::head(unique(as.character(x[,1])), 6L), error=function(e) character(0))
        ex_str <- if (length(ex)) paste(ex, collapse=", ") else "<no preview>"
        dcat(sprintf("%s dims=%d x %d | example col1: %s", tag, nr, nc, ex_str))
      }
      if (nr > 0 && nc > 0) {
        print(utils::head(x[, seq_len(min(nc, 6)), drop=FALSE], n))
      }
    } else if (is.data.frame(x)) {
      dcat(sprintf("%s nrow=%d ncol=%d", tag, nrow(x), ncol(x)))
      print(utils::head(x, n))
      dcat(sprintf("%s colclasses: %s", tag, paste(vapply(x, function(z) class(z)[1], ""), collapse=", ")))
      dcat(sprintf("%s NA col counts: %s",
                   tag,
                   paste(sprintf("%s:%d", names(x), vapply(x, function(z) sum(is.na(z)), 1L)), collapse=" | ")))
    } else if (is.vector(x)) {
      dcat(sprintf("%s length=%d class=%s", tag, length(x), cls))
      print(utils::head(x, n))
    } else {
      dcat(sprintf("%s (unhandled preview type)", tag))
    }
  }
  
  .na_count_df <- function(df) vapply(df, function(z) sum(is.na(z)), 1L)
  
  .dbg_delta_na <- function(before, after, tag="(delta)") {
    nm <- union(names(before), names(after))
    b <- before[nm]; a <- after[nm]
    b[is.na(b)] <- 0L; a[is.na(a)] <- 0L
    del <- a - b
    if (any(del != 0L, na.rm=TRUE)) {
      bad <- nm[del != 0L]
      dcat(sprintf("[NA-DELTA %s] changes: %s",
                   tag,
                   paste(sprintf("%s:%+d", bad, del[bad]), collapse=" | ")), .force=TRUE)
    } else {
      dcat(sprintf("[NA-DELTA %s] no NA changes detected", tag))
    }
  }
  
  .as_num_mat <- function(X, nm="X"){
    dcat(sprintf("[COERCE] .as_num_mat(%s) ENTER", nm))
    dbg_head(X, tag=sprintf("[COERCE] raw %s", nm))
    if (is.matrix(X) && is.numeric(X)) {
      storage.mode(X) <- "double"
      dbg_head(X, tag=sprintf("[COERCE] %s OK numeric matrix (no change)", nm))
      return(X)
    }
    if (is.vector(X) && !is.list(X)) {
      X <- matrix(X, ncol=1L); colnames(X) <- nm
      dcat(sprintf("[COERCE] %s vector->matrix (n=%d)", nm, length(X)))
    }
    Xdf <- as.data.frame(X, stringsAsFactors=FALSE)
    na_before <- .na_count_df(Xdf)
    for (cc in names(Xdf)) {
      v <- Xdf[[cc]]
      if (is.list(v)) {
        v <- vapply(
          v,
          function(z){
            if (is.null(z)) return(NA_real_)
            if (is.list(z)) z <- unlist(z, use.names=FALSE)
            suppressWarnings(as.numeric(if (length(z)) z[1] else NA))
          },
          numeric(1)
        )
      } else if (is.factor(v)) {
        v <- as.character(v)
        suppressWarnings(v <- as.numeric(v))
      } else if (is.logical(v)) {
        v <- as.integer(v)
      } else if (!is.numeric(v)) {
        suppressWarnings(v <- as.numeric(v))
      }
      Xdf[[cc]] <- v
    }
    .dbg_delta_na(na_before, .na_count_df(Xdf), tag=sprintf("%s-num-coerce", nm))
    X <- as.matrix(Xdf)
    storage.mode(X) <- "double"
    dbg_head(X, tag=sprintf("[COERCE] coerced %s", nm))
    X
  }
  
  .as_pred_mat <- function(obj, mode=c("binary","multiclass","regression")){
    mode <- match.arg(mode)
    dcat(sprintf("[PRED-MAT] ENTER mode=%s", mode))
    if (is.list(obj) && !is.null(obj$predicted_output)) {
      P <- obj$predicted_output
      dcat("[PRED-MAT] using obj$predicted_output")
    } else {
      P <- obj
      dcat("[PRED-MAT] using obj directly")
    }
    if (is.data.frame(P)) {
      dcat("[PRED-MAT] P is data.frame -> matrix(double)")
      na_before <- .na_count_df(P)
      P <- as.matrix(P)
      .dbg_delta_na(na_before, .na_count_df(as.data.frame(P)), tag="pred-df->mat")
    }
    if (is.vector(P)) {
      dcat("[PRED-MAT] P is vector -> matrix(ncol=1)")
      P <- matrix(as.numeric(P), ncol=1L)
    }
    if (!is.matrix(P)) stop("[as_pred] unsupported prediction object")
    storage.mode(P) <- "double"
    if (mode=="regression" && ncol(P)>1L) {
      dcat("[PRED-MAT] regression & ncol>1 -> keep first col only")
      P <- P[,1,drop=FALSE]
    }
    dbg_head(P, tag="[PRED-MAT] OUT P")
    P
  }
  
  ## =========================
  ## outdir + config
  ## =========================
  # ===== Output directory handling ===== 
  artifacts_dir <- ddesonn_artifacts_root(OUTPUT_DIR) 
  bm_dir <- ddesonn_artifacts_root(get0(".BM_DIR", inherits = TRUE, ifnotfound = artifacts_dir)) 
  out_norm <- tryCatch(normalizePath(bm_dir, winslash="/", mustWork=FALSE), error=function(e) bm_dir) 
  dcat("OUTPUT_DIR=", out_norm) 
  if (!is.null(OUT_DIR_ASSERT)) { 
    assert_norm <- tryCatch(normalizePath(OUT_DIR_ASSERT, winslash="/", mustWork=FALSE), error=function(e) OUT_DIR_ASSERT) 
    if (!identical(out_norm, assert_norm)) stop(sprintf("OUT_DIR_ASSERT mismatch:\n  OUTPUT_DIR=%s\n  ASSERT=%s", out_norm, assert_norm)) 
  } 
  
  CLASSIFICATION_MODE <- tolower(CLASSIFICATION_MODE)
  if (!CLASSIFICATION_MODE %in% c("binary","multiclass","regression")) stop("bad CLASSIFICATION_MODE")
  CLASS_THRESHOLD <- as.numeric(get0("CLASS_THRESHOLD", inherits=TRUE, ifnotfound=0.5))
  SONN         <- get0("SONN", inherits=TRUE, ifnotfound=NULL)
  dcat("CFG split=", INPUT_SPLIT, " mode=", CLASSIFICATION_MODE, " run=", RUN_INDEX, " seed=", SEED, " slot=", MODEL_SLOT)
  
  ## =========================
  ## load meta
  ## =========================
  .resolve_meta <- function(ENV_META_NAME, MODEL_SLOT, SEED, LOAD_FROM_RDS){
    esc <- function(s) gsub("([.()\\[\\]^$+*?{}|\\\\])","\\\\\\1", s)
    cand <- unique(c(
      ENV_META_NAME,
      sub("(?i)(_model_)\\d+(_metadata)", paste0("\\1", as.integer(MODEL_SLOT), "\\2"), ENV_META_NAME, perl=TRUE),
      paste0(ENV_META_NAME, "_seed", as.character(SEED)),
      sub("(?i)(_seed)\\d+", paste0("\\1", as.character(SEED)), ENV_META_NAME, perl=TRUE),
      sub("(?i)(seed)\\d+",  paste0("\\1", as.character(SEED)), ENV_META_NAME, perl=TRUE)
    ))
    if (!LOAD_FROM_RDS)
      for (nm in cand)
        if (exists(nm, inherits=TRUE)) { m <- get(nm, inherits=TRUE); attr(m,"artifact_path") <- paste0("ENV:", nm); return(m) }
    
    adir_candidates <- ddesonn_legacy_artifacts_candidates(get0(".BM_DIR", inherits=TRUE, ifnotfound=bm_dir)) 
    adir_candidates <- adir_candidates[dir.exists(adir_candidates)] 
    if (!length(adir_candidates)) stop("no RDS artifacts available") 
    files <- unlist(lapply(adir_candidates, function(adir) list.files(adir, pattern="\\.[Rr][Dd][Ss]$", full.names=TRUE, recursive=TRUE)), use.names = FALSE) 
    if (!length(files)) stop("no RDS artifacts in any candidate directory") 
    base_hit <- grepl(sprintf("(?i)%s", esc(ENV_META_NAME)), basename(files), perl=TRUE)
    slot_pat <- sprintf("(?i)_model_%d_", as.integer(MODEL_SLOT))
    seed_pat <- sprintf("(?i)_seed%s(\\.|_|$)", as.character(SEED))
    hit <- base_hit & grepl(slot_pat, basename(files), perl=TRUE) & grepl(seed_pat, basename(files), perl=TRUE)
    if (!any(hit)) hit <- base_hit & grepl(slot_pat, basename(files), perl=TRUE)
    if (!any(hit)) hit <- base_hit
    cand <- files[hit]; info <- file.info(cand); file <- cand[order(info$mtime, decreasing=TRUE)][1L]
    m <- readRDS(file); attr(m,"artifact_path") <- file; m
  }
  meta <- .resolve_meta(ENV_META_NAME, MODEL_SLOT, SEED, LOAD_FROM_RDS)
  dcat("LOAD meta via: ", attr(meta, "artifact_path"))
  
  ## =========================
  ## choose split
  ## =========================
  sl <- tolower(INPUT_SPLIT)
  if (sl == "test")            { Xi_raw <- meta$X_test;       yi_raw <- meta$y_test;       split_used <- "test" }
  else if (sl == "validation") { Xi_raw <- meta$X_validation; yi_raw <- meta$y_validation; split_used <- "validation" }
  else if (sl == "train")      { Xi_raw <- meta$X %||% meta$X_train; yi_raw <- meta$y %||% meta$y_train; split_used <- "train" }
  else {
    Xi_raw <- meta$X_validation; yi_raw <- meta$y_validation; split_used <- "validation"
    if (is.null(Xi_raw) || is.null(yi_raw)) { Xi_raw <- meta$X_test; yi_raw <- meta$y_test; split_used <- "test" }
    if (is.null(Xi_raw) || is.null(yi_raw)) { Xi_raw <- meta$X %||% meta$X_train; yi_raw <- meta$y %||% meta$y_train; split_used <- "train" }
  }
  if (is.null(Xi_raw) || is.null(yi_raw)) stop("Requested split not present in metadata: ", INPUT_SPLIT)
  
  dbg_head(Xi_raw, tag="[SPLIT] Xi_raw")
  dbg_head(yi_raw, tag="[SPLIT] yi_raw")
  
  ## =========================
  ## coerce + align
  ## =========================
  Xi <- .as_num_mat(Xi_raw, "X")
  
  if (CLASSIFICATION_MODE == "multiclass") {
    if (is.matrix(yi_raw) && ncol(yi_raw) > 1L) {
      dcat("[LABEL] multiclass matrix->argmax")
      yi <- as.integer(max.col(yi_raw, ties.method = "first"))
    } else if (is.factor(yi_raw)) {
      dcat("[LABEL] factor->integer")
      yi <- as.integer(yi_raw)
    } else if (is.character(yi_raw)) {
      dcat("[LABEL] character->factor->integer")
      yi <- as.integer(factor(yi_raw))
    } else {
      dcat("[LABEL] generic->integer (with 1-based safety)")
      yi <- suppressWarnings(as.integer(yi_raw))
      if (min(yi, na.rm = TRUE) == 0L) yi <- yi + 1L
    }
  } else if (CLASSIFICATION_MODE == "binary") {
    if (is.factor(yi_raw))         { dcat("[LABEL] binary factor")    ; yi <- as.integer(yi_raw == levels(yi_raw)[length(levels(yi_raw))]) }
    else if (is.character(yi_raw)) { dcat("[LABEL] binary character") ; yi <- as.integer(factor(yi_raw) == levels(factor(yi_raw))[length(levels(factor(yi_raw)))]) }
    else if (is.logical(yi_raw))   { dcat("[LABEL] binary logical")   ; yi <- as.integer(yi_raw) }
    else                           { dcat("[LABEL] binary numeric >min"); yi <- as.integer(yi_raw > min(yi_raw, na.rm=TRUE)) }
  } else {
    dcat("[LABEL] regression numeric")
    yi <- suppressWarnings(as.numeric(yi_raw))
  }
  
  dbg_head(yi, tag="[LABEL] yi (aligned vector pre-trim)")
  
  nX <- nrow(Xi); ny <- length(yi); nmin <- min(nX, ny)
  dcat(sprintf("[ALIGN] nX=%d ny=%d nmin=%d", nX, ny, nmin))
  if (nmin <= 0L) stop(sprintf("[ALIGN] NROW(X)=%d vs len(y)=%d", nX, ny))
  if (nX != nmin) Xi <- Xi[seq_len(nmin), , drop=FALSE]
  if (ny != nmin) yi <- yi[seq_len(nmin)]
  
  dbg_head(Xi, tag="[ALIGN] Xi")
  dbg_head(yi, tag="[ALIGN] yi")
  
  ## =========================
  ## predict
  ## =========================
  t0 <- proc.time()
  dcat("[PRED] calling .safe_run_predict ...")
  out <- .safe_run_predict(
    X = Xi, meta = meta, model_index = as.integer(MODEL_SLOT), ML_NN = ML_NN,
    verbose = isTRUE(get0("VERBOSE_RUNPRED", inherits=TRUE, ifnotfound=FALSE)),
    debug   = isTRUE(get0("DEBUG_RUNPRED",   inherits=TRUE, ifnotfound=FALSE))
  )
  prediction_time <- as.numeric((proc.time() - t0)[["elapsed"]])
  dcat(sprintf("[PRED] .safe_run_predict done in %.3fs", prediction_time))
  
  pred_mode <- switch(CLASSIFICATION_MODE, regression="regression", multiclass="multiclass", binary="binary")
  P_raw <- .as_pred_mat(out, mode = pred_mode)
  if (!is.matrix(P_raw)) P_raw <- as.matrix(P_raw)
  storage.mode(P_raw) <- "double"
  dbg_head(P_raw, tag="[PRED] P_raw")
  
  ## =========================
  ## scores -> probabilities
  ## =========================
  P <- P_raw
  if (CLASSIFICATION_MODE == "multiclass") {
    dcat("[PROB] multiclass softmax (row-wise)")
    if (ncol(P) > 1L) {
      mx <- apply(P, 1L, max)
      ex <- exp(P - mx)
      rs <- rowSums(ex); rs[!is.finite(rs) | rs <= 0] <- 1
      P <- ex / rs
    }
    P[!is.finite(P)] <- 0
    P <- pmin(pmax(P, .Machine$double.eps), 1 - .Machine$double.eps)
  } else if (CLASSIFICATION_MODE == "binary") {
    if (ncol(P) == 1L) {
      dcat("[PROB] binary single-column -> clamp in (0,1)")
      P[!is.finite(P)] <- 0
      P[,1] <- pmin(pmax(P[,1], .Machine$double.eps), 1 - .Machine$double.eps)
    } else {
      dcat("[PROB] binary with 2+ cols -> softmax then take class-2 prob")
      mx <- apply(P, 1L, max)
      ex <- exp(P - mx)
      sm <- rowSums(ex); sm[!is.finite(sm) | sm <= 0] <- 1
      S <- ex / sm
      P <- matrix(S[, min(2L, ncol(S))], ncol=1L)
      P[!is.finite(P)] <- 0
      P[,1] <- pmin(pmax(P[,1], .Machine$double.eps), 1 - .Machine$double.eps)
    }
  } else {
    dcat("[PROB] regression passthrough (1st col if multi)")
    if (ncol(P) > 1L) P <- P[,1,drop=FALSE]
    P[!is.finite(P)] <- NA_real_
  }
  dbg_head(P, tag="[PROB] P (final)")
  base_n <- nrow(P)
  
  ## =========================
  ## label/P sanitize for metrics
  ## =========================
  if (CLASSIFICATION_MODE == "multiclass") {
    K <- ncol(P)
    yi_vec <- suppressWarnings(as.integer(yi))
    yi_vec[yi_vec < 1L | yi_vec > K] <- NA_integer_
    
    sumP <- rowSums(P)
    finite_rows <- rowSums(!is.finite(P)) == 0L
    prob_rows <- is.finite(sumP) & (sumP > 0.999 - 1e-6) & (sumP < 1.001 + 1e-6)
    keep <- finite_rows & prob_rows & is.finite(yi_vec)
    dcat(sprintf("[SAN] MC keep=%d drop=%d of %d", sum(keep), sum(!keep), length(keep)))
    if (!any(keep)) stop("[metrics_align] no valid rows after MC sanitize")
    P <- P[keep,,drop=FALSE]; yi_vec <- yi_vec[keep]
  } else {
    yi_vec <- yi
    keep <- is.finite(yi_vec) & apply(P, 1L, function(r) all(is.finite(r)))
    dcat(sprintf("[SAN] non-MC keep=%d drop=%d of %d", sum(keep), sum(!keep), length(keep)))
    if (!all(keep)) { P <- P[keep,,drop=FALSE]; yi_vec <- yi_vec[keep] }
  }
  base_n <- nrow(P)
  dbg_head(yi_vec, tag="[SAN] yi_vec")
  dbg_head(P, tag="[SAN] P (for metrics)")
  
  ## =========================
  ## base metrics
  ## =========================
  acc <- prec <- rec <- f1s <- NA_real_
  mc_macro_precision <- mc_macro_recall <- mc_macro_f1 <- NA_real_
  mc_micro_precision <- mc_micro_recall <- mc_micro_f1 <- NA_real_
  mc_weighted_precision <- mc_weighted_recall <- mc_weighted_f1 <- NA_real_
  
  if (CLASSIFICATION_MODE == "binary") {
    y_true <- if (all(yi_vec %in% c(0,1), na.rm=TRUE)) as.integer(yi_vec) else as.integer(yi_vec >= 0.5)
    p_pos  <- as.numeric(P[,1]); ok <- is.finite(y_true) & is.finite(p_pos)
    y_true <- y_true[ok]; p_pos <- p_pos[ok]
    yhat <- as.integer(p_pos >= CLASS_THRESHOLD)
    TP <- sum(yhat==1 & y_true==1); FP <- sum(yhat==1 & y_true==0)
    TN <- sum(yhat==0 & y_true==0); FN <- sum(yhat==0 & y_true==1)
    N  <- length(y_true)
    acc <- if (N) (TP + TN) / N else NA_real_
    prec <- if ((TP+FP)>0) TP/(TP+FP) else 0
    rec  <- if ((TP+FN)>0) TP/(TP+FN) else 0
    f1s  <- if ((prec+rec)>0) 2*prec*rec/(prec+rec) else 0
    
  } else if (CLASSIFICATION_MODE == "multiclass") {
    yhat <- max.col(P, ties.method="first")
    ymc  <- as.integer(yi_vec)
    ok   <- is.finite(yhat) & is.finite(ymc); yhat <- yhat[ok]; ymc <- ymc[ok]
    acc  <- mean(yhat == ymc)
    
    K <- max(yhat, ymc, na.rm=TRUE)
    TPk <- FPk <- FNk <- support <- numeric(K)
    pk <- rk <- fk <- numeric(K)
    for (k in seq_len(K)) {
      TPk[k] <- sum(yhat==k & ymc==k)
      FPk[k] <- sum(yhat==k & ymc!=k)
      FNk[k] <- sum(yhat!=k & ymc==k)
      support[k] <- sum(ymc==k)
      pk[k] <- if ((TPk[k]+FPk[k])>0) TPk[k]/(TPk[k]+FPk[k]) else NA_real_
      rk[k] <- if ((TPk[k]+FNk[k])>0) TPk[k]/(TPk[k]+FNk[k]) else NA_real_
      fk[k] <- if (is.finite(pk[k]) && is.finite(rk[k]) && (pk[k]+rk[k])>0) 2*pk[k]*rk[k]/(pk[k]+rk[k]) else NA_real_
    }
    mc_macro_precision <- mean(pk, na.rm=TRUE)
    mc_macro_recall    <- mean(rk, na.rm=TRUE)
    mc_macro_f1        <- mean(fk, na.rm=TRUE)
    
    TPm <- sum(TPk, na.rm=TRUE); FPm <- sum(FPk, na.rm=TRUE); FNm <- sum(FNk, na.rm=TRUE)
    mc_micro_precision <- if ((TPm+FPm)>0) TPm/(TPm+FPm) else NA_real_
    mc_micro_recall    <- if ((TPm+FNm)>0) TPm/(TPm+FNm) else NA_real_
    mc_micro_f1        <- if (is.finite(mc_micro_precision) && is.finite(mc_micro_recall) && (mc_micro_precision+mc_micro_recall)>0)
      2*mc_micro_precision*mc_micro_recall/(mc_micro_precision+mc_micro_recall) else NA_real_
    
    tot_sup <- sum(support, na.rm=TRUE)
    w <- if (tot_sup > 0) support / tot_sup else rep(0, length(support))
    mc_weighted_precision <- sum(w * pk, na.rm=TRUE)
    mc_weighted_recall    <- sum(w * rk, na.rm=TRUE)
    mc_weighted_f1        <- sum(w * fk, na.rm=TRUE)
    
    prec <- mc_macro_precision; rec <- mc_macro_recall; f1s <- mc_macro_f1
  }
  
  dcat(sprintf("METR raw: acc=%.6f | prec=%.6f | rec=%.6f | f1=%.6f | base_n=%d",
               r6(acc), r6(prec), r6(rec), r6(f1s), base_n))
  
  ## =========================
  ## tuned (safe try)
  ## =========================
  tuned <- tryCatch(
    accuracy_precision_recall_f1_tuned(
      SONN=SONN, Rdata=Xi, labels=yi_vec, CLASSIFICATION_MODE=CLASSIFICATION_MODE, predicted_output=P,
      metric_for_tuning=get0("METRIC_FOR_TUNING", inherits=TRUE, ifnotfound="accuracy"),
      threshold_grid=get0("THRESHOLD_GRID", inherits=TRUE, ifnotfound=seq(0.05,0.95,by=0.01)),
      verbose=isTRUE(get0("TUNED_VERBOSE", inherits=TRUE, ifnotfound=FALSE))
    ),
    error=function(e) { dcat(paste0("[TUNED] ERROR: ", conditionMessage(e))); list(accuracy=NA_real_, precision=NA_real_, recall=NA_real_, f1=NA_real_, details=list(best_threshold=NA_real_, tuned_by="error")) }
  )
  chosen_threshold <- if (CLASSIFICATION_MODE=="multiclass") NA_real_ else {
    bt <- tryCatch(tuned$details$best_threshold, error=function(e) tuned$best_threshold)
    if (!is.numeric(bt) || !is.finite(bt)) 0.5 else as.numeric(bt)
  }
  dcat(sprintf("[TUNED] thr=%s acc=%.6f prec=%.6f rec=%.6f f1=%.6f",
               ifelse(is.na(chosen_threshold),"NA",format(chosen_threshold)), r6(tuned$accuracy), r6(tuned$precision), r6(tuned$recall), r6(tuned$f1)))
  
  ## =========================
  ## compact results row
  ## =========================
  row_df <- data.frame(
    run_index=as.integer(RUN_INDEX), seed=as.integer(SEED),
    model_slot=as.integer(MODEL_SLOT), slot=as.integer(MODEL_SLOT),
    split=tolower(split_used), SPLIT=toupper(split_used), .__split__=tolower(split_used),
    CLASSIFICATION_MODE=toupper(CLASSIFICATION_MODE),
    RUN_INDEX=as.integer(RUN_INDEX), SEED=as.integer(SEED), MODEL_SLOT=as.integer(MODEL_SLOT),
    accuracy=r6(acc), precision=r6(prec), recall=r6(rec), f1=r6(f1s),
    precision_macro  = r6(mc_macro_precision),
    recall_macro     = r6(mc_macro_recall),
    f1_macro         = r6(mc_macro_f1),
    precision_micro  = r6(mc_micro_precision),
    recall_micro     = r6(mc_micro_recall),
    f1_micro         = r6(mc_micro_f1),
    precision_weighted = r6(mc_weighted_precision),
    recall_weighted    = r6(mc_weighted_recall),
    f1_weighted        = r6(mc_weighted_f1),
    stringsAsFactors = FALSE, check.names = TRUE
  )
  dbg_head(row_df, tag="[WRITE] row_df (metrics row)")
  
  ## =========================
  ## write metrics (upsert)
  ## =========================
  ts_now <- format(Sys.time(), "%Y%m%d_%H%M%S")
  if (isTRUE(SAVE_METRICS_RDS) && is.null(AGG_METRICS_FILE)) {
    per_seed_metrics_path <- file.path(out_norm, sprintf("%s_run%03d_seed%s_%s_slot%d.rds",
                                                         METRICS_PREFIX, as.integer(RUN_INDEX),
                                                         as.character(SEED), ts_now, as.integer(MODEL_SLOT)))
    saveRDS(row_df, per_seed_metrics_path)
    dcat("WRITE per-seed metrics: ", per_seed_metrics_path)
  }
  if (!is.null(AGG_METRICS_FILE) && nzchar(AGG_METRICS_FILE)) {
    dcat("[UPsert-METR] target=", AGG_METRICS_FILE)
    if (file.exists(AGG_METRICS_FILE)) {
      old <- readRDS(AGG_METRICS_FILE); if (!is.data.frame(old)) old <- as.data.frame(old, stringsAsFactors=FALSE)
      dbg_head(old, tag="[UPsert-METR] old (pre)")
      add_m <- setdiff(names(row_df), names(old)); for (nm in add_m) old[[nm]] <- NA
      add_o <- setdiff(names(old), names(row_df)); for (nm in add_o) row_df[[nm]] <- NA
      keep <- !(old$run_index == as.integer(RUN_INDEX) &
                  old$seed      == as.integer(SEED) &
                  (if ("slot" %in% names(old)) old$slot else old$model_slot) == as.integer(MODEL_SLOT) &
                  tolower(old$split) == tolower(split_used))
      old2 <- old[keep, , drop = FALSE]
      agg_tbl <- if (nrow(old2)) rbind(old2, row_df[, names(old2), drop=FALSE]) else row_df[, names(old), drop=FALSE]
    } else {
      agg_tbl <- row_df
    }
    dbg_head(agg_tbl, tag="[UPsert-METR] agg_tbl (post)")
    saveRDS(agg_tbl, AGG_METRICS_FILE)
    dcat("WRITE agg metrics upsert OK -> ", AGG_METRICS_FILE)
  }
  
  ## =========================
  ## write predictions (upsert)
  ## =========================
  if (!is.null(AGG_PREDICTIONS_FILE) && nzchar(AGG_PREDICTIONS_FILE)) {
    dcat("[UPsert-PRED] target=", AGG_PREDICTIONS_FILE)
    if (CLASSIFICATION_MODE == "binary") {
      y_prob_vec <- as.numeric(P[,1]); y_pred_vec <- as.integer(y_prob_vec >= CLASS_THRESHOLD)
    } else if (CLASSIFICATION_MODE == "multiclass") {
      y_prob_vec <- apply(P, 1, max)
      y_pred_vec <- max.col(P, ties.method="first")
    } else {
      y_prob_vec <- as.numeric(P[,1]); y_pred_vec <- y_prob_vec
    }
    
    SAVE_PREDICTIONS_COLUMN_IN_RDS <- isTRUE(get0("SAVE_PREDICTIONS_COLUMN_IN_RDS", inherits=TRUE, ifnotfound=FALSE))
    pred_list_col <- if (isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) lapply(seq_len(base_n), function(i) as.numeric(P[i, , drop = TRUE])) else NULL
    
    obs_index <- seq_len(base_n)
    
    pred_df <- data.frame(
      run_index = rep.int(as.integer(RUN_INDEX), base_n),
      seed      = rep.int(as.integer(SEED),      base_n),
      model_slot= rep.int(as.integer(MODEL_SLOT),base_n),
      slot      = rep.int(as.integer(MODEL_SLOT),base_n),
      y_true    = as.numeric(yi_vec),
      y_prob    = y_prob_vec,
      y_pred    = y_pred_vec,
      split     = rep.int(tolower(split_used),   base_n),
      SPLIT     = rep.int(toupper(split_used),   base_n),
      .__split__= rep.int(tolower(split_used),   base_n),
      CLASSIFICATION_MODE = rep.int(toupper(CLASSIFICATION_MODE), base_n),
      RUN_INDEX = rep.int(as.integer(RUN_INDEX), base_n),
      SEED      = rep.int(as.integer(SEED),      base_n),
      MODEL_SLOT= rep.int(as.integer(MODEL_SLOT),base_n),
      obs_index = obs_index,
      stringsAsFactors = FALSE, check.names = TRUE
    )
    if (isTRUE(SAVE_PREDICTIONS_COLUMN_IN_RDS)) pred_df$y_prob_full <- pred_list_col
    
    dbg_head(pred_df, tag="[UPsert-PRED] pred_df (new)")
    
    if (file.exists(AGG_PREDICTIONS_FILE)) {
      old <- readRDS(AGG_PREDICTIONS_FILE); if (!is.data.frame(old)) old <- as.data.frame(old, stringsAsFactors=FALSE)
      dbg_head(old, tag="[UPsert-PRED] old (pre)")
      
      ## align columns
      add_m <- setdiff(names(pred_df), names(old)); for (nm in add_m) old[[nm]] <- NA
      add_o <- setdiff(names(old), names(pred_df)); for (nm in add_o) pred_df[[nm]] <- NA
      
      ## 1) drop all rows in old that match this (run, seed, slot, split)
      slot_col  <- if ("slot" %in% names(old)) "slot" else if ("model_slot" %in% names(old)) "model_slot" else "slot"
      split_col <- if ("split" %in% names(old)) "split" else "SPLIT"
      
      old_split_norm <- tolower(as.character(old[[split_col]]))
      keep_old <- !(as.integer(old$run_index) == as.integer(RUN_INDEX) &
                      as.integer(old$seed)      == as.integer(SEED) &
                      as.integer(old[[slot_col]]) == as.integer(MODEL_SLOT) &
                      old_split_norm == tolower(split_used))
      old2 <- old[keep_old, , drop = FALSE]
      
      ## 2) bind
      agg_pred <- if (nrow(old2)) rbind(old2, pred_df[, names(old2), drop=FALSE]) else pred_df[, names(old), drop=FALSE]
      
      ## 3) de-dup on composite key
      slot_col  <- if ("slot" %in% names(agg_pred)) "slot" else if ("model_slot" %in% names(agg_pred)) "model_slot" else "slot"
      split_col <- if ("split" %in% names(agg_pred)) "split" else "SPLIT"
      comp <- paste(
        as.integer(agg_pred$run_index),
        as.integer(agg_pred$seed),
        as.integer(agg_pred[[slot_col]]),
        tolower(as.character(agg_pred[[split_col]])),
        as.integer(agg_pred$obs_index),
        sep="|"
      )
      if (any(duplicated(comp))) {
        dcat(sprintf("[UPsert-PRED] de-dup removed %d rows", sum(duplicated(comp))))
        agg_pred <- agg_pred[!duplicated(comp), , drop=FALSE]
      }
    } else {
      agg_pred <- pred_df
    }
    
    dbg_head(agg_pred, tag="[UPsert-PRED] agg_pred (post)")
    saveRDS(agg_pred, AGG_PREDICTIONS_FILE)
    dcat("WRITE agg preds upsert OK: +", base_n, " rows (this slot)")
    
    ## STRICT WRITE-CHECK
    slot_col  <- if ("slot" %in% names(agg_pred)) "slot" else if ("model_slot" %in% names(agg_pred)) "model_slot" else "slot"
    split_col <- if ("split" %in% names(agg_pred)) "split" else "SPLIT"
    ap <- agg_pred
    ap$run_index <- suppressWarnings(as.integer(ap$run_index))
    ap$seed      <- suppressWarnings(as.integer(ap$seed))
    ap[[slot_col]] <- suppressWarnings(as.integer(ap[[slot_col]]))
    ap[[split_col]] <- tolower(as.character(ap[[split_col]]))
    
    hit <- sum(
      ap$run_index == as.integer(RUN_INDEX) &
        ap$seed      == as.integer(SEED) &
        ap[[slot_col]] == as.integer(MODEL_SLOT) &
        ap[[split_col]] == tolower(split_used)
    )
    dcat(sprintf("[WRITE-CHECK] expect base_n=%d found=%d for key(run,seed,slot,split)=(%s,%s,%s,%s)",
                 base_n, hit, as.character(RUN_INDEX), as.character(SEED), as.character(MODEL_SLOT), split_used))
    if (hit != base_n) {
      stop(sprintf(
        "[WRITE-CHECK] agg preds missing for (run=%s seed=%s slot=%s split=%s): found %d of %d rows",
        as.character(RUN_INDEX), as.character(SEED), as.character(MODEL_SLOT), split_used,
        as.integer(hit), as.integer(base_n)
      ))
    }
  }
  
  ## =========================
  ## final
  ## =========================
  dcat("[OK] seed=", SEED, " slot=", MODEL_SLOT,
       " acc=", r6(acc), " prec=", r6(prec), " rec=", r6(rec), " f1=", r6(f1s))
  
  ret <- list(
    results_compact = data.frame(
      kind = if (LOAD_FROM_RDS) "RDS" else "ENV",
      model = as.integer(MODEL_SLOT),
      split_used = split_used,
      n_pred_rows = as.integer(base_n),
      accuracy = r6(acc), precision = r6(prec), recall = r6(rec), f1 = r6(f1s),
      tuned_threshold = r6(chosen_threshold),
      tuned_accuracy  = r6(tuned$accuracy),
      tuned_precision = r6(tuned$precision),
      tuned_recall    = r6(tuned$recall),
      tuned_f1        = r6(tuned$f1),
      mc_precision_macro   = r6(mc_macro_precision),
      mc_recall_macro      = r6(mc_macro_recall),
      mc_f1_macro          = r6(mc_macro_f1),
      mc_precision_micro   = r6(mc_micro_precision),
      mc_recall_micro      = r6(mc_micro_recall),
      mc_f1_micro          = r6(mc_micro_f1),
      mc_precision_weighted= r6(mc_weighted_precision),
      mc_recall_weighted   = r6(mc_weighted_recall),
      mc_f1_weighted       = r6(mc_weighted_f1),
      stringsAsFactors = FALSE, check.names = TRUE
    ),
    probs = P, split_used = split_used,
    out_dir = out_norm, n_rows = base_n
  )
  dbg_head(ret$results_compact, tag="[RET] results_compact")
  invisible(ret)
}

















# =====================================================================
# FUSED ENSEMBLE DECISION PATH (final consensus predictions)
# Fuse per-slot predictions loaded from an aggregate-per-model RDS
# (filtered by RUN/SEED and optional split) into ONE ensemble output.
#
# Audience (plain English):
# - "Fused" = we COMBINE multiple model outputs into ONE prediction
#   stream that is treated as the final ensemble decision.
#
# What this function does:
# - Reads per-slot predictions (by RUN/SEED/etc.) and builds a matrix.
# - Produces ONE consensus prediction via:
#     * "avg"        -> unweighted mean of probabilities
#     * "wavg"       -> weighted mean (weights from tuned_f1 / f1 / accuracy)
#     * "vote_soft"  -> fraction of models voting positive (soft proportion)
#     * "vote_hard"  -> majority (or quorum) hard vote (0/1 outcome)
# - Computes metrics on these fused outputs (currently binary path shown).
#   (Extend here for multiclass/regression if needed.)
#
# Thresholds & quorum:
# - Voting uses per-slot tuned_threshold when available; otherwise falls
#   back to default_threshold. Quorum defaults to simple majority unless
#   explicitly provided.
#
# How this differs from aggregate_predictions():
# - aggregate_predictions() (mean/median/vote) is a lightweight utility
#   mainly used for reporting flows (e.g., grouped summaries).
# - DDESONN_fuse_from_agg() is THE place that creates the final, fused
#   ensemble outputs for evaluation or downstream consumption, including
#   weighted averaging and explicit soft/hard voting variants.
#
# Notes:
# - If you need median fusion in the final path, add it here explicitly.
# - Ensure row alignment across slots (IDs/order) before forming the matrix.
# =====================================================================


DDESONN_fuse_from_agg <- function(
    AGG_PREDICTIONS_FILE,
    RUN_INDEX,
    SEED,
    y_true,                             # numeric 0/1 (binary). For multiclass, pass ids 1..K
    methods = c("avg","wavg","vote_soft","vote_hard"),
    weight_column = c("tuned_f1","f1","accuracy"),  # for wavg if present
    use_tuned_threshold_for_vote = TRUE,
    default_threshold = 0.5,
    vote_quorum = NULL,                 # NULL = majority
    classification_mode = c("binary","multiclass","regression")
) {
  ## ===================== DEBUG HELPERS =====================
  DEBUG <- TRUE
  dts <- function() format(Sys.time(), "%H:%M:%S")
  dcat <- function(...) if (isTRUE(DEBUG)) cat(sprintf("[FUSE-DBG %s] ", dts()), paste0(..., collapse=""), "\n")
  dhead <- function(x, n=5) { if (isTRUE(DEBUG)) { dcat("head():"); print(utils::head(x, n)); } }
  ## ========================================================
  
  `%||%` <- function(x,y) if (is.null(x)) y else x
  r6 <- function(x) ifelse(is.finite(x), round(x,6), NA_real_)
  weight_column <- match.arg(weight_column)
  methods <- unique(match.arg(methods, several.ok = TRUE))
  classification_mode <- match.arg(classification_mode)
  
  stopifnot(file.exists(AGG_PREDICTIONS_FILE))
  df <- readRDS(AGG_PREDICTIONS_FILE)
  if (!is.data.frame(df) || !nrow(df)) stop("Aggregate predictions file is empty or not a data.frame.")
  dcat("Loaded agg file:", AGG_PREDICTIONS_FILE, " rows=", nrow(df))
  
  ## ---- MINIMAL NA-PREVENTION & TYPING ----
  # keep y_prob_full for MC fusion if present
  if ("y_prob" %in% names(df))  df$y_prob  <- suppressWarnings(as.numeric(df$y_prob))
  if ("y_true" %in% names(df))  df$y_true  <- suppressWarnings(as.numeric(df$y_true))
  if ("y_pred" %in% names(df))  df$y_pred  <- suppressWarnings(as.numeric(df$y_pred))
  if ("obs_index" %in% names(df)) df$obs_index <- suppressWarnings(as.integer(df$obs_index))
  if ("run_index" %in% names(df)) df$run_index <- suppressWarnings(as.integer(df$run_index))
  if ("RUN_INDEX" %in% names(df)) df$RUN_INDEX <- suppressWarnings(as.integer(df$RUN_INDEX))
  if ("seed" %in% names(df))      df$seed      <- suppressWarnings(as.integer(df$seed))
  if ("SEED" %in% names(df))      df$SEED      <- suppressWarnings(as.integer(df$SEED))
  if ("model_slot" %in% names(df)) df$model_slot <- suppressWarnings(as.integer(df$model_slot))
  if ("MODEL_SLOT" %in% names(df)) df$MODEL_SLOT <- suppressWarnings(as.integer(df$MODEL_SLOT))
  if ("split" %in% names(df))     df$split    <- tolower(as.character(df$split))
  if ("SPLIT" %in% names(df))     df$SPLIT    <- toupper(as.character(df$SPLIT))
  if ("CLASSIFICATION_MODE" %in% names(df)) df$CLASSIFICATION_MODE <- tolower(as.character(df$CLASSIFICATION_MODE))
  
  if ("y_prob" %in% names(df) && any(!is.finite(df$y_prob))) stop("Non-finite y_prob in aggregate predictions.")
  if ("y_true" %in% names(df) && any(!is.finite(df$y_true))) stop("Non-finite y_true in aggregate predictions.")
  
  has_col <- function(d, nm) nm %in% names(d)
  pick_col <- function(d, a, b) if (has_col(d, a)) a else if (has_col(d, b)) b else NA_character_
  
  run_col  <- pick_col(df, "run_index",  "RUN_INDEX")
  seed_col <- pick_col(df, "seed",       "SEED")
  slot_col <- pick_col(df, "model_slot", "MODEL_SLOT")
  split_col<- pick_col(df, "split",      "SPLIT")
  
  if (is.na(run_col))  stop("Missing run_index/RUN_INDEX.")
  if (is.na(seed_col)) stop("Missing seed/SEED.")
  if (is.na(slot_col)) stop("Missing model_slot/MODEL_SLOT.")
  
  # Build split for filtering
  df$.__split__ <- if (!is.na(split_col)) tolower(as.character(df[[split_col]])) else NA_character_
  
  keep <- (suppressWarnings(as.integer(df[[run_col]]))  == as.integer(RUN_INDEX)) &
    (suppressWarnings(as.integer(df[[seed_col]])) == as.integer(SEED))
  # Only apply split filter if split exists
  if (any(!is.na(df$.__split__))) {
    keep <- keep & (df$.__split__ %in% c("test","valid","validation","val","holdout"))
  }
  dfx <- df[keep, , drop = FALSE]
  dcat("Filtered rows for RUN=", RUN_INDEX, " SEED=", SEED, " -> rows=", nrow(dfx))
  if (!nrow(dfx)) {
    dcat("Unique runs:", paste(sort(unique(df[[run_col]])), collapse=", "))
    dcat(
      "Unique seeds @run ", RUN_INDEX, ": ",
      paste(
        sort(unique(df[df[[run_col]] == as.integer(RUN_INDEX), seed_col, drop = TRUE])),
        collapse = ", "
      )
    )
    stop(sprintf("No per-slot predictions for RUN_INDEX=%s, SEED=%s (id/split mismatch).",
                 as.character(RUN_INDEX), as.character(SEED)))
  }
  dhead(dfx[, c(run_col, seed_col, slot_col, split_col)[c(TRUE,TRUE,TRUE,!is.na(split_col))], drop=FALSE])
  
  # y_true source
  if (missing(y_true) || is.null(y_true) || !length(y_true) || all(is.na(y_true))) {
    if (!has_col(dfx, "y_true")) stop("y_true not provided and y_true column not found.")
    y_true <- suppressWarnings(as.numeric(dfx$y_true))
    dcat("Using y_true from file; length=", length(y_true))
  } else {
    y_true <- suppressWarnings(as.numeric(y_true))
    dcat("Using y_true passed by caller; length=", length(y_true))
  }
  
  # --- RESCUE: if caller's y_true is unusable, fall back to file ---
  y_true_finite_n <- sum(is.finite(y_true))
  if (y_true_finite_n == 0L) {
    dcat("WARNING: caller y_true is all NA/non-finite; falling back to dfx$y_true")
    if (!("y_true" %in% names(dfx))) {
      stop("y_true provided is all NA and dfx has no y_true column to rescue from.")
    }
    y_true <- suppressWarnings(as.integer(dfx$y_true))
    dcat("Rescued y_true from file; length=", length(y_true),
         " | unique labels: ", paste(sort(unique(na.omit(y_true))), collapse=", "))
  } else {
    dcat("Caller y_true finite count=", y_true_finite_n,
         " | unique labels: ", paste(sort(unique(na.omit(y_true))), collapse=", "))
  }
  
  # Mode auto-detect if CLASSIFICATION_MODE present or if unique labels suggest MC
  cm_hint <- if (has_col(dfx, "CLASSIFICATION_MODE")) unique(dfx$CLASSIFICATION_MODE) else NULL
  if (length(cm_hint)) {
    if (cm_hint[1] %in% c("binary","multiclass","regression")) classification_mode <- cm_hint[1]
  } else {
    if (classification_mode != "regression") {
      uy <- sort(unique(na.omit(y_true)))
      if (length(uy) > 2) classification_mode <- "multiclass"
      if (length(uy) == 2 && all(uy %in% c(0,1))) classification_mode <- "binary"
    }
  }
  dcat("classification_mode:", classification_mode)
  
  # Split by slot
  sp <- split(dfx, dfx[[slot_col]])
  slots <- as.integer(names(sp))
  dcat("Detected slots:", paste(slots, collapse=", "))
  
  # ===== Helper metrics =====
  bin_metrics <- function(p, y, thr=0.5) {
    y01 <- as.integer(y > 0)
    yhat <- as.integer(p >= thr)
    TP <- sum(yhat==1 & y01==1); FP <- sum(yhat==1 & y01==0)
    FN <- sum(yhat==0 & y01==1); TN <- sum(yhat==0 & y01==0)
    acc <- mean(yhat==y01)
    prec <- if ((TP+FP)>0) TP/(TP+FP) else 0
    rec  <- if ((TP+FN)>0) TP/(TP+FN) else 0
    f1   <- if ((prec+rec)>0) 2*prec*rec/(prec+rec) else 0
    list(acc=r6(acc), precision=r6(prec), recall=r6(rec), f1=r6(f1), TP=TP, FP=FP, FN=FN, TN=TN)
  }
  
  mc_metrics <- function(y_pred, y_true, labels=NULL) {
    y <- as.integer(y_true)
    yp <- as.integer(y_pred)
    labs <- sort(unique(c(labels, y[is.finite(y)], yp[is.finite(yp)])))
    K <- length(labs)
    if (K < 2) return(list(acc=NA, precision=NA, recall=NA, f1=NA, K=K,
                           conf_mat=NA, per_class=NA))
    
    # Map to 1..K
    y_m  <- match(y,  labs)
    yp_m <- match(yp, labs)
    
    conf <- matrix(0L, nrow=K, ncol=K, dimnames=list(true=labs, pred=labs))
    for (i in seq_along(y_m)) {
      if (is.na(y_m[i]) || is.na(yp_m[i])) next
      conf[y_m[i], yp_m[i]] <- conf[y_m[i], yp_m[i]] + 1L
    }
    acc <- sum(diag(conf)) / sum(conf)
    
    # per-class counts + metrics (includes TN)
    per_class <- lapply(seq_len(K), function(k){
      TP <- conf[k,k]
      FP <- sum(conf[,k]) - TP
      FN <- sum(conf[k,]) - TP
      TN <- sum(conf) - TP - FP - FN
      P  <- if ((TP+FP)>0) TP/(TP+FP) else 0
      R  <- if ((TP+FN)>0) TP/(TP+FN) else 0
      F1 <- if ((P+R)>0) 2*P*R/(P+R) else 0
      data.frame(class=labs[k], TP=TP, FP=FP, FN=FN, TN=TN,
                 precision=r6(P), recall=r6(R), f1=r6(F1),
                 stringsAsFactors = FALSE)
    })
    per_class <- do.call(rbind, per_class)
    
    list(
      acc = r6(acc),
      precision = r6(mean(per_class$precision)),
      recall    = r6(mean(per_class$recall)),
      f1        = r6(mean(per_class$f1)),
      conf_mat  = conf,
      per_class = per_class,
      K = K
    )
  }
  
  # ===== Build per-slot prediction containers =====
  N0 <- length(y_true)
  dcat("Initial y_true length:", N0, " unique labels:", paste(sort(unique(na.omit(y_true))), collapse=", "))
  
  slot_lengths <- vapply(sp, nrow, integer(1))
  dcat("Slot row counts:", paste(paste0(names(slot_lengths), "=", slot_lengths), collapse=", "))
  
  # ----- Binary containers -----
  get_prob_vec <- function(d) {
    if (has_col(d, "y_prob")) return(suppressWarnings(as.numeric(d$y_prob)))
    if (has_col(d, "y_pred")) return(suppressWarnings(as.numeric(d$y_pred))) # allow if already prob
    stop("No y_prob or y_pred column found for binary mode.")
  }
  
  # ----- Multiclass containers -----
  get_prob_mat_mc <- function(d, labels_ref=NULL) {
    N <- nrow(d)
    # 1) list-col y_prob_full
    if (has_col(d, "y_prob_full") && is.list(d$y_prob_full)) {
      nm_pool <- unique(unlist(lapply(utils::head(d$y_prob_full, 20), names)))
      if (is.null(labels_ref) && length(nm_pool)) {
        labs <- as.integer(sort(as.numeric(nm_pool)))
      } else if (!is.null(labels_ref)) {
        labs <- sort(unique(as.integer(labels_ref)))
      } else {
        labs <- sort(unique(as.integer(unlist(d$y_prob_full))))
      }
      K <- length(labs)
      P <- matrix(0, nrow=N, ncol=K, dimnames=list(NULL, paste0(labs)))
      for (i in seq_len(N)) {
        vi <- d$y_prob_full[[i]]
        if (is.null(vi)) next
        if (!is.null(names(vi))) {
          idx <- match(as.integer(names(vi)), labs)
          ok <- which(is.finite(idx))
          P[i, idx[ok]] <- as.numeric(vi[ok])
        } else {
          if (length(vi) == K) P[i,] <- as.numeric(vi)
        }
      }
      return(list(P=P, labels=labs))
    }
    
    # 2) wide columns prob_1..prob_K
    prob_cols <- grep("^prob_\\d+$", names(d), value = TRUE)
    if (length(prob_cols) > 0) {
      labs <- as.integer(sub("^prob_", "", prob_cols))
      ord <- order(labs); labs <- labs[ord]; prob_cols <- prob_cols[ord]
      P <- as.matrix(d[, prob_cols, drop=FALSE])
      storage.mode(P) <- "double"
      colnames(P) <- paste0(labs)
      return(list(P=P, labels=labs))
    }
    
    # 3) fallback to one-hot from y_pred
    if (has_col(d, "y_pred")) {
      yp <- as.integer(d$y_pred)
      labs <- sort(unique(na.omit(yp)))
      if (!is.null(labels_ref)) labs <- sort(unique(c(labels_ref, labs)))
      K <- length(labs)
      P <- matrix(0, nrow=N, ncol=K, dimnames=list(NULL, paste0(labs)))
      idx <- match(yp, labs)
      ok <- which(is.finite(idx))
      P[cbind(ok, idx[ok])] <- 1
      return(list(P=P, labels=labs))
    }
    
    stop("For multiclass, need y_prob_full, prob_* columns, or y_pred.")
  }
  
  out_rows <- list()
  out_preds <- list()
  
  if (classification_mode == "binary") {
    probs_list <- lapply(sp, get_prob_vec)
    lens <- vapply(probs_list, length, integer(1))
    N <- min(c(length(y_true), lens))
    if (!is.finite(N) || N <= 0L) stop("No usable prediction vectors after alignment (N <= 0).")
    
    probs_mat <- do.call(cbind, lapply(probs_list, function(v) as.numeric(v)[seq_len(N)]))
    if (is.null(dim(probs_mat))) probs_mat <- matrix(probs_mat, nrow = N, ncol = length(probs_list))
    colnames(probs_mat) <- paste0("slot_", slots)
    y <- y_true[seq_len(N)]
    dcat("BINARY N=", N, " S=", ncol(probs_mat))
    
    pick_slot_scalar <- function(d, nm) {
      if (!has_col(d, nm)) return(NA_real_)
      v <- suppressWarnings(as.numeric(d[[nm]]))
      v[which(is.finite(v))[1]] %||% NA_real_
    }
    w_vec <- switch(weight_column,
                    "tuned_f1" = vapply(sp, pick_slot_scalar, numeric(1), nm="tuned_f1"),
                    "f1"       = vapply(sp, pick_slot_scalar, numeric(1), nm="f1"),
                    "accuracy" = vapply(sp, pick_slot_scalar, numeric(1), nm="accuracy"))
    if (!any(is.finite(w_vec))) w_vec[] <- 1 else { w_vec[!is.finite(w_vec)] <- 0; if (sum(w_vec)==0) w_vec[] <- 1 }
    w_norm <- w_vec / sum(w_vec)
    dcat("Weights:", paste(r6(w_norm), collapse=", "))
    
    thr_vec <- if (isTRUE(use_tuned_threshold_for_vote) && has_col(dfx, "tuned_threshold")) {
      vapply(sp, pick_slot_scalar, numeric(1), nm="tuned_threshold")
    } else rep(NA_real_, ncol(probs_mat))
    thr_vec[!is.finite(thr_vec)] <- default_threshold
    dcat("Vote thresholds per slot:", paste(r6(thr_vec), collapse=", "))
    
    if ("avg" %in% methods) {
      p_avg <- rowMeans(probs_mat, na.rm = TRUE)
      m <- bin_metrics(p_avg, y, default_threshold)
      out_rows[["Ensemble_avg"]] <- c(list(kind="Ensemble_avg", n=N, slots=paste(slots, collapse=","), mc_conf_mat=NA, mc_per_class=NA), m)
      out_preds[["Ensemble_avg"]] <- matrix(p_avg, ncol=1, dimnames=list(NULL, "pred"))
      dcat("AVG done.")
    }
    
    if ("wavg" %in% methods) {
      p_w <- as.numeric(probs_mat %*% w_norm)
      m <- bin_metrics(p_w, y, default_threshold)
      out_rows[["Ensemble_wavg"]] <- c(list(kind="Ensemble_wavg", n=N, slots=paste(slots, collapse=","), weights=w_norm, mc_conf_mat=NA, mc_per_class=NA), m)
      out_preds[["Ensemble_wavg"]] <- matrix(p_w, ncol=1, dimnames=list(NULL, "pred"))
      dcat("WAVG done.")
    }
    
    if ("vote_soft" %in% methods || "vote_hard" %in% methods) {
      vote_mat <- sweep(probs_mat, 2, thr_vec, FUN = ">=") * 1L
      vote_frac <- rowMeans(vote_mat, na.rm = TRUE)
      q <- vote_quorum %||% ceiling(ncol(probs_mat)/2)
      vote_hard <- as.integer(rowSums(vote_mat, na.rm = TRUE) >= q)
      
      if ("vote_soft" %in% methods) {
        m <- bin_metrics(vote_frac, y, default_threshold)
        out_rows[["Ensemble_vote_soft"]] <- c(list(kind="Ensemble_vote_soft", n=N, slots=paste(slots, collapse=","), quorum=q, mc_conf_mat=NA, mc_per_class=NA), m)
        out_preds[["Ensemble_vote_soft"]] <- matrix(vote_frac, ncol=1, dimnames=list(NULL, "pred"))
        dcat("VOTE_SOFT done.")
      }
      if ("vote_hard" %in% methods) {
        m <- bin_metrics(as.numeric(vote_hard), y, default_threshold)
        out_rows[["Ensemble_vote_hard"]] <- c(list(kind="Ensemble_vote_hard", n=N, slots=paste(slots, collapse=","), quorum=q, mc_conf_mat=NA, mc_per_class=NA), m)
        out_preds[["Ensemble_vote_hard"]] <- matrix(as.numeric(vote_hard), ncol=1, dimnames=list(NULL, "pred"))
        dcat("VOTE_HARD done.")
      }
    }
    
  } else if (classification_mode == "multiclass") {
    mats <- lapply(sp, get_prob_mat_mc, labels_ref = sort(unique(na.omit(y_true))))
    all_labs <- sort(unique(unlist(lapply(mats, function(x) as.integer(colnames(x$P))))))
    dcat("MC labels (union):", paste(all_labs, collapse=", "))
    
    lens <- vapply(mats, function(m) nrow(m$P), integer(1))
    N <- min(c(length(y_true), lens))
    if (!is.finite(N) || N <= 0L) stop("No usable rows after alignment for MC (N <= 0).")
    y <- as.integer(y_true[seq_len(N)])
    
    align_to <- function(P, labs_target) {
      labs_src <- as.integer(colnames(P))
      idx <- match(labs_target, labs_src)
      Q <- matrix(0, nrow = min(nrow(P), N), ncol = length(labs_target))
      colnames(Q) <- paste0(labs_target)
      ok <- which(is.finite(idx))
      if (length(ok)) Q[, ok] <- P[seq_len(nrow(Q)), idx[ok], drop=FALSE]
      Q
    }
    P_slots <- lapply(mats, function(m) align_to(m$P, all_labs))
    S <- length(P_slots)
    dcat("MC N=", N, " K=", length(all_labs), " S=", S)
    
    pick_slot_scalar <- function(d, nm) {
      if (!has_col(d, nm)) return(NA_real_)
      v <- suppressWarnings(as.numeric(d[[nm]]))
      v[which(is.finite(v))[1]] %||% NA_real_
    }
    w_vec <- switch(weight_column,
                    "tuned_f1" = vapply(sp, pick_slot_scalar, numeric(1), nm="tuned_f1"),
                    "f1"       = vapply(sp, pick_slot_scalar, numeric(1), nm="f1"),
                    "accuracy" = vapply(sp, pick_slot_scalar, numeric(1), nm="accuracy"))
    if (!any(is.finite(w_vec))) w_vec[] <- 1 else { w_vec[!is.finite(w_vec)] <- 0; if (sum(w_vec)==0) w_vec[] <- 1 }
    w_norm <- w_vec / sum(w_vec)
    dcat("Weights:", paste(r6(w_norm), collapse=", "))
    
    argmax_idx <- function(M) max.col(M, ties.method = "first")
    labs_vec <- as.integer(all_labs)
    
    # helper to pack mc diagnostics into out_rows
    pack_mc <- function(yp_labels_vec, kind_label, extra=list()) {
      m <- mc_metrics(yp_labels_vec, y, labels = labs_vec)
      row <- c(
        list(kind=kind_label, n=length(yp_labels_vec), slots=paste(slots, collapse=",")),
        extra,
        list(mc_conf_mat = list(m$conf_mat), mc_per_class = list(m$per_class)),
        m[setdiff(names(m), c("conf_mat", "per_class"))]
      )
      list(row=row, m=m)
    }
    
    if ("avg" %in% methods) {
      P_avg <- Reduce("+", P_slots) / S
      yp <- labs_vec[argmax_idx(P_avg)]
      res <- pack_mc(yp, "Ensemble_avg")
      out_rows[["Ensemble_avg"]] <- res$row
      out_preds[["Ensemble_avg"]] <- matrix(yp, ncol=1, dimnames=list(NULL, "y_pred"))
      dcat("AVG done. acc=", res$m$acc, " f1=", res$m$f1)
    }
    
    if ("wavg" %in% methods) {
      P_w <- P_slots[[1]] * w_norm[1]
      if (S > 1) for (i in 2:S) P_w <- P_w + P_slots[[i]] * w_norm[i]
      yp <- labs_vec[argmax_idx(P_w)]
      res <- pack_mc(yp, "Ensemble_wavg", list(weights=list(w_norm)))
      out_rows[["Ensemble_wavg"]] <- res$row
      out_preds[["Ensemble_wavg"]] <- matrix(yp, ncol=1, dimnames=list(NULL, "y_pred"))
      dcat("WAVG done. acc=", res$m$acc, " f1=", res$m$f1)
    }
    
    if ("vote_soft" %in% methods) {
      P_vs <- Reduce("+", P_slots) / S
      yp <- labs_vec[argmax_idx(P_vs)]
      res <- pack_mc(yp, "Ensemble_vote_soft")
      out_rows[["Ensemble_vote_soft"]] <- res$row
      out_preds[["Ensemble_vote_soft"]] <- matrix(yp, ncol=1, dimnames=list(NULL, "y_pred"))
      dcat("VOTE_SOFT done. acc=", res$m$acc, " f1=", res$m$f1)
    }
    
    if ("vote_hard" %in% methods) {
      yps <- lapply(P_slots, function(P) labs_vec[argmax_idx(P)])
      Yvote <- do.call(cbind, yps)  # N x S
      q <- vote_quorum %||% ceiling(ncol(Yvote)/2)
      vote_label <- integer(N)
      for (i in seq_len(N)) {
        cnt <- tabulate(match(Yvote[i,], labs_vec), nbins=length(labs_vec))
        if (max(cnt) >= q) {
          vote_label[i] <- labs_vec[which.max(cnt)]
        } else {
          vote_label[i] <- labs_vec[which.max(cnt)]
        }
      }
      res <- pack_mc(vote_label, "Ensemble_vote_hard", list(quorum=q))
      out_rows[["Ensemble_vote_hard"]] <- res$row
      out_preds[["Ensemble_vote_hard"]] <- matrix(vote_label, ncol=1, dimnames=list(NULL, "y_pred"))
      dcat("VOTE_HARD done. acc=", res$m$acc, " f1=", res$m$f1)
    }
    
  } else if (classification_mode == "regression") {
    dcat("Regression mode: returning predictions only; metrics placeholders.")
    get_reg_vec <- function(d) {
      if (has_col(d, "y_pred")) return(suppressWarnings(as.numeric(d$y_pred)))
      if (has_col(d, "y_prob")) return(suppressWarnings(as.numeric(d$y_prob)))
      stop("No y_pred/y_prob numeric column for regression.")
    }
    preds_list <- lapply(sp, get_reg_vec)
    lens <- vapply(preds_list, length, integer(1))
    N <- min(c(length(y_true), lens))
    preds_mat <- do.call(cbind, lapply(preds_list, function(v) as.numeric(v)[seq_len(N)]))
    colnames(preds_mat) <- paste0("slot_", slots)
    if ("avg" %in% methods) {
      p_avg <- rowMeans(preds_mat, na.rm = TRUE)
      out_rows[["Ensemble_avg"]] <- list(kind="Ensemble_avg", n=N, slots=paste(slots, collapse=","), acc=NA, precision=NA, recall=NA, f1=NA, TP=NA, FP=NA, FN=NA, TN=NA, mc_conf_mat=NA, mc_per_class=NA)
      out_preds[["Ensemble_avg"]] <- matrix(p_avg, ncol=1, dimnames=list(NULL, "pred"))
    }
    if ("wavg" %in% methods) {
      w_vec <- rep(1, ncol(preds_mat)); w_norm <- w_vec/sum(w_vec)
      p_w <- as.numeric(preds_mat %*% w_norm)
      out_rows[["Ensemble_wavg"]] <- list(kind="Ensemble_wavg", n=N, slots=paste(slots, collapse=","), weights=w_norm, acc=NA, precision=NA, recall=NA, f1=NA, TP=NA, FP=NA, FN=NA, TN=NA, mc_conf_mat=NA, mc_per_class=NA)
      out_preds[["Ensemble_wavg"]] <- matrix(p_w, ncol=1, dimnames=list(NULL, "pred"))
    }
  } else {
    stop("Unknown classification_mode: ", classification_mode)
  }
  
  # ---- format metrics df (now with list-columns for MC) ----
  rows_df <- do.call(rbind, lapply(names(out_rows), function(k) {
    r <- out_rows[[k]]
    # make sure list-cols stay list-cols
    mc_conf <- if (!is.null(r$mc_conf_mat)) r$mc_conf_mat else list(NA)
    mc_pc   <- if (!is.null(r$mc_per_class)) r$mc_per_class else list(NA)
    data.frame(
      kind = as.character(r$kind %||% k),
      n = as.integer(r$n %||% NA),
      slots = as.character(r$slots %||% NA),
      accuracy = r6(r$acc %||% NA),
      precision = r6(r$precision %||% NA),
      recall = r6(r$recall %||% NA),
      f1 = r6(r$f1 %||% NA),
      TP = as.integer(r$TP %||% NA),
      FP = as.integer(r$FP %||% NA),
      FN = as.integer(r$FN %||% NA),
      TN = as.integer(r$TN %||% NA),
      mc_conf_mat = I(mc_conf),    # list-column
      mc_per_class = I(mc_pc),     # list-column (data.frame)
      stringsAsFactors = FALSE
    )
  }))
  rownames(rows_df) <- NULL
  
  dcat("Final metrics table:")
  dhead(rows_df, 10)
  
  list(metrics = rows_df, predictions = out_preds)
}



emit_table <- function(x,
                       title,
                       rows = NULL,
                       verbose = FALSE,
                       viewTables = FALSE) {
  if (!verbose && !viewTables) {
    return(invisible(x))
  }
  
  if (is.null(x)) {
    message(sprintf("%s <NULL>", title))
    return(invisible(x))
  }
  
  n_rows <- tryCatch(NROW(x), error = function(...) NA_integer_)
  if (is.na(n_rows) || n_rows == 0) {
    message(sprintf("%s <empty>", title))
    return(invisible(x))
  }
  
  max_rows <- rows
  if (is.null(max_rows)) {
    max_rows <- if (verbose) n_rows else min(n_rows, 10L)
  }
  
  truncated <- n_rows > max_rows
  to_show <- if (truncated) utils::head(x, max_rows) else x
  
  ddesonn_viewTables(to_show, title = title)

  if (truncated) {
    message(sprintf("... (%d of %d rows shown)", max_rows, n_rows))
  }
  
  invisible(x)
}




# ============================================================
# INTERNAL -- PRESENTATION ONLY
# Fixed decimal places for FINAL SUMMARY / CLASSIFICATION REPORTS.
# Does NOT change training or stored metrics.
# ============================================================
.ddesonn_format_final_summary_decimals <- function(x, decimals) {
  if (is.null(x) || is.null(decimals)) return(x)
  
  d <- suppressWarnings(as.integer(decimals))
  if (!is.finite(d) || is.na(d) || d < 0L) return(x)
  
  if (!is.numeric(x) || length(x) != 1L) return(x)
  
  formatC(as.numeric(x), format = "f", digits = d)
}
