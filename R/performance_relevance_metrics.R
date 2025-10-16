#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#      _     _      _     _      _     _      _     _      _     _      _     _      _     _    $$$$$$$$$$$$$$$$$$$$$$$
#     (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)   $$$$$$$$$$$$$$$$$$$$$$$
#     / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \     $$$$$$$$$$$$$$$$$$$$$$$
#   __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__   $$$$$$$$$$$$$$$$$$$$$$$
#  (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)  $$$$$$$$$$$$$$$$$$$$$$$
#    || M ||      || E ||      || T ||      || R ||      || I ||      || C ||      || S ||      $$$$$$$$$$$$$$$$$$$$$$$
# _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._     $$$$$$$$$$$$$$$$$$$$$$$
#(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)    $$$$$$$$$$$$$$$$$$$$$$$
# `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'     $$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

if (!exists("coerce_to_numeric_matrix", inherits = TRUE)) {
  source("utils/utils.R")
}

quantization_error <- function(SONN, Rdata, run_id, verbose) {
  
  # keep your structure; just coerce data once
  if (!is.matrix(Rdata)) Rdata <- as.matrix(Rdata)
  storage.mode(Rdata) <- "double"
  
  if (SONN$ML_NN) {
    # --- ML path: get a matrix W from SONN$weights (first input-facing layer) ---
    if (is.list(SONN$weights)) {
      # pick the first layer that matches NCOL(Rdata); fallback to first layer
      idx <- which(vapply(SONN$weights, function(w)
        is.matrix(w) && (ncol(w) == NCOL(Rdata) || nrow(w) == NCOL(Rdata)), logical(1L)))[1]
      if (is.na(idx)) idx <- 1L
      W <- as.matrix(SONN$weights[[idx]])
    } else if (is.matrix(SONN$weights)) {
      W <- as.matrix(SONN$weights)
    } else if (is.matrix(SONN)) {
      # legacy: codebook passed directly as a matrix
      W <- as.matrix(SONN)
    } else {
      if (verbose) message("[quantization error]: NA")
      return(NA_real_)
    }
    
  } else {
    # --- SL path (minimal fix): allow length-1 list, force matrix before storage.mode ---
    W <- SONN$weights
    if (is.list(W)) W <- W[[1L]]
    W <- as.matrix(W)
  }
  
  # orient W so columns = features in Rdata
  storage.mode(W) <- "double"
  if (ncol(W) != NCOL(Rdata) && nrow(W) == NCOL(Rdata)) W <- t(W)
  if (ncol(W) != NCOL(Rdata)) {
    if (verbose) message("[quantization error]: NA")
    return(NA_real_)
  }
  
  # === your original distance logic (but use W, not SONN) ===
  distances <- apply(Rdata, 1L, function(x) {
    neuron_distances <- apply(W, 1L, function(w) {
      sqrt(sum((x - w)^2))
    })
    min(neuron_distances)
  })
  
  if (!length(distances) || all(!is.finite(distances))) {
    if (verbose) message("[quantization error]: NA")
    return(NA_real_)
  }
  
  mean(distances, na.rm = TRUE)
}


# Model-only topo error: uses layer-1 weights + map inside SONN
topographic_error <- function(SONN, Rdata, threshold, verbose) {
  # --- normalize to list-of-matrix (preserved) ---
  if (is.matrix(SONN)) SONN <- list(weights = list(as.matrix(SONN)))
  if (is.matrix(SONN$weights)) SONN$weights <- list(as.matrix(SONN$weights))
  if (is.null(SONN$map)) {
    m <- nrow(SONN$weights[[1]])
    r <- max(1L, floor(sqrt(m))); while (m %% r && r > 1L) r <- r - 1L
    c <- max(1L, m %/% r)
    SONN$map <- list(matrix(seq_len(m), nrow = r, ncol = c, byrow = TRUE))
  } else if (!is.list(SONN$map)) {
    SONN$map <- list(as.matrix(SONN$map))
  } else if (!is.matrix(SONN$map[[1]])) {
    SONN$map[[1]] <- as.matrix(SONN$map[[1]])
  }
  
  # --- SL-safe W build (minimal fix): unbox length-1 list, force matrix BEFORE storage.mode ---
  W <- SONN$weights
  if (is.list(W)) W <- W[[1L]]
  W <- as.matrix(W)                    # <— key fix
  M <- as.matrix(SONN$map[[1]])
  X <- as.matrix(Rdata)
  storage.mode(W) <- "double"; storage.mode(X) <- "double"
  
  # ---------- ALIGN W to X (preserved) ----------
  align_to_X <- function(W, X) {
    if (!is.null(colnames(W)) && !is.null(colnames(X))) {
      wanted <- colnames(X)
      miss <- setdiff(wanted, colnames(W))
      if (length(miss)) {
        W <- cbind(W, matrix(0, nrow = nrow(W), ncol = length(miss),
                             dimnames = list(NULL, miss)))
      }
      W <- W[, wanted, drop = FALSE]
      return(W)
    }
    kx <- ncol(X); kw <- ncol(W)
    if (kw > kx) {
      W[, seq_len(kx), drop = FALSE]
    } else if (kw < kx) {
      cbind(W, matrix(0, nrow = nrow(W), ncol = kx - kw))
    } else {
      W
    }
  }
  W <- align_to_X(W, X)
  
  m <- nrow(W); n <- nrow(X)
  if (m < 2L || n < 1L) return(NA_real_)
  
  # map size check (preserved)
  if (length(M) != m) {
    r <- max(1L, floor(sqrt(m))); while (m %% r && r > 1L) r <- r - 1L
    c <- max(1L, m %/% r)
    M <- matrix(seq_len(m), nrow = r, ncol = c, byrow = TRUE)
  }
  
  if (verbose) {
    message(sprintf("[topo] dim(X)=%dx%d | dim(W)=%dx%d | units=%d",
                    nrow(X), ncol(X), nrow(W), ncol(W), m))
    if (!is.null(colnames(X)) && !is.null(colnames(W))) {
      same_names <- identical(colnames(X), colnames(W))
      message(sprintf("[topo] colnames(X)==colnames(W): %s", same_names))
    }
  }
  
  # ---------- Fast pairwise squared distances (preserved) ----------
  X2 <- rowSums(X * X)
  W2 <- rowSums(W * W)
  G  <- X %*% t(W)
  D  <- matrix(X2, n, m) + matrix(W2, n, m, byrow = TRUE) - 2 * G
  
  # BMU / second-BMU (preserved)
  bmu <- max.col(-D, ties.method = "first")
  D[cbind(seq_len(n), bmu)] <- Inf
  sbmu <- max.col(-D, ties.method = "first")
  
  # grid coords (preserved)
  coords <- matrix(NA_integer_, m, 2)
  for (k in 1:m) {
    pos <- which(M == k, arr.ind = TRUE)
    coords[k, ] <- if (length(pos)) c(pos[1, 1], pos[1, 2]) else c(1L, k)
  }
  
  dgrid <- sqrt(rowSums((coords[bmu, , drop = FALSE] - coords[sbmu, , drop = FALSE])^2))
  err <- mean(dgrid > 1)
  
  if (verbose) message(sprintf("[topo] error = %s", err))
  err
}

is.adjacent <- function(map, neuron1, neuron2) {
  # 💡 Ensure map rownames exist and match neuron indices
  if (is.null(rownames(map))) {
    rownames(map) <- as.character(1:nrow(map))
  }
  
  coord1 <- map[as.character(neuron1), , drop = FALSE]
  coord2 <- map[as.character(neuron2), , drop = FALSE]
  
  if (nrow(coord1) == 0 || nrow(coord2) == 0) {
    stop(paste("Neurons", neuron1, "or", neuron2, "not found in the map"))
  }
  
  grid_dist <- sum(abs(coord1 - coord2))
  
  return(grid_dist == 1)
}

# Clustering quality (Davies-Bouldin index)
clustering_quality_db <- function(SONN, Rdata, cluster_assignments, verbose) {
  if (is.null(cluster_assignments)) {
    stop("Cluster assignments not available. Perform kmeans clustering first.")
  }
  
  # Ensure Rdata is a numeric matrix (SL-safe: force matrix BEFORE storage.mode)
  if (!is.matrix(Rdata)) Rdata <- as.matrix(Rdata)
  Rdata <- apply(Rdata, 2, as.numeric)
  storage.mode(Rdata) <- "double"
  
  # Compute centroids and ensure it's a numeric matrix
  centroids <- aggregate(Rdata, by = list(cluster_assignments), FUN = mean)[, -1, drop = FALSE]
  centroids <- as.matrix(centroids)
  centroids <- apply(centroids, 2, as.numeric)
  storage.mode(centroids) <- "double"
  
  n_clusters <- nrow(centroids)
  if (n_clusters < 2L) {
    # DB index undefined with <2 clusters; return NA to avoid bogus division
    if (verbose) message(sprintf("[clustering_quality_db] n_clusters < %d → NA", 2L))
    return(NA_real_)
  }
  
  # Split indices by cluster
  cluster_indices <- split(seq_len(nrow(Rdata)), cluster_assignments)
  
  # Precompute intra-cluster dispersion for each cluster (Si)
  S <- numeric(n_clusters)
  for (i in seq_len(n_clusters)) {
    data_i <- Rdata[cluster_indices[[i]], , drop = FALSE]
    centroid_i <- centroids[i, ]
    centroid_matrix <- matrix(centroid_i, nrow(data_i), ncol(data_i), byrow = TRUE)
    S[i] <- mean(rowSums((data_i - centroid_matrix)^2))
  }
  
  # Precompute inter-cluster distances (squared Euclidean)
  D <- as.matrix(dist(centroids))^2
  storage.mode(D) <- "double"
  
  # Compute Davies-Bouldin index (guard 0 distance)
  db_index <- 0
  for (i in seq_len(n_clusters)) {
    max_ratio <- -Inf
    for (j in seq_len(n_clusters)) {
      if (i != j) {
        denom <- sqrt(D[i, j])
        ratio <- if (denom > 0) (S[i] + S[j]) / denom else Inf
        if (ratio > max_ratio) max_ratio <- ratio
      }
    }
    db_index <- db_index + max_ratio
  }
  
  db_index <- db_index / n_clusters
  
  # if (verbose) {
  #   cat("clustering_quality_db:", db_index, "\n")
  # }
  
  return(db_index)
}


# Debug MSE that USES CLASSIFICATION_MODE for shape handling (silent version)
# Signature: MSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MSE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  # support %||% like base R (define before use)
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # --- coerce to numeric matrices (then SL-style tiny guards) ---
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  if (is.list(lbl))  lbl  <- lbl[[1L]]
  if (is.list(pred)) pred <- pred[[1L]]
  if (!is.matrix(lbl))  lbl  <- as.matrix(lbl)
  if (!is.matrix(pred)) pred <- as.matrix(pred)
  storage.mode(lbl)  <- "double"
  storage.mode(pred) <- "double"
  
  # --- ROW ALIGNMENT (no recycling) ---
  n_common <- min(nrow(lbl), nrow(pred))
  if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # --- MODE-BASED SHAPE HARMONIZATION ---
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  
  if (identical(mode, "multiclass")) {
    # Expect pred: N x K
    K <- ncol(pred)
    if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      # class index 1..K or 0..K-1 -> one-hot
      u <- sort(unique(as.integer(lbl[, 1])))
      if (length(u)) {
        if (min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
          lbl <- lbl + 1
        }
      }
      lbl <- safe_one_hot_matrix(lbl[, 1], K)
    } else if (ncol(lbl) != K) {
      return(NA_real_)
    }
    
  } else if (identical(mode, "binary")) {
    # Conventions:
    #   pred: N x 1 (p_pos) OR N x 2 ([p_neg, p_pos])
    #   lbl : N x 1 (0/1 or two unique values) OR N x 2 one-hot
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      # Reduce labels to 0/1 using column 2 as positive
      lbl <- matrix(lbl[, 2], ncol = 1L)
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      # Expand labels to one-hot [neg, pos]
      v <- as.integer(round(lbl[, 1]))
      v[is.na(v)] <- 0L
      v[v != 0L]  <- 1L
      lbl <- cbind(1 - v, v)
      storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      # Ensure {0,1}
      uniq <- sort(unique(as.integer(lbl[, 1])))
      if (!all(uniq %in% c(0L, 1L))) {
        if (length(uniq) == 2L) {
          lbl[, 1] <- ifelse(lbl[, 1] == uniq[2], 1, 0)
        } else {
          return(NA_real_)
        }
      }
    } else if (ncol(pred) == 2L && ncol(lbl) == 2L) {
      # already aligned
      # nothing to do
    } else {
      return(NA_real_)
    }
    
  } else if (!identical(mode, "regression")) {
    # Unknown mode
    return(NA_real_)
  }
  
  # --- FINAL COLUMN CHECK (avoid replicate/truncate for class probs) ---
  if (ncol(lbl) != ncol(pred)) {
    if (identical(mode, "regression")) {
      # Conform via replicate/truncate
      total_needed <- nrow(lbl) * ncol(lbl)
      if (ncol(pred) < ncol(lbl)) {
        vec <- rep(pred, length.out = total_needed)
        pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
      } else {
        pred <- pred[, 1:ncol(lbl), drop = FALSE]
        pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
      }
    } else {
      return(NA_real_)
    }
  }
  
  # --- NA handling ---
  cc <- stats::complete.cases(cbind(lbl, pred))
  if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]
  pred <- pred[cc, , drop = FALSE]
  if (!nrow(lbl)) return(NA_real_)
  
  # --- Compute error + MSE ---
  err <- pred - lbl
  mse <- mean(err^2)
  
  mse
}

# Debug R^2 that USES CLASSIFICATION_MODE for shape handling (silent version)
# Signature: R2(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
R2 <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # --- coerce (with minimal SL-style guard) ---
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  if (is.list(lbl))  lbl  <- lbl[[1L]]
  if (is.list(pred)) pred <- pred[[1L]]
  if (!is.matrix(lbl))  lbl  <- as.matrix(lbl)
  if (!is.matrix(pred)) pred <- as.matrix(pred)
  storage.mode(lbl)  <- "double"
  storage.mode(pred) <- "double"
  
  # --- row align ---
  n_common <- min(nrow(lbl), nrow(pred)); if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # --- mode-based shape handling ---
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  
  if (identical(mode, "multiclass")) {
    K <- ncol(pred); if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(lbl[, 1])))
      if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) lbl <- lbl + 1
      lbl <- safe_one_hot_matrix(lbl[, 1], K)
    } else if (ncol(lbl) != K) return(NA_real_)
    return(NA_real_)  # R² not meaningful for classification
    
  } else if (identical(mode, "binary")) {
    # Align like in MSE, then bail (R² not meaningful for classification)
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      lbl <- matrix(lbl[, 2], ncol = 1L)
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      v <- as.integer(round(lbl[, 1])); v[is.na(v)] <- 0L; v[v != 0L] <- 1L
      lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      uniq <- sort(unique(as.integer(lbl[, 1])))
      if (!all(uniq %in% c(0L, 1L))) {
        if (length(uniq) == 2L) lbl[, 1] <- ifelse(lbl[, 1] == uniq[2], 1, 0) else return(NA_real_)
      }
    } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) return(NA_real_)
    return(NA_real_)
  } else if (!identical(mode, "regression")) return(NA_real_)
  
  # --- final column check (regression may replicate/truncate) ---
  if (ncol(lbl) != ncol(pred)) {
    total_needed <- nrow(lbl) * ncol(lbl)
    if (ncol(pred) < ncol(lbl)) {
      vec <- rep(pred, length.out = total_needed)
      pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    } else pred <- pred[, 1:ncol(lbl), drop = FALSE]
  }
  
  # --- NA handling ---
  cc <- stats::complete.cases(cbind(lbl, pred)); if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]; pred <- pred[cc, , drop = FALSE]; if (!nrow(lbl)) return(NA_real_)
  
  # --- R² across all columns (if multi-target) ---
  y  <- as.numeric(lbl)
  yhat <- as.numeric(pred)
  ss_res <- sum((y - yhat)^2)
  ss_tot <- sum((y - mean(y))^2)
  if (ss_tot == 0) return(NA_real_)
  1 - ss_res / ss_tot
}

# Signature: MAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MAPE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                 eps_abs = 1e-6, eps_pct = 1e-3) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)

  y   <- coerce_to_numeric_matrix(labels)
  yhat<- coerce_to_numeric_matrix(predicted_output)
  
  n <- min(nrow(y), nrow(yhat)); if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  # Conform columns for regression
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) {
      yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    } else yhat <- yhat[, 1:ncol(y), drop = FALSE]
  }
  
  cc <- stats::complete.cases(cbind(y, yhat)); if (!any(cc)) return(NA_real_)
  y <- y[cc, , drop = FALSE]; yhat <- yhat[cc, , drop = FALSE]; if (!nrow(y)) return(NA_real_)
  
  # Denominator floor: max(|y|, eps_abs, eps_pct*mean|y|)
  mean_abs_y <- max(mean(abs(y)), eps_abs)
  denom_floor <- max(eps_abs, eps_pct * mean_abs_y)
  
  rel <- abs(yhat - y) / pmax(abs(y), denom_floor)
  mean(rel, na.rm = TRUE) * 100
}

# Signature: SMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
SMAPE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                  eps = 1e-6) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)

  y   <- coerce_to_numeric_matrix(labels)
  yhat<- coerce_to_numeric_matrix(predicted_output)
  
  n <- min(nrow(y), nrow(yhat)); if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    else yhat <- yhat[, 1:ncol(y), drop = FALSE]
  }
  
  cc <- stats::complete.cases(cbind(y, yhat)); if (!any(cc)) return(NA_real_)
  y <- y[cc, , drop = FALSE]; yhat <- yhat[cc, , drop = FALSE]; if (!nrow(y)) return(NA_real_)
  
  denom <- (abs(y) + abs(yhat)) / 2
  ok <- is.finite(denom) & (denom > eps)
  if (!any(ok)) return(NA_real_)
  mean(abs(yhat[ok] - y[ok]) / denom[ok]) * 100
}

# Signature: WMAPE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
WMAPE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                  denom_floor = 1e-6) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)

  y   <- coerce_to_numeric_matrix(labels)
  yhat<- coerce_to_numeric_matrix(predicted_output)
  
  n <- min(nrow(y), nrow(yhat)); if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    else yhat <- yhat[, 1:ncol(y), drop = FALSE]
  }
  
  cc <- stats::complete.cases(cbind(y, yhat)); if (!any(cc)) return(NA_real_)
  y <- y[cc, , drop = FALSE]; yhat <- yhat[cc, , drop = FALSE]; if (!nrow(y)) return(NA_real_)
  
  num <- sum(abs(yhat - y))
  den <- sum(abs(y))
  if (!is.finite(den) || den < denom_floor) return(NA_real_)
  (num / den) * 100
}

# Debug MASE that USES CLASSIFICATION_MODE for shape handling (silent version)
# Signature: MASE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MASE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE,
                 lag = 1L) {
  `%||%` <- function(x, y) if (is.null(x)) y else x

  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  if (!identical(mode, "regression")) return(NA_real_)

  y    <- coerce_to_numeric_matrix(labels)
  yhat <- coerce_to_numeric_matrix(predicted_output)
  
  # --- align rows ---
  n <- min(nrow(y), nrow(yhat))
  if (n == 0L) return(NA_real_)
  y    <- y[seq_len(n), , drop = FALSE]
  yhat <- yhat[seq_len(n), , drop = FALSE]
  
  # --- conform columns ---
  if (ncol(yhat) != ncol(y)) {
    total <- nrow(y) * ncol(y)
    if (ncol(yhat) < ncol(y)) {
      yhat <- matrix(rep(yhat, length.out = total), nrow = nrow(y), byrow = FALSE)
    } else {
      yhat <- yhat[, 1:ncol(y), drop = FALSE]
    }
  }
  
  # --- NA handling ---
  cc <- stats::complete.cases(cbind(y, yhat))
  if (!any(cc)) return(NA_real_)
  y    <- y[cc, , drop = FALSE]
  yhat <- yhat[cc, , drop = FALSE]
  if (!nrow(y)) return(NA_real_)
  
  # --- Model error (MAE) ---
  mae_model <- mean(abs(yhat - y))
  
  # --- Baseline error (lag-1 naïve forecast) ---
  if (nrow(y) <= lag) return(NA_real_)
  diffs <- abs(y[(lag + 1):nrow(y), , drop = FALSE] - y[1:(nrow(y) - lag), , drop = FALSE])
  mae_naive <- mean(diffs, na.rm = TRUE)
  
  if (!is.finite(mae_naive) || mae_naive == 0) return(NA_real_)
  
  # --- Ratio: model error / naive error ---
  mase <- mae_model / mae_naive
  mase
}

# Speed
speed_learn <- function(SONN, learn_time, verbose) {
  
  # if (verbose) {
  #   print("speed")
  #   print(learn_time)
  # }
  return(learn_time)
  # if (verbose) {
  #   print("speed complete")
  # }
}

# Speed
speed <- function(SONN, prediction_time, verbose) {
  
  # if (verbose) {
  #   print("speed")
  #   print(prediction_time)
  # }
  return(prediction_time)
  # if (verbose) {
  #   print("speed complete")
  # }
}

# Memory usage
memory_usage <- function(SONN, Rdata, verbose) {
  
  # Calculate the memory usage of the SONN object
  object_size <- object.size(SONN)
  
  # Calculate the memory usage of the Rdata
  Rdata_size <- object.size(Rdata)
  # if (verbose) {
  #   print("memory")
  #   print(object_size + Rdata_size)
  # }
  # Return the total memory usage without the word "bytes"
  return(as.numeric(gsub("bytes", "", object_size + Rdata_size)))
  # if (verbose) {
  #   print("memory complete")
  # }
}

# Robustness (noise/outlier MSE under consistent label/pred harmonization)
robustness <- function(SONN, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num, predicted_output, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # --------- Build noisy data (deterministic) ----------
  Rdata <- as.matrix(Rdata); storage.mode(Rdata) <- "double"
  set.seed(123)
  noisy_Rdata <- Rdata + rnorm(n = nrow(Rdata) * ncol(Rdata), mean = 0, sd = 0.2)
  dim(noisy_Rdata) <- dim(Rdata); storage.mode(noisy_Rdata) <- "double"
  # Inject outliers (~2% rows)
  n_out_rows <- max(1L, min(nrow(noisy_Rdata), as.integer(round(0.02 * nrow(noisy_Rdata)))))
  if (n_out_rows > 0L) {
    idx_rows <- sample.int(nrow(noisy_Rdata), n_out_rows)
    noisy_Rdata[idx_rows, ] <- matrix(
      rnorm(n_out_rows * ncol(noisy_Rdata), mean = 5, sd = 1),
      nrow = n_out_rows, ncol = ncol(noisy_Rdata)
    )
  }
  
  learnOnlyTrainingRun <- get0("learnOnlyTrainingRun", ifnotfound = FALSE, inherits = TRUE)
  plot_robustness      <- get0("plot_robustness",      ifnotfound = FALSE, inherits = TRUE)
  
  # --------- Train-time branch (rare) ----------
  if (isTRUE(learnOnlyTrainingRun)) {
    invisible(SONN$learn(noisy_Rdata, labels, lr))
    return(NA_real_)
  }

  # --------- Predict on noisy data ----------
  # NOTE: predict should be deterministic (no dropout); pass activation_functions only.
  pred_obj <- SONN$predict(noisy_Rdata, SONN$weights, SONN$biases, activation_functions, verbose)
  pred_raw <- pred_obj$predicted_output
  
  # Coerce to numeric matrices
  L <- coerce_to_numeric_matrix(labels)
  P <- coerce_to_numeric_matrix(pred_raw)
  
  # Row alignment
  n_common <- min(nrow(L), nrow(P))
  if (n_common <= 0L) return(NA_real_)
  L <- L[seq_len(n_common), , drop = FALSE]
  P <- P[seq_len(n_common), , drop = FALSE]
  
  # Decide task type
  mode_in <- tolower(as.character(CLASSIFICATION_MODE %||% "auto"))
  if (mode_in %in% c("binary", "multiclass", "regression")) {
    mode <- mode_in
  } else {
    inf <- infer_is_binary(L, P)
    mode <- if (isTRUE(inf$is_binary)) "binary" else {
      if (!is.null(ncol(L)) && ncol(L) >= 3L) "multiclass" else "regression"
    }
  }
  
  # --------- Harmonize shapes ----------
  if (identical(mode, "binary")) {
    # labels -> 0/1 vector; preds -> P(pos) in [0,1]
    y_true01 <- labels_to_binary_vec(L)
    if (is.null(y_true01)) {
      # try to infer from shape (2-col one-hot)
      if (!is.null(ncol(L)) && ncol(L) == 2L) {
        y_true01 <- max.col(L, ties.method = "first") - 1L
      } else if (!is.null(ncol(L)) && ncol(L) == 1L) {
        y_true01 <- ifelse(L[,1] > 0, 1L, 0L)
      } else {
        return(NA_real_)
      }
    }
    p_pos <- preds_to_pos_prob(P)
    if (is.null(p_pos)) return(NA_real_)
    # align rows/finite
    cc <- stats::complete.cases(y_true01, p_pos)
    if (!any(cc)) return(NA_real_)
    y_true01 <- as.numeric(y_true01[cc])
    p_pos    <- as.numeric(p_pos[cc])
    if (!length(y_true01)) return(NA_real_)
    losses <- mean((p_pos - y_true01)^2)
    
  } else if (identical(mode, "multiclass")) {
    # P must be N x K; L should be one-hot N x K
    K <- ncol(P)
    if (is.null(K) || K < 2L) return(NA_real_)
    # If labels are a single col of class ids (0..K-1 or 1..K), expand to one-hot
    if (ncol(L) == 1L) {
      ids <- as.vector(L[,1])
      # allow 0..K-1 by shifting
      ids_num <- suppressWarnings(as.numeric(ids))
      if (all(is.finite(ids_num))) {
        if (min(ids_num, na.rm = TRUE) == 0 && max(ids_num, na.rm = TRUE) == (K - 1)) {
          ids_num <- ids_num + 1
        }
      }
      L <- one_hot_from_ids(ids_num, K = K, N = nrow(L), strict = FALSE)
    }
    if (ncol(L) != K) return(NA_real_)
    # finite rows only
    cc <- stats::complete.cases(cbind(L, P))
    if (!any(cc)) return(NA_real_)
    L <- L[cc, , drop = FALSE]
    P <- P[cc, , drop = FALSE]
    if (!nrow(L)) return(NA_real_)
    losses <- mean((P - L)^2)
    
  } else { # regression
    # Match column counts; recycle/truncate P to L
    if (ncol(L) != ncol(P)) {
      total_needed <- nrow(L) * ncol(L)
      if (ncol(P) < ncol(L)) {
        vec <- rep(P, length.out = total_needed)
        P <- matrix(vec, nrow = nrow(L), ncol = ncol(L), byrow = FALSE)
      } else {
        P <- P[, seq_len(ncol(L)), drop = FALSE]
        P <- matrix(P, nrow = nrow(L), ncol = ncol(L), byrow = FALSE)
      }
    }
    cc <- stats::complete.cases(cbind(L, P))
    if (!any(cc)) return(NA_real_)
    L <- L[cc, , drop = FALSE]
    P <- P[cc, , drop = FALSE]
    if (!nrow(L)) return(NA_real_)
    losses <- mean((P - L)^2)
  }
  
  # Optional plotting (useful only if you later sweep noise SDs)
  if (isTRUE(plot_robustness) && length(losses) > 1L) {
    if (!any(is.nan(losses)) && !any(is.infinite(losses))) {
      plot(
        losses, type = "l",
        main = paste("Loss over noise SD for SONN", model_iter_num),
        xlab = "Noise step", ylab = "MSE", lwd = 2
      )
    }
  }
  
  # Return scalar MSE under noise/outliers
  as.numeric(losses)
}

# Hit Rate
hit_rate <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose) {
  Rdata <- data.frame(Rdata)
  
  # Identify the relevant Rdata points
  relevant_Rdata <- Rdata[Rdata$class == "relevant", ]
  
  # Calculate the hit rate
  if (nrow(relevant_Rdata) == 0L) {
    return(NA_real_)
  }
  
  hit_rate <- sum(predicted_output %in% relevant_Rdata$id) / nrow(relevant_Rdata)
  
  hit_rate
}


# -------------------------
# Accuracy (mode-aware: "binary" | "multiclass" | "regression")
# Signature: accuracy(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass: predicted_output must be N×K probs (K>1). labels = class indices (1..K or 0..K-1) or one-hot N×K.
# - Binary    : predicted_output = N×1 (p_pos) or N×2 ([p_neg, p_pos]); labels = 0/1, two unique values, or one-hot N×2.
# - Regression: accuracy undefined -> returns NA_real_.
accuracy <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  dbg <- function(...) {
    if (verbose) message(paste(..., collapse = " "))
  }
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  
  
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      if (verbose) dbg("[accuracy] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      if (verbose) dbg("[accuracy] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  
  
  # ---------- coerce ----------
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[accuracy] Empty labels or predictions.")
  if (verbose) dbg(sprintf("[accuracy] initial dims: labels=%d x %d | pred=%d x %d",
                                   nrow(lbl), ncol(lbl), nrow(pred), ncol(pred)))
  
  # ---------- row alignment (NO aggressive recycling) ----------
  
  
  if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < 0.9) {
      stop(sprintf("[accuracy] Row mismatch too large (labels=%d, pred=%d). Pass aligned inputs.",
                   nrow(lbl), nrow(pred)))
    }
    if (verbose) dbg(sprintf("[accuracy] Minor row mismatch -> trimming to %d rows", n_common))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (verbose) dbg(paste("[accuracy] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[accuracy] Multiclass mode requires predicted_output with K>1 columns.")
    
    # true classes
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    } else {
      stop(sprintf("[accuracy] Multiclass labels have %d columns; predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    
    # predicted classes
    if (ncol(pred) != K) stop("[accuracy] Multiclass predictions must be N×K probabilities.")
    pred_class <- max.col(pred, ties.method = "first")
    
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[accuracy] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    acc <- mean(pred_class == true_class)
    return(round(as.numeric(acc), 8))
    
  } else if (identical(mode, "binary")) {
    # probabilities for positive class
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[accuracy] Unexpected binary prediction shape: %d columns.", ncol(pred)))
    }
    
    # true labels -> 0/1
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # [neg,pos] -> 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[accuracy] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[accuracy] Unexpected binary label shape: %d columns.", ncol(lbl)))
    }
    
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[accuracy] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    acc <- mean(y_pred == y_true)
    return(round(as.numeric(acc), 8))
    
  } else if (identical(mode, "regression")) {
    if (verbose) dbg("[accuracy] Regression mode: accuracy undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[accuracy] Unhandled mode.")
}

# -------------------------
# Precision (mode-aware: "binary" | "multiclass" | "regression")
# Signature: precision(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass expects predicted_output as N×K probabilities (K>1). labels can be class indices (1..K or 0..K-1) or one-hot N×K.
# - Binary accepts predicted_output as N×1 (p_pos) or N×2 ([p_neg, p_pos]). labels as 0/1, two unique values, or one-hot N×2.
# - Regression: precision is undefined -> returns NA_real_ (with optional debug note).
precision <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  dbg <- function(...) {
    if (verbose) message(paste(..., collapse = " "))
  }
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  
  
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      # called like precision(..., preds, TRUE)
      if (verbose) dbg("[precision] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      # called like precision(..., preds)
      if (verbose) dbg("[precision] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # ---------- coerce ----------
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[precision] Empty labels or predictions.")
  
  # ---------- row alignment (NO aggressive recycling) ----------
  if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < 0.9) {
      stop(sprintf("[precision] Row mismatch too large (labels=%d, pred=%d). Pass aligned inputs.", nrow(lbl), nrow(pred)))
    }
    if (verbose) dbg(sprintf("[precision] Minor row mismatch -> trimming to %d rows", n_common))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (verbose) dbg(paste("[precision] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[precision] Multiclass mode requires predicted_output with K>1 columns.")
    
    # Build true_class from labels
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
      true_class[true_class < 1L] <- 1L
      true_class[true_class > K]  <- K
    } else {
      stop(sprintf("[precision] Multiclass labels have %d columns, predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    
    # Predicted class (argmax over K)
    pred_class <- if (ncol(pred) == K) max.col(pred, ties.method = "first") else
      stop("[precision] Multiclass predictions must be N×K probabilities.")
    
    # drop any NA rows (defensive)
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[precision] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    # Macro-averaged precision
    prec_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fp <- sum(pred_class == k & true_class != k)
      prec_per_class[k] <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    }
    return(as.numeric(mean(prec_per_class)))
    
  } else if (identical(mode, "binary")) {
    # Determine positive probabilities and true labels
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]  # assume column 2 is positive class prob
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[precision] Unexpected binary prediction shape: %d cols.", ncol(pred)))
    }
    
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # map one-hot [neg,pos] to 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          # map smaller -> 0, larger -> 1
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[precision] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[precision] Unexpected binary label shape: %d cols.", ncol(lbl)))
    }
    
    # Threshold (0.5 by default)
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[precision] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    tp <- sum(y_pred == 1L & y_true == 1L)
    fp <- sum(y_pred == 1L & y_true == 0L)
    prec <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    return(round(as.numeric(prec), 8))
    
  } else if (identical(mode, "regression")) {
    if (verbose) dbg("[precision] Regression mode: precision is undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[precision] Unhandled mode.")
}

# -------------------------
# Recall (mode-aware: "binary" | "multiclass" | "regression")
# Signature: recall(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass: macro-averaged recall across K classes.
# - Binary    : standard recall = TP / (TP + FN), using threshold 0.5 (or column 2 for N×2 preds).
# - Regression: undefined -> returns NA_real_.
# Set global option RECALL_MIN_OVERLAP <- 0.80 to control row-trim tolerance (default 0.80).
recall <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = verbose) {
  dbg <- function(...) {
    if (verbose) message(paste(..., collapse = " "))
  }
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")

  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      if (verbose) dbg("[recall] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      if (verbose) dbg("[recall] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # ---------- coerce ----------
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[recall] Empty labels or predictions.")
  if (verbose) dbg(sprintf("[recall] initial dims: labels=%d x %d | pred=%d x %d",
                                   nrow(lbl), ncol(lbl), nrow(pred), ncol(pred)))
  
  # ---------- row alignment (prefer by rownames, else tolerant trim) ----------
  align_by_rownames <- !is.null(rownames(lbl)) && !is.null(rownames(pred))
  if (align_by_rownames) {
    common_ids <- intersect(rownames(lbl), rownames(pred))
    if (length(common_ids) == 0L) stop("[recall] No overlapping rownames between labels and predictions.")
    lbl  <- lbl[common_ids, , drop = FALSE]
    pred <- pred[common_ids, , drop = FALSE]
    if (verbose) dbg(sprintf("[recall] Aligned by rownames: %d rows", length(common_ids)))
  } else if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    min_overlap <- get0("RECALL_MIN_OVERLAP", ifnotfound = 0.80, inherits = TRUE)
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < min_overlap) {
      stop(sprintf("[recall] Row mismatch too large (labels=%d, pred=%d). Overlap=%.3f < %.2f.",
                   nrow(lbl), nrow(pred), ratio, min_overlap))
    }
    if (verbose) dbg(sprintf("[recall] Trimmed to %d rows (overlap=%.3f >= %.2f)", n_common, ratio, min_overlap))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (verbose) dbg(paste("[recall] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[recall] Multiclass mode requires predicted_output with K>1 columns.")
    
    # true classes
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
    } else {
      stop(sprintf("[recall] Multiclass labels have %d columns; predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    true_class[true_class < 1L] <- 1L
    true_class[true_class > K]  <- K
    
    # predicted classes
    if (ncol(pred) != K) stop("[recall] Multiclass predictions must be N×K probabilities.")
    pred_class <- max.col(pred, ties.method = "first")
    
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[recall] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    rec_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fn <- sum(pred_class != k & true_class == k)
      rec_per_class[k] <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    }
    return(as.numeric(mean(rec_per_class)))
    
  } else if (identical(mode, "binary")) {
    # probabilities for positive class
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[recall] Unexpected binary prediction shape: %d columns.", ncol(pred)))
    }
    
    # true labels -> 0/1
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # [neg,pos] -> 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[recall] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[recall] Unexpected binary label shape: %d columns.", ncol(lbl)))
    }
    
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[recall] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    tp <- sum(y_pred == 1L & y_true == 1L)
    fn <- sum(y_pred == 0L & y_true == 1L)
    rec <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    return(round(as.numeric(rec), 8))
    
  } else if (identical(mode, "regression")) {
    if (verbose) dbg("[recall] Regression mode: recall undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[recall] Unhandled mode.")
}

# -------------------------
# F1 Score (mode-aware: "binary" | "multiclass" | "regression")
# Signature: f1_score(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
# Notes:
# - Multiclass: macro-averaged F1 across K classes.
# - Binary    : standard F1 from thresholded predictions (0.5 on p_pos).
# - Regression: undefined -> returns NA_real_.
# Set global option F1_MIN_OVERLAP <- 0.80 to control row-trim tolerance (default 0.80).
f1_score <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = verbose) {
  dbg <- function(...) {
    if (verbose) message(paste(..., collapse = " "))
  }
  
  # ---------- helpers ----------
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")

  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # ---------- argument recovery (handle swapped positional args) ----------
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      if (verbose) dbg("[f1] Swapped args detected -> using CLASSIFICATION_MODE as predicted_output; predicted_output as verbose.")
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      if (verbose) dbg("[f1] Missing predicted_output -> treating CLASSIFICATION_MODE as predicted_output.")
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # ---------- coerce ----------
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  if (nrow(lbl) == 0L || nrow(pred) == 0L) stop("[f1] Empty labels or predictions.")
  if (verbose) dbg(sprintf("[f1] initial dims: labels=%d x %d | pred=%d x %d",
                                   nrow(lbl), ncol(lbl), nrow(pred), ncol(pred)))
  
  # ---------- row alignment (prefer by rownames, else tolerant trim) ----------
  align_by_rownames <- !is.null(rownames(lbl)) && !is.null(rownames(pred))
  if (align_by_rownames) {
    common_ids <- intersect(rownames(lbl), rownames(pred))
    if (length(common_ids) == 0L) stop("[f1] No overlapping rownames between labels and predictions.")
    lbl  <- lbl[common_ids, , drop = FALSE]
    pred <- pred[common_ids, , drop = FALSE]
    if (verbose) dbg(sprintf("[f1] Aligned by rownames: %d rows", length(common_ids)))
  } else if (nrow(lbl) != nrow(pred)) {
    n_common <- min(nrow(lbl), nrow(pred))
    min_overlap <- get0("F1_MIN_OVERLAP", ifnotfound = 0.80, inherits = TRUE)
    ratio <- n_common / max(nrow(lbl), nrow(pred))
    if (ratio < min_overlap) {
      stop(sprintf("[f1] Row mismatch too large (labels=%d, pred=%d). Overlap=%.3f < %.2f.",
                   nrow(lbl), nrow(pred), ratio, min_overlap))
    }
    if (verbose) dbg(sprintf("[f1] Trimmed to %d rows (overlap=%.3f >= %.2f)", n_common, ratio, min_overlap))
    lbl  <- lbl[seq_len(n_common), , drop = FALSE]
    pred <- pred[seq_len(n_common), , drop = FALSE]
  }
  
  # ---------- resolve/infer mode ----------
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  if (verbose) dbg(paste("[f1] MODE =", mode))
  
  # ---------- classification paths ----------
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) stop("[f1] Multiclass mode requires predicted_output with K>1 columns.")
    
    # true classes
    if (ncol(lbl) == K) {
      true_class <- max.col(lbl, ties.method = "first")
    } else if (ncol(lbl) == 1L) {
      true_class <- as.integer(round(lbl[,1]))
      if (length(true_class) && min(true_class, na.rm = TRUE) == 0L) true_class <- true_class + 1L
    } else {
      stop(sprintf("[f1] Multiclass labels have %d columns; predictions have %d. Provide class index or one-hot with K.", ncol(lbl), K))
    }
    true_class[true_class < 1L] <- 1L
    true_class[true_class > K]  <- K
    
    # predicted classes
    if (ncol(pred) != K) stop("[f1] Multiclass predictions must be N×K probabilities.")
    pred_class <- max.col(pred, ties.method = "first")
    
    ok <- stats::complete.cases(true_class, pred_class)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[f1] Dropping %d row(s) with NA in classes", sum(!ok)))
      true_class <- true_class[ok]; pred_class <- pred_class[ok]
    }
    if (!length(true_class)) return(NA_real_)
    
    # macro F1
    f1_per_class <- numeric(K)
    for (k in seq_len(K)) {
      tp <- sum(pred_class == k & true_class == k)
      fp <- sum(pred_class == k & true_class != k)
      fn <- sum(pred_class != k & true_class == k)
      p  <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
      r  <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
      f1_per_class[k] <- if ((p + r) == 0) 0 else 2 * p * r / (p + r)
    }
    return(as.numeric(mean(f1_per_class)))
    
  } else if (identical(mode, "binary")) {
    # probabilities for positive class
    if (ncol(pred) == 2L) {
      p_pos <- pred[,2]
    } else if (ncol(pred) == 1L) {
      p_pos <- pred[,1]
    } else {
      stop(sprintf("[f1] Unexpected binary prediction shape: %d columns.", ncol(pred)))
    }
    
    # true labels -> 0/1
    if (ncol(lbl) == 2L) {
      y_true <- as.integer(lbl[,2] >= lbl[,1])  # [neg,pos] -> 0/1
    } else if (ncol(lbl) == 1L) {
      yv <- as.integer(round(lbl[,1]))
      u  <- sort(unique(yv))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          yv <- ifelse(yv == max(u), 1L, 0L)
        } else {
          stop("[f1] Binary labels must be {0,1}, two unique values, or one-hot N×2.")
        }
      }
      y_true <- yv
    } else {
      stop(sprintf("[f1] Unexpected binary label shape: %d columns.", ncol(lbl)))
    }
    
    y_pred <- as.integer(p_pos >= 0.5)
    
    ok <- stats::complete.cases(y_true, y_pred)
    if (!all(ok)) {
      if (verbose) dbg(sprintf("[f1] Dropping %d row(s) with NA in binary classes", sum(!ok)))
      y_true <- y_true[ok]; y_pred <- y_pred[ok]
    }
    if (!length(y_true)) return(NA_real_)
    
    tp <- sum(y_pred == 1L & y_true == 1L)
    fp <- sum(y_pred == 1L & y_true == 0L)
    fn <- sum(y_pred == 0L & y_true == 1L)
    
    p  <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    r  <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    f1 <- if ((p + r) == 0) 0 else 2 * p * r / (p + r)
    return(round(as.numeric(f1), 8))
    
  } else if (identical(mode, "regression")) {
    if (verbose) dbg("[f1] Regression mode: F1 undefined -> returning NA_real_.")
    return(NA_real_)
  }
  
  stop("[f1] Unhandled mode.")
}

confusion_matrix <- function(SONN, labels, CLASSIFICATION_MODE, predicted_output,
                             threshold = NULL, verbose = FALSE) {
  # only for binary
  if (!identical(CLASSIFICATION_MODE, "binary")) {
    if (verbose) message("[CONFUSION_MATRIX] Only valid for binary classification.")
    return(NULL)
  }
  
  # resolve threshold robustly: arg > SONN$threshold > 0.5
  thr <- if (!is.null(threshold)) {
    as.numeric(threshold)[1]
  } else if (!is.null(SONN$threshold)) {
    as.numeric(SONN$threshold)[1]
  } else {
    0.5
  }
  
  # normalize shapes/types
  if (!is.matrix(labels)) labels <- matrix(as.numeric(labels), ncol = 1)
  if (!is.matrix(predicted_output)) predicted_output <- matrix(as.numeric(predicted_output), ncol = 1)
  
  preds   <- as.integer(predicted_output >= thr)
  actuals <- as.integer(labels)
  
  TP <- sum(preds == 1 & actuals == 1, na.rm = TRUE)
  FP <- sum(preds == 1 & actuals == 0, na.rm = TRUE)
  TN <- sum(preds == 0 & actuals == 0, na.rm = TRUE)
  FN <- sum(preds == 0 & actuals == 1, na.rm = TRUE)
  
  if (verbose) message(sprintf("[CONFUSION_MATRIX] thr=%.3f  TP=%d FP=%d TN=%d FN=%d", thr, TP, FP, TN, FN))
  
  list(TP = TP, FP = FP, TN = TN, FN = FN)
}


# =========================
# Global threshold helpers
# =========================

DDESONN_set_threshold <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value <= 0 || value >= 1) {
    stop("[DDESONN_set_threshold] value must be a single numeric in (0,1).")
  }
  .GlobalEnv$DESONN_THRESHOLDS <- list(binary = as.numeric(value))
  invisible(TRUE)
}

DDESONN_get_threshold <- function(default = NA_real_) {
  x <- get0("DDESONN_THRESHOLDS", envir = .GlobalEnv, inherits = FALSE)
  if (is.null(x) || is.null(x$binary)) return(default)
  as.numeric(x$binary)
}

DDESONN_clear_threshold <- function() {
  if (exists("DDESONN_THRESHOLDS", envir = .GlobalEnv, inherits = FALSE)) {
    rm("DDESONN_THRESHOLDS", envir = .GlobalEnv)
  }
  invisible(TRUE)
}

# =========================
# accuracy_precision_recall_f1_tuned (revised)
# =========================
accuracy_precision_recall_f1_tuned <- function(
    SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output,
    metric_for_tuning = c("accuracy","f1","precision","recall",
                          "macro_f1","macro_precision","macro_recall"),
    threshold_grid = seq(0.05, 0.95, by = 0.01),
    verbose = FALSE
) {
  dbg <- function(...) {
    if (verbose) message(paste(..., collapse = " "))
  }
  metric_for_tuning <- match.arg(metric_for_tuning)
  
  # --- helpers ---
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  infer_mode <- function(L, P) if (max(ncol(L), ncol(P)) > 1L) "multiclass" else "binary"
  
  sanitize_grid_simple <- function(g) {
    if (is.function(g)) {
      attempt <- try(g(), silent = TRUE)
      g <- if (!inherits(attempt, "try-error")) attempt else seq(0.05, 0.95, by = 0.01)
    }
    if (is.language(g) || is.symbol(g)) {
      attempt <- try(eval(g, parent.frame()), silent = TRUE)
      g <- if (!inherits(attempt, "try-error")) attempt else seq(0.05, 0.95, by = 0.01)
    }
    if (is.list(g)) g <- unlist(g, use.names = FALSE)
    g <- suppressWarnings(as.numeric(g))
    g <- g[is.finite(g)]
    g <- sort(unique(g))
    g <- g[g > 0 & g < 1]
    if (!length(g)) g <- seq(0.05, 0.95, by = 0.01)
    g
  }
  
  # --- coerce & trim ---
  L <- coerce_to_numeric_matrix(labels, allow_model_matrix = FALSE)
  P <- coerce_to_numeric_matrix(predicted_output, allow_model_matrix = FALSE)
  n <- min(nrow(L), nrow(P))
  if (n == 0L) stop("[accuracy_precision_recall_f1_tuned] empty inputs after trim.")
  L <- L[seq_len(n), , drop = FALSE]
  P <- P[seq_len(n), , drop = FALSE]
  
  # --- resolve mode ---
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(L, P)
  thr_grid <- sanitize_grid_simple(threshold_grid)
  
  # =========================
  # ===== REGRESSION ========
  # =========================
  if (identical(mode, "regression")) {
    return(list(
      accuracy  = NA_real_,
      precision = NA_real_,
      recall    = NA_real_,
      f1        = NA_real_,
      confusion_matrix = NULL,
      details = list(
        best_threshold = NA_real_,
        y_pred_class   = NA,
        grid_used      = thr_grid,
        tuned_by       = "n/a"
      )
    ))
  }
  
  # ============================
  # ======== BINARY PATH ========
  # ============================
  if (identical(mode, "binary")) {
    # --- Labels (expecting 0/1) ---
    y_true <- if (ncol(L) == 1L) {
      v <- as.numeric(L[,1])
      if (all(v %in% c(0,1))) as.integer(v) else as.integer(v >= 0.5)
    } else {
      as.integer(max.col(L, ties.method = "first") - 1L)
    }
    
    # --- Probabilities (1-col sigmoid output) ---
    if (ncol(P) != 1L) {
      stop("[accuracy_precision_recall_f1_tuned] Binary mode expects 1-column probabilities (sigmoid). Got ", ncol(P), " columns.")
    }
    p_pos <- as.numeric(P[,1])
    
    # --- Global threshold logic ---
    global_th <- DDESONN_get_threshold(default = NA_real_)
    tuned_now <- FALSE
    
    if (is.na(global_th)) {
      dbg("[accuracy_precision_recall_f1_tuned] No global threshold set. Tuning on provided data (use validation here).")
      metrics <- c("accuracy","f1","precision","recall")
      if (!(metric_for_tuning %in% metrics)) {
        metric_for_tuning <- switch(metric_for_tuning,
                                    macro_f1="f1",
                                    macro_precision="precision",
                                    macro_recall="recall",
                                    metric_for_tuning)
      }
      
      best <- list(th = 0.5, score = -Inf)
      for (th in thr_grid) {
        preds <- as.integer(p_pos >= th)
        TP <- sum(preds == 1L & y_true == 1L)
        FP <- sum(preds == 1L & y_true == 0L)
        TN <- sum(preds == 0L & y_true == 0L)
        FN <- sum(preds == 0L & y_true == 1L)
        
        acc <- (TP + TN) / length(y_true)
        pre <- if ((TP + FP) > 0) TP / (TP + FP) else 0
        rec <- if ((TP + FN) > 0) TP / (TP + FN) else 0
        f1  <- if ((pre + rec) > 0) 2 * pre * rec / (pre + rec) else 0
        
        score <- switch(metric_for_tuning,
                        accuracy = acc, f1 = f1, precision = pre, recall = rec)
        if (score > best$score || (abs(score - best$score) < .Machine$double.eps^0.5 &&
                                   abs(th - 0.5) < abs(best$th - 0.5))) {
          best <- list(th=th, score=score)
        }
      }
      DDESONN_set_threshold(best$th)
      global_th <- best$th
      tuned_now <- TRUE
      dbg(sprintf("[accuracy_precision_recall_f1_tuned] Tuned global threshold = %.4f (metric=%s)",
                  global_th, metric_for_tuning))
    } else {
      dbg(sprintf("[accuracy_precision_recall_f1_tuned] Using existing global threshold = %.4f.", global_th))
    }
    
    # --- Apply chosen threshold (tuned or stored) ---
    cm <- confusion_matrix(
      SONN = SONN,
      labels = matrix(y_true, ncol = 1),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1),
      threshold = global_th,
      verbose = FALSE
    )
    
    TP <- cm$TP; FP <- cm$FP; TN <- cm$TN; FN <- cm$FN
    
    total <- length(y_true)
    acc <- (TP + TN) / total
    pre <- if ((TP + FP) > 0) TP / (TP + FP) else 0
    rec <- if ((TP + FN) > 0) TP / (TP + FN) else 0
    f1  <- if ((pre + rec) > 0) 2 * pre * rec / (pre + rec) else 0
    
    y_pred_class <- as.integer(p_pos >= global_th)
    
    return(list(
      accuracy  = as.numeric(acc),
      precision = as.numeric(pre),
      recall    = as.numeric(rec),
      f1        = as.numeric(f1),
      confusion_matrix = cm,
      details   = list(
        best_threshold = as.numeric(global_th),
        y_pred_class   = y_pred_class,
        grid_used      = thr_grid,
        tuned_by       = if (tuned_now) metric_for_tuning else "applied-global"
      )
    ))
  }
  
  # ===============================
  # ===== MULTICLASS (argmax) =====
  # ===============================
  Kp <- ncol(P); Kl <- ncol(L)
  if (Kp > 1L && Kl > 1L && Kp != Kl) {
    K <- min(Kp, Kl)
    Pk <- P[, seq_len(K), drop = FALSE]
    Lk <- L[, seq_len(K), drop = FALSE]
  } else {
    Pk <- if (Kp > 1L) P else NULL
    Lk <- if (Kl > 1L) L else NULL
  }
  
  true_ids <- if (!is.null(Lk)) {
    max.col(Lk, ties.method = "first")
  } else {
    K <- max(2L, Kp)
    v <- as.integer(round(L[,1]))
    if (length(v) && min(v, na.rm = TRUE) == 0L) v <- v + 1L
    v[v < 1L] <- 1L; v[v > K] <- K
    v
  }
  pred_ids <- if (!is.null(Pk)) max.col(Pk, ties.method = "first") else rep(1L, length(true_ids))
  
  acc <- mean(pred_ids == true_ids, na.rm = TRUE)
  
  # macro precision/recall/F1
  K <- max(true_ids, pred_ids, na.rm = TRUE)
  macro_prec <- macro_rec <- macro_f1 <- numeric(K)
  for (k in seq_len(K)) {
    TPk <- sum(pred_ids == k & true_ids == k)
    FPk <- sum(pred_ids == k & true_ids != k)
    FNk <- sum(pred_ids != k & true_ids == k)
    prec <- if ((TPk + FPk) > 0) TPk / (TPk + FPk) else 0
    rec  <- if ((TPk + FNk) > 0) TPk / (TPk + FNk) else 0
    f1   <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0
    macro_prec[k] <- prec; macro_rec[k] <- rec; macro_f1[k] <- f1
  }
  
  list(
    accuracy  = round(as.numeric(acc), 8),
    precision = mean(macro_prec),
    recall    = mean(macro_rec),
    f1        = mean(macro_f1),
    confusion_matrix = NULL,   # only binary returns list(TP,FP,TN,FN)
    details   = list(
      best_threshold = NA_real_,
      y_pred_class   = pred_ids,
      grid_used      = thr_grid,
      tuned_by       = "argmax-macro"
    )
  )
}





# Mean Absolute Error (silent) — uses CLASSIFICATION_MODE for safe shape handling
# Signature: MAE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
MAE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  # Coerce to numeric matrices
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  
  # Row alignment (no recycling)
  n_common <- min(nrow(lbl), nrow(pred))
  if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # Mode-based shape harmonization
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "regression"))
  
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(lbl[, 1])))
      if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
        lbl <- lbl + 1
      }
      lbl <- safe_one_hot_matrix(lbl[, 1], K)
    } else if (ncol(lbl) != K) {
      return(NA_real_)
    }
    
  } else if (identical(mode, "binary")) {
    # pred: N×1 (p_pos) OR N×2 ([p_neg,p_pos]); lbl: N×1 (0/1 or two values) OR N×2 one-hot
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      lbl <- matrix(lbl[, 2], ncol = 1L)
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      v <- as.integer(round(lbl[, 1])); v[is.na(v)] <- 0L; v[v != 0L] <- 1L
      lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      uniq <- sort(unique(as.integer(lbl[, 1])))
      if (!all(uniq %in% c(0L, 1L))) {
        if (length(uniq) == 2L) {
          lbl[, 1] <- ifelse(lbl[, 1] == uniq[2], 1, 0)
        } else {
          return(NA_real_)
        }
      }
    } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) {
      return(NA_real_)
    }
    
  } else if (!identical(mode, "regression")) {
    return(NA_real_)
  }
  
  # Final column check
  if (!identical(mode, "regression") && ncol(lbl) != ncol(pred)) return(NA_real_)
  
  # Only in regression allow replicate/truncate to conform columns
  if (identical(mode, "regression") && ncol(lbl) != ncol(pred)) {
    total_needed <- nrow(lbl) * ncol(lbl)
    if (ncol(pred) < ncol(lbl)) {
      vec <- rep(pred, length.out = total_needed)
      pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    } else {
      pred <- pred[, 1:ncol(lbl), drop = FALSE]
      pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    }
  }
  
  # NA handling
  cc <- stats::complete.cases(cbind(lbl, pred))
  if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]
  pred <- pred[cc, , drop = FALSE]
  if (!nrow(lbl)) return(NA_real_)
  
  # Compute MAE
  mae <- mean(abs(pred - lbl))
  mae
}

# NDCG (Normalized Discounted Cumulative Gain)
ndcg <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, labels, verbose = FALSE) {
  # Convert to data frame if needed
  Rdata <- data.frame(Rdata)
  
  # Define relevance scores: assume binary relevance (1 for "relevant", 0 otherwise)
  relevance_scores <- ifelse(labels == "relevant", 1, 0)
  
  # Create a data frame to pair predicted outputs with relevance
  df <- data.frame(
    prediction = predicted_output,
    relevance  = relevance_scores
  )
  
  # Sort by prediction score descending (as if ranked)
  df_sorted <- df[order(-df$prediction), ]
  
  # Compute DCG
  gains <- (2^df_sorted$relevance - 1) / log2(1 + seq_along(df_sorted$relevance))
  dcg <- sum(gains)
  
  # Compute ideal DCG (perfect ranking)
  ideal_relevance <- sort(df$relevance, decreasing = TRUE)
  ideal_gains <- (2^ideal_relevance - 1) / log2(1 + seq_along(ideal_relevance))
  idcg <- sum(ideal_gains)
  
  # Avoid division by zero
  ndcg_value <- if (idcg == 0) 0 else dcg / idcg
  
  return(ndcg_value)
}


# -------------------------
# Custom Relative Error (binned), mode-aware (silent)
# Signature:
#   custom_relative_error_binned(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose=FALSE)
# Behavior:
#   - regression (default): elementwise |pred - label| / |label| with epsilon guard, then binned.
#   - binary: per-row error = |1 - p_true|, where p_true = p(pos) if y=1; else 1 - p(pos) if y=0.
#   - multiclass: per-row error = |1 - p_true| where p_true is probability of the true class.
# Returns: named LIST of bin means.
custom_relative_error_binned <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  # --- helpers ---
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  infer_mode <- function(L, P) {
    if (max(ncol(L), ncol(P)) > 1L) "multiclass" else "regression"
  }
  
  # bin edges: 0%..100% -> 0..1
  bins <- c(0, 0.05, 0.10, 0.50, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  brks <- bins / 100
  bin_names <- paste0("rel_", bins[-length(bins)], "_", bins[-1], "pct")
  
  # empty-output helper
  empty_out <- function(val = NA_real_) {
    x <- rep(val, length(bin_names))
    names(x) <- bin_names
    as.list(x)
  }
  
  # --- coerce & trim to common rows ---
  L <- coerce_to_numeric_matrix(labels, allow_model_matrix = FALSE)
  P <- coerce_to_numeric_matrix(predicted_output, allow_model_matrix = FALSE)
  n <- min(nrow(L), nrow(P))
  if (n == 0L) return(empty_out())
  
  L <- L[seq_len(n), , drop = FALSE]
  P <- P[seq_len(n), , drop = FALSE]
  
  # --- resolve mode ---
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(L, P)
  
  # --- compute vector of per-sample errors ---
  if (identical(mode, "binary")) {
    # labels -> y_true in {0,1}
    if (ncol(L) == 2L) {
      y_true <- as.integer(L[, 2] >= L[, 1])
    } else {
      v <- as.numeric(L[, 1])
      if (all(v %in% c(0, 1))) y_true <- as.integer(v) else y_true <- as.integer(v >= 0.5)
    }
    # preds -> p(pos)
    p_pos <- if (ncol(P) >= 2L) as.numeric(P[, 2]) else as.numeric(P[, 1])
    p_pos[!is.finite(p_pos)] <- NA_real_
    
    # p_true = p(pos) if y=1, else 1 - p(pos)
    p_true <- ifelse(y_true == 1L, p_pos, 1 - p_pos)
    err <- abs(1 - p_true)          # in [0,1]
    vals <- err[is.finite(err)]
    
  } else if (identical(mode, "multiclass")) {
    # Determine K (prefer predictions)
    Kp <- ncol(P); Kl <- ncol(L)
    if (Kp > 1L && Kl > 1L && Kp != Kl) {
      K  <- min(Kp, Kl)
      Pk <- P[, seq_len(K), drop = FALSE]
      Lk <- L[, seq_len(K), drop = FALSE]
    } else {
      Pk <- if (Kp > 1L) P else NULL
      Lk <- if (Kl > 1L) L else NULL
    }
    
    # true class id (1..K)
    if (!is.null(Lk)) {
      true_ids <- max.col(Lk, ties.method = "first")
      K <- ncol(Lk)
    } else {
      K <- max(2L, Kp)  # at least 2
      v <- as.integer(round(L[, 1]))
      if (length(v) && min(v, na.rm = TRUE) == 0L) v <- v + 1L
      v[v < 1L] <- 1L; v[v > K] <- K
      true_ids <- v
    }
    
    # probability assigned to true class
    if (!is.null(Pk)) {
      idx <- cbind(seq_len(nrow(Pk)), true_ids)
      p_true <- as.numeric(Pk[idx])
    } else {
      # degenerate: no multi-prob predictions; pretend all prob on class 1
      p_true <- as.numeric(true_ids == 1L)
    }
    p_true[!is.finite(p_true)] <- NA_real_
    
    err  <- abs(1 - p_true)         # in [0,1]
    vals <- err[is.finite(err)]
    
  } else {  # regression (default)
    # Align shapes (replicate/truncate columns)
    if (ncol(L) != ncol(P)) {
      if (ncol(P) < ncol(L)) {
        total <- nrow(L) * ncol(L)
        Pm <- matrix(rep(P, length.out = total), nrow = nrow(L), ncol = ncol(L), byrow = FALSE)
      } else {
        Pm <- matrix(P[, 1:ncol(L)], nrow = nrow(L), ncol = ncol(L), byrow = FALSE)
      }
    } else {
      Pm <- P
    }
    error_prediction <- Pm - L
    denom <- abs(L)
    denom[denom == 0 | !is.finite(denom)] <- .Machine$double.eps
    percentage_difference <- as.numeric(abs(error_prediction) / denom)
    vals <- percentage_difference[is.finite(percentage_difference)]
  }
  
  # --- binning ---
  mean_precisions <- numeric(length(brks) - 1)
  if (length(vals) > 0L) {
    idx <- findInterval(vals, brks, rightmost.closed = TRUE, all.inside = TRUE)
    for (j in seq_along(mean_precisions)) {
      v <- vals[idx == j]
      mean_precisions[j] <- if (length(v)) mean(v) else 0
    }
  } else {
    mean_precisions[] <- 0
  }
  
  names(mean_precisions) <- bin_names
  list(custom_relative_error_binned = mean_precisions)
}


# Diversity (Shannon entropy) — robust
# Signature kept: diversity(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose = FALSE)
diversity <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  mode <- tolower(as.character(CLASSIFICATION_MODE %||% "multiclass"))
  eps  <- .Machine$double.eps
  
  # Not meaningful for regression
  if (identical(mode, "regression")) return(NA_real_)
  
  # Helper: safe entropy of a probability vector
  H_vec <- function(p) {
    p <- as.numeric(p)
    if (!length(p)) return(NA_real_)
    p[!is.finite(p) | p < 0] <- 0
    s <- sum(p)
    if (!is.finite(s) || s <= 0) return(NA_real_)
    p <- p / s
    p <- pmax(p, eps)
    -sum(p * log2(p))
  }
  
  # If matrix: treat rows as class distributions (preferred for softmax outputs)
  if (is.matrix(predicted_output)) {
    P <- predicted_output
    storage.mode(P) <- "double"
    P[!is.finite(P)] <- 0
    
    if (ncol(P) == 1L) {
      # Binary prob column → expand to [p_neg, p_pos]
      p1 <- pmin(pmax(P[, 1], 0), 1)       # clamp to [0,1]
      P  <- cbind(1 - p1, p1)
    }
    
    # Row-wise normalization to probabilities
    row_sums <- rowSums(P)
    bad <- !is.finite(row_sums) | row_sums <= 0
    if (any(bad)) {
      # fallback to uniform for bad rows
      P[bad, ] <- 1 / ncol(P)
      row_sums <- rowSums(P)
    }
    P <- P / row_sums
    
    # Per-row Shannon entropy, then aggregate (mean)
    P <- pmax(P, eps)
    H_row <- -rowSums(P * log2(P))
    H <- mean(H_row, na.rm = TRUE)
    if (!is.finite(H)) return(NA_real_)
    return(H)
  }
  
  # If vector:
  v <- as.vector(predicted_output)
  
  # (a) If it looks like class labels (character/factor or integer-ish):
  if (is.factor(v) || is.character(v)) {
    p <- as.numeric(table(v))
    return(H_vec(p))
  }
  
  # (b) Numeric vector → treat as scores/prob mass; normalize safely
  v <- as.numeric(v)
  v[!is.finite(v) | v < 0] <- 0
  if (!length(v) || sum(v) <= 0) return(NA_real_)
  return(H_vec(v))
}

# Generalization ability — train/test split evaluation
# Signature: generalization_ability(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE)
generalization_ability <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
  mode   <- tolower(as.character(CLASSIFICATION_MODE %||% "multiclass"))
  
  if (verbose) message("[GEN] generalization_ability start")
  
  # Guard: if no labels, cannot compute
  if (is.null(labels) || length(labels) != nrow(Rdata)) {
    if (verbose) message("[GEN] labels missing or mismatched — returning NA")
    return(NA_real_)
  }
  
  n <- nrow(Rdata)
  if (n < 2L) return(NA_real_)
  
  # Split train/test (80/20)
  set.seed(123L)
  idx_train <- sample.int(n, max(1L, floor(0.8 * n)))
  idx_test  <- setdiff(seq_len(n), idx_train)
  
  X_train <- as.matrix(Rdata[idx_train, , drop = FALSE])
  X_test  <- as.matrix(Rdata[idx_test, , drop = FALSE])
  y_train <- labels[idx_train]
  y_test  <- labels[idx_test]
  
  # Predict safely
  safe_pred <- function(X) {
    out <- tryCatch(SONN$predict(X), error = function(e) NA)
    if (is.list(out) && !is.null(out$predicted_output)) out$predicted_output else out
  }
  pred_train <- safe_pred(X_train)
  pred_test  <- safe_pred(X_test)
  
  metric <- NA_real_
  
  if (identical(mode, "regression")) {
    # RMSE
    if (!anyNA(pred_test) && length(pred_test) == length(y_test)) {
      pred_test <- as.numeric(pred_test)
      y_test    <- as.numeric(y_test)
      metric <- sqrt(mean((pred_test - y_test)^2, na.rm = TRUE))
    }
  } else {
    # Accuracy
    to_labels <- function(P) {
      if (is.matrix(P)) max.col(P, ties.method = "first")
      else as.integer(as.numeric(P) >= 0.5)
    }
    if (!anyNA(pred_test)) {
      metric <- mean(to_labels(pred_test) == y_test, na.rm = TRUE)
    }
  }
  
  if (verbose) message("[GEN] generalization_ability complete: ", metric)
  return(metric)
}

# Root Mean Squared Error (silent + mode-aware)
# Signature: RMSE(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose)
RMSE <- function(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, verbose) {
  # helpers
  `%||%` <- function(x, y) if (is.null(x)) y else x
  is_valid_mode <- function(x) is.character(x) && length(x) == 1L &&
    tolower(x) %in% c("binary","multiclass","regression")
  infer_mode <- function(lbl, pred) {
    if (ncol(pred) > 1L) return("multiclass")
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (length(u) <= 2L) return("binary")
    }
    "regression"
  }
  
  # argument recovery (handle swapped positional args) — silent
  if (!is_valid_mode(CLASSIFICATION_MODE)) {
    if (!missing(predicted_output) && is.logical(predicted_output) && length(predicted_output) == 1L) {
      verbose <- predicted_output
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    } else if (missing(predicted_output)) {
      predicted_output <- CLASSIFICATION_MODE
      CLASSIFICATION_MODE <- NULL
    }
  }
  
  # coerce to numeric matrices
  lbl  <- coerce_to_numeric_matrix(labels)
  pred <- coerce_to_numeric_matrix(predicted_output)
  
  # row alignment (NO recycling)
  n_common <- min(nrow(lbl), nrow(pred))
  if (n_common == 0L) return(NA_real_)
  lbl  <- lbl[seq_len(n_common), , drop = FALSE]
  pred <- pred[seq_len(n_common), , drop = FALSE]
  
  # resolve/infer mode
  mode <- if (is_valid_mode(CLASSIFICATION_MODE)) tolower(CLASSIFICATION_MODE) else infer_mode(lbl, pred)
  
  # mode-based shape harmonization
  if (identical(mode, "multiclass")) {
    K <- ncol(pred)
    if (K < 2L) return(NA_real_)
    if (ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(lbl[,1])))
      if (length(u) && min(u, na.rm = TRUE) == 0L && max(u, na.rm = TRUE) == (K - 1L)) {
        lbl <- lbl + 1
      }
      lbl <- safe_one_hot_matrix(lbl[,1], K)
    } else if (ncol(lbl) != K) {
      return(NA_real_)
    }
  } else if (identical(mode, "binary")) {
    if (ncol(pred) == 1L && ncol(lbl) == 2L) {
      lbl <- matrix(lbl[,2], ncol = 1L)  # collapse one-hot to 0/1
    } else if (ncol(pred) == 2L && ncol(lbl) == 1L) {
      v <- as.integer(round(lbl[,1])); v[is.na(v)] <- 0L; v[v != 0L] <- 1L
      lbl <- cbind(1 - v, v); storage.mode(lbl) <- "double"
    } else if (ncol(pred) == 1L && ncol(lbl) == 1L) {
      u <- sort(unique(as.integer(round(lbl[,1]))))
      if (!all(u %in% c(0L,1L))) {
        if (length(u) == 2L) {
          lbl[,1] <- ifelse(lbl[,1] == max(u), 1, 0)
        } else {
          return(NA_real_)
        }
      }
    } else if (!(ncol(pred) == 2L && ncol(lbl) == 2L)) {
      return(NA_real_)
    }
  } else if (!identical(mode, "regression")) {
    return(NA_real_)
  }
  
  # final column check
  if (!identical(mode, "regression") && ncol(lbl) != ncol(pred)) return(NA_real_)
  if (identical(mode, "regression") && ncol(lbl) != ncol(pred)) {
    total_needed <- nrow(lbl) * ncol(lbl)
    if (ncol(pred) < ncol(lbl)) {
      vec <- rep(pred, length.out = total_needed)
      pred <- matrix(vec, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    } else {
      pred <- pred[, 1:ncol(lbl), drop = FALSE]
      pred <- matrix(pred, nrow = nrow(lbl), ncol = ncol(lbl), byrow = FALSE)
    }
  }
  
  # NA handling
  cc <- stats::complete.cases(cbind(lbl, pred))
  if (!any(cc)) return(NA_real_)
  lbl  <- lbl[cc, , drop = FALSE]
  pred <- pred[cc, , drop = FALSE]
  if (!nrow(lbl)) return(NA_real_)
  
  # compute RMSE
  rmse <- sqrt(mean((pred - lbl)^2))
  rmse
}


# Serendipity
serendipity <- function(SONN, Rdata, CLASSIFICATION_MODE, predicted_output, verbose = FALSE) {
  # Calculate the average number of times each prediction is made
  prediction_counts <- table(predicted_output)
  
  # Calculate the inverse of the prediction counts
  inverse_prediction_counts <- 1 / prediction_counts
  
  # Return the average inverse prediction count
  mean(inverse_prediction_counts, na.rm = TRUE)
}
