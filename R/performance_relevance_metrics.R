# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#       _     _      _     _      _     _      _     _      _     _      _     _      _     _   $$$$$$$$$$$$$$$$$$$$$$$
#     (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)   $$$$$$$$$$$$$$$$$$$$$$$
#      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \      / ._. \    $$$$$$$$$$$$$$$$$$$$$$$
#   __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__   $$$$$$$$$$$$$$$$$$$$$$$
#  (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)  $$$$$$$$$$$$$$$$$$$$$$$
#    || M ||      || E ||      || T ||      || R ||      || I ||      || C ||      || S ||      $$$$$$$$$$$$$$$$$$$$$$$
# _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._     $$$$$$$$$$$$$$$$$$$$$$$
#(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-`\.-.)    $$$$$$$$$$$$$$$$$$$$$$$
# `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'     $$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



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
  W <- as.matrix(W)                    # <b