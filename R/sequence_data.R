#' Build causal fixed-length market sequences
#'
#' Each output row ends at the corresponding input row; no future observation
#' is ever used.  Incomplete leading windows are either rejected or padded.
#' @param data Numeric matrix/data frame in chronological order.
#' @param sequence_length Positive window length.
#' @param pad Whether to left-pad incomplete windows with `NA_real_`.
#' @return Numeric array (`samples x sequence_length x features`).
#' @export
ddesonn_sequence_data <- function(data, sequence_length = 48L, pad = FALSE) {
  x <- as.matrix(data)
  storage.mode(x) <- "double"
  sequence_length <- as.integer(sequence_length)
  if (length(sequence_length) != 1L || is.na(sequence_length) || sequence_length < 1L)
    stop("sequence_length must be one positive integer.", call. = FALSE)
  if (!nrow(x) || !ncol(x) || any(!is.finite(x)))
    stop("data must be a non-empty, finite numeric matrix.", call. = FALSE)
  if (!pad && nrow(x) < sequence_length)
    stop("data has fewer rows than sequence_length.", call. = FALSE)
  starts <- if (pad) seq_len(nrow(x)) else seq.int(sequence_length, nrow(x))
  out <- array(NA_real_, c(length(starts), sequence_length, ncol(x)))
  for (i in seq_along(starts)) {
    end <- starts[i]; from <- max(1L, end - sequence_length + 1L)
    block <- x[from:end, , drop = FALSE]
    out[i, (sequence_length - nrow(block) + 1L):sequence_length, ] <- block
  }
  attr(out, "end_index") <- starts
  out
}

.ddesonn_validate_sequence <- function(x, sequence_length = NULL, samples = NULL) {
  if (!is.array(x) || length(dim(x)) != 3L || !is.numeric(x))
    stop("sequence_data must be a numeric samples x sequence_length x sequence_features array.", call. = FALSE)
  d <- dim(x)
  if (any(d < 1L) || any(!is.finite(x))) stop("sequence_data must be non-empty and finite.", call. = FALSE)
  if (!is.null(sequence_length) && d[2L] != sequence_length) stop("sequence_data length does not match sequence_length.", call. = FALSE)
  if (!is.null(samples) && d[1L] != samples) stop("sequence_data sample count must match x.", call. = FALSE)
  invisible(d)
}
