# Compare distinct tabular predictor vectors across splits, ignoring labels.
.split_overlap <- function(train, validation = NULL, test = NULL) {
  splits <- list(train, validation, test)
  signatures <- lapply(splits, function(x) {
    if (!NROW(x) || !NCOL(x)) return(NULL)
    unique(apply(as.matrix(x), 1L, function(row) {
      values <- as.double(row)
      values[which(values == 0)] <- 0 # Make positive and negative zero equivalent.
      digest::digest(values, algo = "sha256")
    }))
  })
  count <- function(i, j) {
    if (is.null(signatures[[i]]) || is.null(signatures[[j]]) ||
        NCOL(splits[[i]]) != NCOL(splits[[j]])) return(NA_integer_)
    length(intersect(signatures[[i]], signatures[[j]]))
  }
  c(train_validation = count(1L, 2L), train_test = count(1L, 3L),
    validation_test = count(2L, 3L))
}

.overlap_status <- function(counts) {
  if (any(counts > 0L, na.rm = TRUE)) return("detected")
  if (!length(counts) || all(is.na(counts))) return("unavailable")
  if (anyNA(counts)) return("partial")
  "clear"
}

.overlap_text <- function(counts) {
  status <- .overlap_status(counts)
  label <- switch(status, detected = "Exact predictor overlap detected",
    unavailable = "Check unavailable",
    partial = "No exact predictor overlap in comparable splits",
    clear = "No exact predictor overlap detected")
  if (status == "unavailable") return(label)
  pairs <- c("Train-Validation", "Train-Test", "Validation-Test")
  paste0(label, " (distinct shared vectors: ", paste(paste(pairs, ifelse(is.na(counts),
    "unavailable", counts)), collapse = ", "), ")")
}
