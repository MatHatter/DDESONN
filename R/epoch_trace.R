.trace_methods <- c("train", "train_network", "learn", "predict")

.validate_trace_epochs <- function(trace_epochs) {
  if (is.null(trace_epochs)) return(invisible(NULL))
  methods <- c("all", .trace_methods)
  if (!is.list(trace_epochs) || (length(trace_epochs) &&
      (is.null(names(trace_epochs)) || anyDuplicated(names(trace_epochs)) ||
       any(!names(trace_epochs) %in% methods)))) {
    stop("'trace_epochs' must be a named list with unique method names: ",
      paste(methods, collapse = ", "), ".", call. = FALSE)
  }
  for (method in names(trace_epochs)) {
    value <- trace_epochs[[method]]
    if (is.logical(value) && length(value) == 1L && !is.na(value)) next
    if (identical(method, "all"))
      stop("'trace_epochs$all' must be TRUE or FALSE.", call. = FALSE)
    if (is.numeric(value) && length(value) && all(is.finite(value)) &&
        all(value >= 1 & value == floor(value))) next
    stop("'trace_epochs$", method, "' must be TRUE, FALSE, or positive whole epochs.",
      call. = FALSE)
  }
  invisible(NULL)
}

.resolve_trace_epochs <- function(trace_epochs) {
  all_methods <- trace_epochs$all
  if (is.null(all_methods)) return(trace_epochs)
  defaults <- stats::setNames(rep(list(all_methods), length(.trace_methods)), .trace_methods)
  selected <- intersect(names(trace_epochs), .trace_methods)
  defaults[selected] <- trace_epochs[selected]
  defaults
}

# NULL preserves unfiltered logging. NA suppresses the selected method's marker.
.trace_epoch <- function(selection, epoch) {
  if (is.null(selection)) return(NULL)
  if (isTRUE(selection) || (is.numeric(selection) && epoch %in% selection)) return(epoch)
  NA_integer_
}

.trace_boundary <- function(method, phase, legacy, epoch = NULL, started = NULL) {
  if (is.null(epoch)) {
    if (isTRUE(legacy)) print(paste0(strrep("-", 40), method, "-", phase, strrep("-", 40)))
  } else if (!is.na(epoch)) {
    now <- proc.time()[["elapsed"]]
    elapsed <- if (!is.null(started)) sprintf(" | elapsed=%.3f s", now - started) else ""
    cat(sprintf("[TRACE] %s-%s | epoch=%d%s\n", method, phase, epoch, elapsed))
    return(invisible(now))
  }
  invisible(NULL)
}
