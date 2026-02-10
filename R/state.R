# Private package state for debug and intermediate artifacts
.ddesonn_state <- new.env(parent = emptyenv())

#' Inspect internal DDESONN debug state
#'
#' Returns a named list snapshot of objects currently stored in the private
#' package debug/state environment.
#'
#' @return A named list of objects from DDESONN internal state.
#' @export
ddesonn_debug_state <- function() {
  as.list.environment(.ddesonn_state, all.names = TRUE)
}
