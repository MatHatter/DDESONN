# ============================================================
# FILE: R/aliases.R
# FULL FIXED — LEGACY UPPERCASE ALIASES (CRAN SAFE)
# - PRESERVED structure
# - FIX: attach aliases to lowercase Rd via @rdname
# - FIX: explicitly document ... to eliminate Rd warnings
# - FIX: remove @inheritParams (was failing; alias only has ...)
# - FIX: avoid unresolved roxygen links for plots_dir during document()
# ============================================================


#' Legacy alias for [ddesonn_activation_defaults()]
#'
#' @rdname ddesonn_activation_defaults
#' @aliases DDESONN_activation_defaults
#' @param ... Additional arguments passed through to
#'   [ddesonn_activation_defaults()].
#' @return Same as [ddesonn_activation_defaults()].
#' @export
DDESONN_activation_defaults <- function(...) {
  ddesonn_activation_defaults(...)
}


#' Legacy alias for [ddesonn_artifacts_root()]
#'
#' @rdname ddesonn_artifacts_root
#' @aliases DDESONN_artifacts_root
#' @param ... Additional arguments passed through to
#'   [ddesonn_artifacts_root()].
#' @return Same as [ddesonn_artifacts_root()].
#' @export
DDESONN_artifacts_root <- function(...) {
  ddesonn_artifacts_root(...)
}


#' Legacy alias for [ddesonn_dropout_defaults()]
#'
#' @rdname ddesonn_dropout_defaults
#' @aliases DDESONN_dropout_defaults
#' @param ... Additional arguments passed through to
#'   [ddesonn_dropout_defaults()].
#' @return Same as [ddesonn_dropout_defaults()].
#' @export
DDESONN_dropout_defaults <- function(...) {
  ddesonn_dropout_defaults(...)
}


#' Legacy alias for [ddesonn_fit()]
#'
#' @rdname ddesonn_fit
#' @aliases DDESONN_fit
#' @param ... Additional arguments passed through to
#'   [ddesonn_fit()].
#' @return Same as [ddesonn_fit()].
#' @export
DDESONN_fit <- function(...) {
  ddesonn_fit(...)
}


#' Legacy alias for [ddesonn_model()]
#'
#' @rdname ddesonn_model
#' @aliases DDESONN_model
#' @param ... Additional arguments passed through to
#'   [ddesonn_model()].
#' @return Same as [ddesonn_model()].
#' @export
DDESONN_model <- function(...) {
  ddesonn_model(...)
}


#' Legacy alias for [ddesonn_optimizer_options()]
#'
#' @rdname ddesonn_optimizer_options
#' @aliases DDESONN_optimizer_options
#' @param ... Additional arguments passed through to
#'   [ddesonn_optimizer_options()].
#' @return Same as [ddesonn_optimizer_options()].
#' @export
DDESONN_optimizer_options <- function(...) {
  ddesonn_optimizer_options(...)
}


#' Legacy alias for `ddesonn_plots_dir()`
#'
#' @rdname ddesonn_plots_dir
#' @aliases DDESONN_plots_dir
#' @param ... Additional arguments passed through to
#'   `ddesonn_plots_dir()`.
#' @return Same as `ddesonn_plots_dir()`.
#' @export
DDESONN_plots_dir <- function(...) {
  ddesonn_plots_dir(...)
}


#' Legacy alias for [ddesonn_predict()]
#'
#' @rdname ddesonn_predict
#' @aliases DDESONN_predict
#' @param ... Additional arguments passed through to
#'   [ddesonn_predict()].
#' @return Same as [ddesonn_predict()].
#' @export
DDESONN_predict <- function(...) {
  ddesonn_predict(...)
}


#' Legacy alias for [ddesonn_run()]
#'
#' @rdname ddesonn_run
#' @aliases DDESONN_run
#' @param ... Additional arguments passed through to
#'   [ddesonn_run()].
#' @return Same as [ddesonn_run()].
#' @export
DDESONN_run <- function(...) {
  ddesonn_run(...)
}


#' Legacy alias for [ddesonn_training_defaults()]
#'
#' @rdname ddesonn_training_defaults
#' @aliases DDESONN_training_defaults
#' @param ... Additional arguments passed through to
#'   [ddesonn_training_defaults()].
#' @return Same as [ddesonn_training_defaults()].
#' @export
DDESONN_training_defaults <- function(...) {
  ddesonn_training_defaults(...)
}
