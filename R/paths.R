# =============================================================== 
# Path helpers for artifacts and plots                            
# =============================================================== 

#' Resolve the writable artifacts root for DDESONN.
#'
#' Defaults to a session-scoped temp directory so examples/tests/vignettes
#' do not write into the user home, working directory, or package tree by
#' default.
#'
#' Override order (first non-empty wins):
#' 1) output_root argument
#' 2) Sys.getenv("DDESONN_ARTIFACTS_ROOT")
#' 3) getOption("DDESONN_OUTPUT_ROOT")
#'
#' @param output_root Optional base directory for artifacts. When NULL, a
#'   temp-directory location is selected automatically.
#' @return Absolute path to the artifacts directory (created if missing).
#' @export
ddesonn_artifacts_root <- function(output_root = NULL) { 
  pick_first <- function(...) { 
    for (x in list(...)) { 
      if (!is.null(x) && length(x) && nzchar(x)) return(x) 
    } 
    NULL 
  } 

  fallback <- file.path(tempdir(), "DDESONN") 

  base <- pick_first( 
    output_root, 
    Sys.getenv("DDESONN_ARTIFACTS_ROOT", unset = ""), 
    getOption("DDESONN_OUTPUT_ROOT", default = ""), 
    fallback 
  ) 

  base_norm <- tryCatch(normalizePath(base, winslash = "/", mustWork = FALSE), error = function(e) base) 

  pkg_home <- tryCatch(normalizePath(system.file(package = "DDESONN"), winslash = "/", mustWork = TRUE), error = function(e) NA_character_) 
  if (!is.na(pkg_home) && nzchar(pkg_home)) { 
    pkg_pref <- if (grepl("/$", pkg_home)) pkg_home else paste0(pkg_home, "/") 
    if (identical(base_norm, pkg_home) || startsWith(base_norm, pkg_pref)) { 
      base_norm <- fallback 
    } 
  } 

  root <- if (basename(base_norm) == "artifacts") base_norm else file.path(base_norm, "artifacts") 
  dir.create(root, recursive = TRUE, showWarnings = FALSE) 
  root 
} 

#' Resolve the plots directory inside the artifacts root.
#'
#' @inheritParams ddesonn_artifacts_root
#' @return Absolute path to the plots directory.
#' @export
ddesonn_plots_dir <- function(output_root = NULL) { 
  plots_dir <- file.path(ddesonn_artifacts_root(output_root), "plots") 
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE) 
  plots_dir 
} 
