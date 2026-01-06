# =============================================================== #$$$$$$$$$$$$$
# Path helpers for artifacts and plots                            #$$$$$$$$$$$$$
# =============================================================== #$$$$$$$$$$$$$

# ===== Artifacts root resolver ================================= #$$$$$$$$$$$$$
ddesonn_artifacts_root <- function(output_root = NULL) { #$$$$$$$$$$$$$
  root <- .ddesonn_resolve_artifacts_root(output_root) #$$$$$$$$$$$$$
  dir.create(root, recursive = TRUE, showWarnings = FALSE) #$$$$$$$$$$$$$
  .ddesonn_paths_check(root, context = "artifacts") #$$$$$$$$$$$$$
  root #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$

# ===== Plots dir helper ======================================== #$$$$$$$$$$$$$
ddesonn_plots_dir <- function(output_root = NULL) { #$$$$$$$$$$$$$
  plots_dir <- file.path(ddesonn_artifacts_root(output_root), "plots") #$$$$$$$$$$$$$
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE) #$$$$$$$$$$$$$
  .ddesonn_paths_check(plots_dir, context = "plots") #$$$$$$$$$$$$$
  plots_dir #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$

# ===== Legacy artifacts lookup (read-only) ========================= #$$$$$$$$$$$$$
ddesonn_legacy_artifacts_candidates <- function(output_root = NULL) { #$$$$$$$$$$$$$
  candidates <- list(.ddesonn_resolve_artifacts_root(output_root)) #$$$$$$$$$$$$$
  env_root <- Sys.getenv("DDESONN_ARTIFACTS_ROOT") #$$$$$$$$$$$$$
  if (nzchar(env_root)) { #$$$$$$$$$$$$$
    env_candidate <- if (basename(env_root) == "artifacts") env_root else file.path(env_root, "artifacts") #$$$$$$$$$$$$$
    candidates <- c(candidates, list(env_candidate)) #$$$$$$$$$$$$$
  } #$$$$$$$$$$$$$
  opt_root <- getOption("ddesonn.artifacts_root") #$$$$$$$$$$$$$
  if (!is.null(opt_root) && nzchar(opt_root)) { #$$$$$$$$$$$$$
    opt_candidate <- if (basename(opt_root) == "artifacts") opt_root else file.path(opt_root, "artifacts") #$$$$$$$$$$$$$
    candidates <- c(candidates, list(opt_candidate)) #$$$$$$$$$$$$$
  } #$$$$$$$$$$$$$
  cwd <- getwd() #$$$$$$$$$$$$$
  candidates <- c( #$$$$$$$$$$$$$
    candidates, #$$$$$$$$$$$$$
    list(file.path(cwd, "artifacts")), #$$$$$$$$$$$$$
    list(file.path(cwd, "analysis", "artifacts")) #$$$$$$$$$$$$$
  ) #$$$$$$$$$$$$$
  unique(as.character(candidates)) #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$

# ===== Internal helpers =========================================== #$$$$$$$$$$$$$
.ddesonn_resolve_artifacts_root <- function(output_root = NULL) { #$$$$$$$$$$$$$
  base_dir <- output_root #$$$$$$$$$$$$$
  if (is.null(base_dir) || !nzchar(base_dir)) { #$$$$$$$$$$$$$
    base_dir <- tools::R_user_dir("DDESONN", which = "data") #$$$$$$$$$$$$$
    if (is.null(base_dir) || !nzchar(base_dir)) { #$$$$$$$$$$$$$
      base_dir <- tempdir() #$$$$$$$$$$$$$
    } #$$$$$$$$$$$$$
  } #$$$$$$$$$$$$$
  nr <- tryCatch(normalizePath(base_dir, winslash = "/", mustWork = FALSE), error = function(e) base_dir) #$$$$$$$$$$$$$
  if (basename(nr) == "artifacts") { #$$$$$$$$$$$$$
    return(base_dir) #$$$$$$$$$$$$$
  } #$$$$$$$$$$$$$
  file.path(base_dir, "artifacts") #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$

.ddesonn_paths_check <- function(paths, context = "artifacts") { #$$$$$$$$$$$$$
  src_root <- tryCatch(normalizePath(getwd(), winslash = "/", mustWork = FALSE), error = function(e) getwd()) #$$$$$$$$$$$$$
  is_src_root <- file.exists(file.path(src_root, "DESCRIPTION")) #$$$$$$$$$$$$$
  pkg_root <- tryCatch(normalizePath(system.file(package = "DDESONN"), winslash = "/", mustWork = FALSE), error = function(e) "") #$$$$$$$$$$$$$
  suspicious <- logical(length(paths)) #$$$$$$$$$$$$$
  for (i in seq_along(paths)) { #$$$$$$$$$$$$$
    p <- paths[[i]] #$$$$$$$$$$$$$
    p_norm <- tryCatch(normalizePath(p, winslash = "/", mustWork = FALSE), error = function(e) p) #$$$$$$$$$$$$$
    suspicious[i] <- nzchar(p_norm) && ( #$$$$$$$$$$$$$
      (is_src_root && startsWith(p_norm, paste0(src_root, "/"))) || #$$$$$$$$$$$$$
        (nzchar(pkg_root) && startsWith(p_norm, paste0(pkg_root, "/"))) #$$$$$$$$$$$$$
    ) #$$$$$$$$$$$$$
    if (suspicious[i] && isTRUE(getOption("ddesonn.debug_paths", FALSE))) { #$$$$$$$$$$$$$
      message(sprintf("[ddesonn.%s] path resolved inside package tree: %s", context, p_norm)) #$$$$$$$$$$$$$
    } #$$$$$$$$$$$$$
  } #$$$$$$$$$$$$$
  invisible(!any(suspicious)) #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$
