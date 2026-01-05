# =============================================================== #$$$$$$$$$$$$$
# Path helpers for artifacts and plots                            #$$$$$$$$$$$$$
# =============================================================== #$$$$$$$$$$$$$

# ===== Artifacts root resolver ================================= #$$$$$$$$$$$$$
ddesonn_artifacts_root <- function(output_root = NULL) { #$$$$$$$$$$$$$
  if (is.null(output_root)) { #$$$$$$$$$$$$$
    candidate <- file.path(getwd(), "analysis", "artifacts") #$$$$$$$$$$$$$
    if (dir.exists(candidate)) { #$$$$$$$$$$$$$
      root <- candidate #$$$$$$$$$$$$$
    } else { #$$$$$$$$$$$$$
      root <- file.path(tempdir(), "DDESONN_artifacts") #$$$$$$$$$$$$$
    } #$$$$$$$$$$$$$
  } else { #$$$$$$$$$$$$$
    nr <- tryCatch(normalizePath(output_root, winslash = "/", mustWork = FALSE), error = function(e) output_root) #$$$$$$$$$$$$$
    if (basename(nr) == "artifacts") { #$$$$$$$$$$$$$
      root <- output_root #$$$$$$$$$$$$$
    } else { #$$$$$$$$$$$$$
      root <- file.path(output_root, "artifacts") #$$$$$$$$$$$$$
    } #$$$$$$$$$$$$$
  } #$$$$$$$$$$$$$
  dir.create(root, recursive = TRUE, showWarnings = FALSE) #$$$$$$$$$$$$$
  root #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$

# ===== Plots dir helper ======================================== #$$$$$$$$$$$$$
ddesonn_plots_dir <- function(output_root = NULL) { #$$$$$$$$$$$$$
  plots_dir <- file.path(ddesonn_artifacts_root(output_root), "plots") #$$$$$$$$$$$$$
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE) #$$$$$$$$$$$$$
  plots_dir #$$$$$$$$$$$$$
} #$$$$$$$$$$$$$
