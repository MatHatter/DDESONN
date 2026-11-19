# Run from RStudio with source("dev/check-release.R"), then check_ddesonn_release().
check_ddesonn_release <- function(package = ".",
                                  output = tools::R_user_dir("DDESONN", "cache")) {
  package <- normalizePath(package, winslash = "/", mustWork = TRUE)
  description <- read.dcf(file.path(package, "DESCRIPTION"))
  stopifnot(description[1, "Package"] == "DDESONN")

  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  output <- normalizePath(output, winslash = "/", mustWork = TRUE)
  run_dir <- tempfile("release-check-", tmpdir = output)
  if (!dir.create(run_dir)) stop("Could not create the release check directory.")
  scratch <- file.path(run_dir, "tmp")
  if (!dir.create(scratch)) stop("Could not create the release check temporary directory.")

  # Keep child process files outside RStudio's session-owned temporary directory.
  env <- c(TMPDIR = scratch, TMP = scratch, TEMP = scratch)
  if (.Platform$OS.type == "windows") {
    # Keep the restricted check library, using copies instead of NTFS junctions.
    env <- c(env, R_WIN_NO_JUNCTIONS = "true")
  }
  old_env <- Sys.getenv(names(env), unset = NA_character_)
  on.exit({
    Sys.unsetenv(names(old_env)[is.na(old_env)])
    restore <- old_env[!is.na(old_env)]
    if (length(restore)) do.call(Sys.setenv, as.list(restore))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(env))
  old_dir <- setwd(run_dir)
  on.exit(setwd(old_dir), add = TRUE)

  r <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "R.exe" else "R")
  tarball <- paste0("DDESONN_", description[1, "Version"], ".tar.gz")
  message("Release check output: ", run_dir)
  build_status <- system2(r, c("CMD", "build", shQuote(package)))
  if (build_status != 0L || !file.exists(tarball)) stop("R CMD build failed.")
  check_status <- system2(r, c("CMD", "check", "--as-cran", "--no-manual", shQuote(tarball)))
  log <- file.path(run_dir, "DDESONN.Rcheck", "00check.log")
  if (check_status != 0L || !file.exists(log) ||
      !any(trimws(readLines(log, warn = FALSE)) == "Status: OK")) {
    stop("R CMD check did not report Status: OK. See ", log)
  }
  invisible(log)
}
