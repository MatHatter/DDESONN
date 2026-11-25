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
  env <- c(TMPDIR = scratch, TMP = scratch, TEMP = scratch, "_R_CHECK_TIMINGS_" = "0")
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
  build_seconds <- system.time({
    build_status <- system2(r, c("CMD", "build", shQuote(package)))
  })[["elapsed"]]
  if (build_status != 0L || !file.exists(tarball)) stop("R CMD build failed.")
  check_seconds <- system.time({
    check_status <- system2(r, c("CMD", "check", "--as-cran", "--no-manual", shQuote(tarball)))
  })[["elapsed"]]

  test_files <- list.files(file.path(run_dir, "DDESONN.Rcheck", "tests"),
    pattern = "^testthat[.]Rout$", recursive = TRUE, full.names = TRUE)
  test_times <- vapply(test_files, function(path) {
    lines <- readLines(path, warn = FALSE)
    header <- tail(grep("^[[:space:]]*user[[:space:]]+system[[:space:]]+elapsed[[:space:]]*$",
      lines), 1L)
    if (!length(header) || header == length(lines)) return(NA_real_)
    as.numeric(strsplit(trimws(lines[header + 1L]), "[[:space:]]+")[[1L]][3L])
  }, numeric(1))
  print(round(c(testthat_R_seconds = if (length(test_times)) sum(test_times) else NA_real_,
    check_seconds = check_seconds, build_seconds = build_seconds,
    build_plus_check_seconds = build_seconds + check_seconds,
    check_exit_status = check_status), 2))
  log <- file.path(run_dir, "DDESONN.Rcheck", "00check.log")
  status <- if (file.exists(log)) {
    grep("^Status: ", trimws(readLines(log, warn = FALSE)), value = TRUE)
  }
  status <- tail(status, 1L)
  if (check_status != 0L || length(status) != 1L ||
      !grepl("^Status: (OK|[0-9]+ NOTEs?)$", status)) {
    stop("R CMD check failed or did not finish. See ", log)
  }
  invisible(log)
}
