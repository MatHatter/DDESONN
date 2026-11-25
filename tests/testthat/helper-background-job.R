wait_for_workbench_job <- function(job, timeout = 180000) {
  job$process$wait(timeout)
  if (job$process$is_alive()) {
    logs <- vapply(c(log = job$log, error = job$error), function(path) {
      if (!file.exists(path)) return("(no log file)")
      paste(tail(readLines(path, warn = FALSE), 20L), collapse = "\n")
    }, character(1))
    stop(sprintf("Background training worker still running after %g seconds.\n%s",
                 timeout / 1000, paste(names(logs), logs, sep = ":\n", collapse = "\n")),
         call. = FALSE)
  }
  invisible(job)
}

wait_for_workbench_epoch <- function(job, timeout = 60) {
  deadline <- Sys.time() + timeout
  repeat {
    if (file.exists(job$log) && any(grepl("^Epoch 1 .*Loss:", readLines(job$log, warn = FALSE)))) {
      return(invisible(job))
    }
    if (!job$process$is_alive() || Sys.time() >= deadline) {
      stop("Background training did not reach its first epoch.", call. = FALSE)
    }
    Sys.sleep(.05)
  }
}

# Give Shiny polling a deterministic worker lifecycle for tests of UI state.
workbench_ui_job <- function(config, result = NULL, failure = NULL, log_lines = character()) {
  dir <- tempfile("workbench-ui-job-")
  dir.create(dir)
  log <- file.path(dir, "training.log")
  writeLines(log_lines, log)
  running <- TRUE
  finish <- function() running <<- FALSE
  process <- list(
    is_alive = function() running,
    kill_tree = finish,
    wait = function(timeout) invisible(NULL),
    get_result = function() {
      if (!is.null(failure)) stop(failure)
      result
    }
  )
  list(process = process, finish = finish, dir = dir, log = log,
    error = file.path(dir, "stderr.log"), started = Sys.time(), epochs = config$epochs)
}
