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
