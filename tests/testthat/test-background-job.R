test_that("background waits allow completion and preserve worker errors", {
  skip_if_not_installed("callr")
  process <- callr::r_bg(function() { Sys.sleep(0.2); 42L })
  on.exit(process$kill(), add = TRUE)
  wait_for_workbench_job(list(process = process), timeout = 30000)
  expect_identical(process$get_result(), 42L)

  failed <- callr::r_bg(function() stop("worker training failure"))
  on.exit(failed$kill(), add = TRUE)
  wait_for_workbench_job(list(process = failed), timeout = 30000)
  expect_error(failed$get_result(), "worker training failure")
})

test_that("background timeouts report logs before requesting a running result", {
  skip_if_not_installed("callr")
  log <- tempfile(); error <- tempfile()
  writeLines("last training progress", log)
  writeLines("worker diagnostic", error)
  on.exit(unlink(c(log, error)), add = TRUE)
  process <- callr::r_bg(function() Sys.sleep(60))
  on.exit(process$kill(), add = TRUE)
  job <- list(process = process, log = log, error = error)
  expect_error(wait_for_workbench_job(job, timeout = 1),
               "still running after 0.001 seconds.*last training progress.*worker diagnostic")
  expect_true(process$is_alive())
})
