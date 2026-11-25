library(testthat)
library(DDESONN)
test_results <- test_check("DDESONN")

timings_file <- Sys.getenv("DDESONN_TEST_TIMINGS_FILE", "")
if (nzchar(timings_file)) {
  timing_results <- tryCatch(as.data.frame(test_results), error = function(error) NULL)
  if (!is.null(timing_results) && all(c("file", "real") %in% names(timing_results))) {
    file_times <- aggregate(real ~ file, timing_results, sum, na.rm = TRUE)
    file_times <- file_times[order(file_times$real, decreasing = TRUE), ]
    write.csv(file_times, timings_file, row.names = FALSE)
  }
}
