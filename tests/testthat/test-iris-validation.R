test_that("Iris validation reporting never treats three-class probabilities as binary", {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  original <- getFromNamespace("accuracy_precision_recall_f1_tuned", "DDESONN")
  reported_modes <- character()
  testthat::local_mocked_bindings(
    accuracy_precision_recall_f1_tuned = function(SONN, Rdata, labels,
        CLASSIFICATION_MODE = NULL, predicted_output, ...) {
      reported_modes <<- c(reported_modes, CLASSIFICATION_MODE)
      if (identical(CLASSIFICATION_MODE, "binary"))
        stop("Iris probabilities incorrectly sent to binary reporting")
      original(SONN, Rdata, labels, CLASSIFICATION_MODE, predicted_output, ...)
    }, .package = "DDESONN")
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  cfg <- h$wb_config(hidden = "4", epochs = 2, lr = .001, checkpoint = "best_validation")
  expect_no_error(invisible(capture.output(r <- h$wb_run(p, cfg))))
  expect_true(length(reported_modes) > 0)
  expect_false(any(reported_modes == "binary"))
  expect_true(is.finite(r$selected_epoch))
  expect_true(all(is.finite(r$evaluation$validation$metrics$Value)))
  e <- h$wb_training_error(simpleError(paste0(
    "[accuracy_precision_recall_f1_tuned] Binary mode expects 1-column probabilities (sigmoid). Got 3 columns.")))
  expect_match(e$message, "multiclass predictions as binary")
  expect_match(e$message, "restart R")
  expect_false(grepl("smaller learning rate", e$message, fixed = TRUE))
})
