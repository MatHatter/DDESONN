test_that("uploaded CSV evaluation requires compatible target labels and reports overlap", {
  f <- prediction_flow_fixture()
  x <- iris[1:8, 1:4]
  report <- f$h$wb_prediction_report(f$bundle, x)
  expect_equal(report$rows, 8)
  expect_equal(report$matched_rows, 8)
  expect_null(report$metrics)
  expect_match(report$evaluation_note, "no accuracy", fixed = TRUE)
  report <- f$h$wb_prediction_report(f$bundle, iris[1:8, ])
  predictions <- f$h$wb_new_predictions(f$bundle, iris[1:8, ])
  expect_equal(report$metrics$Value[report$metrics$Metric == "Accuracy"], mean(predictions$predicted == iris$Species[1:8]))
  expect_match(report$evaluation_note, "not an independent", fixed = TRUE)
  bad <- iris[1:8, ]; bad$Species <- as.character(bad$Species); bad$Species[1] <- "unknown"
  invalid <- f$h$wb_prediction_report(f$bundle, bad)
  expect_null(invalid$metrics)
  expect_match(invalid$evaluation_note, "incomplete or incompatible", fixed = TRUE)
  novel <- iris[1:8, ]; novel$Sepal.Length <- novel$Sepal.Length + 100
  report <- f$h$wb_prediction_report(f$bundle, novel)
  expect_equal(report$matched_rows, 0)
  expect_true(all(report$overlap[[2]] == 0))
  expect_true(all(is.finite(report$metrics$Value)))
})
test_that("supplied binary and regression targets produce their own CSV metrics", {
  f <- prediction_flow_fixture()
  for (mode in c("binary", "regression")) {
    target <- if (mode == "binary") "am" else "mpg"
    p <- f$h$wb_prepare(mtcars, target, c("wt", "hp"), mode)
    invisible(capture.output(bundle <- f$h$wb_run(p, f$h$wb_config(hidden = "2", epochs = 1, checkpoint = "final"))))
    d <- head(mtcars, 6)
    report <- f$h$wb_prediction_report(bundle, d)
    predictions <- f$h$wb_new_predictions(bundle, d)
    if (mode == "binary") expect_equal(report$metrics$Value[report$metrics$Metric == "Accuracy"], mean(predictions$predicted == as.character(d[[target]]))) else
      expect_equal(report$metrics$Value[report$metrics$Metric == "RMSE"], sqrt(mean((predictions$prediction - d[[target]])^2)))
    d[[target]][1] <- NA
    expect_null(f$h$wb_prediction_report(bundle, d)$metrics)
  }
})
