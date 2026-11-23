test_that("uploaded CSV evaluation requires compatible target labels and reports overlap", {
  f <- prediction_flow_fixture()
  data <- head(workbench_iris, 8)
  x <- data[1:4]
  report <- f$h$wb_prediction_report(f$bundle, x)
  expect_equal(report$rows, 8)
  expect_equal(report$matched_rows, 8)
  expect_null(report$metrics)
  expect_match(report$evaluation_note, "no accuracy", fixed = TRUE)
  report <- f$h$wb_prediction_report(f$bundle, data)
  predictions <- f$h$wb_new_predictions(f$bundle, data)
  expect_equal(report$metrics$Value[report$metrics$Metric == "Accuracy"], mean(predictions$predicted == data$Species))
  expect_match(report$evaluation_note, "not an independent", fixed = TRUE)
  bad <- data; bad$Species <- as.character(bad$Species); bad$Species[1] <- "unknown"
  invalid <- f$h$wb_prediction_report(f$bundle, bad)
  expect_null(invalid$metrics)
  expect_match(invalid$evaluation_note, "incomplete or incompatible", fixed = TRUE)
  novel <- data; novel$Sepal.Length <- novel$Sepal.Length + 100
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
test_that("prediction-only mode does not score supplied labels", {
  f <- prediction_flow_fixture()
  report <- f$h$wb_prediction_report(f$bundle, head(iris), evaluate = FALSE)
  expect_null(report$metrics)
  expect_match(report$evaluation_note, "Prediction-only mode", fixed = TRUE)
  expect_true(f$h$wb_evaluation_target_check(f$bundle$prepared, head(iris))$valid)
  expect_false(f$h$wb_evaluation_target_check(f$bundle$prepared, head(iris)[1:4])$valid)
  d <- head(iris); d$Species[1] <- NA
  expect_false(f$h$wb_evaluation_target_check(f$bundle$prepared, d)$valid)
})
test_that("the labeled customer example retains known source labels", {
  path <- system.file("shiny", "benchmark", "examples", "customer-segmentation-original-test-example.csv", package = "DDESONN")
  expect_true(nzchar(path))
  example <- read.csv(path, check.names = FALSE)
  source <- read.csv(system.file("extdata", "train_multiclass_customer_segmentation.csv", package = "DDESONN"), check.names = FALSE)
  expect_equal(nrow(example), 20)
  expect_false("ID" %in% names(example))
  expect_true(all(example$Segmentation %in% c("A", "B", "C", "D")))
  # Include the target in the fingerprint: no example answers are fabricated.
  keys <- function(d) vapply(seq_len(nrow(d)), function(i) digest::digest(lapply(d, function(x) as.character(x[i]))), character(1))
  expect_true(all(keys(example) %in% keys(source[names(example)])))
})
