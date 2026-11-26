test_that("fit uses training factor levels to score validation", {
  x <- workbench_mtcars[, c("wt", "hp")]
  y <- factor(ifelse(workbench_mtcars$am == 1, "manual", "automatic"),
              levels = c("automatic", "manual"))
  valid <- seq.int(1L, nrow(x), by = 2L)
  fit_accuracy <- function(validation_y) {
    set.seed(10)
    model <- ddesonn_model(2L, 1L, hidden_sizes = 2L)
    invisible(capture.output(ddesonn_fit(
      model, x[-valid, ], y[-valid], num_epochs = 1L,
      validation = list(x = x[valid, ], y = validation_y)
    )))
    model$ensemble[[1L]]$metadata$best_val_acc
  }
  accuracy <- fit_accuracy(y[valid])
  expect_true(is.finite(accuracy))
  expect_equal(fit_accuracy(factor(y[valid], levels = rev(levels(y)))), accuracy)
  expect_equal(fit_accuracy(workbench_mtcars$am[valid]), accuracy)
})

test_that("fit can train without a validation split by default", {
  x <- workbench_mtcars[, c("wt", "hp")]
  model <- ddesonn_model(2L, 1L, hidden_sizes = 2L,
                        classification_mode = "regression")
  invisible(capture.output(ddesonn_fit(model, x, workbench_mtcars$mpg, num_epochs = 1L)))
  expect_true(is.list(model$last_training))
  expect_equal(nrow(predict(model, x)$predicted_output), nrow(x))
})

test_that("run encodes multiclass indices using the training classes", {
  x <- iris[, 1:4]
  y <- 10L * as.integer(iris$Species)
  train <- c(1:4, 51:54, 101:104)
  valid <- c(5:6, 55:56, 105:106)
  test <- 107:108

  invisible(capture.output(run <- ddesonn_run(
    x[train, ], y[train], classification_mode = "multiclass", hidden_sizes = 2L,
    validation = list(x = x[valid, ], y = y[valid]),
    test = list(x = x[test, ], y = y[test]),
    training_overrides = list(num_epochs = 1L), save_models = FALSE
  )))

  splits <- run$runs[[1L]]$predictions
  expect_equal(ncol(predict(run$model, x[test, ])$predicted_output), 3L)
  expect_equal(splits$validation$y_true, rep(1:3, each = 2))
  expect_equal(splits$test$y_true, rep(3, 2))
  expect_true(run$test_metrics$ok)
  expect_equal(run$test_metrics$accuracy, mean(splits$test$y_pred == 3))

  expect_error(ddesonn_fit(
    run$model, x[train, ], y[train], num_epochs = 1L,
    validation = list(x = x[test, ], y = matrix(2L, nrow = length(test)))
  ), "Validation labels contain missing values or classes absent from training")
})

test_that("binary reports preserve an all-positive encoded split", {
  report <- DDESONN:::.build_binary_report(c(1, 1), c(0.8, 0.2), 0.5)
  expect_equal(report$report[c("0", "1"), "support"], c(0, 2))
})
