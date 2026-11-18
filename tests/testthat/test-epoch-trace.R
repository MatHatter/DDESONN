test_that("method trace selections accept only named logical flags or whole epochs", {
  validate <- DDESONN:::.validate_trace_epochs
  expect_silent(validate(list(train = c(1, 3), train_network = TRUE, learn = FALSE, predict = 2)))
  expect_error(validate(list(1:2)), "named list")
  expect_error(validate(list(fit = TRUE)), "named list")
  expect_error(validate(list(learn = c(0, 1))), "positive whole epochs")
  expect_error(validate(list(learn = c(TRUE, FALSE))), "positive whole epochs")
  expect_error(validate(list(learn = c(1, NA))), "positive whole epochs")
  expect_error(validate(list(all = 1:3)), "must be TRUE or FALSE")
  resolve <- DDESONN:::.resolve_trace_epochs
  expect_identical(resolve(list(all = TRUE, learn = 2:3, predict = FALSE)),
    list(train = TRUE, train_network = TRUE, learn = 2:3, predict = FALSE))
  expect_identical(resolve(list(all = FALSE, train = TRUE)),
    list(train = TRUE, train_network = FALSE, learn = FALSE, predict = FALSE))
  expect_identical(resolve(list(learn = 2:3)), list(learn = 2:3))
})

test_that("method trace filters select independent epochs without changing training", {
  x <- as.matrix(mtcars[1:18, c("disp", "hp")]) / 500
  y <- matrix(mtcars$mpg[1:18] / 100)
  args <- list(x = x[1:12, ], y = y[1:12, , drop = FALSE],
    classification_mode = "regression", hidden_sizes = integer(),
    model_overrides = list(ML_NN = FALSE), save_models = FALSE,
    validation = list(x = x[13:18, ], y = y[13:18, , drop = FALSE]),
    training_overrides = list(num_epochs = 3L, lr = .03,
      batch_normalize_data = FALSE, numeric_columns = integer()))
  expect_false(any(startsWith(capture.output(plain <- do.call(ddesonn_run, args)), "[TRACE] train-")))
  args$training_overrides$trace_epochs <- list(all = TRUE, train = 1:2,
    learn = 2:3, predict = 3L)
  lines <- capture.output(traced <- do.call(ddesonn_run, args))
  for (phase in c("begin", "end")) {
    epochs <- function(method) {
      hits <- grep(paste0("^\\[TRACE\\] ", method, "-", phase, " \\| epoch="), lines, value = TRUE)
      unique(as.integer(sub(".*epoch=", "", hits, fixed = FALSE)))
    }
    expect_identical(epochs("train"), 1:2)
    expect_identical(epochs("train_network"), 1:3)
    expect_identical(epochs("learn"), 2:3)
    expect_identical(epochs("predict"), 3L)
  }
  expect_equal(traced$model$ensemble[[1]]$weights, plain$model$ensemble[[1]]$weights)
  expect_equal(traced$model$ensemble[[1]]$biases, plain$model$ensemble[[1]]$biases)
  expect_identical(traced$runs[[1]]$report_fits[[1]]$members[[1]]$completed, 3L)
})


test_that("Custom Diagnostics converts sliders to the same trace settings as R", {
  skip_if_not_installed("shiny")
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench.R", package = "DDESONN"), h)
  input <- list(wb_trace_enabled = TRUE, wb_trace_all = TRUE,
    wb_trace_mode_train = "off", wb_trace_mode_train_network = "inherit",
    wb_trace_mode_learn = "range", wb_trace_range_learn = c(2, 5),
    wb_trace_mode_predict = "on")
  expect_identical(h$wb_epoch_trace_config(input, 4L),
    list(all = TRUE, train = FALSE, learn = 2:4, predict = TRUE))
  input$wb_trace_enabled <- FALSE
  expect_null(h$wb_epoch_trace_config(input, 4L))
  expect_match(as.character(h$wb_epoch_trace_controls()), "wb_trace_range_learn", fixed = TRUE)
})

test_that("debug fit markers work without an environment gate or epoch selection", {
  withr::local_envvar(DDESONN_DEBUG = NA_character_)
  x <- as.matrix(mtcars[1:12, c("disp", "hp")]) / 500
  y <- matrix(mtcars$am[1:12])
  output <- capture.output(ddesonn_run(x, y, hidden_sizes = integer(),
    model_overrides = list(ML_NN = FALSE), debug = TRUE, verboseLow = TRUE,
    training_overrides = list(num_epochs = 1L, lr = .03, batch_normalize_data = FALSE,
      numeric_columns = integer(), validation_metrics = FALSE), save_models = FALSE))
  expect_true(any(grepl("EVOKE-FIT-BEGIN", output, fixed = TRUE)))
  expect_true(any(grepl("EVOKE-FIT-END", output, fixed = TRUE)))
  expect_false(any(startsWith(output, "[TRACE] ")))
})
