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
      suffix <- if (phase == "end") " [|] elapsed=[0-9]+[.][0-9]{3} s$" else "epoch=[0-9]+$"
      expect_true(all(grepl(suffix, hits)))
      unique(as.integer(sub(".*epoch=([0-9]+).*", "\\1", hits)))
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


test_that("Custom R recipes offer trace examples with tracing off as generated", {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  cfg <- h$wb_custom_config(h$wb_config())
  recipe <- h$wb_fit_code(list(), cfg)
  recipe_env <- new.env(parent = baseenv())
  setup <- Filter(function(expr) is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
    identical(expr[[2L]], quote(config)), as.list(parse(text = recipe)))
  expect_length(setup, 1L)
  eval(setup[[1L]], recipe_env)
  expect_null(recipe_env$config$trace_epochs)

  examples <- grep("^# config\\$trace_epochs", strsplit(recipe, "\n", fixed = TRUE)[[1L]], value = TRUE)
  expect_length(examples, 2L)
  for (example in examples) {
    eval(parse(text = sub("^# ", "", example)), recipe_env)
    expect_silent(DDESONN:::.validate_trace_epochs(recipe_env$config$trace_epochs))
  }
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


test_that("live elapsed labels use complete minutes and seconds", {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  started <- as.POSIXct("2026-11-18 12:00:00", tz = "UTC")
  expect_identical(h$wb_elapsed_text(started, started + 75.9), "Elapsed: 1m 15s")
  expect_identical(h$wb_elapsed_text(started, started - 1), "Elapsed: 0m 00s")
})
