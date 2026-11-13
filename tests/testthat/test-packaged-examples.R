example_fixture <- function() {
  h <- new.env(parent = globalenv())
  path <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(path, "workbench_helpers.R"), h)
  sys.source(file.path(path, "workbench.R"), h)
  list(h = h, path = path)
}

test_that("all packaged inputs load unchanged, export and reproduce their data source", {
  h <- example_fixture()$h
  catalog <- h$wb_example_catalog()
  expect_setequal(names(catalog), c("iris", "mtcars", "heart_failure", "wmt", "customer_train", "customer_test"))
  expect_setequal(unname(h$wb_example_choices()), setdiff(names(catalog), "customer_test"))
  expect_false("customer_test" %in% h$wb_example_choices())
  files <- vapply(Filter(function(x) !is.null(x$file), catalog), `[[`, character(1), "file")
  expect_setequal(unname(files), list.files(system.file("extdata", package = "DDESONN"), pattern = "\\.csv$"))
  sizes <- c(iris = 150L, mtcars = 32L, heart_failure = 5000L, wmt = 13248L,
    customer_train = 8068L, customer_test = 2627L)
  for (id in names(catalog)) {
    ex <- h$wb_example(id)
    expect_equal(nrow(ex$data), unname(sizes[id]))
    env <- new.env(parent = globalenv())
    eval(parse(text = h$wb_data_code(ex$source)), env)
    expect_identical(env$data, ex$data)
    output <- tempfile(fileext = ".csv")
    h$wb_write_example(id, output)
    if (!is.null(ex$path)) expect_identical(readBin(output, "raw", file.info(output)$size),
      readBin(ex$path, "raw", file.info(ex$path)$size)) else {
      exported <- read.csv(output, check.names = FALSE)
      expect_equal(nrow(exported), nrow(ex$data)); expect_identical(names(exported), names(ex$data))
    }
    unlink(output)
    features <- h$wb_default_features(ex$data, ex$target, ex)
    expect_false(any(features %in% c(ex$target, ex$exclude)))
    if (!is.null(ex$target)) {
      p <- h$wb_prepare(ex$data, ex$target, features, ex$mode,
        split = if (is.null(ex$split)) "random" else ex$split)
      expect_equal(p$rows, nrow(ex$data))
    }
  }
  expect_identical(h$wb_default_features(h$wb_example("customer_train")$data,
    "Segmentation", h$wb_example("customer_train")), "Age")
  expect_null(h$wb_example("customer_test")$target)
  expect_true(h$wb_example("customer_test")$prediction_only)
  expect_error(h$wb_example("../../private.csv"), "Choose a packaged example")
  source <- h$wb_example("heart_failure")$source; source$sha256 <- "incorrect"
  expect_error(eval(parse(text = h$wb_data_code(source)), new.env()), "not TRUE")
})

test_that("dataset changes preserve models, unlabeled prediction and guide navigation", {
  skip_if_not_installed("shiny")
  f <- example_fixture(); h <- f$h
  root <- tempfile(); dir.create(root)
  withr::local_options(DDESONN_OUTPUT_ROOT = root)
  withr::local_envvar(DDESONN_ARTIFACTS_ROOT = root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  train <- h$wb_example("customer_train")
  p <- h$wb_prepare(train$data, train$target, train$features, train$mode)
  p$source <- train$source
  invisible(capture.output(bundle <- h$wb_run(p, h$wb_config(hidden = "2", epochs = 2, checkpoint = "final"))))
  weights <- bundle$model$ensemble[[1]]$weights
  server <- h$wb_server; formals(server)$helper <- file.path(f$path, "workbench_helpers.R")
  shiny::testServer(server, {
    begin_attempt(bundle$config)
    add_completed(bundle)
    session$setInputs(wb_example = "heart_failure", wb_demo = 1)
    expect_equal(nrow(raw()), 5000)
    expect_identical(code_source()$package_file, "heart_failure_clinical_records.csv")
    session$setInputs(wb_target = "DEATH_EVENT", wb_features = setdiff(names(raw()), "DEATH_EVENT"),
      wb_mode = "binary", wb_seed = 42, wb_train_fraction = .6,
      wb_val_fraction = .2, wb_scale = TRUE, wb_split = "random", wb_prepare = 1)
    expect_identical(prepared()$source, code_source())
    snapshot <- serialize(prepared(), NULL)
    session$setInputs(wb_example = "customer_test", wb_demo = 2)
    expect_equal(nrow(raw()), 5000)
    expect_identical(serialize(prepared(), NULL), snapshot)
    expect_match(error(), "Predict on new rows")
    expect_identical(result()$model$ensemble[[1]]$weights, weights)
    session$setInputs(wb_example_predict = 1)
    expect_null(model_error())
    expect_equal(nrow(new_predictions()), 2627)
    expect_identical(serialize(prepared(), NULL), snapshot)
    expect_equal(nrow(raw()), 5000)
    expect_identical(result()$model$ensemble[[1]]$weights, weights)
    expect_error(h$wb_new_predictions(result(), iris), "every saved predictor")
    session$setInputs(wb_example = "iris", wb_demo = 3, wb_target = "Species",
      wb_features = names(iris)[1:4], wb_mode = "multiclass")
    session$setInputs(wb_prepare = 2)
    expect_equal(nrow(raw()), 150)
    expect_identical(code_source(), list(example = "iris"))
    snapshot <- serialize(prepared(), NULL)
    messages <- new.env()
    session$sendInputMessage <- function(id, message) messages[[id]] <- message
    for (id in c("wb_data_vignettes", "wb_features_vignettes", "wb_training_vignettes",
        "wb_prediction_vignettes", "wb_experiment_vignettes")) {
      do.call(session$setInputs, stats::setNames(list(1), id))
      expect_identical(messages$workspace$value, "vignettes")
      expect_identical(serialize(prepared(), NULL), snapshot)
    }
  })
})

test_that("normalization preserves dimensions for one and multiple predictors", {
  normalize <- getFromNamespace("DDESONN", "DDESONN")$public_methods$normalize_data
  d <- data.frame(Age = 20:39, other = (1:20)^2)
  for (columns in list("Age", c("Age", "other"))) {
    actual <- normalize(d, columns)
    expected <- d
    expected[, columns] <- scale(d[, columns, drop = FALSE])
    expect_equal(actual, expected)
    expect_identical(dim(actual), dim(d))
  }
})
