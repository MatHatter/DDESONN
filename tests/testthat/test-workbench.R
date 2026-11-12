workbench_fixture <- function() {
  env <- new.env(parent = globalenv())
  path <- system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN")
  sys.source(path, envir = env)
  list(h = env, path = path)
}

test_that("workbench preprocessing is reproducible and fitted only on training rows", {
  h <- workbench_fixture()$h
  d <- data.frame(x = c(1:60, 1001:1040), constant = 1, y = seq_len(100))
  set.seed(17); rng <- .Random.seed
  p <- h$wb_prepare(d, "y", c("x", "constant"), "regression", split = "ordered")
  expect_identical(.Random.seed, rng)
  expect_equal(unname(p$center), c(mean(1:60), 1))
  expect_equal(unname(p$scale[2]), 1)
  expect_equal(colMeans(p$parts$train$x), c(x = 0, constant = 0), tolerance = 1e-10)
  expect_gt(mean(p$parts$test$x[, 1]), 10)
  ids <- unlist(lapply(p$parts, `[[`, "row"), use.names = FALSE)
  expect_equal(sort(ids), 1:100)
  expect_false(anyDuplicated(ids) > 0)
  a <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  b <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  expect_identical(a, b)
  expect_equal(vapply(a$parts, function(x) nrow(x$x), integer(1)), c(train = 90L, validation = 30L, test = 30L))
})

test_that("workbench rejects invalid schemas and model settings", {
  h <- workbench_fixture()$h
  expect_error(h$wb_prepare(iris, "Species", "Species", "multiclass"), "distinct predictor")
  d <- iris; d$Sepal.Length[1] <- NA
  expect_error(h$wb_prepare(d, "Species", names(d)[1:4], "multiclass"), "finite")
  expect_error(h$wb_prepare(iris, "Species", names(iris)[1:4], "binary"), "exactly two")
  expect_error(h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass", train_fraction = .9), "Leave")
  expect_error(h$wb_config(hidden = "16,wrong"), "hidden layers")
  expect_error(h$wb_config(epochs = 0), "epochs")
  expect_error(h$wb_config(lr = 0), "Learning rate")
  expect_error(h$wb_config(optimizer = "adagrad_x"), "Unsupported")
  expect_error(h$wb_validate_bundle(list()), "completed model")
})

test_that("public API training, serialization and inference work for all three tasks", {
  h <- workbench_fixture()$h
  for (mode in c("binary", "multiclass", "regression")) {
    d <- if (mode == "multiclass") iris else mtcars
    target <- switch(mode, binary = "am", multiclass = "Species", regression = "mpg")
    features <- if (mode == "multiclass") names(iris)[1:4] else c("wt", "hp")
    p <- h$wb_prepare(d, target, features, mode)
    cfg <- h$wb_config(hidden = if (mode == "binary") "" else "4", epochs = 2, lr = .001)
    capture.output(r <- h$wb_run(p, cfg))
    expect_true(all(is.finite(r$evaluation$test$metrics$Value)))
    before <- h$wb_new_predictions(r, d[features])
    file <- tempfile(fileext = ".rds"); saveRDS(r, file)
    restored <- h$wb_validate_bundle(readRDS(file))
    expect_equal(h$wb_new_predictions(restored, d[rev(features)]), before)
    expect_equal(nrow(before), nrow(d))
    expect_error(h$wb_new_predictions(restored, d[FALSE, features]), "every saved predictor")
    bad <- d[features]; bad[[1]][1] <- Inf
    expect_error(h$wb_new_predictions(restored, bad), "finite numeric")
    expect_equal(restored$prepared$center, p$center)
    unlink(file)
  }
})

test_that("background training returns a completed model and cancellation discards only the job", {
  skip_if_not_installed("callr")
  f <- workbench_fixture(); h <- f$h
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  j <- h$wb_start(f$path, p, h$wb_config(hidden = "4", epochs = 2))
  on.exit(h$wb_stop(j), add = TRUE)
  j$process$wait(30000)
  expect_false(j$process$is_alive())
  r <- h$wb_validate_bundle(j$process$get_result())
  expect_equal(r$fit_runs, 1L)
  expect_equal(h$wb_log_history(j$log)$epoch, 1:2)
  before <- h$wb_new_predictions(r, iris)
  another <- h$wb_start(f$path, p, h$wb_config(hidden = "4", epochs = 5000), r)
  h$wb_stop(another)
  expect_false(another$process$is_alive())
  expect_false(dir.exists(another$dir))
  expect_identical(h$wb_new_predictions(r, iris), before)
  continuation <- h$wb_start(f$path, p, r$config, r)
  on.exit(h$wb_stop(continuation), add = TRUE)
  continuation$process$wait(30000)
  expect_equal(continuation$process$get_result()$fit_runs, 2L)
})
