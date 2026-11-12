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

test_that("self-organization is optional, affects the fit, and is retained in model bundles", {
  h <- workbench_fixture()$h
  expect_false(h$wb_config()$self_org)
  expect_error(h$wb_config(self_org = "yes"), "TRUE or FALSE")
  for (mode in c("binary", "multiclass", "regression")) {
    data <- if (mode == "multiclass") iris else mtcars
    target <- switch(mode, binary = "am", multiclass = "Species", regression = "mpg")
    features <- if (mode == "multiclass") names(iris)[1:4] else c("wt", "hp")
    p <- h$wb_prepare(data, target, features, mode)
    cfg <- h$wb_config(hidden = if (mode == "binary") "" else "4", epochs = 2, lr = .001)
    capture.output(baseline <- h$wb_run(p, cfg))
    cfg$self_org <- TRUE
    capture.output(organized <- h$wb_run(p, cfg))
    expect_true(organized$config$self_org)
    expect_true(organized$self_org_applied)
    # An additional fit should not reorganize already learned weights.
    copy <- unserialize(serialize(organized, NULL))
    capture.output(continued <- h$wb_run(p, cfg, copy))
    expect_true(continued$config$self_org)
    expect_false(continued$self_org_applied)
    expect_equal(continued$fit_runs, 2L)
    expect_false(isTRUE(all.equal(organized$model$ensemble[[1]]$weights,
      baseline$model$ensemble[[1]]$weights)))
    file <- tempfile(fileext = ".rds"); saveRDS(organized, file)
    restored <- h$wb_validate_bundle(readRDS(file))
    expect_true(restored$config$self_org)
    expect_equal(h$wb_new_predictions(restored, data), h$wb_new_predictions(organized, data))
    unlink(file)
    # Models saved before the checkbox should keep their original off behavior.
    cfg$self_org <- NULL
    capture.output(legacy_config <- h$wb_run(p, cfg))
    expect_false(legacy_config$config$self_org)
    expect_equal(h$wb_new_predictions(legacy_config, data), h$wb_new_predictions(baseline, data))
  }
})

test_that("best-validation restores both tensors and test labels cannot choose the checkpoint", {
  h <- workbench_fixture()$h
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  p$source <- list(example = "iris")
  cfg <- h$wb_config(hidden = "4", epochs = 3, lr = .001)
  expect_identical(cfg$checkpoint, "best_validation")
  invisible(capture.output(best <- h$wb_run(p, cfg)))
  expect_equal(best$selected_epoch, best$model$last_training$predicted_outputAndTime$best_val_epoch)
  expect_lt(best$selected_epoch, cfg$epochs)
  expect_equal(best$model$ensemble[[1]]$weights, best$model$ensemble[[1]]$best_weights)
  expect_equal(best$model$ensemble[[1]]$biases, best$model$ensemble[[1]]$best_biases)
  changed <- p
  changed$parts$test$actual <- rev(changed$parts$test$actual)
  changed$parts$test$y <- changed$parts$test$y[nrow(changed$parts$test$y):1, , drop = FALSE]
  invisible(capture.output(same <- h$wb_run(changed, cfg)))
  expect_equal(same$selected_epoch, best$selected_epoch)
  expect_equal(same$model$ensemble[[1]]$weights, best$model$ensemble[[1]]$weights)
  cfg$checkpoint <- "final"
  invisible(capture.output(final <- h$wb_run(p, cfg)))
  expect_equal(final$selected_epoch, cfg$epochs)
  expect_false(isTRUE(all.equal(best$model$ensemble[[1]]$weights, final$model$ensemble[[1]]$weights)))
  cfg$checkpoint <- NULL # Older saved configurations used final weights.
  invisible(capture.output(old <- h$wb_run(p, cfg)))
  expect_identical(old$config$checkpoint, "final")
  expect_equal(old$model$ensemble[[1]]$weights, final$model$ensemble[[1]]$weights)
})

test_that("exported public-API recipes reproduce preprocessing, predictions and metrics", {
  h <- workbench_fixture()$h
  for (mode in c("binary", "multiclass", "regression")) {
    data <- if (mode == "multiclass") iris else mtcars
    target <- switch(mode, binary = "am", multiclass = "Species", regression = "mpg")
    features <- if (mode == "multiclass") names(iris)[1:4] else c("wt", "hp")
    source <- tempfile("training data ", fileext = ".csv")
    utils::write.csv(data, source, row.names = FALSE)
    data <- utils::read.csv(source, check.names = FALSE)
    p <- h$wb_prepare(data, target, features, mode)
    p$source <- list(file = source)
    cfg <- h$wb_config(hidden = "4", epochs = 3, lr = .001)
    invisible(capture.output(r <- h$wb_run(p, cfg)))
    replay <- new.env(parent = globalenv())
    invisible(capture.output(eval(parse(text = r$recipe), envir = replay)))
    expect_equal(replay$parts$train$x, p$parts$train$x)
    expect_equal(replay$parts$test$y, p$parts$test$y)
    expect_equal(replay$selected_epoch, r$selected_epoch)
    expect_equal(replay$probabilities, h$wb_predictions(r$model, p, p$parts$test))
    expect_equal(unname(replay$metrics), r$evaluation$test$metrics$Value)
    expect_false(grepl("wb_run|wb_prepare", r$recipe))
    if (mode == "multiclass") {
      cfg$epochs <- 2L
      invisible(capture.output(warm <- h$wb_run(p, cfg, r)))
      replay2 <- new.env(parent = globalenv())
      invisible(capture.output(eval(parse(text = warm$recipe), envir = replay2)))
      expect_equal(replay2$probabilities, h$wb_predictions(warm$model, p, p$parts$test))
      expect_equal(replay2$selected_epoch, warm$selected_epoch)
    }
    cfg$checkpoint <- "final"
    invisible(capture.output(final <- h$wb_run(p, cfg)))
    replay_final <- new.env(parent = globalenv())
    invisible(capture.output(eval(parse(text = final$recipe), envir = replay_final)))
    expect_equal(replay_final$probabilities, h$wb_predictions(final$model, p, p$parts$test))
    unlink(source)
  }
})

test_that("preprocessing code also matches ordered splits and unscaled inputs", {
  h <- workbench_fixture()$h
  for (split in c("ordered", "random")) for (standardize in c(TRUE, FALSE)) {
    p <- h$wb_prepare(mtcars, "mpg", c("wt", "hp"), "regression", seed = 7,
      train_fraction = .55, validation_fraction = .25, split = split, standardize = standardize)
    env <- new.env(parent = globalenv()); env$data <- mtcars
    eval(parse(text = h$wb_features_code(p)), envir = env)
    expect_equal(env$parts, p$parts)
  }
})

test_that("checkpoint summaries report the selected tensors and validation criterion", {
  h <- workbench_fixture()$h
  for (mode in c("multiclass", "regression")) for (checkpoint in c("best_validation", "final")) {
    p <- if (mode == "multiclass") h$wb_prepare(iris, "Species", names(iris)[1:4], mode) else
      h$wb_prepare(mtcars, "mpg", c("wt", "hp"), mode)
    invisible(capture.output(r <- h$wb_run(p, h$wb_config(hidden = "4", epochs = 3,
      lr = .001, checkpoint = checkpoint))))
    selected <- h$wb_checkpoint(r)
    if (mode == "regression") {
      prediction <- h$wb_predictions(r$model, p, p$parts$validation)
      expect_equal(selected$score, mean((prediction - p$parts$validation$y)^2))
      if (checkpoint == "best_validation")
        expect_equal(selected$score, r$model$last_training$predicted_outputAndTime$best_val_loss)
    } else expect_equal(selected$score, r$evaluation$validation$metrics$Value[1])
    text <- h$wb_checkpoint_summary(r)
    expect_match(text, paste("Epoch", r$selected_epoch, "of 3"), fixed = TRUE)
    expect_match(text, if (mode == "regression") "validation loss (MSE)" else "validation accuracy", fixed = TRUE)
    expect_match(text, if (checkpoint == "final") "final epoch" else "Best validation checkpoint", fixed = TRUE)
  }
})

test_that("session action states and failed/cancelled workers preserve completed results", {
  skip_if_not_installed("shiny"); skip_if_not_installed("callr")
  f <- workbench_fixture(); h <- f$h
  sys.source(system.file("shiny", "benchmark", "workbench.R", package = "DDESONN"), h)
  server <- h$wb_server
  formals(server)$helper <- f$path
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  invisible(capture.output(completed <- h$wb_run(p, h$wb_config(hidden = "4", epochs = 2))))
  failure <- tempfile(fileext = ".R")
  writeLines('wb_run <- function(...) stop("non-finite predictions from worker")', failure)
  on.exit(unlink(failure), add = TRUE)
  start <- h$wb_start
  shiny::testServer(server, {
    session$flushReact()
    expect_false(action_states()$wb_start)
    expect_false(action_states()$wb_cancel)
    expect_false(action_states()$wb_save)
    prepared(p); result(completed); session$flushReact()
    expect_true(action_states()$wb_start)
    expect_true(action_states()$wb_save)
    expect_true(action_states()$wb_continue)
    original <- serialize(result(), NULL)
    # Launch a real failing callr worker, exercising the completion error path.
    h$wb_start <- function(helper, ...) start(failure, ...)
    launch(p, completed$config)
    session$flushReact()
    expect_false(action_states()$wb_start)
    expect_true(action_states()$wb_cancel)
    expect_false(action_states()$wb_continue)
    expect_false(action_states()$wb_load)
    expect_true(action_states()$wb_save) # Previous immutable model can still be saved.
    job()$process$wait(30000); session$elapse(400)
    expect_null(job())
    expect_identical(serialize(result(), NULL), original)
    expect_match(output$wb_train_error$html, "Technical details")
    expect_match(output$wb_train_error$html, "smaller learning rate")
    expect_match(output$wb_train_error$html, "non-finite")
    # A cancelled real worker cannot mutate the saved R6 model in this session.
    h$wb_start <- start
    cfg <- completed$config; cfg$epochs <- 5000L
    launch(p, cfg, result()); session$flushReact()
    session$setInputs(wb_cancel = 1)
    expect_null(job())
    expect_true(action_states()$wb_start)
    expect_false(action_states()$wb_cancel)
    expect_identical(serialize(result(), NULL), original)
    expect_match(status(), "cancelled")
    prepared(NULL); session$flushReact()
    expect_false(action_states()$wb_start)
    expect_true(action_states()$wb_save)
  })
  # A second session starts with no data preparation or completed-model state.
  shiny::testServer(server, {
    session$flushReact()
    expect_null(result()); expect_null(job()); expect_null(prepared())
    expect_false(action_states()$wb_save)
    expect_false(action_states()$wb_start)
  })
})


test_that("worker wrappers do not disguise numerical or shape errors as installation errors", {
  h <- workbench_fixture()$h
  shape <- h$wb_training_error(simpleError("in callr subprocess: colMeans: x must be an array of at least two dimensions"))
  expect_match(shape$message, "two or more numeric predictors")
  expect_match(shape$details, "colMeans")
  numerical <- h$wb_training_error(simpleError("in callr subprocess: non-finite predictions"))
  expect_match(numerical$message, "smaller learning rate")
  missing <- h$wb_training_error(simpleError("in callr subprocess: there is no package called DDESONN"))
  expect_match(missing$message, "same R library")
})

test_that("every workbench help topic has public R code without changing model or RNG state", {
  f <- workbench_fixture(); h <- f$h
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  cfg <- h$wb_config(hidden = "4", epochs = 2, lr = .001)
  invisible(capture.output(r <- h$wb_run(p, cfg)))
  model <- serialize(r$model, NULL)
  set.seed(791); rng <- .Random.seed
  topics <- h$wb_help_topics()
  expect_equal(length(unique(h$wb_help_key(topics))), length(topics))
  for (topic in topics) {
    code <- h$wb_help_code(topic, p, cfg, r)
    expect_true(nchar(code) > 120, info = topic)
    expect_no_error(parse(text = code))
    expect_false(grepl("wb_[[:alnum:]_]+\\(", code), info = topic)
    expect_no_error(parse(text = h$wb_help_code(topic)))
  }
  expect_identical(.Random.seed, rng)
  expect_identical(serialize(r$model, NULL), model)
  expect_error(h$wb_help_code("unregistered"), "Unknown")
  expect_match(h$wb_help_code("Target"), "Iris illustration")
  expect_match(h$wb_help_code("Additional epochs", more_epochs = 0), "Choose")
})

test_that("help preparation, fit and evaluation snippets reproduce the workbench model", {
  h <- workbench_fixture()$h
  for (mode in c("binary", "multiclass", "regression")) for (checkpoint in c("best_validation", "final")) {
    data <- if (mode == "multiclass") iris else mtcars
    target <- switch(mode, binary = "am", multiclass = "Species", regression = "mpg")
    features <- if (mode == "multiclass") names(iris)[1:4] else c("wt", "hp")
    p <- h$wb_prepare(data, target, features, mode, train_fraction = .55, validation_fraction = .25)
    cfg <- h$wb_config(hidden = "4", epochs = 2, lr = .001, checkpoint = checkpoint,
      reg = "L2", lambda = .001)
    invisible(capture.output(r <- h$wb_run(p, cfg)))
    e <- new.env(parent = globalenv()); e$data <- data
    invisible(capture.output(eval(parse(text = h$wb_help_code("Predictors", p, cfg)), e)))
    expect_equal(e$parts, p$parts)
    invisible(capture.output(eval(parse(text = h$wb_help_code("Regularization strength", p, cfg)), e)))
    invisible(capture.output(eval(parse(text = h$wb_help_code("Evaluation split", p, cfg,
      evaluation_split = "validation")), e)))
    expect_equal(e$selected_epoch, r$selected_epoch)
    expect_equal(e$probabilities, h$wb_predictions(r$model, p, p$parts$validation))
    expect_equal(unname(e$metrics), r$evaluation$validation$metrics$Value)
  }
})
