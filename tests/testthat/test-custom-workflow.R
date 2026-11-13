custom_fixture <- function() {
  env <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), env)
  env
}

test_that("combination distinguishes probabilities, decisions and task-specific votes", {
  combine <- DDESONN:::.ddesonn_combine_predictions
  p <- lapply(c(.9, .2), function(x) matrix(x, 2, 1))
  expect_equal(combine(p, "binary"), matrix(.55, 2, 1))
  expect_equal(combine(p, "binary", "weighted", c(1, 3)), matrix(.375, 2, 1))
  expect_equal(combine(p, "binary", "proportion"), matrix(.5, 2, 1))
  expect_equal(combine(p, "binary", "majority"), matrix(0, 2, 1))
  expect_equal(combine(list(matrix(.5)), "binary", "proportion"), matrix(1))
  expect_error(combine(p, "regression", "majority"), "does not apply")
  expect_error(combine(p, "multiclass", "proportion"), "does not apply")
  expect_error(combine(p, "binary", "weighted", c(-1, 2)), "nonnegative")
  mc <- list(matrix(c(1, 0, 0), 1), matrix(c(0, 1, 0), 1), matrix(c(0, 0, 1), 1))
  expect_equal(combine(mc, "multiclass", "median"), matrix(rep(1/3, 3), 1))
  expect_equal(combine(mc, "multiclass", "majority"), matrix(c(1, 0, 0), 1))
  expect_equal(combine(list(matrix(1), matrix(5)), "regression", "weighted", c(3, 1)), matrix(2))
})

test_that("Custom validates settings and previews all MAIN and TEMP work", {
  h <- custom_fixture()
  base <- h$wb_config(epochs = 2)
  cfg <- h$wb_custom_config(base, "temp", 3, "1,2", 4)
  expect_match(h$wb_workload(cfg), "30 model fits × 2 epochs = 60 total member epochs", fixed = TRUE)
  expect_null(h$wb_config()$workflow)
  expect_error(h$wb_custom_config(base, seeds = "1,1"), "distinct")
  expect_error(h$wb_custom_config(base, members = 1.5, structure = "members"), "integer")
  expect_error(h$wb_custom_config(base, dropout = 1), "Dropout")
  expect_error(h$wb_custom_config(base, schedule = "exponential", minimum = 1), "minimum")
})

test_that("run previews distinguish seed values, complete runs and epochs", {
  h <- custom_fixture()
  base <- h$wb_config(epochs = 100)
  expect_match(h$wb_workload(base), "1 training run, 1 model fit, 100 epochs", fixed = TRUE)
  cfg <- h$wb_custom_config(base, seeds = "42,43")
  preview <- h$wb_workload(cfg)
  expect_match(preview, "2 training runs", fixed = TRUE)
  expect_match(preview, "Run 1 (seed 42): 100 epochs.", fixed = TRUE)
  expect_match(preview, "Run 2 (seed 43): 100 epochs.", fixed = TRUE)
  expect_match(preview, "Total: 2 × 100 = 200 epochs.", fixed = TRUE)
  expect_match(preview, "before the next run starts", fixed = TRUE)
  one <- h$wb_custom_config(base, seeds = "42")
  expect_match(h$wb_workload(one), "1 training run (one complete run", fixed = TRUE)
  big <- h$wb_custom_config(h$wb_config(epochs = 5000), structure = "temp",
    members = 100, temp = 100, seeds = paste(1:100, collapse = ","))
  expect_match(h$wb_workload(big), "5,050,000,000 total member epochs", fixed = TRUE)
})

test_that("Custom uses fixed rows, restores members, saves fusion and warm starts without organization", {
  h <- custom_fixture()
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  p$source <- list(example = "iris")
  original <- serialize(p, NULL)
  cfg <- h$wb_custom_config(h$wb_config(hidden = "2", epochs = 2), "members", 2, "1,2",
    fusion = "weighted", schedule = "exponential", interval = 1)
  lines <- capture.output(r <- h$wb_run(p, cfg))
  expect_false(any(grepl("\\[LEARN\\]|Grad norm|FUSE-DBG|CORE METRICS", lines)))
  expect_true(any(grepl("Epoch 2 .*Loss:", lines)))
  expect_identical(serialize(p, NULL), original)
  payload <- attr(r$model, "workbench_custom")
  expect_identical(attr(r$model, "classification_mode"), "multiclass")
  expect_identical(attributes(h$wb_clone_model(payload$models[[1]])), attributes(payload$models[[1]]))
  copied <- h$wb_clone_model(payload$models[[1]])
  original_weights <- payload$models[[1]]$ensemble[[1]]$weights
  copied$ensemble[[1]]$weights[[1]][1, 1] <- 123
  expect_identical(payload$models[[1]]$ensemble[[1]]$weights, original_weights)
  expect_length(payload$models, 2)
  for (model in payload$models) for (net in model$ensemble) {
    expect_equal(net$weights, net$best_weights)
    expect_equal(net$biases, net$best_biases)
  }
  expect_equal(vapply(payload$weights, sum, numeric(1)), c(1, 1))
  expect_equal(sort(unique(r$seed_comparison$seed)), c(1L, 2L))
  path <- tempfile(fileext = ".rds"); on.exit(unlink(path), add = TRUE)
  saveRDS(r, path); loaded <- h$wb_validate_bundle(readRDS(path))
  expect_equal(h$wb_new_predictions(loaded, iris), h$wb_new_predictions(r, iris))
  cfg$self_org <- TRUE; cfg$epochs <- 1L
  capture.output(warm <- h$wb_run(p, cfg, loaded))
  expect_false(warm$self_org_applied)
  expect_equal(warm$fit_runs, 2L)
  running <- cfg; running$epochs <- 5000L
  helper <- system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN")
  job <- h$wb_start(helper, p, running)
  h$wb_stop(job)
  expect_false(job$process$is_alive())
  expect_false(dir.exists(job$dir))
  expect_equal(h$wb_new_predictions(loaded, iris), h$wb_new_predictions(r, iris))
  # Execute the downloaded full recipe using the installed helpers and fixed split.
  env <- new.env(parent = globalenv())
  capture.output(eval(parse(text = r$recipe), envir = env))
  expect_equal(h$wb_new_predictions(env$bundle, iris), h$wb_new_predictions(r, iris))
})

test_that("source normalization and optimizer diagnostics respect verbosity", {
  model <- ddesonn_model(2, 1, hidden_sizes = integer(), ML_NN = FALSE)
  x <- matrix(1:12, 6, 2)
  expect_output(model$normalize_data(x, 1:2), NA)
  expect_output(model$normalize_data(x, 1:2, verbose = TRUE), "Before normalization")
  expect_output(DDESONN:::optimizers_log_update("sgd", 1, 1, "weights", matrix(1), verboseLow = TRUE), NA)
})


test_that("file-based fusion is quiet and binary majority requires more than half", {
  path <- tempfile(fileext = ".rds"); on.exit(unlink(path), add = TRUE)
  df <- data.frame(run_index = 1L, seed = 1L, model_slot = rep(1:2, each = 2),
    obs_index = rep(1:2, 2), split = "test", y_prob = c(.9, .1, .1, .9))
  saveRDS(df, path)
  expect_output(result <- DDESONN:::DDESONN_fuse_from_agg(path, 1, 1, c(0, 1),
    methods = "vote_hard", classification_mode = "binary"), NA)
  expect_equal(unname(result$predictions$Ensemble_vote_hard), matrix(c(0, 0), 2, 1))
})

test_that("Custom binary and regression use the same saved inference path", {
  h <- custom_fixture()
  for (mode in c("binary", "regression")) {
    target <- if (mode == "binary") "am" else "mpg"
    p <- h$wb_prepare(mtcars, target, c("wt", "hp"), mode)
    cfg <- h$wb_custom_config(h$wb_config(hidden = "2", epochs = 1, checkpoint = "final"),
      "members", 2, "3", fusion = "weighted", dropout = .1)
    capture.output(r <- h$wb_run(p, cfg))
    before <- h$wb_new_predictions(r, mtcars)
    loaded <- h$wb_validate_bundle(unserialize(serialize(r, NULL)))
    expect_equal(h$wb_new_predictions(loaded, mtcars), before)
    expect_true(all(is.finite(r$evaluation$validation$metrics$Value)))
    expect_equal(sum(r$member_weights[[1]]), 1)
  }
})
