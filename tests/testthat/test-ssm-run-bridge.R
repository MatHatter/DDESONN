# [SSM] This entire test file covers the SSM sequence-data bridge.
ssm_bridge_fixture <- function() {
  set.seed(817)
  make_sequence <- function(n, offset) {
    array(rnorm(n * 48L * 13L, mean = offset), c(n, 48L, 13L))
  }
  list(
    train_x = matrix(rnorm(8L * 2L), 8L, 2L),
    train_y = matrix(rnorm(8L), 8L, 1L),
    train_s = make_sequence(8L, 0),
    validation_x = matrix(rnorm(4L * 2L), 4L, 2L),
    validation_y = matrix(rnorm(4L), 4L, 1L),
    validation_s = make_sequence(4L, 10),
    test_x = matrix(rnorm(3L * 2L), 3L, 2L),
    test_y = matrix(rnorm(3L), 3L, 1L),
    test_s = make_sequence(3L, 20)
  )
}

run_ssm_bridge_fixture <- function(d) {
  ddesonn_run(
    x = d$train_x,
    y = d$train_y,
    classification_mode = "regression",
    hidden_sizes = 2L,
    seeds = 17L,
    validation = list(
      x = d$validation_x,
      y = d$validation_y,
      sequence_data = d$validation_s
    ),
    test = list(x = d$test_x, y = d$test_y, sequence_data = d$test_s),
    sequence_encoder = "ssm",
    sequence_data = d$train_s,
    sequence_length = 48L,
    ssm_state_dim = 2L,
    ssm_conv = 2L,
    training_overrides = list(
      num_epochs = 1L,
      validation_metrics = FALSE,
      batch_normalize_data = FALSE,
      viewTables = FALSE,
      verbose = FALSE
    ),
    save_models = FALSE
  )
}

test_that("SSM run bridges separate train, validation, and test sequences", {
  d <- ssm_bridge_fixture()
  result <- run_ssm_bridge_fixture(d)

  expect_s3_class(result, "ddesonn_run_result")
  split_predictions <- result$runs[[1L]]$predictions
  expect_equal(nrow(split_predictions$train), nrow(d$train_x))
  expect_equal(nrow(split_predictions$validation), nrow(d$validation_x))
  expect_equal(nrow(split_predictions$test), nrow(d$test_x))

  encoder <- attr(result$model, "ssm_encoder")
  expect_false(is.null(encoder$scale))
  expect_equal(
    nrow(ddesonn_ssm_encode(encoder, d$validation_s)),
    nrow(d$validation_x)
  )
  expect_equal(nrow(ddesonn_ssm_encode(encoder, d$test_s)), nrow(d$test_x))
})

test_that("SSM run reports missing split sequences without substituting train data", {
  d <- ssm_bridge_fixture()
  common <- list(
    x = d$train_x,
    y = d$train_y,
    classification_mode = "regression",
    hidden_sizes = 2L,
    sequence_encoder = "ssm",
    sequence_data = d$train_s,
    sequence_length = 48L,
    ssm_state_dim = 2L,
    save_models = FALSE
  )

  expect_error(
    do.call(ddesonn_run, c(common, list(
      validation = list(x = d$validation_x, y = d$validation_y)
    ))),
    "validation\\$sequence_data is required for SSM runs"
  )
  expect_error(
    do.call(ddesonn_run, c(common, list(
      test = list(x = d$test_x, y = d$test_y)
    ))),
    "test\\$sequence_data is required for SSM runs"
  )
})

test_that("sequence_encoder none retains split API behavior", {
  d <- ssm_bridge_fixture()
  expect_no_error(ddesonn_run(
    x = d$train_x,
    y = d$train_y,
    classification_mode = "regression",
    hidden_sizes = 2L,
    validation = list(x = d$validation_x, y = d$validation_y),
    test = list(x = d$test_x, y = d$test_y),
    sequence_encoder = "none",
    training_overrides = list(
      num_epochs = 1L,
      validation_metrics = FALSE,
      batch_normalize_data = FALSE,
      viewTables = FALSE
    ),
    save_models = FALSE
  ))
})

test_that("saved SSM run models predict with newly supplied sequences", {
  d <- ssm_bridge_fixture()
  model <- run_ssm_bridge_fixture(d)$model
  path <- tempfile(fileext = ".rds")
  saveRDS(model, path)
  restored <- readRDS(path)

  new_x <- matrix(rnorm(2L * ncol(d$train_x)), 2L, ncol(d$train_x))
  new_sequence <- array(rnorm(2L * 48L * 13L), c(2L, 48L, 13L))
  prediction <- ddesonn_predict(
    restored,
    new_x,
    sequence_data = new_sequence,
    aggregate = "mean"
  )
  expect_length(prediction$prediction, 2L)
  expect_error(
    ddesonn_predict(restored, new_x),
    "sequence_data is required for SSM prediction"
  )
})
