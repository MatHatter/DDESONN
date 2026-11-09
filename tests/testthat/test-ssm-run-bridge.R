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
  train_center <- vapply(
    seq_len(dim(d$train_s)[3L]),
    function(j) mean(d$train_s[, , j]),
    numeric(1L)
  )
  train_scale <- vapply(
    seq_len(dim(d$train_s)[3L]),
    function(j) stats::sd(as.vector(d$train_s[, , j])),
    numeric(1L)
  )
  expect_equal(encoder$scale$center, train_center)
  expect_equal(encoder$scale$scale, train_scale)
  expect_false(isTRUE(all.equal(
    encoder$scale$center,
    vapply(
      seq_len(dim(d$validation_s)[3L]),
      function(j) mean(d$validation_s[, , j]),
      numeric(1L)
    )
  )))
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
  expect_equal(
    attr(restored, "ssm_encoder")$scale,
    attr(model, "ssm_encoder")$scale
  )

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

test_that("binary SSM training persists non-square parameters and Adam state", {
  skip_if_not_installed("knitr")
  d <- ssm_bridge_fixture()
  result <- ddesonn_run(
    x = d$train_x, y = matrix(rep(0:1, 4L), 8L, 1L),
    classification_mode = "binary", hidden_sizes = c(4L, 2L), seeds = 111L,
    validation = list(x = d$validation_x, y = matrix(rep(0:1, 2L), 4L, 1L),
                      sequence_data = d$validation_s),
    test = list(x = d$test_x, y = matrix(c(0, 1, 0), 3L, 1L), sequence_data = d$test_s),
    prediction_data = list(x = d$test_x[1:2, , drop = FALSE],
                           sequence_data = d$test_s[1:2, , , drop = FALSE]),
    sequence_encoder = "ssm", sequence_data = d$train_s,
    sequence_length = 48L, ssm_state_dim = 16L, ssm_conv = 4L,
    training_overrides = list(num_epochs = 3L, lr = 0.001,
      optimizer = "adagrad", loss_type = "CrossEntropy",
      preprocessScaledData = list(already_scaled = TRUE),
      batch_normalize_data = FALSE, validation_metrics = TRUE, viewTables = FALSE),
    save_models = FALSE
  )
  model <- result$model
  enc <- attr(model, "ssm_encoder")
  expect_equal(enc$optimizer$step, 3L)
  expect_equal(enc$scale$center, apply(d$train_s, 3L, mean))
  expect_equal(enc$scale$scale, apply(d$train_s, 3L, sd))
  for (nm in c("params", "m", "v")) {
    values <- if (nm == "params") enc$params else enc$optimizer[[nm]]
    expect_identical(dim(values$W_dt), c(13L, 16L))
    expect_null(dim(values$b_dt))
    expect_true(all(vapply(values, function(x) all(is.finite(x)), logical(1L))))
  }
  # The moments prove the supervised encoder pass updated the encoder.
  expect_gt(sum(abs(enc$optimizer$m$conv)), 0)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(model, path)
  restored <- readRDS(path)
  expect_equal(attr(restored, "ssm_encoder"), enc)
  before <- ddesonn_predict(model, d$test_x, sequence_data = d$test_s)$prediction
  after <- predict(restored, d$test_x, sequence_data = d$test_s)$predicted_output
  expect_equal(as.numeric(after), as.numeric(before))
  expect_true(all(is.finite(before)))
  expect_true(all(before >= 0 & before <= 1))
  expect_equal(nrow(result$runs[[1L]]$predictions$test), 3L)
  expect_length(result$runs[[1L]]$main$predictions$prediction, 2L)
})

test_that("promoted networks retain their own SSM encoder at prediction", {
  d <- ssm_bridge_fixture()
  main <- run_ssm_bridge_fixture(d)$model
  # A different training split gives the candidate a different scaler/encoder.
  candidate_data <- d
  candidate_data$train_s <- d$train_s + 5
  candidate <- run_ssm_bridge_fixture(candidate_data)$model
  candidate_encoder <- attr(candidate, "ssm_encoder")
  expect_equal(attr(candidate$ensemble[[1L]], "ssm_encoder"), candidate_encoder)
  expected <- ddesonn_predict(candidate, d$test_x, sequence_data = d$test_s)$prediction
  # TEMP promotion replaces the member while MAIN retains its model attributes.
  main$ensemble[[1L]] <- candidate$ensemble[[1L]]
  expect_false(identical(attr(main, "ssm_encoder"), candidate_encoder))
  actual <- ddesonn_predict(main, d$test_x, sequence_data = d$test_s)$prediction
  expect_equal(actual, expected)

  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(main, path)
  expect_equal(ddesonn_predict(readRDS(path), d$test_x, sequence_data = d$test_s)$prediction,
               expected)
  # Saved models from before member-level pairing still use their model encoder.
  attr(candidate$ensemble[[1L]], "ssm_encoder") <- NULL
  expect_equal(ddesonn_predict(candidate, d$test_x, sequence_data = d$test_s)$prediction,
               expected)
})
