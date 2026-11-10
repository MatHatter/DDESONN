# [SSM] Verify the full prediction-loss chain, not an auxiliary embedding loss.
joint_fixture <- function(n = 6L) {
  set.seed(947)
  list(x = matrix(rnorm(n * 2L), n, 2L), y = matrix(rep(0:1, length.out = n), n, 1L),
       sequence = array(rnorm(n * 4L * 3L), c(n, 4L, 3L)))
}
joint_model <- function(d, ml = TRUE, members = 1L) {
  ddesonn_model(2L, 1L, hidden_sizes = if (ml) 4L else NULL,
    ML_NN = ml, num_networks = members, classification_mode = "binary",
    sequence_encoder = "ssm", sequence_data = d$sequence,
    sequence_length = 4L, ssm_state_dim = 5L, ssm_conv = 2L)
}
joint_fit <- function(model, d, epochs = 3L, ...) {
  ddesonn_fit(model, d$x, d$y, sequence_data = d$sequence,
    num_epochs = epochs, lr = .001, validation_metrics = FALSE,
    batch_normalize_data = FALSE, viewTables = FALSE,
    dropout_rates = list(NULL, NULL), ...)
}

test_that("final BCE loss differentiates through the head and repeated SSM windows", {
  set.seed(948)
  sequence <- array(rnorm(2L * 3L * 2L), c(2L, 3L, 2L))
  sequence <- sequence[c(2L, 1L, 2L, 2L), , , drop = FALSE]
  x <- matrix(rnorm(8L, sd = .2), 4L, 2L)
  y <- matrix(c(0, 1, 1, 0), 4L, 1L)
  e <- ddesonn_ssm_init(2L, 3L, 2L, 59L)
  e <- attr(ddesonn_ssm_encode(e, sequence, TRUE), "encoder")
  context <- DDESONN:::.ssm_joint_context(e, sequence, engineered_features = 2L)
  model <- ddesonn_model(5L, 1L, hidden_sizes = 4L, classification_mode = "binary",
                          activation_functions = c("tanh", "sigmoid"))
  net <- model$ensemble[[1L]]
  inputs <- DDESONN:::.ssm_joint_inputs(context, cbind(x, matrix(0, 4L, 3L)), cache = TRUE)$train
  acts <- c("tanh", "sigmoid")
  weights <- matrix(c(.5, 1, 1.5, 2), 4L, 1L)
  learned <- net$learn(inputs, y, lr = .001, CLASSIFICATION_MODE = "binary",
                        activation_functions = acts, dropout_rates = list(NULL, NULL),
                        sample_weights = weights, verbose = FALSE)
  loss <- function(input) {
    p <- net$predict(input, net$weights, net$biases,
                      activation_functions_predict = acts)$predicted_output
    -sum(weights * (y * log(p) + (1 - y) * log(1 - p))) / nrow(y)
  }
  for (j in seq_len(ncol(inputs))) {
    plus <- minus <- inputs
    plus[2L, j] <- plus[2L, j] + 1e-6
    minus[2L, j] <- minus[2L, j] - 1e-6
    expect_equal(learned$input_gradient[2L, j] / nrow(y),
                 (loss(plus) - loss(minus)) / 2e-6, tolerance = 2e-5)
  }
  DDESONN:::.ssm_joint_step(context, learned$input_gradient, .001)
  gradient <- lapply(context$encoder$optimizer$m, function(v) v / .1)
  for (nm in names(e$params)) {
    plus <- minus <- e
    plus$params[[nm]][1L] <- plus$params[[nm]][1L] + 1e-6
    minus$params[[nm]][1L] <- minus$params[[nm]][1L] - 1e-6
    numerical <- (loss(cbind(x, ddesonn_ssm_encode(plus, sequence))) -
                    loss(cbind(x, ddesonn_ssm_encode(minus, sequence)))) / 2e-6
    expect_equal(gradient[[nm]][1L], numerical, tolerance = 3e-5, info = nm)
  }
})

test_that("both ML and SL heads update their encoder every requested epoch", {
  d <- joint_fixture()
  for (ml in c(TRUE, FALSE)) {
    set.seed(949)
    model <- joint_model(d, ml)
    initial <- attr(model, "ssm_encoder")$params
    model <- joint_fit(model, d, epochs = 12L)
    encoder <- attr(model, "ssm_encoder")
    history <- attr(model$ensemble[[1L]], "ssm_training_history")
    expect_identical(attr(model, "ssm_training"), "joint")
    expect_equal(encoder$optimizer$step, 12L)
    expect_equal(history$completed_epochs, 12L)
    expect_true(all(history$gradient_norm > 0))
    expect_true(all(history$parameter_change > 0))
    expect_false(isTRUE(all.equal(initial, encoder$params)))
    expect_null(attr(model$ensemble[[1L]], "ssm_joint_context"))
    expect_true(all(is.finite(ddesonn_predict(model, d$x, sequence_data = d$sequence)$prediction)))
  }
})

test_that("a disconnected head gives zero encoder updates despite nonzero labels", {
  d <- joint_fixture()
  model <- joint_model(d, ml = FALSE)
  model$ensemble[[1L]]$weights[] <- 0
  initial <- attr(model, "ssm_encoder")$params
  model <- joint_fit(model, d, update_weights = FALSE, update_biases = FALSE)
  history <- attr(model$ensemble[[1L]], "ssm_training_history")
  expect_equal(history$gradient_norm, rep(0, 3L))
  expect_equal(history$parameter_change, rep(0, 3L))
  expect_identical(attr(model, "ssm_encoder")$params, initial)
})

test_that("validation never changes the encoder update or training scaler", {
  d <- joint_fixture()
  set.seed(950); a <- joint_model(d)
  set.seed(950); b <- joint_model(d)
  validation <- list(x = d$x + 2, y = 1 - d$y, sequence_data = d$sequence + 20)
  set.seed(951); a <- joint_fit(a, d)
  # Supply validation data, but do not use it to select a checkpoint.
  set.seed(951); b <- ddesonn_fit(b, d$x, d$y, validation = validation, sequence_data = d$sequence,
    num_epochs = 3L, lr = .001, validation_metrics = FALSE,
    batch_normalize_data = FALSE, viewTables = FALSE, dropout_rates = list(NULL, NULL))
  expect_equal(attr(a, "ssm_encoder"), attr(b, "ssm_encoder"))
  expect_equal(attr(a, "ssm_encoder")$scale$center, apply(d$sequence, 3L, mean))
})

test_that("joint ensemble members retain independent encoders through serialization", {
  d <- joint_fixture()
  model <- joint_fit(joint_model(d, members = 2L), d)
  encoders <- lapply(model$ensemble, attr, "ssm_encoder")
  expect_false(isTRUE(all.equal(encoders[[1L]]$params, encoders[[2L]]$params)))
  expect_true(all(vapply(model$ensemble, function(net) is.null(attr(net, "ssm_joint_context")), logical(1L))))
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(model, path)
  restored <- readRDS(path)
  expect_equal(ddesonn_predict(restored, d$x, sequence_data = d$sequence)$per_model,
               ddesonn_predict(model, d$x, sequence_data = d$sequence)$per_model)
})

test_that("separate pretraining remains an explicit comparison option", {
  d <- joint_fixture()
  model <- joint_fit(joint_model(d), d, ssm_training = "pretrain")
  expect_identical(attr(model, "ssm_training"), "pretrain")
  expect_null(attr(model$ensemble[[1L]], "ssm_training_history"))
  expect_equal(attr(model, "ssm_encoder")$optimizer$step, 3L)
})

test_that("best-epoch restoration restores the matching head and SSM together", {
  d <- joint_fixture()
  set.seed(952); first <- joint_model(d)
  set.seed(952); restored <- joint_model(d)
  fit_with_validation <- function(model, epochs) {
    set.seed(953)
    ddesonn_fit(model, d$x, d$y, sequence_data = d$sequence,
      validation = list(x = d$x, y = 1 - d$y, sequence_data = d$sequence),
      num_epochs = epochs, lr = .005, validation_metrics = TRUE,
      batch_normalize_data = FALSE, best_weights_on_latest_weights_off = TRUE,
      dropout_rates = list(NULL, NULL), viewTables = FALSE)
  }
  first <- fit_with_validation(first, 1L)
  restored <- fit_with_validation(restored, 8L)
  history <- attr(restored$ensemble[[1L]], "ssm_training_history")
  expect_equal(history$completed_epochs, 8L)
  expect_equal(history$selected_step, 1L)
  expect_equal(attr(restored, "ssm_encoder"), attr(first, "ssm_encoder"))
  expect_identical(restored$ensemble[[1L]]$weights, first$ensemble[[1L]]$weights)
  expect_equal(ddesonn_predict(restored, d$x, sequence_data = d$sequence)$prediction,
               ddesonn_predict(first, d$x, sequence_data = d$sequence)$prediction)
})

test_that("failed head training removes transient sequence caches", {
  d <- joint_fixture()
  model <- joint_model(d)
  expect_error(joint_fit(model, d, activation_functions = "not_an_activation"))
  expect_null(attr(model$ensemble[[1L]], "ssm_joint_context"))
})

test_that("single-layer input dropout is included in the encoder gradient", {
  set.seed(954)
  x <- matrix(rnorm(20L, sd = .2), 4L, 5L)
  y <- matrix(c(0, 1, 1, 0), 4L, 1L)
  model <- ddesonn_model(5L, 1L, hidden_sizes = NULL, ML_NN = FALSE,
                          classification_mode = "binary")
  net <- model$ensemble[[1L]]
  learn <- function(input) {
    set.seed(955)
    net$learn(input, y, lr = .001, CLASSIFICATION_MODE = "binary",
      activation_functions = "sigmoid", dropout_rates = list(.3),
      sample_weights = matrix(1, 4L, 1L), verbose = FALSE)
  }
  analytical <- learn(x)$input_gradient / nrow(x)
  loss <- function(input) {
    p <- learn(input)$learn_output
    -mean(y * log(p) + (1 - y) * log(1 - p))
  }
  for (j in seq_len(ncol(x))) {
    plus <- minus <- x
    plus[2L, j] <- plus[2L, j] + 1e-6
    minus[2L, j] <- minus[2L, j] - 1e-6
    expect_equal(analytical[2L, j], (loss(plus) - loss(minus)) / 2e-6, tolerance = 2e-5)
  }
})

test_that("refitting an ensemble retains each member's encoder and Adam state", {
  d <- joint_fixture()
  model <- joint_fit(joint_model(d, members = 2L), d)
  before <- lapply(model$ensemble, attr, "ssm_encoder")
  # A zero encoder LR makes preservation of both distinct parameter sets exact.
  model <- joint_fit(model, d, epochs = 2L, ssm_lr = 0)
  after <- lapply(model$ensemble, attr, "ssm_encoder")
  for (i in 1:2) {
    expect_identical(after[[i]]$params, before[[i]]$params)
    expect_equal(after[[i]]$optimizer$step, 5L)
  }
  expect_false(identical(after[[1L]]$params, after[[2L]]$params))
  expect_error(joint_fit(model, d, ssm_lr = -1), "ssm_lr")
})
