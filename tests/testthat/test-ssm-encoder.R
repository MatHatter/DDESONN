test_that("causal windows and dimensions are strict", {
  x <- matrix(seq_len(20), 10, 2); s <- ddesonn_sequence_data(x, 3)
  expect_equal(dim(s), c(8L,3L,2L)); expect_equal(s[1,,], x[1:3,])
  expect_error(ddesonn_ssm_forward(ddesonn_ssm_init(3), s), "feature")
})

test_that("forward is deterministic and reset is functional", {
  set.seed(2); x <- array(rnorm(40),c(5,4,2)); e <- ddesonn_ssm_init(2,3,3,7)
  expect_equal(ddesonn_ssm_forward(e,x)$embedding,ddesonn_ssm_forward(e,x)$embedding)
  expect_equal(ddesonn_ssm_reset_state(e)$state,numeric(3))
})

test_that("analytical convolution gradient agrees numerically", {
  set.seed(4); x<-array(rnorm(12),c(2,3,2));e<-ddesonn_ssm_init(2,2,3,9)
  fw<-ddesonn_ssm_forward(e,x,TRUE); g<-matrix(rnorm(4),2,2)
  an<-ddesonn_ssm_backward(e,fw$cache,g)$params$conv[1,1]
  loss<-function(v){q<-e;q$params$conv[1,1]<-v;sum(ddesonn_ssm_forward(q,x)$embedding*g)/2}
  eps<-1e-6;num<-(loss(e$params$conv[1,1]+eps)-loss(e$params$conv[1,1]-eps))/(2*eps)
  expect_equal(an,num,tolerance=2e-5)
})

test_that("backward supports non-square feature and state dimensions", {
  set.seed(817)
  x <- array(rnorm(3L * 48L * 13L), c(3L, 48L, 13L))
  e <- ddesonn_ssm_init(13L, 16L, 4L, 19L)
  fw <- ddesonn_ssm_forward(e, x, TRUE)
  upstream <- matrix(rnorm(3L * 16L), 3L, 16L)

  expect_no_error(gr <- ddesonn_ssm_backward(e, fw$cache, upstream))
  expect_equal(dim(fw$embedding), c(3L, 16L))
  expect_equal(dim(gr$params$W_B), c(13L, 16L))
  expect_equal(dim(gr$params$W_C), c(13L, 16L))
  expect_equal(dim(gr$params$W_dt), c(13L, 16L))
  expect_equal(dim(gr$params$D), c(13L, 16L))
  expect_equal(dim(gr$params$conv), c(4L, 13L))
  expect_equal(dim(gr$input), c(3L, 48L, 13L))
  expect_true(all(vapply(gr$params, function(value) all(is.finite(value)), logical(1L))))
})

test_that("non-square dimensions survive a complete Adam training cycle", {
  set.seed(818)
  x <- array(rnorm(2L * 48L * 13L), c(2L, 48L, 13L))
  e <- ddesonn_ssm_init(13L, 16L, 4L, 31L)
  first <- ddesonn_ssm_forward(e, x, TRUE)
  grads <- ddesonn_ssm_backward(e, first$cache, matrix(rnorm(32L), 2L, 16L))$params
  e <- DDESONN:::.ddesonn_ssm_update(e, grads)
  expect_no_error(second <- ddesonn_ssm_forward(e, x, TRUE))

  matrix_shapes <- list(conv=c(4L,13L), W_dt=c(13L,16L), W_B=c(13L,16L),
                        W_C=c(13L,16L), D=c(13L,16L))
  vector_lengths <- list(conv_bias=13L, b_dt=16L, b_B=16L, b_C=16L, A_log=16L)
  for (nm in names(matrix_shapes)) {
    expect_identical(dim(grads[[nm]]), matrix_shapes[[nm]], info=nm)
    expect_identical(dim(e$params[[nm]]), matrix_shapes[[nm]], info=nm)
    expect_identical(dim(e$optimizer$m[[nm]]), matrix_shapes[[nm]], info=nm)
    expect_identical(dim(e$optimizer$v[[nm]]), matrix_shapes[[nm]], info=nm)
  }
  for (nm in names(vector_lengths)) for (value in list(grads[[nm]], e$params[[nm]],
                                                        e$optimizer$m[[nm]], e$optimizer$v[[nm]])) {
    expect_null(dim(value), info=nm)
    expect_length(value, vector_lengths[[nm]])
  }
  expect_equal(dim(second$embedding), c(2L,16L))
})

test_that("shape invariants diagnose corrupted projection and optimizer shapes", {
  x <- array(0, c(1L,48L,13L)); e <- ddesonn_ssm_init(13L,16L,4L,37L)
  e$params$b_dt <- array(e$params$b_dt, 16L)
  expect_error(ddesonn_ssm_forward(e,x),
               "params\\$b_dt: expected length 16 vector; actual 16")

  e <- ddesonn_ssm_init(13L,16L,4L,37L)
  fw <- ddesonn_ssm_forward(e,x,TRUE)
  grads <- ddesonn_ssm_backward(e,fw$cache,matrix(1,1L,16L))$params
  e$optimizer$m$W_dt <- matrix(0,16L,13L)
  e$optimizer$v$W_dt <- matrix(0,13L,16L)
  expect_error(DDESONN:::.ddesonn_ssm_update(e,grads),
               "optimizer\\$m\\$W_dt: expected 13 x 16; actual 16 x 13")
})

test_that("W_dt finite difference holds for 13 feature by 16 state projection", {
  set.seed(819)
  x <- array(rnorm(48L * 13L), c(1L,48L,13L))
  e <- ddesonn_ssm_init(13L,16L,4L,41L)
  upstream <- matrix(rnorm(16L),1L,16L)
  fw <- ddesonn_ssm_forward(e,x,TRUE)
  analytical <- ddesonn_ssm_backward(e,fw$cache,upstream)$params$W_dt[13L,16L]
  value <- e$params$W_dt[13L,16L]; epsilon <- 1e-6
  loss <- function(candidate) sum(ddesonn_ssm_forward(candidate,x)$embedding*upstream)
  plus <- minus <- e
  plus$params$W_dt[13L,16L] <- value+epsilon
  minus$params$W_dt[13L,16L] <- value-epsilon
  expect_equal(analytical,(loss(plus)-loss(minus))/(2*epsilon),tolerance=3e-5)
})

test_that("non-square projection gradients agree with finite differences", {
  set.seed(23)
  x <- array(rnorm(2L * 3L * 3L), c(2L, 3L, 3L))
  e <- ddesonn_ssm_init(3L, 5L, 2L, 29L)
  upstream <- matrix(rnorm(2L * 5L), 2L, 5L)
  fw <- ddesonn_ssm_forward(e, x, TRUE)
  analytical <- ddesonn_ssm_backward(e, fw$cache, upstream)$params
  loss <- function(parameter, row, column, value) {
    candidate <- e
    candidate$params[[parameter]][row, column] <- value
    sum(ddesonn_ssm_forward(candidate, x)$embedding * upstream) / dim(x)[1L]
  }
  finite_difference <- function(parameter, row, column, epsilon = 1e-6) {
    value <- e$params[[parameter]][row, column]
    (loss(parameter, row, column, value + epsilon) -
       loss(parameter, row, column, value - epsilon)) / (2 * epsilon)
  }

  # W_B, W_C, and W_dt are feature-to-state projections; D is the
  # feature-to-output skip projection. All therefore have features x states.
  for (parameter in c("W_B", "W_C", "W_dt", "D")) {
    expect_equal(
      analytical[[parameter]][2L, 4L],
      finite_difference(parameter, 2L, 4L),
      tolerance = 3e-5,
      info = parameter
    )
  }
})

test_that("training scaling is reused after serialization", {
  x<-array(seq_len(48),c(4,3,4));e<-ddesonn_ssm_init(4,2,3,1)
  z<-ddesonn_ssm_encode(e,x,TRUE);e<-attr(z,"encoder");before<-ddesonn_ssm_encode(e,x)
  p<-tempfile();saveRDS(e,p);expect_equal(ddesonn_ssm_encode(readRDS(p),x),before)
  ddesonn_ssm_encode(e,x+100);expect_equal(e$scale$center,apply(x,3,mean))
})

test_that("none remains the default", {
  expect_identical(formals(ddesonn_model)$sequence_encoder,"none")
  expect_identical(formals(ddesonn_run)$sequence_encoder,"none")
})

test_that("all SSM parameter and input gradients match finite differences", {
  set.seed(820)
  # Include both non-square orientations and singleton sample/time/channel axes.
  for (shape in list(c(2L, 3L, 2L, 3L), c(1L, 3L, 3L, 2L),
                     c(2L, 1L, 1L, 1L))) {
    n <- shape[1L]; h <- shape[4L]
    x <- array(rnorm(prod(shape[1:3])), shape[1:3])
    e <- ddesonn_ssm_init(shape[3L], h, 2L, 43L)
    upstream <- matrix(rnorm(n * h), n, h)
    fw <- ddesonn_ssm_forward(e, x, TRUE)
    analytical <- ddesonn_ssm_backward(e, fw$cache, upstream)
    loss <- function(candidate, input = x) {
      sum(ddesonn_ssm_forward(candidate, input)$embedding * upstream) / n
    }
    epsilon <- 1e-6
    for (nm in names(e$params)) {
      numerical <- e$params[[nm]] * 0
      for (j in seq_along(numerical)) {
        plus <- minus <- e
        plus$params[[nm]][j] <- plus$params[[nm]][j] + epsilon
        minus$params[[nm]][j] <- minus$params[[nm]][j] - epsilon
        numerical[j] <- (loss(plus) - loss(minus)) / (2 * epsilon)
      }
      expect_equal(analytical$params[[nm]], numerical, tolerance = 2e-5, info = nm)
    }
    numerical <- x * 0
    for (j in seq_along(x)) {
      plus <- minus <- x
      plus[j] <- plus[j] + epsilon
      minus[j] <- minus[j] - epsilon
      numerical[j] <- (loss(e, plus) - loss(e, minus)) / (2 * epsilon)
    }
    expect_equal(analytical$input, numerical, tolerance = 2e-5)
  }
})

test_that("SSM training averages the MSE gradient over samples exactly once", {
  set.seed(821)
  x <- array(rnorm(2L * 3L * 2L), c(2L, 3L, 2L))
  y <- matrix(c(-0.2, 0.4), 2L, 1L)
  e <- ddesonn_ssm_init(2L, 3L, 2L, 47L)
  e <- attr(ddesonn_ssm_encode(e, x, TRUE), "encoder")
  scaled <- x
  for (j in 1:2) scaled[, , j] <- (x[, , j] - e$scale$center[j]) / e$scale$scale[j]
  fw <- ddesonn_ssm_forward(e, scaled, TRUE)
  target <- y[, rep(1L, 3L), drop = FALSE]
  grads <- ddesonn_ssm_backward(e, fw$cache, 2 * (fw$embedding - target) / 3)$params
  expected <- DDESONN:::.ddesonn_ssm_update(e, grads, lr = 0.001)
  actual <- DDESONN:::.ddesonn_ssm_train(e, x, y, epochs = 1L)
  expect_equal(actual, expected)

  # Duplicating a batch leaves a mean-loss gradient and Adam moments unchanged.
  duplicated <- x[c(1L, 2L, 1L, 2L), , , drop = FALSE]
  trained <- DDESONN:::.ddesonn_ssm_train(e, x, y, epochs = 3L)
  repeated <- DDESONN:::.ddesonn_ssm_train(e, duplicated, y[c(1L, 2L, 1L, 2L), , drop = FALSE], epochs = 3L)
  expect_equal(repeated, trained, tolerance = 1e-12)
  expect_equal(trained$optimizer$step, 3L)
  expect_lt(mean((ddesonn_ssm_encode(trained, x) - target)^2),
            mean((fw$embedding - target)^2))
})

test_that("repeated-window reuse preserves encoding and weighted training", {
  set.seed(822)
  unique_x <- array(rnorm(3L * 4L * 2L), c(3L, 4L, 2L))
  order <- c(3L, 1L, 3L, 2L, 3L, 1L)
  x <- unique_x[order, , , drop = FALSE]
  # Unequal window frequencies and different labels on the same window.
  y <- matrix(rnorm(12L), 6L, 2L)
  e <- ddesonn_ssm_init(2L, 3L, 2L, 53L)
  embedding <- ddesonn_ssm_encode(e, x, TRUE)
  e <- attr(embedding, "encoder")
  scaled <- x
  for (j in 1:2) scaled[, , j] <- (x[, , j] - e$scale$center[j]) / e$scale$scale[j]
  expect_equal(e$scale$center, apply(x, 3L, mean))
  expect_equal(e$scale$scale, apply(x, 3L, sd))
  expect_equal(unname(embedding), unname(structure(
    ddesonn_ssm_forward(e, scaled)$embedding, encoder = e)))

  expanded <- e
  target <- y[, c(1L, 2L, 1L), drop = FALSE]
  for (epoch in 1:3) {
    fw <- ddesonn_ssm_forward(expanded, scaled, TRUE)
    grads <- ddesonn_ssm_backward(expanded, fw$cache,
                                  2 * (fw$embedding - target) / 3)$params
    expanded <- DDESONN:::.ddesonn_ssm_update(expanded, grads)
  }
  compact <- DDESONN:::.ddesonn_ssm_train(e, x, y, epochs = 3L)
  expect_equal(compact, expanded, tolerance = 1e-12)
  expect_equal(DDESONN:::.ssm_sequence_groups(x)$group, c(1L, 2L, 1L, 3L, 1L, 2L))
})
