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
    expect_length(value, vector_lengths[[nm]], info=nm)
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
