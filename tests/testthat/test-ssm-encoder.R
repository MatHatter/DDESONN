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
