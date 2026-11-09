# Fixed two-step results calculated independently from the published recurrences.
# All matrices below use R column-major ordering; signs and magnitudes differ by
# element so reshaping or recycling an update cannot accidentally pass.
optimizer_examples <- function() list(
  sgd = list(c(.197,-.294,.391,-.488,.585,-.682), c(.203,-.297,.403,-.494,.588,-.691)),
  sgd_momentum = list(c(.197,-.294,.391,-.488,.585,-.682), c(.2006,-.2922,.3958,-.4844,.576,-.6766)),
  nag = list(c(.1946,-.2892,.3838,-.4784,.573,-.6676), c(.20348,-.29076,.39964,-.48152,.5688,-.67228)),
  adagrad = list(c(.170000299997,-.27000014999925,.370000099999667,-.470000074999813,.57000005999988,-.670000049999917),
                 c(.196832995727534,-.283416497864517,.394000051999763,-.483416452864878,.575883532515587,-.683416437864945)),
  rmsprop = list(c(.105134670100083,-.205133170171232,.305132670184408,-.405132420189019,.505132270191154,-.605132170192313),
                 c(.190847731342066,-.249365104853404,.382430967430404,-.449364680950936,.524701984724016,-.649364539648547)),
  adam = list(c(.170000299997,-.27000014999925,.370000099999667,-.470000074999813,.57000005999988,-.670000049999917),
              c(.182454219428007,-.263573345434161,.377487582011661,-.463573249782749,.555782939682924,-.663573217898832)),
  adadelta = list(c(.199905179068814,-.299905143526513,.399905136940219,-.499905134634691,.599905133567505,-.699905132987783),
                  c(.200026354731844,-.29996768714409,.400014448640558,-.499967686281801,.599932808638171,-.699967686122138)),
  lamb = list(c(.185596514102808,-.285582067349602,.385567668511784,-.485553281652903,.58553889958561,-.685524519914116),
              c(.203857654198678,-.275989956678991,.396425474236327,-.475872780388601,.564340036027257,-.675755648324103)),
  ftrl = list(c(.0151776788148687,-.044738815296176,.0851341998000462,-.134042705134614,.189762047590482,-.251015434606011),
              c(.0200800128904947,-.047189982333989,.0931326001199822,-.138187743922956,.191748536879083,-.256401042989676)),
  lookahead = list(c(.170000299997,-.27000014999925,.370000099999667,-.470000074999813,.57000005999988,-.670000049999917),
                   c(.191227109714004,-.28178667271708,.388743791005831,-.481786624891374,.577891469841462,-.681786608949416))
)

optimizer_fixture <- function(ml = TRUE) {
  e <- new.env(parent = emptyenv())
  p <- c(.2,-.3,.4,-.5,.6,-.7)
  e$ML_NN <- ml
  e$num_layers <- if (ml) 2L else 1L
  e$lambda <- .2
  e$groups <- NULL
  e$weights <- if (ml) list(matrix(p,2,3),matrix(p,3,2)) else matrix(p,2,3)
  e$biases <- if (ml) list(matrix(p,1,6),matrix(p,1,6)) else matrix(p,1,6)
  e
}

for (method in names(optimizer_examples())) local({
  optimizer <- method
  test_that(paste(optimizer, "updates SL/ML weights and biases exactly once in order"), {
    expected <- optimizer_examples()[[optimizer]]
    for (ml in c(FALSE, TRUE)) {
      model <- optimizer_fixture(ml)
      ws <- bs <- vector("list", model$num_layers)
      for (step in 1:2) {
        values <- if (step == 1) c(.1,-.2,.3,-.4,.5,-.6) else c(-.2,.1,-.4,.2,-.1,.3)
        wg <- if (ml) list(matrix(values,2,3),matrix(values,3,2)) else list(matrix(values,2,3))
        bg <- rep(list(matrix(values,1,6)),model$num_layers)
        ws <- DDESONN:::update_weights_block(model, TRUE, optimizer, ws, wg,
                  .03, NULL, .8, .9, 1e-6, 99L, 2L)$updated_optimizer_params
        bs <- DDESONN:::update_biases_block(model, TRUE, optimizer, bs, bg,
                  errors = rep(list(matrix(100,7,6)),model$num_layers),
                  .03, NULL, .8, .9, 1e-6, 99L, 2L)$updated_optimizer_params
        weights <- if (ml) model$weights else list(model$weights)
        biases <- if (ml) model$biases else list(model$biases)
        for (layer in seq_len(model$num_layers)) {
          expect_equal(as.numeric(weights[[layer]]), expected[[step]], tolerance=1e-12)
          expect_equal(as.numeric(biases[[layer]]), expected[[step]], tolerance=1e-12)
          expect_identical(dim(weights[[layer]]),dim(wg[[layer]]))
          expect_identical(dim(biases[[layer]]),c(1L,6L))
          expect_equal(ws[[layer]]$param,weights[[layer]])
          expect_equal(bs[[layer]]$param,biases[[layer]])
          # Epoch=99 deliberately differs from update count.
          expect_equal(ws[[layer]]$t,step)
          expect_equal(bs[[layer]]$t,step)
          for (state in intersect(c("m","v","r","momentum","z","n"),names(ws[[layer]]))) {
            expect_identical(dim(ws[[layer]][[state]][[1]]),dim(weights[[layer]]))
            expect_identical(dim(bs[[layer]][[state]][[1]]),dim(biases[[layer]]))
          }
        }
      }
      if (optimizer == "adagrad") expect_equal(ws[[1]]$r[[1]], matrix(c(.05,.05,.25,.20,.26,.45),2,3))
      if (optimizer == "lookahead") {
        expect_equal(ws[[1]]$slow_weights,ws[[1]]$param)
        expect_equal(ws[[1]]$lookahead_counter,0)
      }
    }
  })
})

test_that("initialization binds real values without changing random state", {
  set.seed(2026); rng <- .Random.seed
  p <- matrix(seq_len(6)/10,2,3)
  for (method in setdiff(ddesonn_optimizer_options(),"adagrad_x")) {
    s <- DDESONN:::initialize_optimizer_params(method,list(dim(p)),2L,1L,parameter=p)
    expect_identical(.Random.seed,rng)
    expect_equal(s$param,p)
    if (method == "lookahead") expect_equal(s$slow_weights,p)
  }
  expect_true("adadelta" %in% ddesonn_optimizer_options())
  expect_error(DDESONN:::initialize_optimizer_params("adam",c(0,3),2,1),"dimensions")
})

test_that("invalid gradients and state fail rather than recycle or reset", {
  model <- optimizer_fixture(FALSE)
  s <- list(DDESONN:::initialize_optimizer_params("adam",c(2,3),2,1,parameter=model$weights))
  call <- function(g,state=s) DDESONN:::apply_optimizer_update("adam",state,g,.01,.9,.99,1e-8,1,model,1,"weights")
  before <- model$weights
  expect_error(call(matrix(1,3,2)),"shape")
  expect_error(call(1),"shape")
  expect_error(call(matrix(NA_real_,2,3)),"finite")
  s[[1]]$m <- list(matrix(0,1,6))
  expect_error(call(matrix(1,2,3),s),"state m")
  expect_equal(model$weights,before)
})

test_that("regularization is part of one gradient and update flags freeze tensors", {
  for (ml in c(FALSE,TRUE)) {
    model <- optimizer_fixture(ml)
    w <- if (ml) model$weights else list(model$weights)
    b <- if (ml) model$biases else list(model$biases)
    wg <- lapply(w,function(p) p*0+.1)
    bg <- lapply(b,function(p) p*0-.1)
    DDESONN:::update_weights_block(model,TRUE,"sgd",NULL,wg,.03,"L2",.8,.9,1e-6,1,2)
    DDESONN:::update_biases_block(model,TRUE,"sgd",NULL,bg,NULL,.03,"L2",.8,.9,1e-6,1,2)
    actual_w <- if (ml) model$weights else list(model$weights)
    actual_b <- if (ml) model$biases else list(model$biases)
    for (i in seq_along(w)) {
      expect_equal(actual_w[[i]],w[[i]]-.03*(wg[[i]]+.2*w[[i]]))
      expect_equal(actual_b[[i]],b[[i]]-.03*(bg[[i]]+.2*b[[i]]))
    }
    DDESONN:::update_weights_block(model,FALSE,"adam",NULL,NULL,.03,NULL,.8,.9,1e-6,1,2)
    DDESONN:::update_biases_block(model,FALSE,"adam",NULL,NULL,NULL,.03,NULL,.8,.9,1e-6,1,2)
    expect_equal(if (ml) model$weights else list(model$weights),actual_w)
    expect_equal(if (ml) model$biases else list(model$biases),actual_b)
  }
})

test_that("Adagrad bias gradients do not depend on duplicating the error batch", {
  a <- optimizer_fixture(); b <- optimizer_fixture()
  g <- rep(list(matrix(c(.1,-.2,.3,-.4,.5,-.6),1,6)),2)
  update <- function(model,errors) DDESONN:::update_biases_block(model,TRUE,"adagrad",NULL,g,errors,.03,NULL,.8,.9,1e-6,1,2)
  update(a,rep(list(matrix(1,2,6)),2))
  update(b,rep(list(matrix(1,20,6)),2))
  expect_equal(a$biases,b$biases)
})

test_that("bias gradients preserve unit order for column and vector storage", {
  for (ml in c(FALSE,TRUE)) {
    model <- optimizer_fixture(ml)
    model$biases <- if (ml) list(matrix(1:6,6,1),matrix(1:6,6,1)) else as.numeric(1:6)
    g <- rep(list(matrix(c(.1,-.2,.3,-.4,.5,-.6),1,6)),model$num_layers)
    DDESONN:::update_biases_block(model,TRUE,"sgd",NULL,g,NULL,.03,NULL,.8,.9,1e-6,1,2)
    b <- if (ml) model$biases else list(model$biases)
    for (p in b) {
      expect_equal(as.numeric(p),1:6-.03*as.numeric(g[[1]]))
      expect_identical(dim(p),c(6L,1L))
    }
  }
})

test_that("FTRL threshold and LAMB rate apply to biases as well as weights", {
  p <- matrix(0,1,2)
  s <- DDESONN:::initialize_optimizer_params("ftrl",dim(p),2,1,parameter=p)
  r <- DDESONN:::ftrl_update(s,list(matrix(c(.005,-.005),1)),.1,lambda1=.01)
  expect_equal(r$params$param,p)
  expect_equal(r$weights_update,r$biases_update)
  p <- matrix(c(.3,-.4),1,2)
  s <- DDESONN:::initialize_optimizer_params("lamb",dim(p),2,1,parameter=p)
  a <- DDESONN:::lamb_update(s,list(p),.01,.8,.9,1e-6,0)
  b <- DDESONN:::lamb_update(s,list(p),.02,.8,.9,1e-6,0)
  expect_equal(b$weights_update[[1]],2*a$weights_update[[1]])
  expect_equal(a$weights_update,a$biases_update)
})

test_that("training_overrides selects every optimizer through ddesonn_run", {
  set.seed(107)
  x <- matrix(rnorm(24),8,3); y <- matrix(seq(-.3,.4,length.out=8),8,1)
  calls <- character()
  original <- DDESONN:::apply_optimizer_update
  testthat::local_mocked_bindings(apply_optimizer_update=function(optimizer, ...) {
    calls <<- c(calls,optimizer)
    original(optimizer,...)
  }, .package="DDESONN")
  for (ml in c(TRUE,FALSE)) for (method in setdiff(ddesonn_optimizer_options(),"adagrad_x")) {
    calls <- character()
    capture.output(result <- ddesonn_run(x,y,classification_mode="regression",
      hidden_sizes=if (ml) c(3L,2L) else NULL,seeds=17L,save_models=FALSE,
      model_overrides=list(ML_NN=ml),
      training_overrides=list(optimizer=method,num_epochs=2L,self_org=FALSE,
        ML_NN=ml,lr=.001,lr_decay_rate=1,reg_type=NULL,dropout_rates=NULL,
        validation_metrics=FALSE,batch_normalize_data=FALSE,
        viewTables=FALSE,verbose=FALSE)))
    expect_true(length(calls)>0)
    expect_identical(unique(calls),method)
    pred <- ddesonn_predict(result$model,x)$prediction
    expect_equal(dim(pred),c(8L,1L))
    expect_true(all(is.finite(pred)))
  }
})

test_that("standard Adagrad returns predictions even without a best snapshot", {
  # Correcting the optimizer changes which epoch improves classification
  # accuracy. This documented example previously leaked learn_result when no
  # post-update snapshot beat the pre-update accuracy.
  set.seed(1)
  x <- mtcars[,c("disp","hp","wt","qsec","drat")]
  model <- ddesonn_model(ncol(x),1,hidden_sizes=8)
  capture.output(model <- ddesonn_fit(model,x,mtcars$am,num_epochs=1,
                                    lr=.05,validation_metrics=FALSE,optimizer="adagrad"))
  prediction <- ddesonn_predict(model,x)$prediction
  expect_equal(dim(prediction),c(32L,1L))
  expect_true(all(is.finite(prediction)))
})
