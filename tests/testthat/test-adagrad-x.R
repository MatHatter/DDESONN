# Compatibility fixtures were produced by the unmodified package at the commit
# recorded in fixtures/adagrad-x-baseline.rds, not by the implementation here.
adagrad_x_fixture <- function(ml) {
  e <- new.env(parent=emptyenv())
  e$ML_NN <- ml; e$num_layers <- if (ml) 2L else 1L
  e$lambda <- .2
  e$groups <- if (ml) list(list(1:3),list(1:2)) else NULL
  e$weights <- if (ml) list(matrix(c(.2,-.3,.4,-.5,.6,-.7),2,3),
    matrix(c(-.4,.2,-.1,.3,.5,-.6),3,2)) else matrix(c(.2,-.3,.4,-.5,.6,-.7),2,3)
  e$biases <- if (ml) list(matrix(c(.1,-.2,.3),3,1),matrix(c(-.1,.2),2,1)) else c(.1,-.2,.3)
  e
}

test_that("Adagrad X matches pre-fix updates, state, shapes and RNG exactly", {
  baseline <- readRDS(test_path("fixtures","adagrad-x-baseline.rds"))
  expect_identical(baseline$commit,"5dd541a7fe5c6fc56ddc5b7d4a3e8b0ec7d55373")
  for (case in baseline$cases) {
    model <- adagrad_x_fixture(case$ml)
    ws <- bs <- vector("list",model$num_layers)
    set.seed(901L)
    for (step in 1:3) {
      wg <- if (case$ml) list(matrix(c(.1,-.2,.3,-.4,.5,-.6)*step,2,3),
        matrix(c(-.2,.1,-.4,.2,-.1,.3)*step,3,2)) else list(matrix(c(.1,-.2,.3,-.4,.5,-.6)*step,2,3))
      errors <- if (case$ml) list(matrix(seq(-.6,.5,length.out=12)*step,4,3),
        matrix(seq(-.3,.4,length.out=8)*step,4,2)) else list(matrix(seq(-.6,.5,length.out=12)*step,4,3))
      bg <- lapply(errors,function(x) matrix(colMeans(x),1))
      ws <- DDESONN:::update_weights_block(model,case$weights,"adagrad_x",ws,wg,
        .03,case$reg,.8,.9,1e-6,step,2)$updated_optimizer_params
      bs <- DDESONN:::update_biases_block(model,case$biases,"adagrad_x",bs,bg,errors,
        .03,case$reg,.8,.9,1e-6,step,2)$updated_optimizer_params
      expect_identical(list(weights=model$weights,biases=model$biases,
        groups=model$groups,ws=ws,bs=bs,rng=.Random.seed),case$snapshots[[step]])
    }
  }
})

test_that("Adagrad X is the default and standard Adagrad remains explicit", {
  for (mode in c("binary","multiclass","regression"))
    expect_identical(ddesonn_training_defaults(mode)$optimizer,"adagrad_x")
  expect_true(all(c("adagrad_x","adagrad") %in% ddesonn_optimizer_options()))
})

test_that("default training reproduces pre-fix Adagrad predictions and RNG", {
  baseline <- readRDS(test_path("fixtures","adagrad-x-baseline.rds"))
  for (case in baseline$end_to_end) {
    ml <- case$ml
    capture.output(result <- ddesonn_run(case$x,case$y,classification_mode="regression",
      hidden_sizes=if (ml) c(3L,2L) else NULL,seeds=17L,save_models=FALSE,
      model_overrides=list(ML_NN=ml),
      training_overrides=list(ML_NN=ml,num_epochs=3L,self_org=FALSE,
        lr=.003,lr_decay_rate=1,reg_type="L2",
        dropout_rates=if (ml) list(.2,.2,0) else NULL,
        validation_metrics=FALSE,batch_normalize_data=FALSE,viewTables=FALSE,verbose=FALSE)))
    prediction <- ddesonn_predict(result$model,case$x)$prediction
    expect_identical(prediction,case$prediction)
    expect_identical(lapply(result$model$ensemble,function(n)n$weights),case$weights)
    expect_identical(lapply(result$model$ensemble,function(n)n$biases),case$biases)
    expect_identical(.Random.seed,case$rng)
  }
})
