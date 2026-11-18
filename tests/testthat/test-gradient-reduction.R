gradient_fixture <- function(mode = "binary", ml = TRUE) {
  model <- ddesonn_model(2, if (mode == "multiclass") 2 else 1,
    hidden_sizes = if (ml) 2 else integer(), ML_NN = ml, classification_mode = mode)
  net <- model$ensemble[[1]]
  net$weights <- if (ml) list(matrix(c(.2,.3,-.1,.4),2),
    matrix(seq(.1,.4,length.out=2*(if(mode=="multiclass")2 else 1)),2)) else
    matrix(seq(.1,.4,length.out=2*(if(mode=="multiclass")2 else 1)),2)
  net$biases <- if (ml) list(matrix(.05,2,1),matrix(.02,if(mode=="multiclass")2 else 1,1)) else
    rep(.02,if(mode=="multiclass")2 else 1)
  x <- matrix(c(.1,.2,.4,.7,.3,.8,.2,.6),4,2)
  y <- if(mode=="multiclass") cbind(c(1,0,1,0),c(0,1,0,1)) else matrix(c(0,1,0,1),4)
  acts <- ddesonn_activation_defaults(mode, if(ml)2 else integer())
  list(model=model,net=net,x=x,y=y,acts=acts,ml=ml)
}
gradient_learn <- function(f, average = NULL, reduction = "sum", twice = FALSE) {
  f$net$learn(if(twice)rbind(f$x,f$x) else f$x, if(twice)rbind(f$y,f$y) else f$y,
    lr=.001,CLASSIFICATION_MODE=attr(f$model,"classification_mode"),
    activation_functions=f$acts,dropout_rates=rep(list(NULL),if(f$ml)2 else 1),
    sample_weights=NULL,average_gradients=average,gradient_reduction=reduction)
}

test_that("all four reductions have the expected batch-duplication behavior across tasks and layouts", {
  for(mode in c("binary","multiclass","regression")) for(ml in c(FALSE,TRUE)) {
    f <- gradient_fixture(mode,ml)
    for(average in c(FALSE,TRUE)) {
      a <- gradient_learn(f,average);b <- gradient_learn(f,average,twice=TRUE)
      factor <- if(average)1 else 2
      expect_equal(b$grads_matrix,lapply(a$grads_matrix,function(x)x*factor))
      expect_equal(b$bias_gradients,lapply(a$bias_gradients,function(x)x*factor))
      expect_equal(a$learn_output,b$learn_output[1:4,,drop=FALSE])
    }
    a <- gradient_learn(f,reduction="legacy_mixed")
    b <- gradient_learn(f,reduction="legacy_mixed",twice=TRUE)
    expect_equal(b$grads_matrix,lapply(a$grads_matrix,function(x)x*2))
    expect_equal(b$bias_gradients,a$bias_gradients)
    a <- gradient_learn(f, reduction = "mean_weights")
    b <- gradient_learn(f, reduction = "mean_weights", twice = TRUE)
    expect_equal(b$grads_matrix, a$grads_matrix)
    expect_equal(b$bias_gradients, lapply(a$bias_gradients, function(x) x * 2))
  }
})

test_that("single-layer binary sums and averages match independent weighted BCE derivatives", {
  f <- gradient_fixture("binary",FALSE)
  probability <- 1/(1+exp(-(f$x %*% f$net$weights + .02)))
  delta <- (probability-f$y)*ifelse(f$y==1,2,1)
  sums <- gradient_learn(f,FALSE);means <- gradient_learn(f,TRUE)
  expect_equal(sums$grads_matrix[[1]],t(f$x)%*%delta)
  expect_equal(sums$bias_gradients[[1]],matrix(sum(delta),1,1))
  expect_equal(means$grads_matrix[[1]],t(f$x)%*%delta/4)
  expect_equal(means$bias_gradients[[1]],matrix(mean(delta),1,1))
})

test_that("Boolean validation and model serialization preserve aggregation settings", {
  for(flag in list(NA,1,"sum",logical(),c(TRUE,FALSE)))
    expect_error(ddesonn_model(2,1,average_gradients=flag),"TRUE or FALSE")
  expect_identical(ddesonn_training_defaults()$average_gradients,FALSE)
  for(flag in c(FALSE,TRUE)) {
    model <- ddesonn_model(2,1,average_gradients=flag)
    restored <- unserialize(serialize(model,NULL))
    expect_identical(attr(restored,"average_gradients"),flag)
    expect_identical(attr(restored,"gradient_reduction"),if(flag)"mean" else "sum")
  }
})

test_that("API fit inherits, overrides and records the Boolean reduction", {
  f <- gradient_fixture("regression",FALSE)
  run <- function(model,...) invisible(capture.output(ddesonn_fit(model,f$x,f$y,
    num_epochs=1,optimizer="sgd",lr=.001,reg_type=NULL,
    dropout_rates=NULL,numeric_columns=integer(),batch_normalize_data=FALSE,
    validation_metrics=FALSE,self_org=FALSE,viewTables=FALSE,...)))
  model <- ddesonn_model(2,1,hidden_sizes=integer(),ML_NN=FALSE,
    classification_mode="regression",average_gradients=TRUE)
  run(model)
  expect_identical(attr(model,"gradient_reduction"),"mean")
  run(model,average_gradients=FALSE)
  expect_identical(attr(model,"gradient_reduction"),"sum")
  attr(model,"gradient_reduction") <- NULL;attr(model,"average_gradients") <- NULL
  run(model)
  expect_identical(attr(model,"gradient_reduction"),"legacy_mixed")
})

test_that("workbench config, recipes and old bundles identify the correct mode", {
  h <- new.env(parent=globalenv())
  sys.source(system.file("shiny","benchmark","workbench_helpers.R",package="DDESONN"),h)
  p <- h$wb_prepare(iris,"Species",names(iris)[1:4],"multiclass")
  for(flag in c(FALSE,TRUE)) {
    cfg <- h$wb_config(epochs=1,average_gradients=flag)
    expect_identical(h$wb_gradient_mode(cfg),if(flag)"mean" else "sum")
    expect_match(h$wb_fit_code(p,cfg),paste0("average_gradients = ",flag),fixed=TRUE)
  }
  cfg <- h$wb_config();cfg$average_gradients <- NULL;cfg$gradient_reduction <- NULL
  expect_identical(h$wb_gradient_mode(cfg),"legacy_mixed")
  expect_match(h$wb_fit_code(p,cfg),'gradient_reduction = "legacy_mixed"',fixed=TRUE)
  expect_error(h$wb_config(average_gradients=NA),"TRUE or FALSE")
})

test_that("Shiny gradient choices are independent and reflected in the generated code", {
  h <- new.env(parent=globalenv())
  helper <- system.file("shiny","benchmark","workbench_helpers.R",package="DDESONN")
  sys.source(helper,h)
  sys.source(system.file("shiny","benchmark","workbench.R",package="DDESONN"),h)
  server <- h$wb_server;formals(server)$helper <- helper
  shiny::testServer(server, {
    session$setInputs(wb_hidden="2",wb_epochs=1,wb_lr=.001,wb_optimizer="adagrad",
      wb_checkpoint="final",wb_workflow="basic")
    expect_identical(next_config()$gradient_reduction, "sum")
    session$setInputs(wb_weight_reduction = "mean")
    expect_identical(next_config()$gradient_reduction, "mean_weights")
    expect_null(next_config()$average_gradients)
    p <- h$wb_prepare(iris,"Species",names(iris)[1:4],"multiclass")
    expect_match(h$wb_fit_code(p, next_config()), 'gradient_reduction = "mean_weights"', fixed = TRUE)
    session$setInputs(wb_bias_reduction = "mean")
    expect_identical(next_config()$gradient_reduction, "mean")
    session$setInputs(wb_weight_reduction = "sum")
    expect_identical(next_config()$gradient_reduction, "legacy_mixed")
    expect_identical(DDESONN:::.ddesonn_gradient_choices(h$wb_gradient_mode(next_config())),
      c(weights = "sum", biases = "mean"))
  })
})

test_that("run training overrides preserve the requested mixed reduction", {
  f <- gradient_fixture("regression", FALSE)
  capture.output(run <- ddesonn_run(f$x, f$y, classification_mode = "regression",
    hidden_sizes = integer(), model_overrides = list(ML_NN = FALSE),
    training_overrides = list(gradient_reduction = "mean_weights", lr = .001,
      num_epochs = 1L, validation_metrics = FALSE, batch_normalize_data = FALSE,
      numeric_columns = integer()), save_models = FALSE))
  expect_identical(attr(run$model, "gradient_reduction"), "mean_weights")
  expect_null(attr(run$model, "average_gradients"))
  expect_true("Gradient reduction: Weights: Average | Biases: Sum" %in% run$run_report$details)
})
