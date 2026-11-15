# Capture both output streams: capture.output() alone misses message() diagnostics.
optimizer_output <- function(code) {
  messages <- warnings <- character()
  output <- capture.output(value <- withCallingHandlers(force(code),
    message = function(cnd) {
      messages <<- c(messages, conditionMessage(cnd))
      invokeRestart("muffleMessage")
    },
    warning = function(cnd) {
      warnings <<- c(warnings, conditionMessage(cnd))
      invokeRestart("muffleWarning")
    }))
  list(value = value, output = output, messages = messages, warnings = warnings)
}

for (method in DDESONN::ddesonn_optimizer_options()) local({
  optimizer <- method
  test_that(paste(optimizer, "training and prediction respect quiet mode in SL and ML models"), {
    set.seed(107)
    x <- matrix(rnorm(24), 8, 3)
    y <- matrix(seq(-.3, .4, length.out = 8), 8, 1)
    for (ml in c(FALSE, TRUE)) {
      captured <- optimizer_output({
        run <- ddesonn_run(x, y, classification_mode = "regression",
          hidden_sizes = if (ml) c(3L, 2L) else NULL, seeds = 17L,
          save_models = FALSE, verbose = FALSE, verboseLow = FALSE, debug = FALSE,
          model_overrides = list(ML_NN = ml),
          training_overrides = list(optimizer = optimizer, num_epochs = 2L,
            self_org = FALSE, ML_NN = ml, lr = .001, lr_decay_rate = 1,
            reg_type = NULL, dropout_rates = NULL, validation_metrics = FALSE,
            batch_normalize_data = FALSE, viewTables = FALSE,
            verbose = FALSE, verboseLow = FALSE))
        ddesonn_predict(run$model, x, verbose = FALSE, verboseLow = FALSE,
          debug = FALSE)$prediction
      })
      expect_identical(captured$output, character())
      expect_identical(captured$messages, character())
      expect_identical(captured$warnings, character())
      expect_equal(dim(captured$value), c(8L, 1L))
      expect_true(all(is.finite(captured$value)))
    }
  })

  test_that(paste(optimizer, "workbench progress does not leak optimizer diagnostics"), {
    h <- new.env(parent = globalenv())
    sys.source(system.file("shiny", "benchmark", "workbench_helpers.R",
      package = "DDESONN"), h)
    prepared <- h$wb_prepare(mtcars, "mpg", c("wt", "hp"), "regression")
    captured <- optimizer_output(h$wb_run(prepared,
      h$wb_config(hidden = "2", epochs = 1, lr = .001, optimizer = optimizer)))
    expect_match(paste(captured$output, collapse = "\n"), "Epoch 1 .*Loss:")
    expect_false(any(grepl("\\[OPT=|Grad norm|\\[LEARN\\]|FUSE-DBG",
      c(captured$output, captured$messages))))
    expect_identical(captured$messages, character())
    expect_identical(captured$warnings, character())
    expect_identical(captured$value$config$optimizer, optimizer)
    expect_true(all(is.finite(captured$value$evaluation$test$metrics$Value)))
  })
})
