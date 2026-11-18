test_that("the console prints one report before the evaluation sections", {
  x <- as.matrix(mtcars[1:20, c("disp", "hp")]) / 500
  y <- matrix(mtcars$am[1:20], ncol = 1)
  out <- capture.output(run <- ddesonn_run(x[1:12, ], y[1:12, , drop = FALSE],
    hidden_sizes = integer(), seeds = 17L, dataset_name = "C:\\Data\\cars.csv",
    model_overrides = list(ML_NN = FALSE),
    training_overrides = list(num_epochs = 1L, lr = .03, batch_normalize_data = FALSE,
      numeric_columns = integer(), validation_metrics = TRUE),
    validation = list(x = x[13:16, ], y = y[13:16, , drop = FALSE]),
    test = list(x = x[17:20, ], y = y[17:20, , drop = FALSE]), save_models = FALSE))
  report <- run$run_report
  expect_length(which(out == "DDESONN run report"), 1L)
  expect_lt(match("Effective configuration", out), grep("TRAIN|Classification Report", out)[1])
  expect_true(all(c("Scenario: A | Single model", "Task: Binary classification",
    "Dataset: cars.csv", "Training: 12 | 60%", "Validation: 4 | 20%", "Test: 4 | 20%",
    "Prediction checkpoint: Final epoch") %in% report$overview))
  expect_true(all(c("Training seeds: 17", "Training batch size: 12 (full training split)",
    "Optimizer: adagrad", "Initial learning rate: 0.03",
    "Early stopping: Disabled") %in% report$details))
  expect_match(report$overview[grepl("^Best validation loss", report$overview)],
    "at epoch 1", fixed = TRUE)
  expect_identical(run$runs[[1]]$report_fits[[1]]$members[[1]]$completed, 1L)
})

test_that("scenario labels reflect the ensemble switches and counts", {
  scenario <- DDESONN:::.report_scenario
  expect_identical(scenario(FALSE, 1L, 0L), "A | Single model")
  expect_identical(scenario(FALSE, 2L, 4L), "B | Multiple independently trained models")
  expect_identical(scenario(TRUE, 3L, 0L), "C | Main ensemble")
  expect_identical(scenario(TRUE, 3L, 1L), "D | Main ensemble with TEMP models")
})

test_that("TEMP reports keep MAIN and TEMP counts and effective settings distinct", {
  x <- as.matrix(mtcars[, c("disp", "hp")]) / 500
  y <- mtcars$mpg / 100
  tr <- 1:14
  va <- 15:21
  run <- ddesonn_run(x[tr, ], y[tr], classification_mode = "regression",
    hidden_sizes = integer(), seeds = 17L, dataset_name = "mtcars",
    do_ensemble = TRUE, num_networks = 2L, num_temp_iterations = 1L,
    model_overrides = list(ML_NN = FALSE),
    training_overrides = list(num_epochs = 1L, lr = .03, batch_normalize_data = FALSE,
      numeric_columns = integer(), validation_metrics = TRUE),
    temp_overrides = list(num_epochs = 2L, lr = .04, batch_normalize_data = FALSE,
      numeric_columns = integer(), validation_metrics = TRUE),
    validation = list(x = x[va, ], y = y[va]), save_models = FALSE, print_report = FALSE)
  expect_true(all(c("Scenario: D | Main ensemble with TEMP models",
    "Prediction models per seed: 2", "TEMP models trained: 2",
    "Epochs completed / requested per model (latest stage): 1\u20132 / 1\u20132") %in% run$run_report$overview))
  expect_true(all(c("Initial learning rate: 0.03", "Initial learning rate: 0.04") %in% run$run_report$details))
})

test_that("recent details and downloads follow the recorded run and preserve stages", {
  f <- prediction_flow_fixture()
  initial <- f$bundle
  initial$prepared$source <- list(file = "C:\\Private\\flowers.csv")
  initial$model_name <- "First model"
  cfg <- initial$config
  cfg$epochs <- 1L
  cfg$lr <- .02
  cfg$checkpoint <- "best_validation"
  invisible(capture.output(continued <- f$h$wb_run(initial$prepared, cfg,
    f$h$wb_clone_model(initial))))
  continued$model_name <- "Continued model"
  expect_identical(continued$report_stages[[1]], initial$report_stages[[1]])
  expect_equal(continued$report_stages[[2]]$cumulative, 3)
  expect_equal(continued$report_stages[[2]]$settings[["Epochs added per model"]], 1)

  old <- initial
  old$report_stages <- NULL
  old$report_r_version <- NULL
  legacy <- f$h$wb_run_report(old)
  expect_true("Earlier stage settings: Unavailable in this saved bundle" %in% legacy$details)
  expect_true("Cumulative epochs per model across recorded stages: Unavailable" %in% legacy$overview)
  expect_false(any(grepl("Private", legacy$overview, fixed = TRUE)))

  shiny::testServer(f$server, {
    remember_recent(initial, "Newly trained model")
    remember_recent(continued, "Further-trained saved model")
    session$setInputs(wb_recent_id = "recent-2", wb_lr = .9, workspace = "recent")
    expected <- selected_run_report()
    expect_true("Model: Continued model" %in% expected$overview)
    expect_true("Dataset: flowers.csv" %in% expected$overview)
    expect_true("Prediction checkpoint: Best validation checkpoint" %in% expected$overview)
    expect_true("Checkpoint selection criterion: Highest validation accuracy" %in% expected$details)
    expect_true("Cumulative epochs per model across recorded stages: 3" %in% expected$overview)
    expect_true(all(c("Initial learning rate: 0.01", "Initial learning rate: 0.02") %in% expected$details))
    expect_false("Initial learning rate: 0.9" %in% expected$details)
    expect_match(output$wb_recent_details$html, "Download run report")
    expect_identical(enc2utf8(readLines(output$wb_recent_report, encoding = "UTF-8")),
      enc2utf8(DDESONN:::.report_text(expected)))
    session$setInputs(wb_recent_id = "recent-1")
    expect_true("Model: First model" %in% selected_run_report()$overview)
    expect_false("Initial learning rate: 0.02" %in% selected_run_report()$details)
  })
})
