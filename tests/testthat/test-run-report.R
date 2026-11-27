test_that("the console defaults to evaluation tables and preserves the full report", {
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
  expect_false(any(grepl("DDESONN run (report|summary)|Effective configuration", out)))
  expect_true(all(c("===== TRAIN =====", "===== VALIDATION =====", "===== TEST =====") %in% out))
  render <- DDESONN:::.report_console_text
  medium <- render(report, FALSE, TRUE)
  expect_identical(medium[1:4], c("", "===== DDESONN run summary =====",
    "", "Scenario: A | Single model"))
  expect_true(all(c("===== DDESONN run summary =====", "Epochs completed / requested per model: 1 / 1",
    "Training: 12 | 60%") %in% medium))
  expect_false(any(grepl("Effective configuration", medium)))
  expect_true(any(startsWith(medium, "Split overlap:")))
  expect_identical(render(report, TRUE, TRUE), DDESONN:::.report_text(report))
  full <- render(report, TRUE, FALSE)
  expect_identical(full, DDESONN:::.report_text(report))
  sections <- match(paste("=====", c("DDESONN run report", "Data splits", "Training and checkpoint",
    "Provenance", "Effective configuration", "Training stage 1 (initial)", "Per-model outcomes"),
    "====="), full)
  expect_false(anyNA(sections))
  expect_true(all(diff(sections) > 0))
  expect_lt(tail(sections, 1), grep("^  Model 1:", full)[1])
  expect_true("===== Timing =====" %in% full)
  expect_true(any(startsWith(full, "**Model fits**:")))
  expect_true(any(grepl("included in fit time above", full, fixed = TRUE)))
  expect_length(render(report, FALSE, FALSE), 0L)
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
    validation = list(x = x[va, ], y = y[va]), save_models = FALSE)
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

  workbench_test_server(f$server, {
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
    expect_match(output$wb_recent_details$html, "Exact predictor overlap", ignore.case = TRUE)
    expect_identical(expected$overlap, initial$prepared$overlap)
    expect_identical(enc2utf8(readLines(output$wb_recent_report, encoding = "UTF-8")),
      enc2utf8(DDESONN:::.report_text(expected)))
    session$setInputs(wb_recent_id = "recent-1")
    expect_true("Model: First model" %in% selected_run_report()$overview)
    expect_false("Initial learning rate: 0.02" %in% selected_run_report()$details)
  })
})


test_that("stage timing keeps main and TEMP fits separate from prediction and other work", {
  stage <- list(elapsed = 12, prediction_elapsed = 3, timing_scope = "Measured stage",
    fits = list(list(label = "Seed 1 \u00b7 MAIN", elapsed = 4),
      list(label = "Seed 1 \u00b7 TEMP 1", elapsed = 2)))
  timing <- DDESONN:::.report_timing(stage)
  expect_true(all(c("**MAIN fits**: 4.00 s", "TEMP fits: 2.00 s",
    "Measured prediction / scoring outside fits: 3.00 s", "Other run work: 3.00 s",
    "Total measured: 12.00 s") %in% timing))
  stage$prediction_elapsed <- NULL
  expect_length(DDESONN:::.report_timing(stage), 0L)
})
