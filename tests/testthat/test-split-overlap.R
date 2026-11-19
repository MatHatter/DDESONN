test_that("overlap counts distinct predictor vectors for each split pair", {
  check <- DDESONN:::.split_overlap
  status <- DDESONN:::.overlap_status
  train <- matrix(c(0, 1, 2, 2), ncol = 1)
  validation <- matrix(c(-0, 2, 3), ncol = 1)
  test <- matrix(c(3L, 4L), ncol = 1)
  counts <- check(train, validation, test)
  expect_identical(counts, c(train_validation = 2L, train_test = 0L, validation_test = 1L))
  expect_identical(status(counts), "detected")
  expect_identical(status(check(train, matrix(9), test)), "clear")
  expect_identical(status(check(train, test = test)), "partial")
  expect_identical(status(check(train)), "unavailable")
  expect_match(DDESONN:::.overlap_text(counts), "Train-Validation 2", fixed = TRUE)
})

test_that("prepared regression data retains overlap and Shiny shows actionable status", {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  sys.source(system.file("shiny", "benchmark", "workbench.R", package = "DDESONN"), h)
  d <- data.frame(x = rep(1:2, 6), y = seq_len(12))
  p <- h$wb_prepare(d, "y", "x", "regression", split = "ordered")
  expect_identical(p$overlap, c(train_validation = 2L, train_test = 2L, validation_test = 2L))
  if (requireNamespace("shiny", quietly = TRUE)) {
    ui <- as.character(h$wb_overlap_ui(p$overlap))
    expect_match(ui, "wb-overlap-report", fixed = TRUE)
    expect_match(ui, 'aria-label="Help: Split overlap"', fixed = TRUE)
    expect_match(ui, "text-danger", fixed = TRUE)
    expect_match(ui, "subject ID", fixed = TRUE)
    d$x <- seq_len(nrow(d))
    clean <- h$wb_prepare(d, "y", "x", "regression", split = "ordered")
    expect_match(as.character(h$wb_overlap_ui(clean$overlap)), "text-success", fixed = TRUE)
    expect_match(as.character(h$wb_overlap_ui(NULL)), "Check unavailable", fixed = TRUE)
  }
})

test_that("Workbench help and split preparation work without the namespace overlap helpers", {
  skip_if_not_installed("shiny")
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  sys.source(system.file("shiny", "benchmark", "workbench.R", package = "DDESONN"), h)
  d <- data.frame(x = rep(1:2, 6), y = seq_len(12))
  expected <- h$wb_prepare(d, "y", "x", "regression", split = "ordered")$overlap
  h$asNamespace <- function(...) new.env(parent = emptyenv())

  expect_match(as.character(h$wb_overlap_ui(NULL)), "Check unavailable", fixed = TRUE)
  prepared <- h$wb_prepare(d, "y", "x", "regression", split = "ordered")
  expect_identical(prepared$overlap, expected)
  expect_match(as.character(h$wb_overlap_ui(prepared$overlap)), "Train-Validation 2", fixed = TRUE)
  expect_match(h$wb_help_code("Start with Data"), "iris", fixed = TRUE)
  expect_match(h$wb_help_code("Inspect your dataset"), "data", fixed = TRUE)
})
