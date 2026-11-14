inspector_fixture <- function() {
  h <- new.env(parent = globalenv())
  root <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(root, "workbench_helpers.R"), h)
  sys.source(file.path(root, "workbench.R"), h)
  list(h = h, root = root)
}

test_that("row browsing and R expressions match the selected original data", {
  h <- inspector_fixture()$h
  data <- datasets::mtcars
  original <- serialize(data, NULL)
  for (view in c("head", "tail", "range")) {
    slice <- h$wb_inspect_rows(data, view, 5, 12, c("mpg", "hp"))
    env <- list2env(list(data = data))
    expect_identical(eval(parse(text = slice$expression), env), slice$data)
    expect_identical(names(slice$data), c("mpg", "hp"))
    expect_equal(nrow(slice$data), 5)
    expect_identical(slice$rows, switch(view, head = 1:5, tail = 28:32, range = 12:16))
  }
  expect_equal(nrow(h$wb_inspect_rows(iris, "head", 1000)$data), 100)
  expect_equal(h$wb_inspect_rows(data, "range", 10, 1000)$rows, 32L)
  expect_equal(nrow(h$wb_inspect_rows(data, "head", NA_real_)$data), 5)
  expect_equal(nrow(h$wb_inspect_rows(data[FALSE, ], "tail")$data), 0)
  expect_identical(serialize(data, NULL), original)
})

test_that("profiles count missing values and summarize finite data without mutation", {
  h <- inspector_fixture()$h
  d <- data.frame(x = c(1, 2, 3, NA, Inf), category = c("a", "", NA, "a", "b"), constant = 7)
  original <- serialize(d, NULL)
  p <- h$wb_data_profile(d)
  expect_identical(p$Column, names(d))
  expect_equal(p$Missing, c(1, 2, 0))
  expect_equal(p$Nonfinite, c(1, 0, 0))
  expect_equal(p$Distinct, c(4, 2, 1))
  expect_equal(p$Mean[c(1, 3)], c(2, 7))
  expect_equal(p$SD[c(1, 3)], c(1, 0))
  expect_equal(p$Median[c(1, 3)], c(2, 7))
  expect_true(all(is.na(p[2, c("Mean", "SD", "Min", "Q1", "Median", "Q3", "Max")])))
  expect_identical(serialize(d, NULL), original)
})

test_that("dropdown changes load inspector data without needing Reload", {
  f <- inspector_fixture(); h <- f$h
  server <- h$wb_server; formals(server)$helper <- file.path(f$root, "workbench_helpers.R")
  shiny::testServer(server, {
    session$setInputs(wb_example = "iris")
    session$setInputs(wb_example = "mtcars")
    expect_identical(raw(), datasets::mtcars)
    expect_identical(code_source(), list(example = "mtcars"))
    session$setInputs(wb_inspect_view = "tail", wb_inspect_n = 5, wb_inspect_columns = c("mpg", "hp"))
    expect_identical(inspected()$data, tail(datasets::mtcars[, c("mpg", "hp")], 5))
    expect_match(output$wb_data_code, "datasets::mtcars", fixed = TRUE)
    expect_match(output$wb_data_code, "tail(", fixed = TRUE)
    session$setInputs(wb_example = "customer_train")
    expect_equal(nrow(raw()), 8068)
    expect_identical(dataset_preset()$target, "Segmentation")
    session$setInputs(wb_example = "wmt")
    expect_equal(nrow(raw()), 13248)
    expect_match(code_source()$package_file, "WMT")
    session$setInputs(wb_example = "iris")
    expect_identical(raw(), datasets::iris)
  })
})
