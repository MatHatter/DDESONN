dashboard_fixture <- function() {
  path <- system.file("shiny", "benchmark", package = "DDESONN")
  helper <- new.env(parent = globalenv())
  sys.source(file.path(path, "helpers.R"), envir = helper)
  list(helper = helper, data = utils::read.csv(file.path(path, "data", "heart-failure-100.csv")))
}

test_that("dashboard validates paired inputs and recomputes uploaded differences", {
  f <- dashboard_fixture(); validate <- f$helper$dashboard_validate
  d <- data.frame(seed = c(3, 1), accuracy_legacy = c(.8, .9),
    accuracy_corrected = c(.9, .8), accuracy_difference = 99)
  checked <- validate(d)
  expect_equal(checked$seed, c(1, 3))
  expect_equal(checked$accuracy_difference, c(-.1, .1))
  expect_error(validate(transform(d, seed = c(1, 1))), "unique positive")
  expect_error(validate(transform(d, accuracy_legacy = c(NA, .5))), "finite")
  expect_error(validate(transform(d, accuracy_corrected = c(1.1, .5))), "finite")
  expect_error(validate(d[, c("seed", "accuracy_legacy")]), "Include both")
  expect_error(validate(data.frame(seed = 1)), "at least one")
  expect_error(validate(d[FALSE, ]), "at least one seed")
  expect_error(validate(transform(d, seed = c(1.5, 2))), "unique positive")
  expect_error(validate(data.frame(seed = 1, log_loss_legacy = -.1,
    log_loss_corrected = .2)), "nonnegative")
})

test_that("dashboard reproduces benchmark summaries and metric direction", {
  f <- dashboard_fixture(); h <- f$helper
  d <- h$dashboard_validate(f$data)
  s <- h$dashboard_summary(d)
  accuracy <- s[s$Metric == "Accuracy", ]
  expect_equal(accuracy$Standard, .99968, tolerance = 1e-10)
  expect_equal(accuracy$Legacy, .9994266666666667, tolerance = 1e-10)
  expect_equal(unlist(accuracy[c("Improved", "Tied", "Worsened")], use.names = FALSE),
    c(25, 58, 17))
  expect_equal(sum(round((1 - d$accuracy_legacy) * 750)), 43)
  expect_equal(sum(round((1 - d$accuracy_corrected) * 750)), 24)
  one <- data.frame(seed = 1, log_loss_legacy = .2, log_loss_corrected = .1)
  s <- h$dashboard_summary(h$dashboard_validate(one))
  expect_equal(s$Improved, 1)
  expect_true(is.na(s$Difference_SD))
})

test_that("dashboard filters, replaces valid uploads and retains data after invalid uploads", {
  skip_if_not_installed("shiny")
  f <- dashboard_fixture(); h <- f$helper; d <- h$dashboard_validate(f$data)
  shiny::testServer(h$dashboard_server(d), {
    session$setInputs(seeds = c(1, 50), metric = "accuracy")
    expect_equal(nrow(selected()), 50)
    expect_match(as.character(output$cards$html), "SELECTED SEEDS")
    expect_equal(summary()$Seeds, rep(50, 7))
    csv <- tempfile(fileext = ".csv")
    utils::write.csv(data.frame(seed = 101, brier_legacy = .2, brier_corrected = .1),
      csv, row.names = FALSE)
    session$setInputs(upload = data.frame(datapath = csv, name = "next.csv",
      size = file.info(csv)$size))
    session$setInputs(seeds = c(101, 101), metric = "brier")
    expect_equal(selected()$seed, 101)
    expect_equal(summary()$Improved, 1)
    bad <- tempfile(fileext = ".csv")
    writeLines("seed,accuracy_legacy\n1,0.5", bad)
    session$setInputs(upload = data.frame(datapath = bad, name = "bad.csv",
      size = file.info(bad)$size))
    expect_match(upload_error(), "Include both")
    expect_equal(current()$seed, 101)
    session$setInputs(reset = 1, seeds = c(1, 100))
    expect_equal(nrow(selected()), 100)
    expect_null(upload_error())
    unlink(c(csv, bad))
  })
})
