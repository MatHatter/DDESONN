study_fixture <- function() {
  app <- normalizePath(testthat::test_path("..", "..", "inst", "shiny", "benchmark"))
  env <- new.env(parent = globalenv())
  sys.source(file.path(app, "studies.R"), env)
  list(h = env, data = utils::read.csv(file.path(app, "data", "clean-matched-100.csv")), app = app)
}

test_that("packaged clean results retain all conditions and reproduce paired evidence", {
  f <- study_fixture()
  for (r in c("sum", "mean")) {
    p <- f$h$study_clean_pairs(f$data, r)
    expect_equal(p$seed, 1:100)
    blocks <- f$h$study_seed_summary(p)
    expect_true(all(blocks$advantage_pp > 0))
    final <- tail(f$h$study_seed_summary(p, TRUE), 1)
    expect_equal(final$advantage_pp, if (r == "sum") 3.876558603491271 else 2.769326683291771)
    expect_equal(final$wins, if (r == "sum") 85L else 75L)
    expect_equal(final$low_pp, if (r == "sum") 3.1075261830677507 else 1.9186046051897052)
  }
  expect_error(f$h$study_clean_pairs(f$data[-1, ], "sum"))
  bad <- f$data; bad$seed[2] <- 200
  expect_error(f$h$study_clean_pairs(bad, "sum"))
  bad <- f$data; bad$test_rows[1] <- 750
  expect_error(f$h$study_clean_pairs(bad, "sum"))
})

test_that("clean comparison switching uses the right records and labels", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("ggplot2")
  f <- study_fixture()
  shiny::testServer(f$h$clean_study_server(f$data), {
    session$setInputs(clean_reduction = "sum")
    expect_equal(mean(pairs()$difference_pp), 3.876558603491271)
    expect_equal(tail(cumulative()$wins, 1), 85)
    session$setInputs(clean_reduction = "mean")
    expect_equal(mean(pairs()$difference_pp), 2.769326683291771)
    expect_equal(tail(cumulative()$wins, 1), 75)
    expect_true(nzchar(output$clean_convergence$src))
  })
  ui <- as.character(f$h$clean_study_ui())
  expect_match(ui, "not tested here", fixed = TRUE)
  expect_match(ui, "retrospective", fixed = TRUE)
  app <- paste(readLines(file.path(f$app, "app.R")), collapse = "\n")
  expect_match(app, 'selected = "clean"', fixed = TRUE)
  expect_match(app, "738 also appear in training", fixed = TRUE)
})
