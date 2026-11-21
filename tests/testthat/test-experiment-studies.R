study_fixture <- function() {
  app <- system.file("shiny", "benchmark", package = "DDESONN", mustWork = TRUE)
  env <- new.env(parent = globalenv())
  sys.source(file.path(app, "studies.R"), env)
  list(h = env, data = utils::read.csv(file.path(app, "data", "ddesonn-vs-keras-100-seeds-clean-split.csv")), app = app)
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
  expect_match(ui, "The completed benchmark uses uniform reductions.", fixed = TRUE)
  expect_match(ui, "retrospective", fixed = TRUE)
  app <- paste(readLines(file.path(f$app, "app.R")), collapse = "\n")
  expect_match(app, 'selected = "clean"', fixed = TRUE)
  expect_match(app, "738 also appear in training", fixed = TRUE)
})

test_that("downloaded analysis code reproduces the selected study without training", {
  f <- study_fixture()
  for (r in c("sum", "mean")) {
    code <- f$h$study_analysis_code(f$data, r)
    env <- new.env(parent = globalenv())
    grDevices::pdf(tempfile(fileext = ".pdf"))
    invisible(capture.output(eval(parse(text = code), env)))
    grDevices::dev.off()
    expect_equal(tail(env$cumulative$advantage_pp, 1),
      if (r == "sum") 3.876558603491271 else 2.769326683291771)
    expect_false(grepl("run_study\\(", code))
  }
  sys.source(file.path(f$app, "helpers.R"), f$h)
  data <- f$h$dashboard_validate(data.frame(seed = 12:13,
    accuracy_legacy = c(.7, .8), accuracy_corrected = c(.8, .9)))
  code <- f$h$study_analysis_code(data, historical = TRUE, metric = "accuracy")
  env <- new.env(parent = globalenv())
  grDevices::pdf(tempfile(fileext = ".pdf"))
  invisible(capture.output(eval(parse(text = code), env)))
  grDevices::dev.off()
  expect_equal(env$data$seed, 12:13)
  expect_equal(env$delta, c(.1, .1))
})

test_that("installed reproduction bundle contains exact frozen training sources", {
  f <- study_fixture()
  archive <- file.path(f$app, "reproduction.zip")
  entries <- utils::unzip(archive, list = TRUE)$Name
  expect_true(all(c("RUN_STUDY.R", "prepare_original.py", "keras_mean.R",
    "keras_sum.R", "ddesonn_mean.R", "ddesonn_sum.R") %in% entries))
  temp <- tempfile(); dir.create(temp)
  on.exit(unlink(temp, recursive = TRUE))
  utils::unzip(archive, exdir = temp)
  protocol <- jsonlite::fromJSON(file.path(temp, "dev/research/keras-clean-reference/protocol.json"))
  for (name in names(protocol$adapter_sha256)) {
    expect_identical(digest::digest(file = file.path(temp, name), algo = "sha256"),
      protocol$adapter_sha256[[name]], info = name)
  }
  # Core native fitting/metric block is unchanged by portable compilation.
  old <- paste(readLines(file.path(temp, "dev/research/keras-clean-reference/run_ddesonn.R")), collapse="\n")
  portable <- paste(readLines(file.path(temp, "dev/research/keras-clean-reference/community/run_native.R")), collapse="\n")
  expect_identical(strsplit(old,"x <- matrix",fixed=TRUE)[[1]][2],
    strsplit(portable,"x <- matrix",fixed=TRUE)[[1]][2])
})
