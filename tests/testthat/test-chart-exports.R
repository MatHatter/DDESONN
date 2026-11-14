chart_fixture <- function() {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  h
}

test_that("training logs retain fit boundaries and local epochs", {
  h <- chart_fixture(); file <- tempfile()
  on.exit(unlink(file), add = TRUE)
  writeLines(c("ignore", "Epoch 1 Loss: 0.4", "Epoch 2 Loss: 0.2", "Epoch 1 Loss: 0.5", "Epoch 2 Loss: 0.3"), file)
  d <- h$wb_log_history(file)
  expect_equal(d$epoch, 1:4)
  expect_equal(d$fit, c(1, 1, 2, 2))
  expect_equal(d$fit_epoch, c(1, 2, 1, 2))
  expect_equal(d$loss, c(.4, .2, .5, .3))
  writeLines(character(), file)
  expect_equal(nrow(h$wb_log_history(file)), 0)
  old <- data.frame(epoch = 1:6, loss = seq(.1, .6, .1))
  d <- h$wb_chart_history(old, list(workflow = "custom", epochs = 2))
  expect_equal(d$fit, rep(1:3, each = 2))
  expect_true(attr(d, "inferred_boundaries"))
  expect_equal(h$wb_chart_history(old)$fit, rep(1, 6))
})

test_that("charts separate fits and both graphics formats export", {
  h <- chart_fixture()
  d <- data.frame(epoch = 1:6, loss = c(.4, .3, .2, .5, .4, .3), fit = rep(1:2, each = 3), fit_epoch = rep(1:3, 2))
  p <- h$wb_loss_plot(d, selected_epoch = 2)
  built <- ggplot2::ggplot_build(p)
  expect_equal(length(unique(built$data[[1]]$group)), 2)
  expect_equal(built$data[[3]]$xintercept, 3.5)
  expect_match(p$labels$caption, "2 separate fits", fixed = TRUE)
  expect_no_warning(ggplot2::ggplotGrob(h$wb_loss_plot(d[c(1, 4), ])))
  single <- h$wb_loss_plot(d[1:3, ], selected_epoch = 2)
  expect_equal(ggplot2::ggplot_build(single)$data[[3]]$xintercept, 2)
  png <- tempfile(fileext = ".png"); svg <- tempfile(fileext = ".svg")
  on.exit(unlink(c(png, svg)), add = TRUE)
  ggplot2::ggsave(png, p, width = 10, height = 4, dpi = 100)
  ggplot2::ggsave(svg, p, device = grDevices::svg, width = 10, height = 4)
  expect_true(file.info(png)$size > 1000)
  expect_match(paste(readLines(svg, warn = FALSE), collapse = ""), "<svg", fixed = TRUE)
})

test_that("all preprocessing controls have explained, parseable focused help", {
  h <- chart_fixture()
  info <- h$wb_pp_help()
  expect_length(info, 10)
  expect_null(names(h$wb_help_topics()))
  expect_true(all(vapply(info, function(x) nchar(x$text) > 100, logical(1))))
  for (item in info) {
    expect_true(item$title %in% h$wb_help_topics())
    expect_true(length(parse(text = h$wb_help_code(item$title))) > 0)
    expect_identical(h$wb_help_code(item$title), item$code)
  }
})
