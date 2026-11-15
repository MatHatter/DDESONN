test_that("loading the package does not launch or load Shiny", {
  skip_if_not_installed("callr")
  result <- callr::r(function() {
    loadNamespace("DDESONN")
    "shiny" %in% loadedNamespaces()
  })
  expect_false(result)
})

test_that("the optional app explains missing Shiny and callr dependencies", {
  app <- system.file("shiny", "benchmark", "app.R", package = "DDESONN")
  for (missing in c("shiny", "callr")) {
    scope <- new.env(parent = baseenv())
    scope$requireNamespace <- function(package, ...) !identical(package, missing)
    expect_error(sys.source(app, envir = scope),
      "Install optional packages 'shiny' and 'callr' to use the workbench.", fixed = TRUE)
  }
})
