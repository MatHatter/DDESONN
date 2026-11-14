test_that("CSV choices explain holdout and production with compact independent help", {
  h <- new.env(parent = globalenv())
  root <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(root, "workbench.R"), h)
  html <- as.character(h$wb_csv_purpose_control())
  expect_match(html, "Evaluate a holdout sample (known answers)", fixed = TRUE)
  expect_match(html, "Make production predictions (answers unknown)", fixed = TRUE)
  expect_match(html, "Typical holdout testing", fixed = TRUE)
  expect_match(html, "actual repayment outcomes", fixed = TRUE)
  expect_match(html, "not a live update or retraining", fixed = TRUE)
  expect_equal(lengths(regmatches(html, gregexpr('class="wb-help-icon"', html, fixed = TRUE))), 2)
  expect_false(grepl("Equivalent R", html, fixed = TRUE))
  # The details controls are siblings of radio labels, not their descendants.
  collect <- function(node) {
    if (inherits(node, "shiny.tag") && identical(node$attribs$class, "radio wb-purpose-choice")) return(list(node))
    if (inherits(node, "shiny.tag")) return(unlist(lapply(node$children, collect), recursive = FALSE))
    if (is.list(node)) return(unlist(lapply(node, collect), recursive = FALSE))
    list()
  }
  rows <- collect(h$wb_csv_purpose_control())
  expect_length(rows, 2)
  for (row in rows) {
    expect_identical(row$children[[1]]$name, "label")
    expect_identical(row$children[[2]]$name, "details")
  }
})
