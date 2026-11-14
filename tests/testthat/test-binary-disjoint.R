binary_split_fixture <- function() {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  h
}
assert_disjoint <- function(p) {
  sets <- lapply(p$parts, function(part) apply(part$x, 1, function(v) paste(v, collapse = "|")))
  expect_length(intersect(sets$train, sets$validation), 0)
  expect_length(intersect(sets$train, sets$test), 0)
  expect_length(intersect(sets$validation, sets$test), 0)
}
test_that("binary groups ignore labels and preserve every original row", {
  h <- binary_split_fixture()
  d <- data.frame(x = rep(1:30, each = 4), z = rep(31:60, each = 4), y = rep(rep(0:1, each = 15), each = 4))
  d$y[1] <- 1 # same predictors, conflicting label: must stay together
  original <- serialize(d, NULL)
  p <- h$wb_prepare(d, "y", c("x", "z"), "binary")
  assert_disjoint(p)
  expect_identical(sort(unlist(lapply(p$parts, `[[`, "row"), use.names = FALSE)), seq_len(nrow(d)))
  expect_identical(serialize(d, NULL), original)
  expect_true(all(p$split_audit$overlap_groups == 0))
  expect_identical(p, h$wb_prepare(d, "y", c("x", "z"), "binary"))
  env <- list2env(list(data = d)); eval(parse(text = h$wb_features_code(p)), env)
  expect_identical(lapply(env$parts, `[[`, "row"), lapply(p$parts, `[[`, "row"))
  expect_equal(lapply(env$parts, `[[`, "x"), lapply(p$parts, `[[`, "x"))
})
test_that("binary grouping excludes identifiers removed by preprocessing", {
  h <- binary_split_fixture()
  d <- data.frame(ID = 1:120, x = rep(1:30, each = 4), y = rep(rep(0:1, each = 15), each = 4))
  p <- h$wb_prepare(d, "y", c("ID", "x"), "binary", preprocessing = h$wb_pp_config())
  expect_identical(p$features, "x"); assert_disjoint(p)
})
test_that("ordered duplicates and too few independent groups fail explicitly", {
  h <- binary_split_fixture()
  d <- data.frame(x = rep(1:20, 3), y = rep(rep(0:1, each = 10), 3))
  expect_error(h$wb_prepare(d, "y", "x", "binary", split = "ordered"), "cross the ordered split")
  d$x <- d$y
  expect_error(h$wb_prepare(d, "y", "x", "binary"), "three independent predictor groups")
})
test_that("heart failure split has zero pairwise predictor overlap", {
  h <- binary_split_fixture()
  d <- read.csv(system.file("extdata", "heart_failure_clinical_records.csv", package = "DDESONN"))
  features <- setdiff(names(d), "DEATH_EVENT")
  original <- serialize(d, NULL)
  for (seed in c(1, 42, 100)) {
    p <- h$wb_prepare(d, "DEATH_EVENT", features, "binary", seed = seed)
    assert_disjoint(p)
    expect_true(all(p$split_audit$overlap_groups == 0))
    expect_equal(sum(vapply(p$parts, function(part) nrow(part$x), integer(1))), nrow(d))
    expect_equal(unname(p$center), unname(colMeans(as.matrix(d[p$parts$train$row, features]))))
  }
  expect_identical(serialize(d, NULL), original)
})
test_that("training-derived filling cannot silently create cross-split overlap", {
  h <- binary_split_fixture()
  d <- data.frame(x = rep(c(0:29, NA), each = 4), y = rep(c(rep(0:1, each = 15), 0), each = 4))
  rejected <- FALSE
  for (seed in 1:6) {
    p <- tryCatch(h$wb_prepare(d, "y", "x", "binary", seed = seed,
      preprocessing = h$wb_pp_config(numeric = "zero")), error = identity)
    if (inherits(p, "error")) {
      expect_match(conditionMessage(p), "Preprocessing created identical inputs")
      rejected <- TRUE
    } else assert_disjoint(p)
  }
  expect_true(rejected)
})
