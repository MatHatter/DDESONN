clean_pilot_fixture <- function() {
  h <- new.env(parent = globalenv())
  sys.source(system.file("scripts", "heart_failure_clean_split.R", package = "DDESONN"), h)
  h
}

test_that("clean pilot freezes group-disjoint rows and fits scaling only on training", {
  h <- clean_pilot_fixture()
  d <- data.frame(x = rep(1:30, each = 4), z = rep(31:60, each = 4),
    DEATH_EVENT = rep(rep(0:1, each = 15), each = 4))
  p <- h$clean_split_prepare(d)
  expect_identical(p, h$clean_split_prepare(d))
  groups <- lapply(p$parts, `[[`, "group")
  expect_length(intersect(groups$train, groups$test), 0)
  expect_length(intersect(groups$train, groups$validation), 0)
  expect_length(intersect(groups$validation, groups$test), 0)
  expect_true(all(p$independent_overlap_audit == 0))
  expect_identical(sort(unlist(lapply(p$parts, `[[`, "row"), use.names = FALSE)), seq_len(nrow(d)))
  expect_equal(unname(p$center), unname(colMeans(d[p$parts$train$row, c("x", "z")])))
  expect_equal(max(abs(p$parts$train$x)), 1)

})

test_that("pilot reports duplicate-weighted and group-weighted scores separately", {
  h <- clean_pilot_fixture()
  m <- h$clean_split_metrics(c(0, 0, 0, 1), c(.1, .1, .1, .1), c("a", "a", "a", "b"))
  expect_equal(m$accuracy, .75)
  expect_equal(m$group_weighted_accuracy, .5)
  expect_equal(m$predictor_groups, 2)
  expect_equal(m$TN, 3); expect_equal(m$FN, 1)
  expect_error(h$clean_split_metrics(c(0, 1), c(.2, NA), c("a", "b")), "Invalid")
  expect_error(h$clean_split_metrics(c(0, 1), c(.2, 1.1), c("a", "b")), "Invalid")
})

test_that("pilot refuses invalid seeds and refuses to overwrite an experiment", {
  h <- clean_pilot_fixture()
  output <- tempfile(); dir.create(output)
  on.exit(unlink(output, recursive = TRUE), add = TRUE)
  writeLines("preserve", file.path(output, "existing.txt"))
  expect_error(h$clean_split_main(c(output, "1,1")), "unique")
  expect_error(h$clean_split_main(c(output, "1", "0")), "positive")
  expect_error(h$clean_split_main(c(output, "1", "360", "42", "invalid")), "FALSE, TRUE")
  expect_error(h$clean_split_main(output), "never overwritten")
  expect_identical(readLines(file.path(output, "existing.txt")), "preserve")
})

test_that("pilot checkpoints and fitted tensors do not depend on held-out test labels", {
  h <- clean_pilot_fixture()
  d <- data.frame(x = rep(1:120, each = 4), z = rep(121:240, each = 4),
    DEATH_EVENT = rep(rep(0:1, each = 60), each = 4))
  p <- h$clean_split_prepare(d)
  scratch <- tempfile(); dir.create(scratch)
  previous <- setwd(scratch)
  on.exit({ setwd(previous); unlink(scratch, recursive = TRUE) }, add = TRUE)
  device <- grDevices::dev.cur()
  first <- h$clean_split_fit(p, seed = 7, epochs = 2)
  expect_identical(grDevices::dev.cur(), device)
  expect_false(file.exists("Rplots.pdf"))
  p$parts$test$y <- 1 - p$parts$test$y
  second <- h$clean_split_fit(p, seed = 7, epochs = 2)
  expect_identical(first$model$ensemble[[1]]$weights, second$model$ensemble[[1]]$weights)
  expect_identical(first$model$ensemble[[1]]$biases, second$model$ensemble[[1]]$biases)
  expect_identical(first$metrics$selected_epoch, second$metrics$selected_epoch)
  expect_identical(first$predictions, second$predictions)
  restored <- unserialize(serialize(first$model, NULL))
  expect_identical(as.numeric(ddesonn_predict(restored, p$parts$test$x)$prediction), first$predictions$test)
})
