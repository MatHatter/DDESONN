pp_fixture <- function() {
  h <- new.env(parent = globalenv())
  root <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(root, "workbench_helpers.R"), h)
  sys.source(file.path(root, "workbench.R"), h)
  list(h = h, root = root)
}

test_that("preprocessing learns numeric fills and levels from training rows only", {
  h <- pp_fixture()$h
  d <- data.frame(ID = 1:30, x = c(NA, 2:18, rep(1e6, 12)),
    group = c(rep(c("a", "b", NA), 6), rep("unseen", 12)), y = rep(c("A", "B"), 15))
  d$group <- factor(d$group) # Includes held-out levels, which must not enter the fitted schema.
  original <- serialize(d, NULL)
  p <- h$wb_prepare(d, "y", c("ID", "x", "group"), "multiclass", split = "ordered",
    preprocessing = h$wb_pp_config())
  expect_identical(p$features, c("x", "group"))
  expect_equal(p$preprocessing$specs[[1]]$fill, median(2:18))
  expect_identical(p$preprocessing$specs[[2]]$levels, c("a", "b"))
  expect_false("unseen" %in% p$preprocessing$specs[[2]]$levels)
  expect_true(all(vapply(p$parts, function(part) all(is.finite(part$x)), logical(1))))
  unscaled <- h$wb_pp_apply(d, p$preprocessing)
  expect_equal(unname(unscaled[1, "x"]), 10)
  expect_equal(unname(unscaled[19, "group__[Unknown]"]), 1)
  expect_equal(unname(unscaled[3, "group__[Missing]"]), 1)
  expect_equal(p$center, colMeans(unscaled[1:18, , drop = FALSE]))
  expect_equal(p$rows, 30)
  expect_equal(sum(vapply(p$parts, function(part) nrow(part$x), integer(1))), 30)
  expect_identical(serialize(d, NULL), original)
  e <- new.env(); e$data <- d
  eval(parse(text = h$wb_features_code(p)), e)
  expect_identical(e$parts, p$parts)
  expect_identical(e$preprocessing, p$preprocessing)
})

test_that("manual steps, identifier decisions and ordinal overrides are explicit", {
  h <- pp_fixture()$h
  d <- data.frame(ID = 1:12, amount = c(NA, 2:12), grade = rep(c("Low", "High", NA), 4))
  expect_error(h$wb_pp_apply(d, h$wb_pp_fit(d, names(d), h$wb_pp_config("manual"))), "Choose one-hot|Choose a missing")
  cfg <- h$wb_pp_config("manual", numeric = "mean", encoding = "ordinal", categorical_missing = "mode",
    overrides = list(grade = list(order = c("Low", "High")), ID = list(action = "exclude")))
  fitted <- h$wb_pp_fit(d, names(d), cfg)
  expect_identical(fitted$features, c("amount", "grade"))
  x <- h$wb_pp_apply(d, fitted)
  expect_equal(unname(x[1, "amount"]), mean(2:12))
  expect_equal(x[1:2, "grade"], c(1, 2))
  future <- data.frame(amount = c(5, NA), grade = c("Novel", NA))
  expect_equal(h$wb_pp_apply(future, fitted)[, "grade"], c(-1, 2))
  cfg$categorical_missing <- "level"
  expect_equal(h$wb_pp_apply(future, h$wb_pp_fit(d, names(d), cfg))[, "grade"], c(-1, 0))
  cfg$overrides$grade$order <- c("Low")
  expect_error(h$wb_pp_fit(d, names(d), cfg), "ordinal order")
  keep <- h$wb_pp_fit(d, names(d), h$wb_pp_config(overrides = list(ID = list(action = "keep"))))
  expect_true("ID" %in% keep$features)
  expect_error(h$wb_pp_fit(d, "ID", h$wb_pp_config()), "excludes every")
  expect_error(h$wb_pp_fit(data.frame(z = c(NA_real_, NA_real_)), "z", h$wb_pp_config()), "no finite training")
  expect_equal(h$wb_pp_apply(data.frame(z = NA_real_), h$wb_pp_fit(data.frame(z = NA_real_), "z",
    h$wb_pp_config(numeric = "zero")))[1, 1] |> unname(), 0)
  expect_equal(unname(h$wb_pp_apply(data.frame(amount = Inf, grade = "Low"), fitted)[1, "amount"]), mean(2:12))
  expect_error(h$wb_pp_apply(data.frame(amount = "bad", grade = "Low"), fitted), "remain numeric")
  expect_error(h$wb_pp_apply(data.frame(amount = 1), fitted), "every saved predictor")
})

test_that("one-hot missing and unknown slots cannot collide with real category names", {
  h <- pp_fixture()$h
  d <- data.frame(g = c("[Missing]", "[Unknown]", NA, "A B", "A_B"))
  fitted <- h$wb_pp_fit(d, "g", h$wb_pp_config())
  expect_false(anyDuplicated(fitted$columns) > 0)
  x <- h$wb_pp_apply(d, fitted)
  expect_equal(rowSums(x), rep(1, 5))
  expect_false(identical(x[1, ], x[3, ]))
  expect_false(identical(x[2, ], h$wb_pp_apply(data.frame(g = "new"), fitted)[1, ]))
  expect_error(h$wb_pp_fit(data.frame(g = as.character(1:201)), "g", h$wb_pp_config()), "over 200")
  empty <- h$wb_pp_fit(data.frame(g = c(NA_character_, NA_character_)), "g", h$wb_pp_config())
  expect_true(all(is.finite(h$wb_pp_apply(data.frame(g = c(NA, "future")), empty))))
})

test_that("customer preprocessing survives Basic/Custom training, replay and saved prediction", {
  h <- pp_fixture()$h
  root <- tempfile(); dir.create(root)
  withr::local_options(DDESONN_OUTPUT_ROOT = root)
  withr::local_envvar(DDESONN_ARTIFACTS_ROOT = root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  ex <- h$wb_example("customer_train")
  p <- h$wb_prepare(ex$data, ex$target, setdiff(names(ex$data), ex$target), "multiclass", preprocessing = h$wb_pp_config())
  p$source <- ex$source
  test <- h$wb_example("customer_test")$data
  expect_equal(ncol(p$parts$train$x), length(p$encoded_features))
  expect_false("ID" %in% p$features)
  expect_true(all(is.finite(p$parts$validation$x)))
  cfg <- h$wb_config(hidden = "2", epochs = 2, checkpoint = "final")
  invisible(capture.output(b <- h$wb_run(p, cfg)))
  predictions <- h$wb_new_predictions(b, test)
  expect_equal(nrow(predictions), 2627)
  expect_identical(h$wb_new_predictions(b, test[, setdiff(names(test), "ID")]), predictions)
  expect_true(all(predictions$predicted %in% c("A", "B", "C", "D")))
  expect_equal(rowSums(predictions[, -(1:2)]), rep(1, 2627), tolerance = 1e-6)
  restored <- h$wb_validate_bundle(unserialize(serialize(b, NULL)))
  expect_identical(h$wb_new_predictions(restored, test), predictions)
  e <- new.env(parent = globalenv())
  invisible(capture.output(eval(parse(text = b$recipe), e)))
  expect_identical(e$parts, p$parts)
  expect_equal(h$wb_predictions(e$model, p, p$parts$test), h$wb_predictions(b$model, p, p$parts$test), tolerance = 1e-10)
  invisible(capture.output(custom <- h$wb_run(p, h$wb_custom_config(cfg, seeds = "42"))))
  custom <- h$wb_validate_bundle(unserialize(serialize(custom, NULL)))
  expect_equal(nrow(h$wb_new_predictions(custom, test)), 2627)
  expect_true(all(is.finite(h$wb_prediction_inputs(test, custom$prepared))))
  replay <- new.env(parent = globalenv())
  invisible(capture.output(eval(parse(text = custom$recipe), replay)))
  expect_identical(replay$bundle$prepared$preprocessing, custom$prepared$preprocessing)
  expect_equal(h$wb_new_predictions(replay$bundle, test), h$wb_new_predictions(custom, test), tolerance = 1e-10)
  expect_error(h$wb_prepare(transform(ex$data, Segmentation = NA_character_), "Segmentation", "Age", "multiclass",
    preprocessing = h$wb_pp_config()), "target contains missing")
})

test_that("Shiny prepares the full customer example automatically and invalidates changed rules", {
  f <- pp_fixture(); h <- f$h
  server <- h$wb_server; formals(server)$helper <- file.path(f$root, "workbench_helpers.R")
  shiny::testServer(server, {
    session$setInputs(wb_example = "iris")
    session$setInputs(wb_example = "customer_train")
    session$setInputs(wb_target = "Segmentation", wb_features = setdiff(names(raw()), "Segmentation"),
      wb_mode = "multiclass", wb_seed = 42, wb_train_fraction = .6, wb_val_fraction = .2, wb_scale = TRUE, wb_split = "random")
    session$setInputs(wb_prepare = 1)
    expect_null(error())
    expect_equal(prepared()$rows, 8068)
    expect_false("ID" %in% prepared()$features)
    expect_equal(ncol(prepared()$parts$train$x), 40)
    session$setInputs(wb_pp_numeric = "mean")
    expect_null(prepared())
    session$setInputs(wb_pp_column = "ID", wb_pp_role = "keep")
    session$setInputs(wb_pp_apply = 1)
    expect_identical(pp_overrides()$ID$action, "keep")
    session$setInputs(wb_example = "iris")
    expect_length(pp_overrides(), 0)
  })
})
