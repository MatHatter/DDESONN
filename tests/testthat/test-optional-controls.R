test_that("optional controls disable their backend effects and restore saved enablement", {
  skip_if_not_installed("shiny")
  h <- new.env(parent = globalenv())
  path <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(path, "workbench_helpers.R"), h)
  sys.source(file.path(path, "workbench.R"), h)
  server <- h$wb_server; formals(server)$helper <- file.path(path, "workbench_helpers.R")
  shiny::testServer(server, {
    session$setInputs(wb_hidden = "2", wb_epochs = 2, wb_lr = .01,
      wb_optimizer = "adagrad", wb_reg = "L2", wb_lambda = .002,
      wb_workflow = "custom", wb_structure = "members", wb_members = 2,
      wb_seeds = "1,2", wb_temp = 1, wb_activation = "relu", wb_init = "he",
      wb_dropout = .25, wb_schedule = "exponential", wb_minimum = .0001,
      wb_decay = .8, wb_interval = 5, wb_fusion = "mean", wb_seed_aggregate = "mean",
      wb_reg_enabled = FALSE, wb_dropout_enabled = FALSE, wb_schedule_enabled = FALSE)
    off <- next_config()
    expect_identical(off$reg, "none"); expect_equal(off$lambda, 0)
    expect_equal(off$dropout, 0); expect_identical(off$schedule, "fixed")
    expect_equal(off$decay, 1); expect_equal(off$minimum, off$lr)
    session$setInputs(wb_reg_enabled = TRUE, wb_dropout_enabled = TRUE, wb_schedule_enabled = TRUE)
    on <- next_config()
    expect_identical(on$reg, "L2"); expect_equal(on$lambda, .002)
    expect_equal(on$dropout, .25); expect_identical(on$schedule, "exponential")
    expect_equal(on$decay, .8); expect_equal(on$minimum, .0001)
    # Regularization is shared and survives workflow changes.
    session$setInputs(wb_workflow = "basic")
    expect_identical(next_config()$reg, "L2")
    expect_equal(next_config()$lambda, .002)
    for (penalty in c("L1", "L1_L2", "Group_Lasso", "Max_Norm")) {
      session$setInputs(wb_reg = penalty)
      expect_identical(next_config()$reg, penalty)
      expect_equal(next_config()$lambda, .002)
    }
    session$setInputs(wb_reg_enabled = FALSE)
    expect_identical(next_config()$reg, "none")
    expect_equal(next_config()$lambda, 0)
    session$setInputs(wb_reg_enabled = TRUE, wb_reg = "L2")
    for (optimizer in DDESONN::ddesonn_optimizer_options()) {
      session$setInputs(wb_optimizer = optimizer)
      expect_identical(next_config()$optimizer, optimizer)
      session$setInputs(wb_workflow = "custom")
      expect_identical(next_config()$optimizer, optimizer)
      session$setInputs(wb_workflow = "basic")
      expect_identical(next_config()$optimizer, optimizer)
    }
    session$setInputs(wb_optimizer = "adagrad")
    basic <- next_config()
    session$setInputs(wb_workflow = "custom")
    expect_equal(next_config(), on)
    messages <- new.env()
    session$sendInputMessage <- function(inputId, message) messages[[inputId]] <- message
    sync_controls(on, "multiclass")
    for (id in c("wb_reg_enabled", "wb_dropout_enabled", "wb_schedule_enabled"))
      expect_true(messages[[id]]$value)
    sync_controls(off, "multiclass")
    for (id in c("wb_reg_enabled", "wb_dropout_enabled", "wb_schedule_enabled"))
      expect_false(messages[[id]]$value)
    sync_controls(basic, "multiclass")
    expect_true(messages$wb_reg_enabled$value)
    expect_equal(as.numeric(messages$wb_lambda$value), .002)
    expect_identical(messages$wb_reg$value, "L2")
    session$setInputs(wb_reg_enabled = FALSE, wb_dropout_enabled = FALSE, wb_schedule_enabled = FALSE)
    expect_equal(next_config(), off)
  })
})

test_that("Basic training applies shared penalties and retains them in saved bundles", {
  h <- new.env(parent = globalenv())
  sys.source(system.file("shiny", "benchmark", "workbench_helpers.R", package = "DDESONN"), h)
  root <- tempfile(); dir.create(root)
  withr::local_options(DDESONN_OUTPUT_ROOT = root)
  withr::local_envvar(DDESONN_ARTIFACTS_ROOT = root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  base <- h$wb_config(hidden = "2", epochs = 2, lr = .01, checkpoint = "final")
  invisible(capture.output(off <- h$wb_run(p, base)))
  for (penalty in c("L1", "L2", "L1_L2", "Group_Lasso", "Max_Norm")) {
    cfg <- base; cfg$reg <- penalty; cfg$lambda <- .01
    invisible(capture.output(on <- h$wb_run(p, cfg)))
    expect_true(all(is.finite(h$wb_predictions(on$model, p, p$parts$test))))
    expect_equal(on$model$ensemble[[1]]$lambda, .01)
    restored <- h$wb_validate_bundle(unserialize(serialize(on, NULL)))
    expect_identical(restored$config$reg, penalty)
    expect_equal(restored$config$lambda, .01)
    if (penalty == "L2") expect_false(isTRUE(all.equal(
      on$model$ensemble[[1]]$weights, off$model$ensemble[[1]]$weights)))
  }
  cfg <- base; cfg$optimizer <- "ftrl"
  invisible(capture.output(ftrl <- h$wb_run(p, cfg)))
  expect_true(all(is.finite(h$wb_predictions(ftrl$model, p, p$parts$test))))
  expect_identical(ftrl$config$optimizer, "ftrl")
})
