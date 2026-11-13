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
    messages <- new.env()
    session$sendInputMessage <- function(inputId, message) messages[[inputId]] <- message
    sync_controls(on, "multiclass")
    for (id in c("wb_reg_enabled", "wb_dropout_enabled", "wb_schedule_enabled"))
      expect_true(messages[[id]]$value)
    sync_controls(off, "multiclass")
    for (id in c("wb_reg_enabled", "wb_dropout_enabled", "wb_schedule_enabled"))
      expect_false(messages[[id]]$value)
    session$setInputs(wb_reg_enabled = FALSE, wb_dropout_enabled = FALSE, wb_schedule_enabled = FALSE)
    expect_equal(next_config(), off)
  })
})
