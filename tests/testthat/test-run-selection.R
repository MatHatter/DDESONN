test_that("completed run selection keeps controls, plots and outputs separate from attempts", {
  skip_if_not_installed("shiny"); skip_if_not_installed("callr")
  path <- system.file("shiny", "benchmark", package = "DDESONN")
  h <- new.env(parent = globalenv())
  sys.source(file.path(path, "workbench_helpers.R"), h)
  sys.source(file.path(path, "workbench.R"), h)
  server <- h$wb_server
  formals(server)$helper <- file.path(path, "workbench_helpers.R")
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  p$source <- list(example = "iris")
  cfg <- h$wb_config(hidden = "4", epochs = 2, lr = .001)
  invisible(capture.output(best <- h$wb_run(p, cfg)))
  cfg$checkpoint <- "final"; cfg$lr <- .002
  invisible(capture.output(final <- h$wb_run(p, cfg)))
  best$history <- data.frame(epoch = 1:2, loss = c(.5, .3))
  final$history <- data.frame(epoch = 1:2, loss = c(.6, .4))
  failure <- tempfile(fileext = ".R"); success <- tempfile(fileext = ".R")
  saved <- tempfile(fileext = ".rds"); saveRDS(final, saved)
  writeLines('wb_run <- function(...) stop("No valid validation checkpoint was recorded")', failure)
  writeLines(paste0('wb_run <- function(...) { cat("Epoch 1 Loss: 0.6\\nEpoch 2 Loss: 0.4\\n"); readRDS(',
    deparse(saved), ') }'), success)
  on.exit(unlink(c(failure, success, saved)), add = TRUE)
  start <- h$wb_start
  shiny::testServer(server, {
    session$flushReact()
    prepared(p)
    begin_attempt(best$config); add_completed(best); finish_attempt("Completed")
    begin_attempt(final$config); add_completed(final); finish_attempt("Completed")
    session$flushReact()
    expect_identical(names(completed_runs()), c("1", "2"))
    expect_identical(selected_run(), "2")
    expect_identical(result()$config$checkpoint, "final")
    expect_identical(plot_history(), final$history)
    expect_match(output$wb_chart_model, "Run 2")
    expect_match(output$wb_run_choices$html, "radio")
    original <- lapply(completed_runs(), serialize, connection = NULL)
    session$setInputs(wb_selected_run = "1")
    expect_identical(selected_run(), "1")
    expect_identical(result()$config$checkpoint, "best_validation")
    expect_identical(plot_history(), best$history)
    expect_match(output$wb_settings_origin, "Run 1")
    expect_match(output$wb_chart_model, "Run 1")
    # Browser radio rebinding echoes its current value; this is not a new selection.
    new_predictions(data.frame(prediction = 1))
    select_completed("1")
    expect_identical(new_predictions(), data.frame(prediction = 1))
    # Inspect a point without replacing the model, weights, checkpoint or run.
    hover <- list(x = 1, y = .5, mapping = list(x = "epoch", y = "loss"),
      domain = list(left = 0, right = 3, bottom = 0, top = 1),
      range = list(left = 0, right = 300, bottom = 200, top = 0),
      coords_img = list(x = 100, y = 100), img_css_ratio = list(x = 1, y = 1),
      log = list(x = NULL, y = NULL))
    session$setInputs(wb_loss_hover = hover)
    expect_match(output$wb_loss_inspection, "Epoch 1")
    expect_match(output$wb_loss_marker$html, "left:100.00px;top:100.00px", fixed = TRUE)
    expect_identical(lapply(completed_runs(), serialize, connection = NULL), original)
    expect_identical(selected_run(), "1")
    # A failing real worker is an attempt, never a selectable completed model.
    h$wb_start <- function(helper, ...) start(failure, ...)
    launch(p, best$config); session$flushReact()
    history(final$history) # unrelated live output cannot replace the selected plot
    expect_identical(plot_history(), best$history)
    job()$process$wait(30000); session$elapse(400)
    expect_null(job())
    expect_identical(attempts()[["3"]]$state, "Failed")
    expect_length(completed_runs(), 2)
    expect_identical(selected_run(), "1")
    expect_match(output$wb_train_error$html, "checkpoint")
    expect_match(output$wb_train_error$html, "No usable validation checkpoint")
    expect_match(output$wb_train_error$html, "Final epoch")
    expect_match(output$wb_attempt_history$html, "Run 3")
    # Preparing data again must not relabel the latest failed attempt.
    last_status <- status()
    session$setInputs(wb_target = "Species", wb_features = names(iris)[1:4],
      wb_mode = "multiclass", wb_seed = 42, wb_train_fraction = .6,
      wb_val_fraction = .2, wb_scale = TRUE, wb_split = "random")
    session$setInputs(wb_prepare = 1)
    expect_identical(status(), last_status)
    # Cancellation leaves both completed snapshots and the selected chart intact.
    h$wb_start <- start
    long <- best$config; long$epochs <- 5000L
    launch(p, long, result()); session$setInputs(wb_cancel = 1)
    expect_null(job())
    expect_identical(attempts()[["4"]]$state, "Cancelled")
    expect_identical(lapply(completed_runs(), serialize, connection = NULL), original)
    expect_identical(plot_history(), best$history)
    # A later success clears the active warning but retains the failed record.
    h$wb_start <- function(helper, ...) start(success, ...)
    launch(p, final$config); session$flushReact()
    expect_null(training_error())
    job()$process$wait(30000); session$elapse(400)
    expect_null(job()); expect_null(training_error())
    expect_identical(attempts()[["5"]]$state, "Completed")
    expect_identical(attempts()[["3"]]$state, "Failed")
    expect_identical(selected_run(), "5")
    expect_length(completed_runs(), 3)
    expect_identical(plot_history()$epoch, 1:2)
    expect_match(output$wb_status, "Completed")
    expect_identical(lapply(completed_runs()[c("1", "2")], serialize, connection = NULL), original)
    # Loading adds another selectable snapshot and never reuses another run's loss.
    legacy <- final; legacy$history <- NULL; saveRDS(legacy, saved)
    session$setInputs(wb_load = list(datapath = saved, size = file.info(saved)$size))
    expect_identical(selected_run(), "6")
    expect_length(completed_runs(), 4)
    expect_equal(nrow(plot_history()), 0)
    expect_null(training_error())
  })
  shiny::testServer(server, {
    session$flushReact()
    expect_length(completed_runs(), 0); expect_length(attempts(), 0)
    expect_null(selected_run()); expect_null(result()); expect_null(job())
    expect_false(action_states()$wb_save)
  })
})
