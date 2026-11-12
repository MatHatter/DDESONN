wb_ui <- function() {
  list(
    shiny::tabPanel("Data", value = "data",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          shiny::h3("Start with your data"),
          shiny::fileInput("wb_data", "Training data (CSV, up to 10 MB)", accept = ".csv"),
          shiny::actionButton("wb_demo", "Use iris example"),
          shiny::p("The example is ready for multiclass training. For binary classification choose data with two target classes."),
          shiny::uiOutput("wb_columns"),
          shiny::selectInput("wb_mode", "Task", c("Multiclass classification" = "multiclass", "Binary classification" = "binary", "Regression" = "regression")),
          shiny::selectInput("wb_split", "Split method", c("Random (stratified for classification)" = "random", "Ordered (preserve row order)" = "ordered")),
          shiny::numericInput("wb_train_fraction", "Training fraction", .6, min = .1, max = .9, step = .05),
          shiny::numericInput("wb_val_fraction", "Validation fraction", .2, min = .05, max = .4, step = .05),
          shiny::numericInput("wb_seed", "Random seed", 42, min = 0, max = 2147483646, step = 1),
          shiny::checkboxInput("wb_scale", "Standardize using training means and SDs", TRUE),
          shiny::actionButton("wb_prepare", "Prepare data", class = "btn-primary")),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom", shiny::h3("Inspect before training"),
            shiny::p("Numeric predictors only. Encode categorical predictors and handle missing values before uploading. The target may be a class label or a numeric regression value."),
            shiny::textOutput("wb_data_source"),
            shiny::div(class = "scroll-table", shiny::tableOutput("wb_preview"))),
          shiny::div(class = "panel-body-custom", shiny::h3("Prepared split"),
            shiny::textOutput("wb_prepared"),
            shiny::p("Any change to data, columns, task, split or seed requires Prepare data again. Preprocessing is fitted only on training rows and reused for validation, test and new predictions. Random splits do not keep duplicate inputs or related subjects together; prepare grouped data separately when needed.")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_error"))))),
    shiny::tabPanel("Build & train", value = "train",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          shiny::h3("Build a single model"),
          shiny::textInput("wb_hidden", "Hidden-layer sizes", "16,8"),
          shiny::p("Comma-separated units; leave blank for no hidden layer. Task-appropriate activations are selected by the API."),
          shiny::numericInput("wb_epochs", "Epochs", 50, min = 1, max = 5000, step = 1),
          shiny::selectInput("wb_optimizer", "Optimizer", c("Standard AdaGrad" = "adagrad", "Adam" = "adam", "RMSprop" = "rmsprop", "SGD" = "sgd")),
          shiny::numericInput("wb_lr", "Learning rate", .01, min = .000001, max = 1, step = .001),
          shiny::tags$details(shiny::tags$summary("Advanced settings"),
            shiny::selectInput("wb_reg", "Regularization", c("None" = "none", "L1", "L2")),
            shiny::numericInput("wb_lambda", "Regularization strength", 0, min = 0, step = .0001)),
          shiny::actionButton("wb_start", "Train new model", class = "btn-primary"),
          shiny::br(), shiny::br(), shiny::actionButton("wb_cancel", "Cancel training")),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom", shiny::h3("Training progress"),
            shiny::textOutput("wb_status"), shiny::uiOutput("wb_progress"),
            shiny::plotOutput("wb_loss", height = "280px")),
          shiny::div(class = "context", "Training uses the public DDESONN API in a separate R process. Final-epoch weights are retained; no early stopping or test-based selection. Self-organization, dropout and batch normalization are off for this first workflow. Cancelled or failed runs do not replace a previously completed model."),
          shiny::div(class = "panel-body-custom", shiny::h3("Current completed model"),
            shiny::textOutput("wb_model_info"),
            shiny::p("Settings above apply to the next new model. They do not alter a completed model. To start another fit from its weights, use Models.")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_train_error"))))),
    shiny::tabPanel("Evaluate", value = "evaluate",
      shiny::div(class = "panel-body-custom",
        shiny::h3("Evaluate the completed model"),
        shiny::p("These results belong to the saved model and its original split, even if you have since changed the Data page. Use validation for development; repeated test inspection can influence model selection."),
        shiny::selectInput("wb_eval_split", "Evaluation split", c("Validation" = "validation", "Test" = "test", "Training" = "train")),
        shiny::tableOutput("wb_metrics"),
        shiny::h4("Confusion matrix (classification)"), shiny::tableOutput("wb_confusion"),
        shiny::plotOutput("wb_diagnostics", height = "300px"),
        shiny::downloadButton("wb_predictions_download", "Download evaluated predictions"),
        shiny::div(class = "scroll-table", shiny::tableOutput("wb_predictions_preview")))),
    shiny::tabPanel("Models", value = "models",
      shiny::fluidRow(
        shiny::column(6, shiny::div(class = "panel-body-custom", shiny::h3("Save or restore"),
          shiny::textOutput("wb_saved_info"),
          shiny::downloadButton("wb_save", "Download model bundle (.rds)"),
          shiny::p("Includes the model, training configuration, split data, labels and fitted preprocessing. Store it as you would the source dataset."),
          shiny::fileInput("wb_load", "Load trusted workbench model (.rds, up to 50 MB)", accept = ".rds"),
          shiny::p("Load only workbench bundles from sources you trust. Other RDS model formats are not supported."),
          shiny::h3("Another fit from saved weights"),
          shiny::numericInput("wb_more_epochs", "Additional epochs", 10, min = 1, max = 5000, step = 1),
          shiny::actionButton("wb_continue", "Continue from saved weights"),
          shiny::p("Uses the model's original data, preprocessing and settings. The API restarts optimizer accumulators and epoch schedules on each fit; this is a warm start, not an exact interrupted-run resume. Changes on Build & train are ignored here."))),
        shiny::column(6, shiny::div(class = "panel-body-custom", shiny::h3("Predict on new rows"),
          shiny::p("Provide the saved numeric predictor columns. The target is optional; saved preprocessing is reused without refitting."),
          shiny::fileInput("wb_new_data", "New data (CSV, up to 10 MB)", accept = ".csv"),
          shiny::downloadButton("wb_new_download", "Download new predictions"),
          shiny::div(class = "scroll-table", shiny::tableOutput("wb_new_preview")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_model_error"))))))
  )
}

wb_server <- function(input, output, session, helper) {
  raw <- shiny::reactiveVal(datasets::iris)
  data_source <- shiny::reactiveVal("Included iris example · 150 rows")
  prepared <- shiny::reactiveVal(NULL)
  result <- shiny::reactiveVal(NULL)
  job <- shiny::reactiveVal(NULL)
  status <- shiny::reactiveVal("Prepare data, then train a new model.")
  error <- shiny::reactiveVal(NULL)
  model_error <- shiny::reactiveVal(NULL)
  new_predictions <- shiny::reactiveVal(NULL)
  history <- shiny::reactiveVal(data.frame(epoch = integer(), loss = numeric()))
  attempt <- function(code, destination = error) {
    tryCatch({ force(code); destination(NULL) }, error = function(e) destination(conditionMessage(e)))
  }
  busy <- function() if (!is.null(job())) stop("Finish or cancel the running job first.")
  read_csv <- function(upload, limit = 10 * 1024^2) {
    shiny::req(upload)
    if (upload$size > limit) stop("File exceeds the upload limit.")
    utils::read.csv(upload$datapath, check.names = FALSE)
  }
  output$wb_columns <- shiny::renderUI({
    cols <- names(raw())
    shiny::tagList(shiny::selectInput("wb_target", "Target", cols, selected = tail(cols, 1)),
      shiny::selectizeInput("wb_features", "Predictors", cols, selected = head(cols, -1), multiple = TRUE))
  })
  shiny::observeEvent(input$wb_target, {
    shiny::updateSelectizeInput(session, "wb_features", selected = setdiff(names(raw()), input$wb_target))
  })
  shiny::observeEvent(input$wb_data, attempt({
    busy(); candidate <- read_csv(input$wb_data)
    if (nrow(candidate) < 10 || anyDuplicated(names(candidate))) stop("Use at least 10 rows and unique column names.")
    raw(candidate); prepared(NULL); data_source(paste("Uploaded:", basename(input$wb_data$name)))
  }))
  shiny::observeEvent(input$wb_demo, attempt({
    busy(); raw(datasets::iris); prepared(NULL); data_source("Included iris example · 150 rows")
    shiny::updateSelectInput(session, "wb_mode", selected = "multiclass")
    shiny::updateSelectInput(session, "wb_target", selected = "Species")
    shiny::updateSelectizeInput(session, "wb_features", selected = names(datasets::iris)[1:4])
  }))
  # Invalidate only preparation. The immutable completed model is still usable.
  signature <- shiny::reactive(list(input$wb_target, input$wb_features, input$wb_mode,
    input$wb_seed, input$wb_train_fraction, input$wb_val_fraction, input$wb_scale, input$wb_split))
  shiny::observeEvent(signature(), prepared(NULL), ignoreInit = TRUE)
  shiny::observeEvent(input$wb_prepare, attempt({
    busy()
    p <- wb_prepare(raw(), input$wb_target, input$wb_features, input$wb_mode,
      input$wb_seed, input$wb_train_fraction, input$wb_val_fraction, input$wb_scale, input$wb_split)
    prepared(p); status("Data prepared. Ready to train a new model.")
    shiny::updateTabsetPanel(session, "workspace", selected = "train")
  }))
  launch <- function(p, cfg, old = NULL) {
    busy()
    next_job <- wb_start(helper, p, cfg, old)
    job(next_job); history(data.frame(epoch = integer(), loss = numeric()))
    status("Starting training in a separate R process…")
  }
  shiny::observeEvent(input$wb_start, attempt({
    if (is.null(prepared())) stop("Prepare data on the Data page before training.")
    cfg <- wb_config(input$wb_hidden, input$wb_epochs, input$wb_lr, input$wb_optimizer,
      input$wb_lambda, input$wb_reg)
    launch(prepared(), cfg)
  }))
  shiny::observeEvent(input$wb_continue, attempt({
    if (is.null(result())) stop("Train or load a completed model first.")
    old <- result(); cfg <- old$config
    cfg$epochs <- wb_config(epochs = input$wb_more_epochs)$epochs
    launch(old$prepared, cfg, old)
  }))
  shiny::observeEvent(input$wb_cancel, attempt({
    if (!is.null(job())) {
      wb_stop(job()); job(NULL)
      status("Training cancelled. The previous completed model, if any, was retained.")
    }
  }))
  shiny::observe({
    shiny::invalidateLater(300, session)
    active <- shiny::isolate(job())
    if (!is.null(active)) {
      h <- wb_log_history(active$log); history(h)
      elapsed <- as.integer(difftime(Sys.time(), active$started, units = "secs"))
      status(sprintf("Running · %s seconds · %s of %s epochs logged", elapsed,
        if (nrow(h)) max(h$epoch) else 0L, active$epochs))
      if (!active$process$is_alive()) {
        tryCatch({
          complete <- active$process$get_result()
          complete$history <- h
          complete <- wb_validate_bundle(complete)
          result(complete); new_predictions(NULL); model_error(NULL); error(NULL)
          status(sprintf("Completed %s epochs. Model ready to evaluate or save.", active$epochs))
        }, error = function(e) {
          error(paste("Training failed:", conditionMessage(e)))
          status("Training failed. Any previous completed model was retained.")
        })
        wb_stop(active); job(NULL)
      }
    }
  })
  session$onSessionEnded(function() wb_stop(shiny::isolate(job())))
  shiny::observeEvent(input$wb_load, attempt({
    busy()
    if (input$wb_load$size > 50 * 1024^2) stop("Model bundle exceeds 50 MB.")
    candidate <- wb_validate_bundle(readRDS(input$wb_load$datapath))
    result(candidate); new_predictions(NULL)
    history(if (is.data.frame(candidate$history)) candidate$history else data.frame(epoch = integer(), loss = numeric()))
    status("Completed model loaded. Its original split and preprocessing are retained.")
  }, model_error))
  shiny::observeEvent(input$wb_new_data, attempt({
    new_predictions(NULL)
    if (is.null(result())) stop("Train or load a model before predicting.")
    new_predictions(wb_new_predictions(result(), read_csv(input$wb_new_data)))
  }, model_error))
  model_info <- shiny::reactive({
    r <- result()
    if (is.null(r)) return("No completed model yet.")
    sprintf("%s · %s predictors → %s · hidden layers: %s · %s · %s fit run(s) · saved package %s · completed %s",
      r$prepared$mode, length(r$prepared$features), r$prepared$target,
      if (length(r$config$hidden)) paste(r$config$hidden, collapse = ", ") else "none",
      r$config$optimizer, r$fit_runs, r$package_version, r$completed)
  })
  output$wb_model_info <- shiny::renderText(model_info())
  output$wb_saved_info <- shiny::renderText(model_info())
  output$wb_error <- shiny::renderText(error())
  output$wb_train_error <- shiny::renderText(error())
  output$wb_model_error <- shiny::renderText(model_error())
  output$wb_data_source <- shiny::renderText(data_source())
  output$wb_preview <- shiny::renderTable(utils::head(raw(), 10), rownames = FALSE)
  output$wb_prepared <- shiny::renderText({
    p <- prepared()
    if (is.null(p)) return("Not prepared. Choose your columns and task, then click Prepare data.")
    counts <- vapply(p$parts, function(x) nrow(x$x), integer(1))
    paste(paste(sprintf("%s: %s rows", names(counts), counts), collapse = " · "),
      if (p$mode == "binary") paste("· positive class:", p$classes[2]) else "")
  })
  output$wb_status <- shiny::renderText(status())
  output$wb_progress <- shiny::renderUI({
    active <- job(); shiny::req(active)
    h <- history(); n <- if (nrow(h)) max(h$epoch) else 0
    shiny::tags$progress(value = n, max = active$epochs, style = "width:100%;", `aria-label` = "Logged training epochs")
  })
  output$wb_loss <- shiny::renderPlot({
    h <- history(); shiny::validate(shiny::need(nrow(h) > 0, "Training loss appears as epochs are logged."))
    ggplot2::ggplot(h, ggplot2::aes(x = epoch, y = loss)) +
      (if (nrow(h) > 1) ggplot2::geom_line(color = "#087f8c")) +
      ggplot2::geom_point(color = "#087f8c") + ggplot2::labs(x = "Epoch (current fit)", y = "Logged training loss") +
      ggplot2::theme_minimal(base_size = 13)
  })
  evaluation <- shiny::reactive({
    shiny::validate(shiny::need(!is.null(result()), "Train or load a completed model first."))
    part <- if (is.null(input$wb_eval_split)) "validation" else input$wb_eval_split
    result()$evaluation[[part]]
  })
  output$wb_metrics <- shiny::renderTable(evaluation()$metrics, digits = 6)
  output$wb_confusion <- shiny::renderTable(evaluation()$confusion, rownames = TRUE)
  output$wb_predictions_preview <- shiny::renderTable(utils::head(evaluation()$predictions, 20), digits = 6)
  output$wb_diagnostics <- shiny::renderPlot({
    r <- result(); shiny::req(r)
    d <- evaluation()$predictions
    if (r$prepared$mode == "regression") {
      ggplot2::ggplot(d, ggplot2::aes(x = actual, y = prediction)) +
        ggplot2::geom_point(color = "#087f8c") + ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) +
        ggplot2::theme_minimal(base_size = 13)
    } else {
      ggplot2::ggplot(as.data.frame(table(Actual = d$actual, Predicted = d$predicted)),
        ggplot2::aes(x = Predicted, y = Actual, fill = Freq)) + ggplot2::geom_tile() +
        ggplot2::geom_text(ggplot2::aes(label = Freq)) + ggplot2::scale_fill_gradient(low = "#eaf5f4", high = "#087f8c") +
        ggplot2::theme_minimal(base_size = 13)
    }
  })
  output$wb_save <- shiny::downloadHandler(filename = function() "ddesonn-workbench-model.rds",
    content = function(file) { shiny::req(result()); saveRDS(result(), file) })
  output$wb_predictions_download <- shiny::downloadHandler(filename = function() "ddesonn-evaluated-predictions.csv",
    content = function(file) utils::write.csv(evaluation()$predictions, file, row.names = FALSE))
  output$wb_new_preview <- shiny::renderTable({ shiny::req(new_predictions()); utils::head(new_predictions(), 20) }, digits = 6)
  output$wb_new_download <- shiny::downloadHandler(filename = function() "ddesonn-new-predictions.csv",
    content = function(file) { shiny::req(new_predictions()); utils::write.csv(new_predictions(), file, row.names = FALSE) })
  # Returned reactives make session state testable without exporting app internals.
  invisible(list(prepared = prepared, result = result, job = job, status = status))
}
