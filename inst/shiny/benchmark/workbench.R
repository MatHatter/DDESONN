# Native disclosure controls keep help available without another JavaScript package.
wb_help <- function(topic, text, api = NULL) {
  shiny::tags$details(class = "wb-inline-help",
    shiny::tags$summary(title = text, `aria-label` = paste("Help:", topic), "i"),
    shiny::div(class = "wb-help-copy", text, if (!is.null(api)) shiny::tags$pre(class = "wb-help-code", api)))
}

wb_heading <- function(title, help, level = 3) {
  shiny::div(class = "wb-title-row",
    if (level == 3) shiny::h3(title) else shiny::h4(title), wb_help(title, help))
}

wb_field <- function(id, title, help, control, api = NULL) {
  shiny::div(class = "wb-field",
    shiny::div(class = "wb-control-title",
      shiny::tags$label(id = paste0(id, "-label"), `for` = id, title), wb_help(title, help, api)), control)
}

wb_switch <- function(id, title, help, value = FALSE, api = NULL) {
  shiny::div(class = "wb-switch-row shiny-input-container",
    shiny::tags$label(id = paste0(id, "-label"), `for` = id, title),
    shiny::div(class = "wb-switch-actions",
      shiny::tags$label(class = "wb-switch",
        shiny::tags$input(id = id, type = "checkbox", role = "switch",
          `aria-labelledby` = paste0(id, "-label"), checked = if (value) "checked" else NULL),
        shiny::span(class = "wb-switch-track", `aria-hidden` = "true")),
      wb_help(title, help, api)))
}

wb_code_panel <- function(output_id, title = "Equivalent R code") {
  shiny::tags$details(class = "wb-code-panel", shiny::tags$summary(title),
    shiny::verbatimTextOutput(output_id))
}

# Control availability ourselves, including on Shiny versions before the
# downloadButton(enabled=) argument. Reapply states when output URLs arrive.
wb_download_button <- function(id, label) {
  button <- shiny::downloadButton(id, label)
  button$attribs[["data-shiny-disable-auto-enable"]] <- ""
  button$attribs[["aria-disabled"]] <- "true"
  button$attribs$tabindex <- "-1"
  button$attribs$class <- paste(button$attribs$class, "disabled")
  button
}

wb_ui <- function() {
  performance <- shiny::tabPanel("Performance", value = "evaluate",
    shiny::div(class = "panel-body-custom",
      wb_heading("Model performance", "Performance measures how well the completed model predicts known outcomes. These results use its saved data split and preprocessing, not any new selections on Features."),
      shiny::p(class = "wb-guidance", "Finally, review results and outputs. Use validation to compare your choices, and reserve test results for your final assessment."),
      wb_field("wb_eval_split", "Evaluation split", "Training measures fit to examples the model learned from. Validation helps you assess settings. Test is held aside for a final assessment; repeatedly choosing settings from it weakens that assessment.",
        shiny::selectInput("wb_eval_split", NULL, c("Test" = "test", "Validation" = "validation", "Training" = "train"))),
      shiny::textOutput("wb_selected_model"),
      shiny::textOutput("wb_checkpoint_results"),
      shiny::tableOutput("wb_metrics"),
      wb_code_panel("wb_evaluation_code", "Equivalent R code — displayed results"),
      wb_heading("Confusion matrix (classification)", "Rows are actual classes and columns are predicted classes. Counts on the diagonal are correct predictions; the other cells show which classes were confused.", level = 4),
      shiny::tableOutput("wb_confusion"), shiny::plotOutput("wb_diagnostics", height = "300px"),
      wb_download_button("wb_predictions_download", "Download evaluated predictions"),
      shiny::div(class = "scroll-table", shiny::tableOutput("wb_predictions_preview"))))
  models <- shiny::tabPanel("Saved models & predictions", value = "models",
    shiny::fluidRow(
      shiny::column(6, shiny::div(class = "panel-body-custom",
        wb_heading("Save or restore", "A model bundle contains the completed model plus the settings and preprocessing it needs to make consistent predictions later."),
        shiny::textOutput("wb_saved_info"), wb_download_button("wb_save", "Download model bundle (.rds)"),
        shiny::br(), shiny::br(), shiny::uiOutput("wb_completed_recipe_button"),
        wb_code_panel("wb_completed_code", "Equivalent R code — completed model"),
        shiny::p("The bundle also contains split data and labels. Store it as you would the source dataset."),
        shiny::fileInput("wb_load", "Load trusted workbench model (.rds, up to 50 MB)", accept = ".rds"),
        shiny::p("Load only workbench bundles from sources you trust. Other RDS model formats are not supported."),
        wb_heading("Another fit from saved weights", "This starts another training run using the saved weights, data and settings. Optimizer accumulators and epoch schedules restart, so it is a warm start rather than an exact resume."),
        shiny::numericInput("wb_more_epochs", "Additional epochs", 10, min = 1, max = 5000, step = 1),
        shiny::actionButton("wb_continue", "Continue from saved weights", disabled = "disabled", `aria-disabled` = "true"),
        shiny::p("Uses the original data, preprocessing and settings, without repeating initial self-organization. Changes on Build & Train are ignored here."))),
      shiny::column(6, shiny::div(class = "panel-body-custom",
        wb_heading("Predict on new rows", "Use the completed model to predict outcomes for rows it has not seen. Include the saved predictor columns; the target column is optional. Training preprocessing is reused."),
        shiny::fileInput("wb_new_data", "New data (CSV, up to 10 MB)", accept = ".csv"),
        wb_download_button("wb_new_download", "Download new predictions"),
        wb_code_panel("wb_model_code", "Equivalent R code — saved model and prediction"),
        shiny::div(class = "scroll-table", shiny::tableOutput("wb_new_preview")),
        shiny::div(class = "error", role = "alert", shiny::textOutput("wb_model_error"))))))
  list(
    shiny::tabPanel("1. Data", value = "data",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Start with Data", "This is where you load a CSV dataset or choose the included example. Each row is an observation and each column is a variable. Choosing what to predict comes next, on Features."),
          shiny::fileInput("wb_data", "Training data (CSV, up to 10 MB)", accept = ".csv"),
          shiny::actionButton("wb_demo", "Use iris example"),
          shiny::p(class = "wb-guidance", "Start with Data. Inspect the rows, then choose the target and predictors in Features."),
          shiny::actionButton("wb_next_features", "Next: choose features", class = "btn-primary")),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom",
            wb_heading("Inspect your dataset", "Check the column names, values and overall structure before selecting features. The preview shows the first ten rows; the full dataset is used when you prepare it."),
            shiny::textOutput("wb_data_source"),
            shiny::p("Predictors must be numeric. Encode categories and handle missing values before training. A classification target can contain class labels."),
            shiny::div(class = "scroll-table", shiny::tableOutput("wb_preview")),
            wb_code_panel("wb_data_code")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_data_error"))))),
    shiny::tabPanel("2. Features", value = "features",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Choose what the model learns", "The target is the answer you want to predict. Predictors supply the evidence. Task type determines the kind of answer the model produces and the metrics used to assess it."),
          shiny::uiOutput("wb_columns"),
          wb_field("wb_mode", "Task", "Regression predicts a number, such as price. Binary classification chooses between two classes. Multiclass classification chooses among several classes. Choose a task that matches your target.",
            shiny::selectInput("wb_mode", NULL, c("Multiclass classification" = "multiclass", "Binary classification" = "binary", "Regression" = "regression"))),
          wb_field("wb_split", "Train / validation / test split", "Training rows teach the model. Validation rows help you assess development choices. Test rows are held aside for a final assessment. The fractions below allocate training and validation; remaining rows become test data.",
            shiny::selectInput("wb_split", NULL, c("Random (stratified for classification)" = "random", "Ordered (preserve row order)" = "ordered"))),
          shiny::numericInput("wb_train_fraction", "Training fraction", .6, min = .1, max = .9, step = .05),
          shiny::numericInput("wb_val_fraction", "Validation fraction", .2, min = .05, max = .4, step = .05),
          wb_field("wb_seed", "Random seed", "A seed makes random splitting and model initialization repeatable with the same data, settings and software. Changing it can change results.",
            shiny::numericInput("wb_seed", NULL, 42, min = 0, max = 2147483646, step = 1)),
          wb_switch("wb_scale", "Standardize using training means and SDs",
            "Subtracts each predictor's training average and divides by its training standard deviation (SD, a measure of spread). This puts inputs on comparable scales. Validation, test and new rows reuse those same training values; the target is unchanged.", TRUE,
            api = paste("center <- colMeans(x_train)", "spread <- apply(x_train, 2, sd)",
              "spread[!is.finite(spread) | spread == 0] <- 1",
              "x_train_scaled <- sweep(sweep(x_train, 2, center, '-'), 2, spread, '/')", sep = "\n")),
          shiny::actionButton("wb_prepare", "Prepare data", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true")),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom",
            wb_heading("Prepare for training", "Prepare data validates your choices, creates the three row splits, and fits any scaling using training rows only. The same scaling is then reused for validation, test and new data."),
            shiny::p(class = "wb-guidance", "Next, choose the target and predictors. Click Prepare data to continue to Build & Train."),
            shiny::textOutput("wb_prepared"),
            shiny::p("Target + predictors determine what the model learns from. Task determines its output and evaluation metrics. Changing any selection here requires Prepare data again."),
            wb_code_panel("wb_features_code", "Equivalent R code — prepared split and scaling")),
          shiny::div(class = "panel-body-custom",
            wb_heading("About your split", "Random classification splits keep examples from each class in each split. Ordered splitting preserves row order, which can be useful for time-based data. Neither option groups duplicate inputs or related subjects automatically."),
            shiny::p("Preprocessing is fitted only on training rows. Prepare grouped data separately when related subjects or duplicate inputs must stay together.")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_error"))))),
    shiny::tabPanel("3. Build & Train", value = "train",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Build & Train", "Configure the model and how it learns, then start a new training run on your prepared data. Architecture controls model capacity; optimizer, learning rate and epochs control how weights are updated."),
          wb_field("wb_hidden", "Hidden-layer sizes", "Hidden layers transform the predictors before the output. Enter units separated by commas, such as 16,8 for two layers. Leave blank for no hidden layer. Larger models can take longer and may overfit.",
            shiny::textInput("wb_hidden", NULL, "16,8")),
          wb_field("wb_epochs", "Epochs", "An epoch is one pass through the training data. More epochs give the model more opportunities to learn, but do not guarantee better predictions on new data.",
            shiny::numericInput("wb_epochs", NULL, 50, min = 1, max = 5000, step = 1)),
          wb_field("wb_optimizer", "Optimizer", "The optimizer is the rule used to update model weights from prediction errors. Different rules can learn at different speeds; compare them using validation data.",
            shiny::selectInput("wb_optimizer", NULL, c("Standard AdaGrad" = "adagrad", "Adam" = "adam", "RMSprop" = "rmsprop", "SGD" = "sgd"))),
          wb_field("wb_lr", "Learning rate", "The learning rate sets the size of each weight update. Too large can make training unstable; too small can make learning slow. It interacts with the optimizer and data scaling.",
            shiny::numericInput("wb_lr", NULL, .01, min = .000001, max = 1, step = .001)),
          wb_switch("wb_self_org", "Self-organize before initial training",
            "Runs DDESONN's input-space self-organization phase before supervised training. It organizes neighborhood structure, rather than directly minimizing prediction loss. This optional phase is off by default and runs only for a new model. Warm starts preserve the organized weights without repeating it. It can destabilize training; a smaller learning rate may help.",
            api = "ddesonn_fit(model, x_train, y_train, self_org = TRUE)"),
          wb_field("wb_checkpoint", "Weights used for results",
            "Best validation checkpoint uses the highest validation accuracy for classification or lowest validation loss for regression within this fit. Final epoch uses the last weights after the loop finishes. Both run all requested epochs; this is not early stopping. Test never selects a checkpoint. The equivalent code below follows your choice.",
            shiny::selectInput("wb_checkpoint", NULL,
              c("Best validation checkpoint" = "best_validation", "Final epoch (last weights)" = "final")),
            api = paste("ddesonn_fit(model, x, y, validation = list(x = x_val, y = y_val),",
              "  validation_metrics = TRUE, best_weights_on_latest_weights_off = TRUE)",
              "# Restore BOTH tensors before prediction:",
              "model$ensemble[[1]]$weights <- model$ensemble[[1]]$best_weights",
              "model$ensemble[[1]]$biases <- model$ensemble[[1]]$best_biases", sep = "\n")),
          shiny::tags$details(shiny::tags$summary("Advanced settings"),
            shiny::selectInput("wb_reg", "Regularization", c("None" = "none", "L1", "L2")),
            shiny::numericInput("wb_lambda", "Regularization strength", 0, min = 0, step = .0001)),
          shiny::actionButton("wb_start", "Train new model", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true"),
          shiny::br(), shiny::br(), shiny::actionButton("wb_cancel", "Cancel training", disabled = "disabled", `aria-disabled` = "true")),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom",
            wb_heading("Training progress", "Training runs separately so you can still use the interface. This plot shows training loss as epochs are logged. The selected epoch is marked after a completed fit; selection uses validation, not this training curve. Lower training loss alone does not establish better performance on held-out data."),
            shiny::p(class = "wb-guidance", "Then configure and train the model. When it finishes, review Model / Results."),
            shiny::textOutput("wb_status"), shiny::uiOutput("wb_progress"), shiny::plotOutput("wb_loss", height = "280px",
              hover = shiny::hoverOpts("wb_loss_hover", delay = 100, delayType = "throttle", nullOutside = TRUE)),
            shiny::div(`aria-live` = "polite", shiny::textOutput("wb_loss_inspection")),
            shiny::tags$details(shiny::tags$summary("Training loss values (epoch and loss)"),
              shiny::div(class = "wb-loss-values scroll-table", shiny::tableOutput("wb_loss_values")))),
          shiny::div(class = "context", "Training runs for all requested epochs. Choose best-validation or final-epoch weights using the dropdown. Automatic early stopping is not implemented. Self-organization is optional; dropout and batch normalization are off in this workbench. Cancelled or failed runs retain any previous completed model."),
          shiny::div(class = "panel-body-custom",
            wb_heading("Current completed model", "This is the most recent successfully trained or loaded model. Changing controls affects the next new model, not this completed model or its results."),
            shiny::textOutput("wb_model_info"),
            shiny::textOutput("wb_checkpoint_summary"),
            wb_code_panel("wb_training_code", "Equivalent R code — next new model"),
            shiny::uiOutput("wb_next_recipe_button"),
            shiny::br(), shiny::br(), shiny::actionButton("wb_review_results", "Next: review results", disabled = "disabled", `aria-disabled` = "true")),
          shiny::uiOutput("wb_train_error")))),
    shiny::tabPanel("4. Model / Results", value = "results",
      shiny::p(class = "wb-step-intro", "Finally, review performance and outputs, save the model, or use it for new predictions. Results reflect the saved choices from the earlier steps."),
      shiny::tabsetPanel(id = "wb_results_tabs", performance, models))
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
  training_error <- shiny::reactiveVal(NULL)
  new_predictions <- shiny::reactiveVal(NULL)
  history <- shiny::reactiveVal(data.frame(epoch = integer(), loss = numeric()))
  attempt <- function(code, destination = error) {
    tryCatch({ force(code); destination(NULL) }, error = function(e) destination(conditionMessage(e)))
  }
  training_attempt <- function(code) attempt(code, function(e) {
    training_error(if (is.null(e)) NULL else wb_training_error(simpleError(e)))
  })
  # Per-session states are also enforced by server guards, not just the browser.
  action_states <- shiny::reactive({
    running <- !is.null(job()); completed <- !is.null(result())
    list(wb_start = !is.null(prepared()) && !running, wb_cancel = running,
      wb_continue = completed && !running, wb_save = completed,
      wb_review_results = completed, wb_predictions_download = completed,
      wb_new_download = !is.null(new_predictions()), wb_new_data = completed,
      wb_load = !running,
      wb_prepare = !running && length(input$wb_target) == 1 &&
        length(input$wb_features) > 0 && !input$wb_target %in% input$wb_features,
      wb_data = !running, wb_demo = !running)
  })
  shiny::observe(session$sendCustomMessage("wb-action-states", action_states()))
  busy <- function() if (!is.null(job())) stop("Finish or cancel the running job first.")
  read_csv <- function(upload, limit = 10 * 1024^2) {
    shiny::req(upload)
    if (upload$size > limit) stop("File exceeds the upload limit.")
    utils::read.csv(upload$datapath, check.names = FALSE)
  }
  output$wb_columns <- shiny::renderUI({
    cols <- names(raw())
    shiny::tagList(
      wb_field("wb_target", "Target", "The target is the variable you want the model to predict, such as a class label or a numeric outcome. It is kept out of the predictors.",
        shiny::selectInput("wb_target", NULL, cols, selected = tail(cols, 1))),
      wb_field("wb_features", "Predictors", "Predictors are the input features used to predict the target. Select numeric columns available when you will make future predictions; do not include the target or information derived from the answer.",
        shiny::selectizeInput("wb_features", NULL, cols, selected = head(cols, -1), multiple = TRUE)))
  })
  shiny::observeEvent(input$wb_next_features,
    shiny::updateTabsetPanel(session, "workspace", selected = "features"))
  shiny::observeEvent(input$wb_review_results, attempt({
    if (is.null(result())) stop("Train or load a completed model before reviewing results.")
    shiny::updateTabsetPanel(session, "workspace", selected = "results")
    shiny::updateTabsetPanel(session, "wb_results_tabs", selected = "evaluate")
  }))
  code_source <- shiny::reactive({
    if (startsWith(data_source(), "Included iris")) list(example = "iris") else
      list(file = basename(input$wb_data$name))
  })
  next_config <- shiny::reactive(wb_config(input$wb_hidden, input$wb_epochs, input$wb_lr,
    input$wb_optimizer, input$wb_lambda, input$wb_reg, self_org = isTRUE(input$wb_self_org),
    checkpoint = if (is.null(input$wb_checkpoint)) "best_validation" else input$wb_checkpoint))
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
    p$source <- code_source()
    prepared(p); status("Data prepared. Ready to train a new model.")
    shiny::updateTabsetPanel(session, "workspace", selected = "train")
  }))
  launch <- function(p, cfg, old = NULL) {
    busy()
    next_job <- wb_start(helper, p, cfg, old)
    job(next_job); history(data.frame(epoch = integer(), loss = numeric()))
    status("Starting training in a separate R process…")
  }
  shiny::observeEvent(input$wb_start, training_attempt({
    if (is.null(prepared())) stop("Choose target and predictors on Features, then click Prepare data before training.")
    cfg <- next_config()
    launch(prepared(), cfg)
  }))
  shiny::observeEvent(input$wb_continue, training_attempt({
    if (is.null(result())) stop("Train or load a completed model first.")
    old <- result(); cfg <- old$config
    cfg$epochs <- wb_config(epochs = input$wb_more_epochs)$epochs
    launch(old$prepared, cfg, old)
  }))
  shiny::observeEvent(input$wb_cancel, training_attempt({
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
          training_error(NULL)
          status(paste(wb_checkpoint_summary(complete), "· Model ready to evaluate or save."))
        }, error = function(e) {
          training_error(wb_training_error(e, if (file.exists(active$error))
            readLines(active$error, warn = FALSE) else character()))
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
    result(candidate); new_predictions(NULL); training_error(NULL)
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
    sprintf("%s · %s predictors → %s · hidden layers: %s · %s · %s checkpoint, epoch %s of %s · initial self-organization %s · %s fit run(s) · saved package %s · completed %s",
      r$prepared$mode, length(r$prepared$features), r$prepared$target,
      if (length(r$config$hidden)) paste(r$config$hidden, collapse = ", ") else "none",
      r$config$optimizer, if (identical(r$config$checkpoint, "best_validation")) "best validation" else "final",
      if (is.null(r$selected_epoch)) "not recorded" else r$selected_epoch, r$config$epochs,
      if (isTRUE(r$config$self_org)) "on" else "off",
      r$fit_runs, r$package_version, r$completed)
  })
  output$wb_data_code <- shiny::renderText(wb_data_code(code_source()))
  output$wb_features_code <- shiny::renderText({
    shiny::validate(shiny::need(!is.null(prepared()), "Choose features and click Prepare data first."))
    paste("# Run the Data snippet first.", wb_features_code(prepared()), sep = "\n")
  })
  output$wb_training_code <- shiny::renderText({
    shiny::validate(shiny::need(!is.null(prepared()), "Prepare data before generating a new-model recipe."))
    cfg <- tryCatch(next_config(), error = function(e) NULL)
    shiny::validate(shiny::need(!is.null(cfg), "Check the model settings to generate a recipe."))
    paste("# Run Data and Features snippets first, or download the complete recipe.",
      wb_fit_code(prepared(), cfg), sep = "\n")
  })
  output$wb_model_code <- shiny::renderText(wb_bundle_code())
  output$wb_completed_code <- shiny::renderText({
    shiny::req(result())
    if (is.null(result()$recipe)) "# This older bundle does not contain a replay recipe. Use the saved-model API code." else result()$recipe
  })
  output$wb_next_recipe_button <- shiny::renderUI({
    cfg <- tryCatch(next_config(), error = function(e) NULL)
    if (!is.null(prepared()) && !is.null(cfg))
      shiny::downloadButton("wb_next_recipe", "Download next-model R recipe")
  })
  output$wb_completed_recipe_button <- shiny::renderUI({
    if (!is.null(result()$recipe)) shiny::downloadButton("wb_completed_recipe", "Download completed-model R recipe")
  })
  output$wb_next_recipe <- shiny::downloadHandler(filename = function() "ddesonn-next-model.R",
    content = function(file) { shiny::req(prepared()); writeLines(wb_recipe(prepared(), next_config()), file) })
  output$wb_completed_recipe <- shiny::downloadHandler(filename = function() "ddesonn-completed-model.R",
    content = function(file) { shiny::req(result()$recipe); writeLines(result()$recipe, file) })
  output$wb_selected_model <- shiny::renderText(model_info())
  output$wb_model_info <- shiny::renderText(model_info())
  output$wb_saved_info <- shiny::renderText(model_info())
  output$wb_error <- shiny::renderText(error())
  output$wb_data_error <- shiny::renderText(error())
  output$wb_train_error <- shiny::renderUI({
    e <- training_error(); if (is.null(e)) return(NULL)
    shiny::div(class = "wb-training-error",
      shiny::div(class = "error", role = "alert", e$message,
        shiny::p("Any previous completed model is still available. Failed runs never replace it.")),
      shiny::tags$details(shiny::tags$summary("Technical details"), shiny::tags$pre(e$details)))
  })
  output$wb_checkpoint_summary <- shiny::renderText(wb_checkpoint_summary(result()))
  output$wb_checkpoint_results <- shiny::renderText(wb_checkpoint_summary(result()))
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
  output$wb_loss_inspection <- shiny::renderText({
    h <- history()
    if (is.null(input$wb_loss_hover) || !nrow(h))
      return("Hover near a point to inspect epoch and loss, or expand Training loss values. Inspection does not change model weights.")
    point <- shiny::nearPoints(h, input$wb_loss_hover, xvar = "epoch", yvar = "loss",
      threshold = 20, maxpoints = 1)
    if (!nrow(point)) return("Move closer to a point to inspect its epoch and loss.")
    sprintf("Epoch %s · training loss: %.8g", point$epoch, point$loss)
  })
  output$wb_loss_values <- shiny::renderTable(history(), digits = 8)
  output$wb_loss <- shiny::renderPlot({
    h <- history(); shiny::validate(shiny::need(nrow(h) > 0, "Training loss appears as epochs are logged."))
    r <- result()
    selected <- if (is.null(job()) && !is.null(r) && identical(h, r$history)) r$selected_epoch else NULL
    ggplot2::ggplot(h, ggplot2::aes(x = epoch, y = loss)) +
      (if (length(selected) == 1 && is.finite(selected))
        ggplot2::geom_vline(xintercept = selected, linetype = 2, color = "#69788c")) +
      (if (nrow(h) > 1) ggplot2::geom_line(color = "#087f8c")) +
      ggplot2::geom_point(color = "#087f8c") + ggplot2::labs(x = "Epoch (current fit)", y = "Logged training loss",
        subtitle = if (length(selected) == 1 && is.finite(selected))
          sprintf("Selected epoch %s · %s", selected,
            if (identical(r$config$checkpoint, "best_validation")) "best validation" else "final epoch") else NULL) +
      ggplot2::theme_minimal(base_size = 13)
  })
  evaluation <- shiny::reactive({
    shiny::validate(shiny::need(!is.null(result()), "Train or load a completed model first."))
    part <- if (is.null(input$wb_eval_split)) "test" else input$wb_eval_split
    result()$evaluation[[part]]
  })
  output$wb_evaluation_code <- shiny::renderText({
    shiny::req(result())
    part <- if (is.null(input$wb_eval_split)) "test" else input$wb_eval_split
    wb_evaluation_code(result()$prepared, part)
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
