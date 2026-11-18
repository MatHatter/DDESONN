# Native disclosure controls keep help available without another JavaScript package.
wb_help <- function(topic, text, api = NULL, show_code = TRUE) {
  shiny::tags$details(class = "wb-inline-help",
    shiny::tags$summary(title = text, `aria-label` = paste("Help:", topic),
      shiny::tags$svg(class = "wb-help-icon", viewBox = "0 0 20 20", width = "20", height = "20",
        `aria-hidden` = "true", focusable = "false",
        shiny::tags$circle(class = "wb-help-ring", cx = "10", cy = "10", r = "8.4"),
        shiny::tags$text(class = "wb-help-letter", x = "10", y = "14.25", `text-anchor` = "middle", "i"))),
    shiny::div(class = "wb-help-copy", tabindex = "0", role = "region",
      `aria-label` = paste(topic, if (show_code) "help and equivalent R" else "help"), text,
      if (show_code) shiny::tagList(shiny::div(class = "wb-help-code-label", "Equivalent R"),
      shiny::tags$pre(id = wb_help_key(topic), class = "shiny-text-output wb-help-code",
        tabindex = "0", `aria-label` = paste("Equivalent R code:", topic),
        if (is.null(api)) "# Open help to load the equivalent R example." else api),
      shiny::p(class = "wb-help-prerequisites", "This fragment covers only this topic. Show full R code at the bottom of the page for the complete workflow and required objects."))))
}

# Keep help beside each radio label, outside the label so opening it never
# changes the purpose or clears already computed results.
wb_csv_purpose_control <- function() {
  topics <- c("Evaluate a holdout sample", "Make production predictions")
  explanations <- c(
    "Typical holdout testing: set aside data with known correct answers before training. Load the saved weights, predict those rows, then compare the predictions with the answers to measure accuracy. Bank example: test on historical loan applicants whose repayment outcomes are already known, using cases kept out of training. Include the correct answer column in your CSV. This evaluates the existing model; it does not train it again.",
    "Production use: a bank estimates whether a new applicant will repay before deciding whether to lend. The saved weights and the applicant's input values produce a learned estimate; the real outcome is not known yet, so accuracy cannot be calculated now. Save the predictions and compare them with actual repayment outcomes when they become available. This is one prediction run, not a live update or retraining. An unknown answer can also be a current unknown, such as a customer's segment, rather than a future event.")
  control <- shiny::radioButtons("wb_csv_purpose", NULL,
    c("Evaluate a holdout sample (known answers)" = "evaluate",
      "Make production predictions (answers unknown)" = "predict"), selected = "evaluate", width = "100%")
  control$attribs[["aria-label"]] <- "Choose holdout evaluation or production prediction"
  index <- 0L
  add_help <- function(node) {
    if (inherits(node, "shiny.tag")) {
      if (identical(node$attribs$class, "radio")) {
        index <<- index + 1L
        node$attribs$class <- "radio wb-purpose-choice"
        node$children <- c(node$children, list(wb_help(topics[index], explanations[index], show_code = FALSE)))
        return(node)
      }
      node$children <- lapply(node$children, add_help)
    } else if (is.list(node)) node <- lapply(node, add_help)
    node
  }
  add_help(control)
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

wb_pp_field <- function(id, title, control) {
  help <- wb_pp_help()[[id]]
  wb_field(id, title, help$text, control, help$code)
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
  wb_download_info(id, button)
}

# Download help describes the exact artifact, independently of other controls.
wb_download_info <- function(id, button) {
  info <- switch(id,
    wb_save =, wb_save_selected = list("A selected-model bundle (.rds) with complete retained trained tensors and all retained Custom seeds/members, settings, preprocessing, prepared split data and labels, evaluations, history, retained logs and recipe. It excludes discarded TEMP candidate models and the session's other completed runs. Preview summarizes these contents; the file retains full tensors and data.", "saveRDS(bundle, 'ddesonn-workbench-model.rds')"),
    wb_predictions_download = list("CSV of every evaluated row in the selected Train/Validation/Test split: row index, actual outcome and prediction, plus classification probabilities. The table previews the first 20 rows. This CSV contains no trained model.", "write.csv(bundle$evaluation[[split]]$predictions, 'ddesonn-evaluated-predictions.csv', row.names = FALSE)"),
    wb_new_download = list("CSV of all processed new rows: row index and prediction, plus classification probabilities. The table previews the first 20 rows. It does not include model tensors or the training dataset.", "write.csv(predictions, 'ddesonn-new-predictions.csv', row.names = FALSE)"),
    wb_next_recipe = list("An R script for the currently prepared dataset and next-model settings. It references the original CSV or included example, prepares splits/scaling, trains and evaluates. It is code, not a trained model or a copy of your raw CSV. Preview shows the complete script.", "# Review the complete script before running it.\nsource('ddesonn-next-model.R')"),
    wb_completed_recipe = list("The selected completed model's saved training recipe, unaffected by current control edits. It is an R script, not trained weights. Warm-start recipes identify the prior bundle they require. Preview shows the complete saved script.", "# Uses the source data or prior bundle named in the saved recipe.\nsource('ddesonn-completed-model.R')"),
    wb_full_download = list("Exactly the R script displayed in this full-code viewer: next-model training, the completed model's saved recipe, or the saved-model prediction example, according to Code for. It contains code rather than a trained model export.", "# The displayed text and downloaded .R file are identical.\n# Review required input paths before sourcing the script."),
    download = list("CSV containing every currently selected experiment row and its metric columns. The Explore seeds table previews at most 100 rows; the download includes all selected rows. It contains saved benchmark results, not models or a new training run.", "write.csv(selected_rows, 'selected-results.csv', row.names = FALSE)"),
    download_summary = list("CSV of the experiment metric summary for the current filters: the same summary shown on Overview. It contains summary statistics, not individual models or their weights.", "write.csv(metric_summary, 'summary.csv', row.names = FALSE)"))
  if (is.null(info)) return(button)
  title <- switch(id, wb_save = "Download model bundle", wb_save_selected = "Save selected model",
    wb_predictions_download = "Download evaluated predictions", wb_new_download = "Download new predictions",
    wb_next_recipe = "Download next-model recipe", wb_completed_recipe = "Download completed-model recipe",
    wb_full_download = "Download displayed R script", download = "Download experiment rows", download_summary = "Download experiment summary")
  shiny::div(class = "wb-download-row", button, wb_help(title, info[[1]], info[[2]]))
}

wb_vignette_link <- function(id, label = "See related vignettes") {
  shiny::div(class = "wb-vignette-link", shiny::actionLink(id, label))
}

wb_saved_pager <- function(id) {
  shiny::div(class = "wb-saved-pager",
    shiny::actionButton(paste0(id, "_prev"), "Previous"),
    shiny::textOutput(paste0(id, "_page"), inline = TRUE),
    shiny::actionButton(paste0(id, "_next"), "Next"))
}

wb_optimizer_choices <- function() {
  values <- DDESONN::ddesonn_optimizer_options()
  labels <- c(adagrad = "Standard AdaGrad", adam = "Adam", lamb = "LAMB",
    sgd = "SGD", sgd_momentum = "SGD with momentum", nag = "Nesterov accelerated gradient",
    rmsprop = "RMSprop", ftrl = "FTRL", lookahead = "Lookahead", adadelta = "AdaDelta")
  display <- values
  known <- values %in% names(labels)
  display[known] <- labels[values[known]]
  stats::setNames(values, display)
}

wb_ui <- function() {
  performance <- shiny::tabPanel("Performance", value = "evaluate",
    shiny::div(class = "panel-body-custom wb-performance",
      wb_heading("Model performance", "Performance measures how well the completed model predicts known outcomes. These results use its saved data split and preprocessing, not any new selections on Features."),
      shiny::p(class = "wb-guidance", "Finally, review results and outputs. Use validation to compare your choices, and reserve test results for your final assessment."),
      wb_field("wb_eval_split", "Evaluation split", "Training measures fit to examples the model learned from. Validation helps you assess settings. Test is held aside for a final assessment; repeatedly choosing settings from it weakens that assessment.",
        shiny::selectInput("wb_eval_split", NULL, c("Test" = "test", "Validation" = "validation", "Training" = "train"), width = "180px")),
      shiny::textOutput("wb_selected_model"),
      shiny::textOutput("wb_checkpoint_results"),
      shiny::tableOutput("wb_metrics"),
      shiny::tags$details(shiny::tags$summary("Training-seed validation comparison and member weights"), shiny::tableOutput("wb_seed_comparison"), shiny::verbatimTextOutput("wb_member_weights")),
      wb_code_panel("wb_evaluation_code", "Equivalent R code — displayed results"),
      wb_heading("Confusion matrix (classification)", "Rows are actual classes and columns are predicted classes. Counts on the diagonal are correct predictions; the other cells show which classes were confused.", level = 4),
      shiny::tableOutput("wb_confusion"), shiny::plotOutput("wb_diagnostics", height = "300px"),
      shiny::div(class = "wb-evaluation-exports",
        wb_download_button("wb_predictions_download", "Download evaluated predictions"),
        shiny::div(class = "wb-evaluation-actions",
          shiny::tags$details(shiny::tags$summary("Preview evaluated CSV (first 20 rows)"),
            shiny::div(class = "scroll-table", shiny::tableOutput("wb_predictions_preview"))),
          shiny::actionButton("wb_use_trained_model", "Evaluate or predict with this trained model", class = "btn-primary")))))
  recent <- shiny::tabPanel("Recent runs", value = "recent",
    shiny::div(class = "panel-body-custom",
      shiny::h3("Global recent runs — this session"),
      shiny::p("Keeps the latest 10 completed training models, loaded models and CSV evaluation/prediction runs in memory for this Shiny session. Nothing here is automatically saved to disk. Restore a model and use Save & export to download its .rds before ending the session."),
      shiny::p("Clear saved-model workspace does not remove this history. Remove selected item and Clear recent history remove only archive entries: they do not close an active model, erase training results or delete files."),
      shiny::uiOutput("wb_recent_items"),
      shiny::actionButton("wb_recent_restore", "Restore selected item", disabled = "disabled"),
      shiny::actionButton("wb_recent_remove", "Remove selected history item", disabled = "disabled"),
      shiny::actionButton("wb_recent_clear", "Clear recent history", disabled = "disabled"),
      shiny::div(`aria-live` = "polite", shiny::textOutput("wb_recent_notice")),
      shiny::div(class = "error", role = "alert", shiny::textOutput("wb_recent_error")),
      shiny::uiOutput("wb_recent_details")))
  models <- shiny::tabPanel("Saved model → evaluate or predict", value = "saved",
    shiny::div(class = "panel-body-custom",
      shiny::h3("1. Choose a model"),
      shiny::p("A saved model already contains trained weights. Loading it lets you evaluate or predict without training again. Additional training also stays in this saved-model workspace; Data, Features and Build & Train stay unchanged."),
      shiny::actionButton("wb_clear_saved", "Clear saved-model workspace"),
      shiny::p("Starts this Saved model → evaluate or predict tab blank: clears its open models/selection, model summary and history display, CSV, results, file pickers, errors and purpose choice. Keeps Recent runs, all training data/settings/results and files on your computer. A replacement file becomes active only after it validates successfully."),
      shiny::p("To evaluate the downloadable example, only two uploads are needed: the .rds model here, then its labeled CSV below. No other files or retraining are required."),
      shiny::p("Load only workbench bundles from sources you trust. Other RDS model formats are not supported."),
      shiny::p("Loading a model does not create a new file on disk. Recent runs keeps session snapshots; Save & export downloads a model file for permanent keeping."),
      shiny::div(class = "wb-model-upload",
        shiny::fileInput("wb_load", "Load trusted workbench model (.rds, up to 50 MB)", accept = ".rds", width = "100%"),
        shiny::tags$p(class = "wb-model-status", `aria-live` = "polite", shiny::tags$em(shiny::textOutput("wb_active_model", inline = TRUE)))),
      shiny::div(class = "error", role = "alert", shiny::textOutput("wb_load_error")),
      shiny::div(`aria-live` = "polite", shiny::textOutput("wb_workspace_notice")),
      shiny::uiOutput("wb_prediction_model_ui"),
      shiny::conditionalPanel("output.wb_has_saved_model === 'yes'",
        shiny::tags$details(class = "wb-disclosure", shiny::tags$summary("View trained model details"), shiny::textOutput("wb_saved_info"))),
      shiny::conditionalPanel("output.wb_has_saved_model === 'yes'",
        shiny::div(class = "wb-result-area", shiny::h3("Saved model evaluation"),
          shiny::p("These are the original scores from the data splits saved with the model. If you only want to review those results, no CSV upload is needed. Upload another CSV below to evaluate labeled data or predict unknown answers."),
          shiny::tableOutput("wb_saved_evaluation")),
        shiny::tags$details(shiny::tags$summary("Saved training history — not live training"),
          shiny::textOutput("wb_saved_history_note"),
          shiny::plotOutput("wb_saved_loss", height = "250px"),
          shiny::verbatimTextOutput("wb_saved_logs"))),
        shiny::tags$details(class = "wb-model-help",
          shiny::tags$summary("Need help preparing a saved model?"),
          shiny::p("Use ChatGPT separately to help prepare a compatible model. ChatGPT is not built into this app. You can also train here and download the completed model bundle."),
          shiny::tags$ol(
            shiny::tags$li("Reference ", shiny::tags$a(href = "https://github.com/MatHatter/DDESONN", target = "_blank", rel = "noopener noreferrer", "MatHatter/DDESONN on GitHub"),
              " and give your installed DDESONN version, task, target and predictor columns. For customer segmentation, the task is multiclass classification and the packaged target is Segmentation."),
            shiny::tags$li("Ask for an actually trained workbench .rds bundle with saved preprocessing and class labels, a labeled evaluation CSV with known answers, and a separate unlabeled prediction CSV. Ask where evaluation rows came from and whether they repeat the saved test set. If ChatGPT cannot run R, ask for a reproducible R script to run yourself; generated instructions alone are not a trained model."),
            shiny::tags$li("Choose Use a saved model at the top. Under 1. Choose a model, upload the trusted workbench .rds bundle using Load trusted workbench model."),
            shiny::tags$li("Choose Evaluate a holdout sample and upload predictors plus known answers to measure accuracy. Or choose Make production predictions to estimate unknown answers from predictors. Saved preprocessing is reused."),
            shiny::tags$li("Read the evaluation or predictions below your CSV, then optionally choose Download results (CSV). Check the model's held-out evaluation before relying on its predictions.")),
          shiny::p("Example request to copy and adapt:"),
          shiny::tags$pre("Use https://github.com/MatHatter/DDESONN and my installed DDESONN version [version]. Train a multiclass customer-segmentation model with target Segmentation and these predictors: [columns]. Export a compatible Shiny workbench .rds bundle with fitted preprocessing and class labels, plus a labeled evaluation CSV with known answers and a separate unlabeled prediction CSV. State whether the evaluation CSV reuses saved test rows or contains new data. Use separate training, validation and test data and report the evaluation. If you cannot execute R, provide a reproducible R script instead of claiming to have trained the model.")),
      shiny::tabsetPanel(id = "wb_model_tools", selected = "predict",
        shiny::tabPanel("Evaluate or predict", value = "predict",
          shiny::h3("2. What do you want to do with your CSV?"),
          wb_csv_purpose_control(),
          shiny::p(class = "wb-guidance", shiny::textOutput("wb_csv_instructions")),
          shiny::fileInput("wb_new_data", "Your data (CSV, up to 10 MB)", accept = ".csv", width = "360px"),
          shiny::tags$details(shiny::tags$summary("CSV columns this loaded model needs"),
            shiny::p("The loaded .rds model determines this list: these are the input columns it used during training. Your CSV must contain those same column names and compatible values. The model reuses its saved preprocessing. To measure accuracy, also include its target column with the known correct answers; for production predictions, answers are not needed."),
            shiny::textOutput("wb_prediction_columns")),
          shiny::conditionalPanel("input.wb_csv_purpose === 'predict'", shiny::tags$details(shiny::tags$summary("Optional: try an unlabeled customer example"),
            shiny::p("These example rows have no correct Segmentation labels, so they demonstrate prediction only, not accuracy. They may match inputs stored in a compatible customer model."),
            shiny::actionButton("wb_example_predict", "Use packaged customer rows"),
            shiny::downloadButton("wb_prediction_example_download", "Download example input CSV"),
            wb_vignette_link("wb_prediction_vignettes", "Packaged-data and prediction guide"))),
          shiny::div(class = "wb-result-area wb-csv-performance",
          shiny::h3("3. Model performance / predictions"),
          shiny::actionButton("wb_predict", "Evaluate model", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true"),
          shiny::div(`aria-live` = "polite", shiny::textOutput("wb_prediction_status")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_model_error")),
          shiny::conditionalPanel("output.wb_has_predictions === 'yes'",
            shiny::div(class = "wb-result-area", shiny::h4(shiny::textOutput("wb_csv_result_title")),
            shiny::h3(shiny::textOutput("wb_csv_primary_score")),
            shiny::textOutput("wb_prediction_accuracy_note"),
            shiny::textOutput("wb_prediction_result_info"),
            shiny::plotOutput("wb_new_chart", height = "300px"),
            shiny::div(class = "scroll-table", shiny::tableOutput("wb_new_preview")),
            shiny::conditionalPanel("output.wb_csv_has_metrics === 'yes'", shiny::tags$details(shiny::tags$summary("Additional evaluation metrics"), shiny::tableOutput("wb_prediction_accuracy"))),
            shiny::tags$details(class = "wb-overlap-report", shiny::tags$summary("Optional data checks: matching inputs (not accuracy)"),
              shiny::p("These are counts of identical inputs, not scores. A zero means no matching rows, not 0% accuracy."),
              shiny::textOutput("wb_prediction_overlap_note"), shiny::tableOutput("wb_prediction_overlap_table")),
            shiny::div(class = "wb-export-right", wb_download_info("wb_new_download", shiny::downloadButton("wb_new_download", "Download results (CSV)")))))),
          wb_code_panel("wb_model_code", "R example for prediction")),
        shiny::tabPanel("Save & export", value = "export",
          wb_heading("Save a copy of this trained model", "Export a copy of the selected model, or its training recipe. A recipe is code, not trained weights."),
          shiny::p("You do not need to download a model you just uploaded. These optional exports download the selected model or its recipe to your computer. Exporting a copy does not change either workspace."),
          shiny::uiOutput("wb_export_filename_ui"),
          shiny::p(class = "help-block", "A filename is suggested from the active model. Edit it if you wish; .rds is added automatically. Your browser controls where the file is saved."),
          wb_download_button("wb_save", "Download model bundle (.rds)"),
          shiny::br(), shiny::br(), shiny::uiOutput("wb_completed_recipe_button"),
          wb_code_panel("wb_completed_code", "Equivalent R code — completed model"),
          shiny::actionButton("wb_preview_bundle", "Preview bundle contents")),
        shiny::tabPanel("Train further — optional", value = "continue",
          shiny::conditionalPanel("output.wb_has_saved_model !== 'yes'",
            shiny::p("Load or restore a trained model first. This optional section is not needed for evaluation or prediction.")),
          shiny::conditionalPanel("output.wb_has_saved_model === 'yes'",
            shiny::h3("This model is already trained"),
            shiny::p("You can evaluate or predict now without more training. Only use this section if you want to train the selected model further from its saved weights."),
            shiny::textOutput("wb_warm_start_summary"),
            wb_field("wb_more_epochs", "Additional epochs", "Number of extra epochs to run when you click Start additional training. Starts from the selected saved weights, keeps the original data/settings and restarts optimizer schedules. It does not repeat the previous epochs.",
              shiny::numericInput("wb_more_epochs", NULL, 10, min = 1, max = 5000, step = 1, width = "110px")),
            shiny::div(class = "wb-epoch-shortcuts", `aria-label` = "Set additional epochs",
              shiny::actionButton("wb_epochs_10", "10"), shiny::actionButton("wb_epochs_100", "100"),
              shiny::actionButton("wb_epochs_1000", "1000"), shiny::actionButton("wb_epochs_clear", "Clear")),
            shiny::p(class = "help-block", "Shortcuts only set the number; Clear empties it. Enter 1–5000 extra epochs before starting. For example, 10 runs 10 more epochs from the saved weights."),
            shiny::uiOutput("wb_saved_checkpoint_ui"),
            shiny::actionButton("wb_continue", "Start additional training", disabled = "disabled", `aria-disabled` = "true"),
            shiny::tags$details(id = "wb_saved_fit_panel", open = "open",
              shiny::tags$summary("Additional training — progress and results"),
              shiny::textOutput("wb_saved_fit_status"),
              shiny::uiOutput("wb_saved_fit_progress"),
              shiny::div(class = "wb-saved-cancel", shiny::actionButton("wb_saved_cancel", "Cancel additional training", disabled = "disabled")),
              shiny::div(class = "error", role = "alert", shiny::textOutput("wb_saved_fit_error")),
              shiny::textOutput("wb_saved_fit_checkpoint"),
              shiny::div(class = "wb-loss-chart",
                shiny::plotOutput("wb_saved_fit_loss", height = "280px",
                  hover = shiny::hoverOpts("wb_saved_loss_hover", delay = 100, delayType = "throttle", nullOutside = TRUE)),
                shiny::uiOutput("wb_saved_loss_marker")),
              shiny::textOutput("wb_saved_loss_inspection"),
              shiny::tags$details(shiny::tags$summary("Additional training loss values"),
                wb_saved_pager("wb_saved_values"), shiny::tableOutput("wb_saved_fit_values")),
              shiny::tags$details(shiny::tags$summary("Additional training log"),
                shiny::p("25 lines per page from the most recent 2,000 retained log lines."),
                wb_saved_pager("wb_saved_log"), shiny::verbatimTextOutput("wb_saved_fit_log")),
              shiny::h4("New copy: saved-split evaluation"),
              shiny::p("These scores use the original saved training/validation/test splits after this extra fit. Reusing the test set to choose further training weakens its value as an independent final assessment."),
              shiny::tableOutput("wb_saved_fit_evaluation"),
              shiny::actionButton("wb_saved_use_fit", "Use this newly trained copy", disabled = "disabled")),
            shiny::p("This button trains a copy here, using the selected saved model’s weights, original data and settings. Your original model and the Train a new model workspace stay unchanged. The progress and results above belong to this extra fit; when complete, use the new copy for evaluation or export."))))))


  list(
    shiny::tabPanel("1. Data", value = "data",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Start with Data", "Choose a labeled dataset or upload a CSV with known target values. Shiny divides this one dataset into training, validation and test sets on Features. Each row is an observation and each column is a variable. Loading data does not train a model. Choose the target and predictors next, on Features."),
          shiny::selectInput("wb_example", "Packaged training example", wb_example_choices(), selected = "iris", selectize = FALSE),
          shiny::div(class = "wb-guidance", shiny::textOutput("wb_example_note")),
          shiny::actionButton("wb_demo", "Reload selected example"),
          shiny::downloadButton("wb_example_download", "Download example CSV"),
          wb_vignette_link("wb_data_vignettes", "Packaged-data guide in Vignettes"),
          shiny::tags$a(href = "https://github.com/MatHatter/DDESONN/tree/main/inst/extdata", target = "_blank", rel = "noopener noreferrer", "Browse dataset source on GitHub"),
          shiny::fileInput("wb_data", "Or upload training data (CSV, up to 10 MB)", accept = ".csv"),
          shiny::p(class = "wb-guidance", "Start with Data. Inspect the rows, then choose the target and predictors in Features."),
          shiny::div(class = "wb-action-right", shiny::actionButton("wb_next_features", "Next: choose features", class = "btn-primary"))),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom",
            wb_heading("Inspect your dataset", "Check the column names, values and overall structure before selecting features. Choosing a packaged example loads it immediately. Browse first, last or specific rows and choose columns. Inspection does not change training data."),
            shiny::textOutput("wb_data_source"),
            shiny::p(class = "wb-guidance", "Optional discovery only: browse rows, descriptive statistics and box plots to understand your dataset. Profiling does not clean, scale or otherwise change the data, and it is not required before training. Choose and apply any needed preprocessing on Features."),
            shiny::fluidRow(
              shiny::column(4, shiny::selectInput("wb_inspect_view", "Rows to view", c("First rows (head)" = "head", "Last rows (tail)" = "tail", "From row number" = "range"), selectize = FALSE)),
              shiny::column(4, shiny::numericInput("wb_inspect_n", "Number of rows (1–100)", 5, min = 1, max = 100)),
              shiny::column(4, shiny::numericInput("wb_inspect_start", "Starting row (From row number)", 1, min = 1))),
            shiny::checkboxInput("wb_inspect_all", "Show all columns", value = TRUE),
            shiny::conditionalPanel("input.wb_inspect_all === false",
              shiny::uiOutput("wb_inspect_columns_ui")),
            shiny::textOutput("wb_inspect_range"),
            shiny::div(class = "scroll-table", shiny::tableOutput("wb_preview")),
            wb_code_panel("wb_data_code", "R code — load data and inspect these rows"),
            shiny::tags$details(open = "open", shiny::tags$summary("Dataset profiling — descriptive summary"),
              shiny::p("All rows and columns, independent of the row preview. Missing counts include blank text. Numeric summaries use finite values; NA means not applicable or unavailable. No data is removed or imputed."),
              shiny::div(class = "scroll-table", shiny::tableOutput("wb_data_profile"))),
            shiny::tags$details(shiny::tags$summary("Box-and-whisker plot"),
              shiny::uiOutput("wb_box_column_ui"),
              shiny::p("Uses all finite values in the chosen numeric column. The box shows the middle 50%, the line is the median, and whiskers extend to values within 1.5 × IQR. Points beyond are potential outliers; they are kept in your data."),
              shiny::plotOutput("wb_data_box", height = "280px"))),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_data_error"))))),
    shiny::tabPanel("2. Features", value = "features",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Choose what the model learns", "The target is the answer you want to predict. Predictors supply the evidence. Task type determines the kind of answer the model produces and the metrics used to assess it."),
          shiny::div(class = "wb-guidance", shiny::textOutput("wb_features_source")),
          shiny::uiOutput("wb_columns"),
          shiny::tags$details(class = "wb-settings-section", open = "open", shiny::tags$summary("Preprocessing — Auto / Manual"),
            wb_pp_field("wb_pp_mode", "Overall preprocessing", shiny::radioButtons("wb_pp_mode", NULL, c("Auto" = "auto", "Manual" = "manual"), selected = "auto", inline = TRUE)),
            shiny::p("Preprocessing prepares predictors for the model, for binary classification, multiclass classification and regression alike. Use category encoding for text/category inputs and missing-value treatment for incomplete predictors. Exclude identifiers or answer-derived columns when they should not be learned."),
            shiny::p("For example, customer segmentation has category inputs that need encoding and missing predictors may need filling. Complete numeric examples such as iris or heart failure do not need category encoding or missing-value filling when their selected predictors are already complete and finite. Scaling can still help when numeric ranges differ."),
            shiny::p("Auto proposes rules locally; a step makes no changes when no treatment is needed. Manual requires complete numeric inputs unless you choose treatments. Standardization is a separate switch below, enabled by default in both modes. Review the plan, then click Prepare data: the split happens first, rules are fitted on training rows only, and validation, test and future predictions reuse them. Target labels are not imputed."),
            wb_pp_field("wb_pp_numeric", "Missing numeric values / infinities", shiny::selectInput("wb_pp_numeric", NULL, c("Use Auto default (see info)" = "auto", "Training median" = "median", "Training mean" = "mean", "Fill with zero" = "zero", "Require complete values" = "error"))),
            wb_pp_field("wb_pp_encoding", "Categorical encoding", shiny::selectInput("wb_pp_encoding", NULL, c("Use Auto default (see info)" = "auto", "One-hot: separate 0/1 inputs" = "onehot", "Ordinal: use an explicit order per column" = "ordinal", "Require numeric predictors" = "error"))),
            wb_pp_field("wb_pp_missing", "Missing categorical values", shiny::selectInput("wb_pp_missing", NULL, c("Use Auto default (see info)" = "auto", "Separate Missing level" = "level", "Most frequent training category" = "mode", "Require complete categories" = "error"))),
            wb_pp_field("wb_pp_ids", "Likely identifier columns", shiny::selectInput("wb_pp_ids", NULL, c("Use Auto default (see info)" = "auto", "Exclude names ID, *_id, id_*" = "exclude", "Keep selected columns" = "keep"))),
            shiny::tags$details(shiny::tags$summary("Override one column"),
              shiny::uiOutput("wb_pp_column_ui"),
              wb_pp_field("wb_pp_role", "Column role", shiny::selectInput("wb_pp_role", NULL, c("Use identifier rule above" = "auto", "Keep" = "keep", "Exclude" = "exclude"))),
              wb_pp_field("wb_pp_column_numeric", "Numeric missing-value override", shiny::selectInput("wb_pp_column_numeric", NULL, c("Use numeric missing-value rule above" = "auto", "Median" = "median", "Mean" = "mean", "Zero" = "zero", "Require complete" = "error"))),
              wb_pp_field("wb_pp_column_encoding", "Category encoding override", shiny::selectInput("wb_pp_column_encoding", NULL, c("Use categorical encoding rule above" = "auto", "One-hot" = "onehot", "Ordinal" = "ordinal", "Require numeric" = "error"))),
              shiny::textInput("wb_pp_order", "Ordinal categories, low to high (separate with |)", value = ""),
              shiny::p("For ordinal encoding, enter categories separated by |, for example Low|Average|High. Known levels become 1..K; missing becomes 0 and unseen becomes -1. One-hot categories have no numerical order."),
              wb_pp_field("wb_pp_column_missing", "Categorical missing-value override", shiny::selectInput("wb_pp_column_missing", NULL, c("Use categorical missing-value rule above" = "auto", "Missing level" = "level", "Most frequent" = "mode", "Require complete" = "error"))),
              shiny::actionButton("wb_pp_apply", "Apply column override"),
              shiny::actionButton("wb_pp_reset", "Clear all column overrides"),
              shiny::textOutput("wb_pp_override_status")),
            shiny::p("One-hot uses training categories plus reserved Missing/Unknown inputs. A new category never changes the saved input layout. Review suggested exclusions: an identifier name is a hint, not proof.")),
          wb_field("wb_mode", "Task", "Regression predicts a number, such as price. Binary classification chooses between two classes. Multiclass classification chooses among several classes. Choose a task that matches your target.",
            shiny::selectInput("wb_mode", NULL, c("Multiclass classification" = "multiclass", "Binary classification" = "binary", "Regression" = "regression"))),
          wb_field("wb_split", "Train / validation / test split", "Training rows teach the model. Validation rows help you assess development choices. Test rows are held aside for a final assessment. The fractions below allocate training and validation; remaining rows become test data.",
            shiny::selectInput("wb_split", NULL, c("Random (binary: keep identical inputs together)" = "random", "Ordered (preserve row order)" = "ordered"))),
          wb_field("wb_train_fraction", "Training fraction", "Fraction of rows allocated to training. Binary random splits allocate whole groups of identical retained predictors with approximate class balance. Requested fractions apply to groups, so row counts can vary. Other classification tasks split within each class.",
            shiny::numericInput("wb_train_fraction", NULL, .6, min = .1, max = .9, step = .05)),
          wb_field("wb_val_fraction", "Validation fraction", "Fraction of rows held for validation; remaining rows become Test. Test is not used to select checkpoints.",
            shiny::numericInput("wb_val_fraction", NULL, .2, min = .05, max = .4, step = .05)),
          shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Reproducibility"),
          wb_field("wb_seed", "Data-split seed", "Controls repeatable random row splitting. Ordered splits keep row order. Basic reuses this value for its one training run; Custom uses the separate Training-run seed values on Build & Train. A value of 42 is one reproducibility seed, not 42 runs.",
            shiny::numericInput("wb_seed", NULL, 42, min = 0, max = 2147483646, step = 1)),
          shiny::p(class = "wb-guidance", "42 is a seed value, not a run count. Basic reuses it for one training run. Custom has a separate list of training-run seeds on Build & Train.")),
          wb_switch("wb_scale", "Scale inputs using training means and SDs",
            "Subtracts each predictor's training average and divides by its training standard deviation (SD, a measure of spread). This puts inputs on comparable scales and can improve training when ranges differ, regardless of task type. It is preprocessing too, separate from category encoding and missing-value filling; Auto/Manual does not turn it on or off. Validation, test and new rows reuse those same training values; the target is unchanged.", TRUE,
            api = paste("center <- colMeans(x_train)", "spread <- apply(x_train, 2, sd)",
              "spread[!is.finite(spread) | spread == 0] <- 1",
              "x_train_scaled <- sweep(sweep(x_train, 2, center, '-'), 2, spread, '/')", sep = "\n")),
          shiny::div(class = "wb-action-right", shiny::actionButton("wb_prepare", "Prepare data", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true"))),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom",
            wb_heading("Prepare for training", "Prepare data creates the three splits, then fits encoding, missing-value rules and scaling using training rows only. Saved rules are reused for validation, test and new data."),
            shiny::p(class = "wb-guidance", "Next, choose the target and predictors. Click Prepare data to continue to Build & Train."),
            shiny::tags$details(open = "open", shiny::tags$summary("Suggested preprocessing plan"),
              shiny::p("This is a proposed plan for selected predictors. Values, category levels and scaling are learned only after splitting. Review it, then click Prepare data. The full source dataset is retained. Scroll horizontally to see all rules."),
              shiny::div(class = "scroll-table wb-preprocessing-table", shiny::tableOutput("wb_pp_plan")),
              wb_code_panel("wb_pp_plan_code", "R code — preprocessing choices")),
            shiny::textOutput("wb_prepared"),
            shiny::tags$details(shiny::tags$summary("Fitted preprocessing — training rows only"),
              shiny::p("These saved replacements and category rules are reused for all other rows. At most 20 category labels are shown per column; the complete rules are retained in the model bundle and R code."),
              shiny::div(class = "scroll-table wb-preprocessing-table", shiny::tableOutput("wb_pp_fitted"))),
            shiny::p("Target + predictors determine what the model learns from. Task determines its output and evaluation metrics. Changing any selection here requires Prepare data again."),
            wb_code_panel("wb_features_code", "Equivalent R code — prepared split and scaling")),
          shiny::div(class = "panel-body-custom",
            wb_heading("About your split", "Binary random splits keep identical selected predictor inputs together, with approximate class balance; group sizes can change row fractions. Binary ordered splits reject repeated inputs crossing boundaries. Other tasks retain row splitting. Patient identities and related subjects require external grouping."),
            shiny::p("Binary splits are checked again after preprocessing; preparation stops if transformed inputs overlap. Rules are fitted on training rows only. Identical inputs do not establish patient identity; related subjects still require external grouping."),
            wb_vignette_link("wb_features_vignettes", "Data preparation and limitations in Vignettes")),
          shiny::div(class = "error", role = "alert", shiny::textOutput("wb_error"))))),
    shiny::tabPanel("3. Build & Train", value = "train",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Build & Train", "Configure the model and how it learns, then start a new training run on your prepared data. Architecture controls model capacity; optimizer, learning rate and epochs control how weights are updated."),
          wb_field("wb_workflow", "Workflow", "Basic keeps the streamlined single-model workflow. Custom exposes DDESONN's member, seed and MAIN/TEMP orchestration plus optional training controls. Both use the same native R engine. Keras is a separate general-purpose deep-learning API; this workbench does not use it or imply a performance advantage.",
            shiny::div(class = "wb-workflow-pills", shiny::radioButtons("wb_workflow", NULL, c("Basic" = "basic", "Custom" = "custom"), inline = TRUE))),
          shiny::conditionalPanel("input.wb_workflow === 'custom'",
            shiny::p(class = "wb-custom-brief", "DDESONN is a native R neural-network framework with transparent weight updates, optional self-organization and evolving ensembles. Custom opens its member, seed and MAIN/TEMP controls; Basic uses the same engine with a streamlined setup.")),
          shiny::tags$script(shiny::HTML("$(document).on('shiny:connected', function() { var m; try { m = localStorage.getItem('DDESONN.workflow'); } catch(e) {} if (m === 'custom' || m === 'basic') { Shiny.setInputValue('wb_remembered_workflow', m); } }); $(document).on('change', 'input[name=wb_workflow]', function() { try { localStorage.setItem('DDESONN.workflow', this.value); } catch(e) {} });")),
          wb_field("wb_hidden", "Hidden-layer sizes", "Hidden layers transform the predictors before the output. Enter units separated by commas, such as 16,8 for two layers. Leave blank for no hidden layer. Larger models can take longer and may overfit.",
            shiny::textInput("wb_hidden", NULL, "16,8")),
          wb_field("wb_epochs", "Epochs per model fit", "An epoch is one pass through the training data. Each model fit completes this many passes. With Single model, two training-run seeds and 100 epochs means run 1 trains for 100 epochs, then run 2 trains for 100: 200 total. Ensemble members and MAIN/TEMP fits each receive this epoch budget. Both checkpoint modes run all requested epochs.",
            shiny::numericInput("wb_epochs", NULL, 50, min = 1, max = 5000, step = 1)),
          wb_field("wb_optimizer", "Optimizer", "The optimizer is the rule used to update model weights from prediction errors. Different rules can learn at different speeds; compare them using validation data.",
            shiny::selectInput("wb_optimizer", NULL, wb_optimizer_choices(), selected = "adagrad")),
          shiny::div(class = "wb-field", role = "group", `aria-labelledby` = "wb-gradient-reduction-label",
            shiny::div(class = "wb-control-title",
              shiny::tags$span(id = "wb-gradient-reduction-label", "Gradient reduction"),
              wb_help("Gradient reduction",
                paste("Gradient reduction combines per-example gradients into the update signal the optimizer uses for weights and biases.",
                  "DDESONN defaults to Sum for both. Keras binary cross-entropy defaults to averaging over the batch.",
                  "In the clean 100-seed binary study, DDESONN averaged 91.36% test accuracy with Sum for both and 90.39% with Average for both.",
                  "Its advantage over Keras under the matching reduction was 3.88 and 2.77 percentage points, respectively.",
                  "That study used one dataset split. Mixed settings and other datasets still need testing."),
                show_code = FALSE)),
            shiny::div(class = "wb-workflow-pills wb-gradient-pills",
              shiny::radioButtons("wb_weight_reduction", "Weight gradients",
                c(Sum = "sum", Average = "mean"), selected = "sum", inline = TRUE),
              shiny::radioButtons("wb_bias_reduction", "Bias gradients",
                c(Sum = "sum", Average = "mean"), selected = "sum", inline = TRUE))),
          wb_field("wb_lr", "Learning rate", "The learning rate sets the size of each weight update. Too large can make training unstable; too small can make learning slow. It interacts with the optimizer and data scaling. Basic keeps it fixed. Custom also offers exponential decay in Learning-rate scheduling.",
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
          shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Regularization"),
            wb_switch("wb_reg_enabled", "Enable regularization", "Optional penalties: L1, L2, mixed L1_L2, Group_Lasso and Max_Norm. Off by default; turn on to choose a penalty and strength.",
              api = "reg_type <- 'L2'\nlambda <- 0.0001"),
            shiny::conditionalPanel("input.wb_reg_enabled",
              wb_field("wb_reg", "Regularization", "L1 encourages sparse parameters; L2 shrinks their magnitudes. L1_L2 mixes L1 and L2 equally, Group_Lasso encourages group sparsity, and Max_Norm is a penalty toward unit-norm parameters, not a hard constraint. These options are available in both Basic and Custom.",
                shiny::selectInput("wb_reg", NULL, c("L2", "L1", "L1_L2", "Group_Lasso", "Max_Norm"))),
              wb_field("wb_lambda", "Regularization strength", "Controls the selected penalty strength. A strength of zero has no effect.",
                shiny::numericInput("wb_lambda", NULL, .0001, min = 0, step = .0001)))),
          shiny::conditionalPanel("input.wb_workflow !== 'custom'", shiny::div(class = "wb-workload-explanation", shiny::textOutput("wb_basic_workload"))),
          shiny::conditionalPanel("input.wb_workflow === 'custom'",
            shiny::div(class = "wb-custom-divider", shiny::h4("Advanced Custom settings")),
            shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Training runs and model structure"),
              wb_field("wb_structure", "Run structure", "Single model trains one member. Multiple members inside one model trains several members without MAIN/TEMP orchestration. MAIN is the incumbent or champion ensemble. TEMP trains challengers, also called candidates or competitors, for the existing replacement process. These are names for the same MAIN/TEMP roles, not extra run types.",
                shiny::selectInput("wb_structure", NULL, c("Single model" = "single", "Multiple members inside one model" = "members", "Main ensemble (MAIN)" = "main", "Main ensemble with temporary iterations (MAIN + TEMP)" = "temp"))),
              shiny::conditionalPanel("input.wb_structure !== 'single'",
              wb_field("wb_members", "Members per model", "Each member is a separately initialized network inside a model. This count applies to MAIN and to each TEMP candidate model. It is distinct from the number of training seeds.",
                shiny::numericInput("wb_members", NULL, 3, min = 1, max = 100))),
              shiny::conditionalPanel("input.wb_structure === 'temp'",
              wb_field("wb_temp", "Temporary iterations", "Each iteration trains a TEMP challenger/candidate ensemble. The existing backend selects a candidate and replaces a MAIN incumbent slot using its configured metric. A warm start fits the saved MAIN members without repeating this search.",
                shiny::numericInput("wb_temp", NULL, 1, min = 1, max = 100))),
              wb_field("wb_seeds", "Training-run seed values", "Each comma-separated seed value starts one complete training run. 42 means one run; 42,43 means two runs. With Single model and 100 epochs, the first run completes all 100 epochs before the second run starts its 100 epochs. Every run reuses the same prepared rows and scaling; these seeds never resplit the data.",
                shiny::textInput("wb_seeds", NULL, "42")),
              shiny::p(class = "wb-guidance", "42 = 1 training run. 42,43 = 2 training runs. Example for Single model: 100 epochs means run 1 trains for 100, then run 2 trains for 100 (200 total). These values change training randomness, not the prepared data split.")),
            shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Architecture and initialization"),
              wb_field("wb_activation", "Hidden activation", "ReLU keeps positive inputs; leaky ReLU also keeps a small negative response. Tanh is bounded around zero, sigmoid maps to 0–1, and SELU is a scaled exponential linear unit. The selected function applies to hidden layers. Output activation follows the task: sigmoid, softmax or identity. With no hidden layers this choice has no effect.",
                shiny::selectInput("wb_activation", NULL, c("relu", "tanh", "sigmoid", "selu", "leaky_relu"))),
              wb_field("wb_init", "Weight initialization", "He scales initial weights by input width and is commonly paired with ReLU; LeCun uses a smaller input-width scale. Xavier and Glorot use input/output widths, with Glorot uniform drawing from a uniform range. Orthogonal uses a QR-based construction; variance scaling uses the engine's smaller width-based scale. These initialize new members; warm starts retain saved tensors. Predictor standardization is separate on Features.",
                shiny::selectInput("wb_init", NULL, c("he", "xavier", "lecun", "orthogonal", "variance_scaling", "glorot_uniform"))),
              wb_switch("wb_dropout_enabled", "Enable dropout", "Randomly drops units during training. Off uses zero dropout; inference uses the full network."),
              shiny::conditionalPanel("input.wb_dropout_enabled",
              wb_field("wb_dropout", "Dropout rate", "Fraction dropped during training, from zero to below one. Applies to hidden layers, or to inputs when there is no hidden layer. Prediction does not randomly drop units.",
                shiny::numericInput("wb_dropout", NULL, .1, min = 0, max = .99, step = .05)))),
            shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Learning-rate scheduling"),
              wb_switch("wb_schedule_enabled", "Enable learning-rate scheduling", "Off keeps the Learning rate fixed. On reveals the supported schedule and decay controls. Warm starts restart the schedule."),
              shiny::conditionalPanel("input.wb_schedule_enabled",
              wb_field("wb_schedule", "Schedule", "Exponential decay uses max(minimum rate, initial rate times multiplier raised to floor(epoch / interval)). The Learning rate above supplies the initial rate. This is the existing engine formula, including epoch-one behavior.",
                shiny::selectInput("wb_schedule", NULL, c("Exponential decay" = "exponential"))),
              wb_field("wb_minimum", "Minimum rate", "Positive floor for the scheduled learning rate. It cannot exceed the initial Learning rate. Scheduling off sets this floor equal to the fixed rate.",
                shiny::numericInput("wb_minimum", NULL, .00001, min = .00000001)),
              wb_field("wb_decay", "Decay multiplier", "Fraction of the rate retained at each decay interval. A multiplier of 0.5 halves it; 1 retains the rate. The minimum floor still applies.",
                shiny::numericInput("wb_decay", NULL, .5, min = .000001, max = 1, step = .05)),
              wb_field("wb_interval", "Decay interval (epochs)", "Positive epoch interval used by floor(epoch / interval). Each new member fit and warm start restarts its epoch schedule.",
                shiny::numericInput("wb_interval", NULL, 20, min = 1, max = 5000)))),
            shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Ensemble prediction combination"),
              wb_field("wb_fusion", "Combine members within each training seed", "Combines member predictions within one seed before combining seeds. Mean averages probabilities or regression predictions. Weighted mean uses normalized validation accuracy or inverse validation MSE, never test labels; zero total accuracy uses equal weights. Binary members vote positive at probability >= 0.5; a majority tie is negative. Multiclass uses plurality, with ties choosing the first saved class. No quorum is exposed. Voting returns decisions or vote proportions, not calibrated probabilities. Multiclass medians are renormalized; an all-zero row falls back to the mean. Saved weights are reused for new data.",
                shiny::selectInput("wb_fusion", NULL, c("Average probabilities / predictions" = "mean", "Validation-weighted average" = "weighted", "Median" = "median", "Majority voting" = "majority"))),
              wb_field("wb_seed_aggregate", "Combine across training seeds", "Combines the already-combined prediction from each training seed, by mean or median. Each seed contributes equally at this stage. This is separate from weighting members within a seed. Binary classification then uses >= 0.5; multiclass chooses the first maximum.",
                shiny::selectInput("wb_seed_aggregate", NULL, c("Average" = "mean", "Median" = "median")))),
            shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Diagnostics"),
              wb_switch("wb_diagnostics", "Detailed training diagnostics",
                paste("Show tensor, gradient, optimizer and training summaries in the Training log.",
                  "For specific methods and epochs, open Equivalent R code and uncomment a trace example to run in R.")))),
          shiny::conditionalPanel("input.wb_workflow === 'custom'", shiny::div(class = "wb-workload-explanation", shiny::textOutput("wb_workload"))),
          wb_vignette_link("wb_training_vignettes", "Training and plotting guides in Vignettes"),
          shiny::div(class = "wb-training-actions",
            shiny::actionButton("wb_cancel", "Cancel training", disabled = "disabled", `aria-disabled` = "true"),
            shiny::actionButton("wb_start", "Train new model", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true"))),
        shiny::mainPanel(
          shiny::div(class = "wb-guidance", shiny::textOutput("wb_training_context")),
          shiny::uiOutput("wb_split_overlap"),
          shiny::conditionalPanel("output.wb_running_flag !== 'yes' || output.wb_error_flag === 'yes'",
            shiny::div(class = "wb-latest-attempt wb-attempt-above", role = "status",
              shiny::conditionalPanel("output.wb_running_flag !== 'yes'",
                shiny::strong(shiny::textOutput("wb_attempt_title", inline = TRUE)), shiny::textOutput("wb_status")),
              shiny::uiOutput("wb_train_error"))),
          shiny::div(class = "panel-body-custom", id = "wb_training_panel",
            wb_heading("Training progress", "Training runs separately so you can still use the interface. The curve appears when epochs are logged. Hover to inspect an epoch and loss without selecting weights; the table provides the same values. Best-validation selection uses validation rather than this training curve. Lower training loss alone does not establish held-out performance."),
            shiny::conditionalPanel("output.wb_running_flag === 'yes'",
              shiny::div(class = "wb-latest-attempt wb-attempt-running", role = "status",
                shiny::strong("Latest attempt"), shiny::textOutput("wb_running_status"), shiny::uiOutput("wb_progress"))),
            shiny::conditionalPanel("output.wb_chart_ready === 'yes'",
              shiny::textOutput("wb_chart_model"),
              shiny::div(class = "wb-loss-chart",
                shiny::plotOutput("wb_loss", height = "280px",
                  hover = shiny::hoverOpts("wb_loss_hover", delay = 100, delayType = "throttle", nullOutside = TRUE)),
                shiny::uiOutput("wb_loss_marker")),
              shiny::div(`aria-live` = "polite", shiny::textOutput("wb_loss_inspection")),
              shiny::div(class = "wb-guidance", shiny::textOutput("wb_loss_explanation")),
              shiny::div(class = "wb-chart-exports",
                shiny::downloadButton("wb_loss_png", "Download PNG"),
                shiny::downloadButton("wb_loss_svg", "Download SVG"),
                shiny::downloadButton("wb_loss_csv", "Download loss data"),
                shiny::tags$button(id = "wb_copy_loss", type = "button", class = "btn btn-default", "Copy chart image")),
              shiny::div(id = "wb_copy_loss_status", role = "status", `aria-live` = "polite"),
              shiny::tags$details(shiny::tags$summary("Training loss values (epoch and loss)"),
                shiny::div(class = "wb-loss-values scroll-table", shiny::tableOutput("wb_loss_values")))),
            shiny::conditionalPanel("output.wb_has_model === 'yes' && output.wb_chart_ready !== 'yes'",
              shiny::p("This saved model has no recorded training-loss history.")),
            shiny::conditionalPanel("output.wb_has_logs === 'yes'",
              shiny::tags$details(shiny::tags$summary("Training log (most recent 2,000 lines)"),
                shiny::tags$pre(style = "max-height:320px;overflow:auto;", shiny::textOutput("wb_training_log"))))),
          shiny::conditionalPanel("output.wb_chart_ready === 'yes'",
          shiny::div(class = "context", "Training runs for all requested epochs. Choose best-validation or final-epoch weights using the dropdown. Automatic early stopping is not implemented. Self-organization is optional; Custom exposes dropout. Internal batch normalization is off. Cancelled or failed runs retain any previous completed model.")),
          shiny::div(class = "panel-body-custom",
            wb_heading("Local completed model", "Select a successful training run below. Uploaded models live separately in Use a saved model. Its saved settings, chart, results and downloads follow your selection. Failed or cancelled attempts are kept separately in Attempt history. Changing controls affects the next new model, not this completed model."),
            shiny::conditionalPanel("output.wb_has_model === 'yes'", shiny::p("Select a completed run to show its settings, chart and results. Completed models and Recent runs are in this session only; download the .rds to keep it after the session ends.")),
            shiny::uiOutput("wb_run_choices"),
            shiny::textOutput("wb_settings_origin"),
            shiny::p(class = "wb-guidance", shiny::textOutput("wb_training_success")),
            wb_vignette_link("wb_community_vignettes", "Share results or contribute"),
            shiny::textOutput("wb_model_info"),
            shiny::textOutput("wb_checkpoint_summary"),
            wb_code_panel("wb_training_code", "Equivalent R code — next new model"),
            shiny::uiOutput("wb_next_recipe_button"),
            shiny::br(), shiny::br(), shiny::actionButton("wb_review_results", "View selected results", disabled = "disabled", `aria-disabled` = "true"),
            shiny::uiOutput("wb_selected_save"),
            shiny::tags$details(class = "wb-attempt-history", shiny::tags$summary("Attempt history"),
              shiny::uiOutput("wb_attempt_history")))))),
    shiny::tabPanel("4. Model / Results", value = "results",
      shiny::p(class = "wb-step-intro", "Finally, review performance and outputs, save the model, or use it for new predictions. Results reflect the saved choices from the earlier steps."),
      shiny::tabsetPanel(id = "wb_results_tabs", performance)),
    models, recent
  )
}

wb_server <- function(input, output, session, helper) {
  raw <- shiny::reactiveVal(datasets::iris)
  data_revision <- shiny::reactiveVal(0L)
  data_source <- shiny::reactiveVal("Included iris example · 150 rows")
  data_origin <- shiny::reactiveVal(list(example = "iris"))
  dataset_preset <- shiny::reactiveVal(wb_example("iris"))
  pp_overrides <- shiny::reactiveVal(list())
  input_default <- function(value, default) if (is.null(value)) default else value
  pp_config <- shiny::reactive(wb_pp_config(
    input_default(input$wb_pp_mode, "auto"), input_default(input$wb_pp_numeric, "auto"),
    input_default(input$wb_pp_encoding, "auto"), input_default(input$wb_pp_missing, "auto"),
    input_default(input$wb_pp_ids, "auto"), pp_overrides()))
  shiny::observeEvent(input$wb_pp_mode, {
    auto <- identical(input$wb_pp_mode, "auto")
    choices <- list(
      wb_pp_numeric = c(stats::setNames("auto", if (auto) "Use Auto default: training median" else "Use Manual default: require complete"), "Training median" = "median", "Training mean" = "mean", "Fill with zero" = "zero", "Require complete values" = "error"),
      wb_pp_encoding = c(stats::setNames("auto", if (auto) "Use Auto default: one-hot" else "Use Manual default: require numeric"), "One-hot: separate 0/1 inputs" = "onehot", "Ordinal: explicit order per column" = "ordinal", "Require numeric predictors" = "error"),
      wb_pp_missing = c(stats::setNames("auto", if (auto) "Use Auto default: Missing level" else "Use Manual default: require complete"), "Separate Missing level" = "level", "Most frequent training category" = "mode", "Require complete categories" = "error"),
      wb_pp_ids = c(stats::setNames("auto", if (auto) "Use Auto default: exclude ID-like names" else "Use Manual default: keep selected"), "Exclude names ID, *_id, id_*" = "exclude", "Keep selected columns" = "keep"))
    for (id in names(choices)) shiny::updateSelectInput(session, id, choices = choices[[id]], selected = shiny::isolate(input_default(input[[id]], "auto")))
  })
  output$wb_pp_column_ui <- shiny::renderUI(wb_pp_field("wb_pp_column", "Predictor to override", shiny::selectInput("wb_pp_column", NULL, input$wb_features)))
  shiny::observeEvent(input$wb_pp_apply, attempt({
    busy()
    column <- input$wb_pp_column
    if (length(column) != 1 || !column %in% input$wb_features) stop("Choose a selected predictor to override.")
    overrides <- pp_overrides()
    order <- trimws(strsplit(input_default(input$wb_pp_order, ""), "|", fixed = TRUE)[[1]])
    overrides[[column]] <- list(action = input_default(input$wb_pp_role, "auto"),
      numeric = input_default(input$wb_pp_column_numeric, "auto"),
      encoding = input_default(input$wb_pp_column_encoding, "auto"),
      missing = input_default(input$wb_pp_column_missing, "auto"), order = order[nzchar(order)])
    pp_overrides(overrides)
  }))
  shiny::observeEvent(input$wb_pp_reset, pp_overrides(list()))
  shiny::observeEvent(input$wb_pp_column, {
    o <- pp_overrides()[[input$wb_pp_column]]
    shiny::updateSelectInput(session, "wb_pp_role", selected = input_default(o$action, "auto"))
    shiny::updateSelectInput(session, "wb_pp_column_numeric", selected = input_default(o$numeric, "auto"))
    shiny::updateSelectInput(session, "wb_pp_column_encoding", selected = input_default(o$encoding, "auto"))
    shiny::updateSelectInput(session, "wb_pp_column_missing", selected = input_default(o$missing, "auto"))
    shiny::updateTextInput(session, "wb_pp_order", value = paste(o$order, collapse = "|"))
  })
  output$wb_pp_override_status <- shiny::renderText(paste("Column overrides:",
    if (length(pp_overrides())) paste(names(pp_overrides()), collapse = ", ") else "none"))
  output$wb_pp_plan <- shiny::renderTable({
    features <- intersect(input$wb_features, names(raw()))
    shiny::req(length(features))
    wb_pp_plan(raw(), features, pp_config())
  })
  output$wb_pp_plan_code <- shiny::renderText(paste0(
    "# These choices are fitted after splitting, using training rows only.\n",
    "preprocessing_config <- ", wb_literal(pp_config()),
    "\n# Prepare data to see the complete, runnable encoding and imputation code."))
  prepared <- shiny::reactiveVal(NULL)
  result <- shiny::reactiveVal(NULL)
  completed_runs <- shiny::reactiveVal(list())
  selected_run <- shiny::reactiveVal(NULL)
  recent_runs <- shiny::reactiveVal(list())
  recent_counter <- shiny::reactiveVal(0L)
  recent_notice <- shiny::reactiveVal(NULL)
  recent_error <- shiny::reactiveVal(NULL)
  workspace_notice <- shiny::reactiveVal(NULL)
  restoring_purpose <- shiny::reactiveVal(NULL)
  training_success <- shiny::reactiveVal(NULL)
  remember_recent <- function(bundle, kind, run = NULL, key = NULL) {
    records <- recent_runs()
    same <- if (is.null(key)) integer() else which(vapply(records, function(record) identical(record$key, key), logical(1)))
    if (length(same)) id <- names(records)[same[1]] else {
      recent_counter(recent_counter() + 1L); id <- paste0("recent-", recent_counter())
    }
    record <- unserialize(serialize(list(id = id, kind = kind, key = key, bundle = bundle, run = run), NULL))
    records <- c(stats::setNames(list(record), id), records[setdiff(names(records), id)])
    recent_runs(utils::head(records, 10L))
  }
  selected_recent <- shiny::reactive({
    id <- input$wb_recent_id
    if (length(id) != 1L || !id %in% names(recent_runs())) return(NULL)
    recent_runs()[[id]]
  })
  saved_models <- shiny::reactiveVal(list())
  selected_saved <- shiny::reactiveVal(NULL)
  active_model <- shiny::reactive({ if (is.null(selected_saved())) NULL else saved_models()[[selected_saved()]] })
  task_flow <- shiny::reactiveVal("train")
  last_training_page <- shiny::reactiveVal("data")
  set_flow <- function(flow, page = NULL, sync_radio = TRUE) {
    if (!flow %in% c("train", "saved")) return(invisible(NULL))
    if (identical(flow, task_flow()) && is.null(page)) return(invisible(NULL))
    task_flow(flow)
    if (sync_radio) shiny::updateRadioButtons(session, "wb_task_flow", selected = flow)
    for (tab in c("data", "features", "train", "results")) {
      if (flow == "train") shiny::showTab("workspace", tab, session = session) else shiny::hideTab("workspace", tab, session = session)
    }
    if (flow == "saved") shiny::showTab("workspace", "saved", session = session) else shiny::hideTab("workspace", "saved", session = session)
    shiny::updateTabsetPanel(session, "workspace", selected = if (flow == "saved") "saved" else if (is.null(page)) last_training_page() else page)
    session$sendCustomMessage("wb-flow-theme", list(flow = flow))
  }
  shiny::observeEvent(input$wb_task_flow, set_flow(input$wb_task_flow, sync_radio = FALSE))
  shiny::observeEvent(input$workspace, {
    if (input$workspace %in% c("data", "features", "train", "results")) last_training_page(input$workspace)
  })
  shiny::observeEvent(TRUE, set_flow("train", "data"), once = TRUE)
  output$wb_flow_guide <- shiny::renderText(if (task_flow() == "saved")
    "Use a saved model: upload .rds → read its summary → choose labeled evaluation or unlabeled prediction → upload CSV → run → view results. Training tabs stay unchanged." else
    "Train a new model: Data → Features → Build & Train → Evaluate → Predict. Saved models use their own workflow.")
  code_model <- shiny::reactive(if (task_flow() == "saved") active_model() else result())
  attempts <- shiny::reactiveVal(list())
  latest_attempt <- shiny::reactiveVal(NULL)
  job <- shiny::reactiveVal(NULL)
  saved_fit <- shiny::reactiveVal(NULL)
  saved_fit_counter <- shiny::reactiveVal(0L)
  training_context <- shiny::reactiveVal(NULL)
  status <- shiny::reactiveVal("Prepare data, then train a new model.")
  error <- shiny::reactiveVal(NULL)
  model_error <- shiny::reactiveVal(NULL)
  load_error <- shiny::reactiveVal(NULL)
  training_error <- shiny::reactiveVal(NULL)
  new_predictions <- shiny::reactiveVal(NULL)
  prediction_data <- shiny::reactiveVal(NULL)
  prediction_source <- shiny::reactiveVal(NULL)
  prediction_provenance <- shiny::reactiveVal(NULL)
  prediction_report <- shiny::reactiveVal(NULL)
  history <- shiny::reactiveVal(data.frame(epoch = integer(), loss = numeric()))
  attempt <- function(code, destination = error) {
    tryCatch({ force(code); destination(NULL) }, error = function(e) destination(conditionMessage(e)))
  }
  read_job_log <- function(active) {
    if (is.null(active)) return(character())
    unlist(lapply(c(active$log, active$error), function(path)
      if (file.exists(path)) tail(readLines(path, warn = FALSE), 2000L) else character()), use.names = FALSE)
  }
  begin_attempt <- function(cfg = NULL, kind = "New training") {
    id <- as.character(length(attempts()) + 1L)
    records <- attempts()
    records[[id]] <- list(id = id, kind = kind, state = "Running", config = cfg,
      started = format(Sys.time(), tz = "UTC", usetz = TRUE), error = NULL,
      history = data.frame(epoch = integer(), loss = numeric()))
    attempts(records); latest_attempt(id); training_error(NULL)
    history(data.frame(epoch = integer(), loss = numeric()))
    id
  }
  finish_attempt <- function(state, detail = NULL) {
    id <- latest_attempt()
    if (is.null(id)) return(invisible(NULL))
    records <- attempts(); record <- records[[id]]
    record$state <- state; record$error <- detail; record$history <- history()
    if (!is.null(job())) record$training_log <- read_job_log(job())
    records[[id]] <- record; attempts(records)
    training_error(detail)
    status(paste0("Run ", id, " · ", if (state == "Cancelled") "Training cancelled" else state, " · ", record$kind,
      if (!is.null(record$config)) paste0(" · ",
        if (identical(record$config$checkpoint, "best_validation")) "Best validation checkpoint" else "Final epoch") else ""))
  }
  sync_controls <- function(cfg, mode = NULL) {
    shiny::updateTextInput(session, "wb_hidden", value = paste(cfg$hidden, collapse = ","))
    shiny::updateNumericInput(session, "wb_epochs", value = cfg$epochs)
    shiny::updateNumericInput(session, "wb_lr", value = cfg$lr)
    shiny::updateSelectInput(session, "wb_optimizer", choices = wb_optimizer_choices(), selected = cfg$optimizer)
    reduction <- DDESONN:::.ddesonn_gradient_choices(wb_gradient_mode(cfg))
    shiny::updateRadioButtons(session, "wb_weight_reduction", selected = reduction[["weights"]])
    shiny::updateRadioButtons(session, "wb_bias_reduction", selected = reduction[["biases"]])
    shiny::updateSelectInput(session, "wb_checkpoint", selected = cfg$checkpoint)
    shiny::updateSelectInput(session, "wb_reg", choices = c("L2", "L1", "L1_L2", "Group_Lasso", "Max_Norm"), selected = if (identical(cfg$reg, "none")) "L2" else cfg$reg)
    shiny::updateNumericInput(session, "wb_lambda", value = if (identical(cfg$reg, "none") && cfg$lambda == 0) .0001 else cfg$lambda)
    shiny::updateCheckboxInput(session, "wb_self_org", value = isTRUE(cfg$self_org))
    shiny::updateCheckboxInput(session, "wb_reg_enabled", value = !identical(cfg$reg, "none"))
    shiny::updateRadioButtons(session, "wb_workflow", selected = if (identical(cfg$workflow, "custom")) "custom" else "basic")
    if (identical(cfg$workflow, "custom")) {
      for (name in c("structure", "activation", "init", "seed_aggregate"))
        shiny::updateSelectInput(session, paste0("wb_", name), selected = cfg[[name]])
      for (name in c("members", "temp", "minimum", "decay", "interval"))
        shiny::updateNumericInput(session, paste0("wb_", name), value = cfg[[name]])
      shiny::updateCheckboxInput(session, "wb_schedule_enabled", value = identical(cfg$schedule, "exponential"))
      shiny::updateSelectInput(session, "wb_schedule", selected = "exponential")
      shiny::updateCheckboxInput(session, "wb_dropout_enabled", value = cfg$dropout > 0)
      shiny::updateNumericInput(session, "wb_dropout", value = if (cfg$dropout > 0) cfg$dropout else .1)
      shiny::updateSelectInput(session, "wb_fusion", choices = wb_fusion_choices(mode), selected = cfg$fusion)
      shiny::updateTextInput(session, "wb_seeds", value = paste(cfg$seeds, collapse = ","))
      shiny::updateCheckboxInput(session, "wb_diagnostics", value = isTRUE(cfg$diagnostics))
    }

  }
  select_completed <- function(id) {
    r <- completed_runs()[[id]]
    if (is.null(r)) return(invisible(NULL))
    # Rebinding the radio list echoes the already selected value. Do not let
    # that echo overwrite settings the user has started editing for the next fit.
    if (identical(selected_run(), id) && identical(result(), r)) return(invisible(NULL))
    selected_run(id); result(r)
    sync_controls(r$config, r$prepared$mode)
    session$sendCustomMessage("wb-clear-loss-hover", list())
  }
  add_completed <- function(bundle) {
    if (length(bundle$model_name) != 1L || is.na(bundle$model_name) || !nzchar(bundle$model_name))
      bundle$model_name <- paste0(bundle$prepared$target, " · trained run ", latest_attempt())
    records <- completed_runs(); records[[latest_attempt()]] <- bundle
    completed_runs(records); select_completed(latest_attempt())
  }
  training_attempt <- function(code) {
    tryCatch(force(code), error = function(e) {
      # A rejected duplicate click must not relabel or replace a running job.
      if (!is.null(job())) { training_error(wb_training_error(e)); return(invisible(NULL)) }
      records <- attempts(); id <- latest_attempt()
      if (is.null(id) || !identical(records[[id]]$state, "Running")) begin_attempt()
      finish_attempt("Failed", wb_training_error(e))
    })
  }
  shiny::observeEvent(input$wb_selected_run, select_completed(input$wb_selected_run), ignoreInit = TRUE)
  clear_saved_rows <- function() {
    workspace_notice(NULL); restoring_purpose(NULL)
    prediction_data(NULL); prediction_source(NULL); new_predictions(NULL)
    prediction_provenance(NULL); prediction_report(NULL); model_error(NULL)
    session$sendCustomMessage("wb-reset-files", list(ids = c("wb_new_data")))
  }
  select_saved <- function(id, refresh = FALSE) {
    if (is.null(saved_models()[[id]]) || (identical(selected_saved(), id) && !refresh)) return(invisible(NULL))
    selected_saved(id)
    clear_saved_rows()
  }
  add_saved <- function(bundle) {
    bundle <- unserialize(serialize(bundle, NULL))
    if (length(bundle$model_name) != 1L || is.na(bundle$model_name) || !nzchar(bundle$model_name)) bundle$model_name <- paste(bundle$prepared$target, "saved model")
    models <- saved_models()
    same <- if (is.null(bundle$upload_sha256)) integer() else which(vapply(models,
      function(model) identical(model$upload_sha256, bundle$upload_sha256), logical(1)))
    id <- if (length(same)) names(models)[same[1]] else paste0("saved-", length(models) + 1L)
    models[[id]] <- bundle; saved_models(models); select_saved(id, refresh = TRUE)
  }
  shiny::observeEvent(input$wb_clear_saved, attempt({
    busy()
    selected_saved(NULL); saved_models(list()); saved_fit(NULL); clear_saved_rows(); load_error(NULL)
    session$sendCustomMessage("wb-reset-files", list(ids = c("wb_load", "wb_new_data")))
    shiny::updateRadioButtons(session, "wb_csv_purpose", selected = "evaluate")
    shiny::updateTabsetPanel(session, "wb_model_tools", selected = "predict")
  }, load_error))
  shiny::observeEvent(input$wb_prediction_model, select_saved(input$wb_prediction_model), ignoreInit = TRUE)
  output$wb_prediction_model_ui <- shiny::renderUI({
    models <- saved_models(); if (!length(models)) return(NULL)
    labels <- vapply(names(models), function(id) paste0(models[[id]]$model_name, " · ", id), character(1))
    shiny::selectInput("wb_prediction_model", "Local saved models", choices = stats::setNames(names(models), labels), selected = selected_saved(), selectize = FALSE, width = "360px")
  })
  shiny::observeEvent(input$wb_use_trained_model, attempt({
    if (is.null(result())) stop("Train a model before using it for prediction.")
    add_saved(result()); set_flow("saved")
  }, model_error))
  output$wb_recent_items <- shiny::renderUI({
    records <- recent_runs()
    if (!length(records)) return(shiny::p("No recent runs in this session yet. Successful training, model uploads and completed CSV runs will appear here."))
    current <- shiny::isolate(input$wb_recent_id)
    if (length(current) != 1L || !current %in% names(records)) current <- names(records)[1]
    shiny::radioButtons("wb_recent_id", "Choose a recent item (newest first)", width = "100%",
      selected = current, choiceValues = names(records), choiceNames = unname(lapply(records, function(record) {
        detail <- if (is.null(record$run)) paste(record$bundle$prepared$mode, "model") else
          paste0(record$run$source, " · ", nrow(record$run$data), " rows", if (!is.null(record$run$report$metrics)) {
            metrics <- record$run$report$metrics
            metric <- if (record$bundle$prepared$mode == "regression") "RMSE" else "Accuracy"
            value <- metrics$Value[match(metric, metrics$Metric)]
            if (metric == "Accuracy") sprintf(" · CSV accuracy %.2f%%", value * 100) else sprintf(" · CSV RMSE %.4f", value)
          } else " · predictions only")
        shiny::tagList(shiny::strong(paste(record$kind, record$bundle$model_name, sep = " · ")),
          shiny::tags$span(class = "wb-run-summary", paste(record$id, if (is.null(record$bundle$model_origin)) "Saved model" else record$bundle$model_origin, detail, sep = " · ")))
      })))
  })
  selected_run_report <- shiny::reactive({
    record <- selected_recent()
    if (is.null(record)) return(NULL)
    wb_run_report(record$bundle, record)
  })
  output$wb_recent_details <- shiny::renderUI({
    report <- selected_run_report()
    if (is.null(report)) return(NULL)
    shiny::tagList(shiny::h4("Run details"), wb_overlap_ui(report$overlap),
      shiny::tags$pre(style = "white-space: pre-wrap;",
        paste(report$overview[!startsWith(report$overview, "Split overlap: ")], collapse = "\n")),
      shiny::tags$details(shiny::tags$summary("Effective configuration and training stages"),
        shiny::tags$pre(style = "white-space: pre-wrap;", paste(report$details, collapse = "\n"))),
      shiny::downloadButton("wb_recent_report", "Download run report"))
  })
  output$wb_recent_report <- shiny::downloadHandler(
    filename = function() {
      record <- selected_recent()
      shiny::req(record)
      paste0("ddesonn-", record$id, "-report.txt")
    },
    content = function(file) {
      report <- selected_run_report()
      shiny::req(report)
      writeLines(enc2utf8(DDESONN:::.report_text(report)), file, useBytes = TRUE)
    })
  shiny::observeEvent(input$wb_recent_remove, attempt({
    record <- selected_recent(); if (is.null(record)) stop("Select a recent item first.")
    records <- recent_runs(); records[[record$id]] <- NULL; recent_runs(records)
    recent_notice("Removed that history item only. Active saved models, CSV/results, training and disk files are unchanged.")
  }, recent_error))
  shiny::observeEvent(input$wb_recent_clear, {
    recent_runs(list()); recent_error(NULL)
    recent_notice("Recent history cleared for this session only. Active saved models, CSV/results, training and disk files are unchanged.")
  })
  shiny::observeEvent(input$wb_recent_restore, attempt({
    busy(); record <- selected_recent(); if (is.null(record)) stop("Select a recent item first.")
    add_saved(record$bundle); load_error(NULL); set_flow("saved", "saved")
    session$sendCustomMessage("wb-reset-files", list(ids = c("wb_load", "wb_new_data")))
    shiny::updateTabsetPanel(session, "wb_model_tools", selected = "predict")
    if (!is.null(record$run)) {
      run <- record$run
      restoring_purpose(if (!identical(csv_purpose(), run$purpose)) run$purpose else NULL)
      shiny::updateRadioButtons(session, "wb_csv_purpose", selected = run$purpose)
      prediction_data(run$data); prediction_source(run$source)
      prediction_report(run$report); new_predictions(run$predictions); prediction_provenance(run$provenance)
      workspace_notice(paste0("Restored model and CSV results from ", record$id, " in this session. File pickers are blank because no files were re-uploaded. Training is unchanged."))
    } else workspace_notice(paste0("Restored model from ", record$id, " in this session. Upload a CSV to evaluate or predict. Training is unchanged."))
    recent_notice("Restored a copy into Saved model → evaluate or predict. The history item remains available.")
  }, recent_error))
  output$wb_recent_notice <- shiny::renderText(recent_notice())
  output$wb_recent_error <- shiny::renderText(recent_error())
  output$wb_workspace_notice <- shiny::renderText(workspace_notice())
  output$wb_training_success <- shiny::renderText(training_success())
  csv_purpose <- shiny::reactive(if (is.null(input$wb_csv_purpose)) "evaluate" else input$wb_csv_purpose)
  csv_target_check <- shiny::reactive({
    if (is.null(active_model()) || is.null(prediction_data())) return(list(valid = FALSE, message = "Upload a model and CSV first."))
    wb_evaluation_target_check(active_model()$prepared, prediction_data())
  })
  shiny::observeEvent(input$wb_csv_purpose, {
    if (!is.null(restoring_purpose()) && identical(input$wb_csv_purpose, restoring_purpose())) {
      restoring_purpose(NULL); return(invisible(NULL))
    }
    restoring_purpose(NULL); workspace_notice(NULL)
    new_predictions(NULL); prediction_provenance(NULL); prediction_report(NULL); model_error(NULL)
  }, ignoreInit = TRUE)
  shiny::observe(shiny::updateActionButton(session, "wb_predict", label = if (csv_purpose() == "evaluate") "Evaluate model" else "Run predictions"))
  output$wb_csv_instructions <- shiny::renderText({
    target <- if (is.null(active_model())) "the model's target column" else active_model()$prepared$target
    if (csv_purpose() == "evaluate") paste0("Include your predictor columns and the known answers in ", target,
      ". This compares predictions with those answers; it does not retrain the model. Open the circled i for a holdout example.") else
      "Use the saved weights to estimate unknown answers from your input rows. No accuracy is available until actual answers are known. Open the circled i for the bank example."
  })
  # Per-session states are also enforced by server guards, not just the browser.
  action_states <- shiny::reactive({
    running <- !is.null(job()); completed <- !is.null(result()); saved <- !is.null(active_model())
    list(wb_start = !is.null(prepared()) && !running, wb_cancel = running && !identical(job()$destination, "saved"),
      wb_continue = saved && !running && valid_extra_epochs(), wb_save = saved,
      wb_saved_cancel = running && identical(job()$destination, "saved"),
      wb_saved_use_fit = !running && !is.null(saved_fit()$bundle),
      wb_saved_values_prev = saved_values_page()$page > 1,
      wb_saved_values_next = saved_values_page()$page < saved_values_page()$pages,
      wb_saved_log_prev = saved_log_page()$page > 1,
      wb_saved_log_next = saved_log_page()$page < saved_log_page()$pages,
      wb_review_results = completed, wb_predictions_download = completed,
      wb_new_download = !is.null(new_predictions()), wb_new_data = saved,
      wb_use_trained_model = completed && !running,
      wb_load = !running, wb_clear_saved = !running,
      wb_recent_restore = !is.null(selected_recent()) && !running,
      wb_recent_remove = !is.null(selected_recent()), wb_recent_clear = length(recent_runs()) > 0,
      wb_prepare = !running && length(input$wb_target) == 1 && nzchar(input$wb_target) && input$wb_target %in% names(raw()) &&
        length(input$wb_features) > 0 && !input$wb_target %in% input$wb_features,
      wb_data = !running, wb_demo = !running, wb_example = !running,
      wb_example_predict = saved && !running,
      wb_predict = saved && !is.null(prediction_data()) && !running && (csv_purpose() == "predict" || csv_target_check()$valid))
  })
  shiny::observe(session$sendCustomMessage("wb-action-states", action_states()))
  busy <- function() if (!is.null(job())) stop("Finish or cancel the running job first.")
  read_csv <- function(upload, limit = 10 * 1024^2) {
    shiny::req(upload)
    if (upload$size > limit) stop("File exceeds the upload limit.")
    utils::read.csv(upload$datapath, check.names = FALSE)
  }
  output$wb_columns <- shiny::renderUI({
    data_revision() # Explicit reloads reset defaults even if the data is identical.
    cols <- names(raw())
    preset <- dataset_preset()
    target <- if (is.null(preset)) tail(cols, 1) else if (is.null(preset$target)) "" else preset$target
    shiny::tagList(
      wb_field("wb_target", "Target", "The target is the variable you want the model to predict, such as a class label or a numeric outcome. It is kept out of the predictors.",
        shiny::selectInput("wb_target", NULL, c("Choose a target" = "", stats::setNames(cols, cols)), selected = target)),
      wb_field("wb_features", "Predictors", "Predictors are the input features used to predict the target. All columns except the target are selected by default. Remove identifiers or information derived from the answer. Review the preprocessing plan below to encode categories and fill missing predictor values.",
        shiny::selectizeInput("wb_features", NULL, setdiff(cols, target), selected = wb_default_features(raw(), target, preset), multiple = TRUE)),
      shiny::p(class = "wb-guidance", "All non-target columns start selected. Auto preprocessing proposes identifier exclusions and handles categories/missing values; review the plan and leakage before preparing data."))
  })
  # Refresh hidden Features controls before either navigation path reveals them.
  shiny::outputOptions(output, "wb_columns", suspendWhenHidden = FALSE)
  output$wb_features_source <- shiny::renderText(paste("Dataset:", data_source()))
  shiny::observeEvent(data_revision(), {
    session$onFlushed(function() {
      shiny::isolate({
        cols <- names(raw()); preset <- dataset_preset()
        target <- if (is.null(preset)) tail(cols, 1) else preset$target
        shiny::updateSelectInput(session, "wb_target",
          choices = c("Choose a target" = "", stats::setNames(cols, cols)), selected = target)
        features <- wb_default_features(raw(), target)
        shiny::updateSelectizeInput(session, "wb_features", choices = features, selected = features)
      })
    }, once = TRUE)
  }, ignoreInit = TRUE)
  for (id in c("wb_data_vignettes", "wb_features_vignettes", "wb_training_vignettes", "wb_prediction_vignettes", "wb_experiment_vignettes", "wb_community_vignettes")) local({
    link_id <- id
    shiny::observeEvent(input[[link_id]], shiny::updateTabsetPanel(session, "workspace", selected = "vignettes"))
  })
  shiny::observeEvent(input$wb_next_features,
    shiny::updateTabsetPanel(session, "workspace", selected = "features"))
  shiny::observeEvent(input$wb_review_results, attempt({
    if (is.null(result())) stop("Train or load a completed model before reviewing results.")
    shiny::updateTabsetPanel(session, "workspace", selected = "results")
    shiny::updateTabsetPanel(session, "wb_results_tabs", selected = "evaluate")
  }))
  code_source <- shiny::reactive(data_origin())
  shiny::observeEvent(input$wb_remembered_workflow, {
    if (input$wb_remembered_workflow %in% c("basic", "custom"))
      shiny::updateRadioButtons(session, "wb_workflow", selected = input$wb_remembered_workflow)
  }, once = TRUE)
  shiny::observeEvent(input$wb_mode, {
    choices <- wb_fusion_choices(input$wb_mode)
    shiny::updateSelectInput(session, "wb_fusion", choices = choices,
      selected = if (length(input$wb_fusion) == 1 && input$wb_fusion %in% choices) input$wb_fusion else "mean")
  })
  next_config <- shiny::reactive({
    regularize <- isTRUE(input$wb_reg_enabled)
    cfg <- wb_config(input$wb_hidden, input$wb_epochs, input$wb_lr,
      input$wb_optimizer, if (regularize) input$wb_lambda else 0,
      if (regularize) input$wb_reg else "none", self_org = isTRUE(input$wb_self_org),
      checkpoint = if (is.null(input$wb_checkpoint)) "best_validation" else input$wb_checkpoint,
      gradient_reduction = wb_gradient_selection(input$wb_weight_reduction, input$wb_bias_reduction))
    if (!identical(input$wb_workflow, "custom")) return(cfg)
    wb_custom_config(cfg, input$wb_structure, input$wb_members, input$wb_seeds,
      input$wb_temp, input$wb_activation, input$wb_init,
      if (isTRUE(input$wb_dropout_enabled)) input$wb_dropout else 0,
      if (isTRUE(input$wb_schedule_enabled)) input$wb_schedule else "fixed", input$wb_minimum, input$wb_decay, input$wb_interval,
      input$wb_fusion, input$wb_seed_aggregate, input$wb_diagnostics)
  })
  output$wb_basic_workload <- shiny::renderText(tryCatch(wb_workload(next_config()), error = function(e) conditionMessage(e)))
  output$wb_workload <- shiny::renderText(tryCatch(wb_workload(next_config()), error = function(e) conditionMessage(e)))
  output$wb_training_log <- shiny::renderText({
    shiny::invalidateLater(500, session)
    lines <- if (!is.null(job()) && !identical(job()$destination, "saved")) read_job_log(job()) else if (!is.null(result())) result()$training_log else if (!is.null(latest_attempt()))
      attempts()[[latest_attempt()]]$training_log else character()
    paste(tail(lines, 2000L), collapse = "\n")
  })
  output$wb_seed_comparison <- shiny::renderTable({ shiny::req(result()); result()$seed_comparison })
  output$wb_member_weights <- shiny::renderPrint({ shiny::req(result()); result()$member_weights })
  shiny::observeEvent(list(raw(), input$wb_target, data_revision()), {
    target <- input$wb_target
    if (length(target) != 1L || (nzchar(target) && !target %in% names(raw()))) return(invisible(NULL))
    features <- wb_default_features(raw(), target)
    shiny::updateSelectizeInput(session, "wb_features", choices = features, selected = features)
  })
  shiny::observeEvent(input$wb_data, attempt({
    busy(); candidate <- read_csv(input$wb_data)
    if (nrow(candidate) < 10 || anyDuplicated(names(candidate))) stop("Use at least 10 rows and unique column names.")
    pp_overrides(list()); dataset_preset(NULL); raw(candidate); data_revision(data_revision() + 1L); prepared(NULL)
    data_origin(list(file = basename(input$wb_data$name),
      sha256 = digest::digest(file = input$wb_data$datapath, algo = "sha256")))
    data_source(paste("Uploaded:", basename(input$wb_data$name)))
  }))
  example_id <- shiny::reactive(if (is.null(input$wb_example)) "iris" else input$wb_example)
  output$wb_example_note <- shiny::renderText(wb_example_catalog()[[example_id()]]$note)
  load_example <- function() {
    busy(); example <- wb_example(example_id())
    if (isTRUE(example$prediction_only)) stop("Use unlabeled examples under Predict on new rows, after training a compatible model.")
    pp_overrides(list()); dataset_preset(example); raw(example$data); data_revision(data_revision() + 1L); prepared(NULL); data_origin(example$source)
    data_source(paste0("Included ", example$label, " · ", nrow(example$data), " rows"))
    if (!is.null(example$mode)) shiny::updateSelectInput(session, "wb_mode", selected = example$mode)
    shiny::updateSelectInput(session, "wb_split", selected = if (is.null(example$split)) "random" else example$split)
  }
  shiny::observeEvent(input$wb_demo, attempt(load_example()))
  shiny::observeEvent(input$wb_example, attempt(load_example()), ignoreInit = TRUE)
  output$wb_example_download <- shiny::downloadHandler(
    filename = function() paste0("DDESONN-", example_id(), ".csv"),
    content = function(file) wb_write_example(example_id(), file))
  output$wb_prediction_example_download <- shiny::downloadHandler(
    filename = function() "DDESONN-customer-prediction.csv",
    content = function(file) wb_write_example("customer_test", file))
  shiny::observeEvent(input$wb_example_predict, attempt({
    busy(); new_predictions(NULL); prediction_provenance(NULL); prediction_report(NULL)
    if (is.null(active_model())) stop("Train or load a compatible model before choosing example rows.")
    example <- wb_example("customer_test")
    prediction_data(example$data); prediction_source("Packaged customer rows")
    set_flow("saved")
    shiny::updateTabsetPanel(session, "wb_model_tools", selected = "predict")
  }, model_error))

  # Invalidate only preparation. The immutable completed model is still usable.
  signature <- shiny::reactive(list(input$wb_target, input$wb_features, input$wb_mode,
    input$wb_seed, input$wb_train_fraction, input$wb_val_fraction, input$wb_scale, input$wb_split, pp_config()))
  shiny::observeEvent(signature(), prepared(NULL), ignoreInit = TRUE)
  shiny::observeEvent(input$wb_prepare, attempt({
    busy()
    p <- wb_prepare(raw(), input$wb_target, input$wb_features, input$wb_mode,
      input$wb_seed, input$wb_train_fraction, input$wb_val_fraction, input$wb_scale, input$wb_split, preprocessing = pp_config())
    p$source <- code_source()
    prepared(p)
    if (is.null(latest_attempt())) status("Data prepared. Ready to train a new model.")
    shiny::updateTabsetPanel(session, "workspace", selected = "train")
  }))
  launch <- function(p, cfg, old = NULL, destination = "train") {
    busy()
    if (destination == "saved") {
      saved_fit_counter(saved_fit_counter() + 1L)
      id <- paste0("saved-fit-", saved_fit_counter())
    } else {
      training_success(NULL)
      id <- begin_attempt(cfg, "New training")
    }
    next_job <- wb_start(helper, p, cfg, old)
    next_job$destination <- destination
    next_job$run_id <- id
    next_job$parent_name <- if (is.null(old)) NULL else old$model_name
    job(next_job)
    if (destination == "train") {
      history(data.frame(epoch = integer(), loss = numeric()))
      status(paste0("Run ", id, " · Starting training in a separate R process…"))
    }
  }
  shiny::observeEvent(input$wb_start, training_attempt({
    if (is.null(prepared())) stop("Choose target and predictors on Features, then click Prepare data before training.")
    cfg <- next_config()
    training_context(NULL)
    launch(prepared(), cfg)
  }))
  valid_extra_epochs <- shiny::reactive({
    n <- input$wb_more_epochs
    length(n) == 1L && is.finite(n) && n >= 1 && n <= 5000 && n == floor(n)
  })
  for (n in c(10, 100, 1000)) local({
    value <- n
    shiny::observeEvent(input[[paste0("wb_epochs_", value)]], shiny::updateNumericInput(session, "wb_more_epochs", value = value))
  })
  shiny::observeEvent(input$wb_epochs_clear, shiny::updateNumericInput(session, "wb_more_epochs", value = NA_real_))
  output$wb_saved_checkpoint_ui <- shiny::renderUI({
    b <- active_model(); shiny::req(b)
    shiny::tagList(shiny::selectInput("wb_saved_checkpoint", "Weights to keep after this extra fit",
      c("Last epoch weights" = "final", "Best validation weights" = "best_validation"),
      selected = b$config$checkpoint, width = "260px"),
      shiny::p(class = "help-block", "Last epoch keeps the final weights. Best validation keeps the epoch with the highest validation accuracy for classification, or the lowest validation loss for regression. It does not select the lowest training loss. The completed graph marks the selected epoch when a single checkpoint applies; multiple Custom members may keep different epochs."))
  })
  shiny::observeEvent(input$wb_continue, attempt({
    if (is.null(active_model())) stop("Load or select a saved model first.")
    busy()
    old <- unserialize(serialize(active_model(), NULL)); cfg <- old$config
    if (!valid_extra_epochs()) stop("Enter a whole number of additional epochs from 1 to 5000.")
    cfg$epochs <- wb_config(epochs = input$wb_more_epochs)$epochs
    if (length(input$wb_saved_checkpoint) == 1L && input$wb_saved_checkpoint %in% c("final", "best_validation")) cfg$checkpoint <- input$wb_saved_checkpoint
    launch(old$prepared, cfg, old, destination = "saved")
    saved_fit(list(parent = old$model_name, state = "Running", config = cfg, history = data.frame(epoch = integer(), loss = numeric()), bundle = NULL, error = NULL, log = character()))
    shiny::updateTabsetPanel(session, "wb_model_tools", selected = "continue")
  }, model_error))
  shiny::observeEvent(input$wb_saved_use_fit, attempt({
    shiny::req(saved_fit()$bundle)
    busy(); add_saved(saved_fit()$bundle)
    shiny::updateTabsetPanel(session, "wb_model_tools", selected = "predict")
  }, model_error))
  shiny::observeEvent(input$wb_saved_cancel, {
    active <- job()
    if (!is.null(active) && identical(active$destination, "saved")) {
      h <- wb_log_history(active$log)
      fit <- saved_fit(); fit$history <- h; fit$log <- read_job_log(active); fit$state <- "Cancelled"; saved_fit(fit)
      job(NULL); wb_stop(active)
    }
  })
  shiny::observeEvent(input$wb_cancel, training_attempt({
    active <- job()
    if (!is.null(active) && !identical(active$destination, "saved")) {
      history(wb_log_history(active$log))
      finish_attempt("Cancelled")
      # Detach before waiting for process shutdown so polling cannot record failure.
      job(NULL); wb_stop(active)
    }
  }))
  shiny::observe({
    shiny::invalidateLater(300, session)
    active <- shiny::isolate(job())
    if (!is.null(active)) {
      h <- wb_log_history(active$log)
      if (identical(active$destination, "saved")) {
        fit <- shiny::isolate(saved_fit()); fit$history <- h; fit$log <- read_job_log(active); saved_fit(fit)
      } else {
        history(h)
        elapsed <- as.integer(difftime(Sys.time(), active$started, units = "secs"))
        status(sprintf("Run %s · Running · %s seconds · %s of %s epochs logged", active$run_id, elapsed,
          if (nrow(h)) max(h$epoch) else 0L, active$epochs))
      }
      if (!active$process$is_alive()) {
        tryCatch({
          complete <- active$process$get_result()
          if (!is.null(active$parent_name)) complete$model_name <- paste0(active$parent_name, " · extra fit ", active$run_id)
          complete$history <- h
          complete$training_log <- read_job_log(active)
          complete <- wb_validate_bundle(complete)
          if (identical(active$destination, "saved")) {
            complete$model_origin <- paste0("Further-trained saved model from ", active$parent_name)
            complete$upload_sha256 <- NULL
            fit <- saved_fit(); fit$bundle <- complete; fit$state <- "Completed"; saved_fit(fit)
            remember_recent(complete, "Further-trained saved model", key = paste0("saved-fit-", active$run_id))
          } else {
            complete$model_origin <- "Trained in Train a new model"
            add_completed(complete); error(NULL)
            remember_recent(result(), "Newly trained model", key = paste0("trained-", selected_run()))
          }
          if (!identical(active$destination, "saved")) {
            finish_attempt("Completed")
            training_success("Model trained—nice work. Review its held-out results before using it for new predictions.")
          }
        }, error = function(e) {
          if (identical(active$destination, "saved")) {
            fit <- saved_fit(); fit$state <- "Failed"; fit$error <- paste(unlist(wb_training_error(e)), collapse = "\n\n"); saved_fit(fit)
          }
          if (!identical(active$destination, "saved")) finish_attempt("Failed", wb_training_error(e, if (file.exists(active$error))
            readLines(active$error, warn = FALSE) else character()))
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
    candidate$model_origin <- "Uploaded saved model (.rds)"
    candidate$model_name <- if (is.null(input$wb_load$name)) "Loaded workbench model" else basename(input$wb_load$name)
    candidate$source_filename <- candidate$model_name
    candidate$upload_sha256 <- digest::digest(file = input$wb_load$datapath, algo = "sha256")
    add_saved(candidate)
    remember_recent(active_model(), "Uploaded saved model", key = paste0("upload-", candidate$upload_sha256))
    set_flow("saved")
    shiny::updateTabsetPanel(session, "wb_model_tools", selected = "predict")
  }, load_error))
  shiny::observeEvent(input$wb_new_data, attempt({
    workspace_notice(NULL)
    new_predictions(NULL); prediction_provenance(NULL); prediction_report(NULL); prediction_data(NULL); prediction_source(NULL)
    prediction_data(read_csv(input$wb_new_data))
    prediction_source(basename(input$wb_new_data$name))
  }, model_error))
  shiny::observeEvent(input$wb_predict, attempt({
    busy(); new_predictions(NULL); prediction_provenance(NULL); prediction_report(NULL)
    if (is.null(active_model())) stop("Load or select a model first.")
    if (is.null(prediction_data())) stop("Upload a prediction CSV first.")
    if (csv_purpose() == "evaluate" && !csv_target_check()$valid) stop(csv_target_check()$message)
    predictions <- wb_new_predictions(active_model(), prediction_data())
    report <- wb_prediction_report(active_model(), prediction_data(), evaluate = csv_purpose() == "evaluate")
    prediction_report(report)
    if (!is.null(report$metrics)) predictions <- cbind(predictions[1], actual = prediction_data()[[active_model()$prepared$target]], predictions[-1])
    new_predictions(predictions)
    prediction_provenance(list(model = active_model()$model_name, source = prediction_source(), rows = nrow(predictions), purpose = csv_purpose()))
    remember_recent(active_model(), if (csv_purpose() == "evaluate") "CSV evaluation" else "Production prediction",
      run = list(data = prediction_data(), source = prediction_source(), purpose = csv_purpose(),
        report = prediction_report(), predictions = new_predictions(), provenance = prediction_provenance()))
  }, model_error))
  output$wb_load_error <- shiny::renderText(if (!is.null(load_error())) paste0("Saved-model action failed: ", load_error(), " The previous active model, if any, is still selected."))
  output$wb_active_model <- shiny::renderText({
    bundle <- active_model()
    if (is.null(bundle)) return("No model selected. Upload a .rds file first.")
    paste("Trained model loaded:", bundle$model_name,
      DDESONN:::.ddesonn_gradient_label(wb_gradient_mode(bundle$config)), sep = " ")
  })
  output$wb_prediction_columns <- shiny::renderText({
    if (is.null(active_model())) return("Load a .rds model first. Its saved training inputs determine the CSV columns listed here.")
    paste0("Input columns saved in this model: ", paste(active_model()$prepared$features, collapse = ", "), if (csv_purpose() == "evaluate") paste0("; plus ", active_model()$prepared$target, " with known answers. Saved preprocessing is reused.") else ". Target labels are not evaluated in prediction-only mode. Saved preprocessing is reused.")
  })
  output$wb_prediction_status <- shiny::renderText({
    if (is.null(active_model())) return("Step 1: load or select a model.")
    if (is.null(prediction_data())) return("Model ready. Choose evaluation or prediction, then upload your CSV. Reviewing the saved scores alone requires no upload.")
    if (csv_purpose() == "evaluate" && !csv_target_check()$valid)
      return(paste0(csv_target_check()$message, " If the answers are unknown, choose Make production predictions instead."))
    if (!is.null(model_error())) return("Could not complete. Correct the error below and try again.")
    if (is.null(new_predictions())) return(paste0(prediction_source(), " · ", nrow(prediction_data()), " rows ready. Click ", if (csv_purpose() == "evaluate") "Evaluate model" else "Run predictions", "."))
    paste0(if (csv_purpose() == "evaluate") "Evaluation" else "Predictions", " complete for ", nrow(new_predictions()), " rows. Results are below; download is optional.")
  })
  output$wb_csv_result_title <- shiny::renderText({
    shiny::req(prediction_provenance())
    if (prediction_provenance()$purpose == "evaluate") "Evaluation of your labeled CSV" else "Predictions for your unlabeled CSV"
  })
  output$wb_csv_primary_score <- shiny::renderText({
    report <- prediction_report(); shiny::req(report)
    if (is.null(report$metrics)) return(paste(report$rows, "predictions generated"))
    metric <- if (active_model()$prepared$mode == "regression") "RMSE" else "Accuracy"
    value <- report$metrics$Value[match(metric, report$metrics$Metric)]
    if (metric == "Accuracy") sprintf("CSV accuracy: %.2f%% (%d rows)", value * 100, report$rows) else
      sprintf("CSV RMSE: %.4f (lower is better; %d rows)", value, report$rows)
  })
  output$wb_prediction_result_info <- shiny::renderText({
    provenance <- prediction_provenance(); shiny::req(provenance)
    paste0(provenance$model, " · ", provenance$source, " · ", provenance$rows, " predictions. Showing the first ", min(20L, provenance$rows), " rows; the optional CSV contains every row.")
  })
  output$wb_saved_fit_status <- shiny::renderText({
    fit <- saved_fit(); if (is.null(fit)) return("No additional training started. Choose the number of extra epochs, then click Start additional training.")
    paste0(fit$state, " · additional training from saved model: ", fit$parent,
      if (fit$state == "Running") paste0(" · ", if (nrow(fit$history)) max(fit$history$epoch) else 0, " epochs logged") else
        if (fit$state == "Completed") " · new copy ready; original unchanged" else " · original unchanged")
  })
  output$wb_saved_fit_error <- shiny::renderText(if (!is.null(saved_fit()$error)) saved_fit()$error else model_error())
  output$wb_saved_fit_progress <- shiny::renderUI({
    active <- job(); shiny::req(identical(active$destination, "saved"))
    h <- saved_fit()$history
    shiny::tags$progress(value = if (nrow(h)) max(h$epoch) else 0, max = active$epochs, style = "width:100%;")
  })
  saved_plot_history <- shiny::reactive({
    fit <- saved_fit(); wb_chart_history(fit$history, fit$config)
  })
  output$wb_saved_fit_loss <- shiny::renderPlot({
    fit <- saved_fit(); shiny::req(nrow(saved_plot_history()) > 0)
    wb_loss_plot(saved_plot_history(), fit$config, if (!is.null(fit$bundle)) fit$bundle$selected_epoch else NULL)
  })
  output$wb_saved_fit_checkpoint <- shiny::renderText({
    fit <- saved_fit(); if (is.null(fit)) return(NULL)
    if (!is.null(fit$bundle)) paste0(wb_checkpoint_summary(fit$bundle),
      if (length(unique(saved_plot_history()$fit)) == 1L) ". Dashed line marks the retained epoch." else ". Separate member fits can retain different checkpoints; no single checkpoint line is shown.") else
      paste0("This extra fit will retain ", if (identical(fit$config$checkpoint, "best_validation")) "best validation weights" else "last epoch weights", ". The retained checkpoint is shown after completion.")
  })
  bind_saved_pager <- function(id, source) {
    force(id); force(source)
    page <- shiny::reactiveVal(1L)
    view <- shiny::reactive(wb_page(source(), page()))
    shiny::observeEvent(input[[paste0(id, "_prev")]], page(max(1L, view()$page - 1L)))
    shiny::observeEvent(input[[paste0(id, "_next")]], page(min(view()$pages, view()$page + 1L)))
    shiny::observeEvent(input$wb_continue, page(1L), ignoreInit = TRUE)
    output[[paste0(id, "_page")]] <- shiny::renderText({
      v <- view(); paste0("Page ", v$page, " of ", v$pages, " · ", v$total, " ", if (id == "wb_saved_log") "lines" else "rows", " · 25 per page")
    })
    view
  }
  saved_values_page <- bind_saved_pager("wb_saved_values", saved_plot_history)
  saved_log_page <- bind_saved_pager("wb_saved_log", shiny::reactive(tail(saved_fit()$log, 2000L)))
  output$wb_saved_fit_values <- shiny::renderTable(saved_values_page()$data, digits = 8)
  output$wb_saved_fit_log <- shiny::renderText(paste(saved_log_page()$data, collapse = "\n"))
  output$wb_saved_fit_evaluation <- shiny::renderTable({
    b <- saved_fit()$bundle; shiny::req(b)
    do.call(rbind, lapply(c("train", "validation", "test"), function(split) {
      m <- b$evaluation[[split]]$metrics; cbind(Split = split, m)
    }))
  })
  output$wb_warm_start_summary <- shiny::renderText({
    r <- active_model(); if (is.null(r)) return("Load or select a model before training further.")
    paste0("Selected model: ", r$model_name, ". Its previous fit requested ", r$config$epochs,
      " epochs per member", if (length(r$selected_epoch) == 1L && is.finite(r$selected_epoch)) paste0("; saved weights are from epoch ", r$selected_epoch) else "", ".")
  })
  output$wb_saved_evaluation <- shiny::renderTable({
    r <- active_model(); shiny::req(r)
    do.call(rbind, lapply(c("train", "validation", "test"), function(split) {
      m <- r$evaluation[[split]]$metrics
      metric <- if (r$prepared$mode == "regression") "RMSE" else "Accuracy"
      score <- m$Value[match(metric, m$Metric)]
      data.frame(Split = switch(split, train = "Training", validation = "Validation", test = "Test"),
        Rows = nrow(r$prepared$parts[[split]]$x), Metric = metric,
        Value = if (metric == "Accuracy") sprintf("%.2f%%", score * 100) else sprintf("%.4f", score))
    }))
  })
  output$wb_saved_history_note <- shiny::renderText({
    r <- active_model(); shiny::req(r)
    paste0("This model is already trained. Loading it does not start new epochs. ",
      if (is.data.frame(r$history) && nrow(r$history)) "The curve below is the history saved in this bundle, not live progress." else "No training-loss history was saved in this bundle.")
  })
  output$wb_saved_loss <- shiny::renderPlot({
    r <- active_model(); shiny::req(r)
    shiny::validate(shiny::need(is.data.frame(r$history) && nrow(r$history) > 0, "No saved loss curve is available."))
    wb_loss_plot(r$history, r$config, r$selected_epoch)
  })
  output$wb_saved_logs <- shiny::renderText({
    r <- active_model(); shiny::req(r)
    if (length(r$training_log)) paste(utils::tail(r$training_log, 200), collapse = "\n") else "No training log was saved."
  })
  output$wb_prediction_overlap_note <- shiny::renderText({
    report <- prediction_report(); shiny::req(report)
    paste0(report$rows, " prediction rows checked against this model's saved inputs. A row may match more than one split. ",
      if (report$matched_rows) paste0(report$matched_rows, " match at least one saved split; these rows are not an independent unseen-input evaluation.") else "No exact prepared-input matches found. This does not establish different people or detect near-duplicates.")
  })
  output$wb_prediction_overlap_table <- shiny::renderTable({ shiny::req(prediction_report()); prediction_report()$overlap })
  output$wb_prediction_accuracy_note <- shiny::renderText({ shiny::req(prediction_report()); prediction_report()$evaluation_note })
  output$wb_prediction_accuracy <- shiny::renderTable({
    shiny::req(prediction_report()); metrics <- prediction_report()$metrics
    if (is.null(metrics)) return(NULL)
    metrics$Value <- ifelse(metrics$Metric == "Accuracy", sprintf("%.2f%%", metrics$Value * 100), sprintf("%.6f", metrics$Value))
    metrics
  })
  model_info <- shiny::reactive({
    r <- result()
    if (is.null(r)) return("No completed model yet.")
    info <- sprintf("%s · %s predictors → %s · hidden layers: %s · %s · %s checkpoint, epoch %s of %s · initial self-organization %s · %s fit run(s) · saved package %s · completed %s",
      r$prepared$mode, length(r$prepared$features), r$prepared$target,
      if (length(r$config$hidden)) paste(r$config$hidden, collapse = ", ") else "none",
      r$config$optimizer, if (identical(r$config$checkpoint, "best_validation")) "best validation" else "final",
      if (is.null(r$selected_epoch)) "not recorded" else r$selected_epoch, r$config$epochs,
      if (isTRUE(r$config$self_org)) "on" else "off",
      r$fit_runs, r$package_version, r$completed)
    paste0(r$model_name, " · ", if (!is.null(selected_run())) paste0("Run ", selected_run(), " · ") else "", info)
  })
  # Each disclosure is an independent, lazily rendered code output. Reading help
  # never launches a fit or mutates a completed model.
  for (topic in wb_help_topics()) local({
    selected_topic <- topic
    output[[wb_help_key(selected_topic)]] <- shiny::renderText({
      wb_help_code(selected_topic, prepared = prepared(),
        config = tryCatch(next_config(), error = function(e) NULL),
        completed = code_model(), source = code_source(),
        evaluation_split = if (is.null(input$wb_eval_split)) "test" else input$wb_eval_split,
        more_epochs = if (is.null(input$wb_more_epochs)) 10L else input$wb_more_epochs)
    })
  })
  full_choice <- shiny::reactiveVal("next")
  shiny::observeEvent(input$wb_full_source, full_choice(input$wb_full_source))
  full_script <- shiny::reactive({
    selected <- full_choice()
    tryCatch(wb_full_code(prepared(), tryCatch(next_config(), error = function(e) NULL), code_model(), selected),
      error = function(e) { shiny::validate(shiny::need(FALSE, conditionMessage(e))) })
  })
  show_full_code <- function(choice = NULL) {
    choices <- if (task_flow() == "saved") character() else c("Next model / iris illustration" = "next")
    if (!is.null(code_model()$recipe)) choices <- c(choices, "Completed model's saved recipe" = "completed")
    choices <- c(choices, "R example: load a saved model and predict new rows" = "prediction")
    selected <- if (!is.null(code_model()$recipe)) "completed" else if (!is.null(code_model())) "prediction" else "next"
    if (!is.null(choice)) selected <- choice
    if (task_flow() == "saved" && selected == "next") selected <- "prediction"
    full_choice(selected)
    shiny::showModal(shiny::modalDialog(title = "Full R code", size = "l", easyClose = TRUE,
      shiny::radioButtons("wb_full_source", "Code for", choices, selected = selected),
      shiny::textOutput("wb_full_notice"),
      shiny::tags$pre(id = "wb_full_code", class = "shiny-text-output wb-full-script", tabindex = "0",
        `aria-label` = "Complete R script"),
      shiny::uiOutput("wb_full_download_button"),
      footer = shiny::modalButton("Close")))
  }
  shiny::observeEvent(input$wb_show_full_code, show_full_code())
  shiny::observeEvent(input$wb_preview_next_recipe, show_full_code("next"))
  shiny::observeEvent(input$wb_preview_completed_recipe, show_full_code("completed"))
  show_bundle_preview <- function() {
    shiny::showModal(shiny::modalDialog(title = "Selected model bundle contents", size = "l", easyClose = TRUE,
      shiny::tableOutput("wb_bundle_manifest"),
      footer = shiny::modalButton("Close")))
  }
  shiny::observeEvent(input$wb_preview_bundle, show_bundle_preview())
  shiny::observeEvent(input$wb_preview_selected_bundle, show_bundle_preview())
  output$wb_bundle_manifest <- shiny::renderTable({
    shiny::validate(shiny::need(!is.null(code_model()), "Choose a model to preview its bundle."))
    wb_bundle_manifest(code_model())
  })
  output$wb_full_notice <- shiny::renderText(full_script()$notice)
  output$wb_full_code <- shiny::renderText(full_script()$code)
  output$wb_full_download_button <- shiny::renderUI({
    script <- tryCatch(full_script(), error = function(e) NULL)
    if (is.null(script)) shiny::tags$button(type = "button", class = "btn btn-default disabled",
      disabled = "disabled", `aria-disabled` = "true", "Download full R script") else
      wb_download_info("wb_full_download", shiny::downloadButton("wb_full_download", "Download full R script"))
  })
  output$wb_full_download <- shiny::downloadHandler(filename = function() full_script()$filename,
    content = function(file) writeLines(full_script()$code, file))
  inspected <- shiny::reactive({
    columns <- if (!identical(input$wb_inspect_all, FALSE)) names(raw()) else input$wb_inspect_columns
    shiny::validate(shiny::need(length(columns) > 0, "Choose at least one column, or select Show all columns."))
    wb_inspect_rows(raw(),
    if (is.null(input$wb_inspect_view)) "head" else input$wb_inspect_view,
    if (is.null(input$wb_inspect_n)) 5 else input$wb_inspect_n,
    if (is.null(input$wb_inspect_start)) 1 else input$wb_inspect_start,
    columns)
  })
  output$wb_data_code <- shiny::renderText(paste(wb_data_code(code_source()),
    "# Inspect a subset; the full data remains unchanged.", inspected()$expression, sep = "\n\n"))
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
    shiny::req(active_model())
    if (is.null(active_model()$recipe)) "# This older bundle does not contain a replay recipe. Use the saved-model API code." else active_model()$recipe
  })
  output$wb_next_recipe_button <- shiny::renderUI({
    cfg <- tryCatch(next_config(), error = function(e) NULL)
    if (!is.null(prepared()) && !is.null(cfg))
      shiny::tagList(wb_download_info("wb_next_recipe", shiny::downloadButton("wb_next_recipe", "Download next-model R recipe")), shiny::actionButton("wb_preview_next_recipe", "Preview recipe"))
  })
  output$wb_completed_recipe_button <- shiny::renderUI({
    if (!is.null(active_model()$recipe)) shiny::tagList(wb_download_info("wb_completed_recipe", shiny::downloadButton("wb_completed_recipe", "Download completed-model R recipe")), shiny::actionButton("wb_preview_completed_recipe", "Preview saved recipe"))
  })
  output$wb_next_recipe <- shiny::downloadHandler(filename = function() "ddesonn-next-model.R",
    content = function(file) { shiny::req(prepared()); writeLines(wb_recipe(prepared(), next_config()), file) })
  output$wb_completed_recipe <- shiny::downloadHandler(filename = function() "ddesonn-completed-model.R",
    content = function(file) { shiny::req(active_model()$recipe); writeLines(active_model()$recipe, file) })
  output$wb_run_choices <- shiny::renderUI({
    runs <- completed_runs()
    if (!length(runs)) return(shiny::p("No completed training models yet. Train here; uploaded models stay in Use a saved model."))
    ids <- rev(names(runs)); records <- shiny::isolate(attempts())
    shiny::radioButtons("wb_selected_run", "Local completed runs", width = "100%", selected = selected_run(),
      choiceValues = ids, choiceNames = lapply(ids, function(id) shiny::tagList(
        shiny::strong(paste0(runs[[id]]$model_name, " · Run ", id, " · ", records[[id]]$kind)),
        shiny::tags$span(class = "wb-run-summary", wb_checkpoint_summary(runs[[id]])))))
  })
  output$wb_settings_origin <- shiny::renderText({
    if (is.null(selected_run())) return(NULL)
    paste0("Settings from Run ", selected_run(),
      " · edits apply to the next new run on the currently prepared data. Saved model settings remain unchanged.")
  })
  output$wb_selected_save <- shiny::renderUI({
    if (!is.null(result())) shiny::tagList(wb_download_info("wb_save_selected", shiny::downloadButton("wb_save_selected", "Save selected model")), shiny::actionButton("wb_preview_selected_bundle", "Preview bundle contents"))
  })
  output$wb_attempt_history <- shiny::renderUI({
    records <- attempts()
    if (!length(records)) return(shiny::p("No attempts in this session yet."))
    shiny::tagList(lapply(rev(records), function(record) shiny::div(class = "wb-attempt-row",
      shiny::strong(paste0("Run ", record$id, " · ", record$state, " · ", record$kind)),
      shiny::p(record$started),
      if (!is.null(record$config)) shiny::p(
        if (identical(record$config$checkpoint, "best_validation")) "Best validation checkpoint" else "Final epoch"),
      if (!is.null(record$error)) shiny::tagList(shiny::p(record$error$message),
        shiny::tags$details(shiny::tags$summary("Technical details"), shiny::tags$pre(record$error$details))))))
  })
  output$wb_selected_model <- shiny::renderText(model_info())
  output$wb_model_info <- shiny::renderText(if (!is.null(result())) model_info())
  output$wb_saved_info <- shiny::renderText({
    r <- active_model(); shiny::req(r)
    paste0(r$model_name, " · ", r$prepared$mode, " → ", r$prepared$target,
      " · ", length(r$prepared$features), " predictors · saved package ", r$package_version,
      " · checkpoint ", r$selected_epoch, " of ", r$config$epochs, " requested epochs")
  })
  output$wb_error <- shiny::renderText(error())
  output$wb_data_error <- shiny::renderText(error())
  output$wb_train_error <- shiny::renderUI({
    e <- training_error(); if (is.null(e)) return(NULL)
    shiny::div(class = "wb-training-error",
      shiny::div(class = "error", role = "alert", e$message,
        shiny::p(if (!is.null(result())) "Completed models remain available below. This message belongs to the latest attempt, not the selected model." else "No completed model was produced. Correct the settings or data and retry.")),
      shiny::tags$details(shiny::tags$summary("Technical details"), shiny::tags$pre(e$details)))
  })
  output$wb_checkpoint_summary <- shiny::renderText(if (!is.null(result())) wb_checkpoint_summary(result()))
  output$wb_checkpoint_results <- shiny::renderText(if (!is.null(result())) wb_checkpoint_summary(result()))
  output$wb_model_error <- shiny::renderText(model_error())
  output$wb_data_source <- shiny::renderText(data_source())
  output$wb_inspect_columns_ui <- shiny::renderUI(shiny::selectizeInput("wb_inspect_columns", "Choose columns to view", names(raw()), selected = names(raw()), multiple = TRUE))
  shiny::observeEvent(raw(), shiny::updateNumericInput(session, "wb_inspect_start", value = 1, max = max(1, nrow(raw()))), ignoreInit = TRUE)
  output$wb_inspect_range <- shiny::renderText({
    ids <- inspected()$rows
    paste0(if (length(ids)) paste0("Rows ", min(ids), "–", max(ids)) else "No rows",
      " of ", nrow(raw()), " · ", length(inspected()$columns), " of ", ncol(raw()), " columns. Row names identify the original rows.")
  })
  output$wb_preview <- shiny::renderTable({
    data <- inspected()$data
    shiny::validate(shiny::need(ncol(data) > 0, "Choose columns from the loaded dataset."))
    data
  }, rownames = TRUE)
  output$wb_data_profile <- shiny::renderTable(wb_data_profile(raw()), digits = 4, na = "NA")
  output$wb_box_column_ui <- shiny::renderUI({
    choices <- names(raw())[vapply(raw(), is.numeric, logical(1))]
    if (!length(choices)) return(shiny::p("No numeric columns in this dataset."))
    shiny::selectInput("wb_box_column", "Numeric column", choices, selectize = FALSE)
  })
  output$wb_data_box <- shiny::renderPlot({
    column <- input$wb_box_column
    shiny::validate(shiny::need(length(column) == 1 && column %in% names(raw()) && is.numeric(raw()[[column]]), "Choose a numeric column."))
    values <- raw()[[column]]
    finite <- values[is.finite(values)]
    shiny::validate(shiny::need(length(finite) > 0, "No finite numeric values to plot."))
    graphics::boxplot(finite, horizontal = TRUE, main = column, xlab = "Value", col = "#dce9f6", border = "#326ea5")
    graphics::mtext(paste(length(finite), "finite values;", length(values) - length(finite), "missing/nonfinite omitted from plot only"), side = 3, cex = .8)
  })
  output$wb_pp_fitted <- shiny::renderTable({
    shiny::req(prepared(), prepared()$preprocessing)
    wb_pp_summary(prepared()$preprocessing)
  })
  output$wb_prepared <- shiny::renderText({
    p <- prepared()
    if (is.null(p)) return("Not prepared. Choose your columns and task, then click Prepare data.")
    counts <- vapply(p$parts, function(x) nrow(x$x), integer(1))
    paste(paste(sprintf("%s: %s rows", names(counts), counts), collapse = " · "),
      paste0("· ", length(p$features), " source predictors → ", ncol(p$parts$train$x), " numeric model inputs"),
      if (p$mode == "binary") paste("· positive class:", p$classes[2]) else "")
  })
  output$wb_split_overlap <- shiny::renderUI(wb_overlap_ui(wb_recorded_overlap(prepared())))
  output$wb_training_context <- shiny::renderText(training_context())
  output$wb_status <- shiny::renderText(status())
  output$wb_running_status <- shiny::renderText(status())
  output$wb_attempt_title <- shiny::renderText(if (is.null(latest_attempt())) "Training status" else "Latest attempt")
  output$wb_running_flag <- shiny::renderText(if (!is.null(job()) && !identical(job()$destination, "saved")) "yes" else "no")
  output$wb_error_flag <- shiny::renderText(if (!is.null(training_error())) "yes" else "no")
  output$wb_csv_has_metrics <- shiny::renderText(if (!is.null(prediction_report()$metrics)) "yes" else "no")
  output$wb_has_saved_model <- shiny::renderText(if (!is.null(active_model())) "yes" else "no")
  output$wb_has_predictions <- shiny::renderText(if (!is.null(new_predictions())) "yes" else "no")
  output$wb_has_model <- shiny::renderText(if (!is.null(result())) "yes" else "no")
  output$wb_chart_ready <- shiny::renderText(if (nrow(plot_history()) > 0) "yes" else "no")
  output$wb_has_logs <- shiny::renderText({
    lines <- if (!is.null(job())) TRUE else if (!is.null(result())) length(result()$training_log) > 0 else {
      id <- latest_attempt()
      !is.null(id) && length(attempts()[[id]]$training_log) > 0
    }
    if (lines) "yes" else "no"
  })
  for (flag in c("wb_running_flag", "wb_error_flag", "wb_has_model", "wb_chart_ready", "wb_has_logs", "wb_has_predictions", "wb_has_saved_model", "wb_csv_has_metrics"))
    shiny::outputOptions(output, flag, suspendWhenHidden = FALSE)
  output$wb_progress <- shiny::renderUI({
    active <- job(); shiny::req(active)
    h <- history(); n <- if (nrow(h)) max(h$epoch) else 0
    shiny::tags$progress(value = n, max = active$epochs, style = "width:100%;", `aria-label` = "Logged member epochs")
  })
  raw_plot_history <- shiny::reactive({
    r <- result()
    if (!is.null(r)) {
      if (is.data.frame(r$history)) r$history else data.frame(epoch = integer(), loss = numeric())
    } else if (!is.null(job()) && !identical(job()$destination, "saved")) history() else {
      id <- latest_attempt()
      record <- if (is.null(id)) NULL else attempts()[[id]]
      if (is.data.frame(record$history)) record$history else data.frame(epoch = integer(), loss = numeric())
    }
  })
  plot_history <- shiny::reactive(wb_chart_history(raw_plot_history(),
    if (!is.null(result())) result()$config else NULL))
  output$wb_loss_explanation <- shiny::renderText({
    h <- plot_history()
    if (!nrow(h)) return(NULL)
    if (length(unique(h$fit)) > 1) paste0("These are ", length(unique(h$fit)),
      " separate fits, not one model restarting. Custom can log multiple members, seeds and MAIN/TEMP fits. Loss can start higher for a new fit because its weights or training stage differ. Fit numbers follow log order; they do not identify a specific seed or MAIN/TEMP role.",
      if (isTRUE(attr(h, "inferred_boundaries"))) " Older-bundle boundaries are inferred from the configured epochs per fit." else "") else
      "One fit is shown. A dashed checkpoint line, when present, marks the selected epoch; it does not stop training."
  })
  output$wb_chart_model <- shiny::renderText({
    if (!is.null(result())) paste0("Showing selected model: Run ",
      if (is.null(selected_run())) "(loaded)" else selected_run(), " · ", wb_checkpoint_summary(result())) else
      if (!is.null(job())) "Live training loss" else if (!is.null(latest_attempt()) && nrow(plot_history()))
        paste0("Training loss from Run ", latest_attempt(), " · ", attempts()[[latest_attempt()]]$state,
          " · loss curve retained for inspection.") else NULL
  })
  saved_inspected_point <- shiny::reactive({
    h <- saved_plot_history()
    if (is.null(input$wb_saved_loss_hover) || !nrow(h)) return(NULL)
    point <- shiny::nearPoints(h, input$wb_saved_loss_hover, xvar = "epoch", yvar = "loss",
      threshold = 20, maxpoints = 1)
    if (nrow(point)) point else NULL
  })
  output$wb_saved_loss_inspection <- shiny::renderText({
    point <- saved_inspected_point()
    if (is.null(point)) return(NULL)
    sprintf("Fit %s · epoch %s within fit · logged position %s · training loss: %.8g", point$fit, point$fit_epoch, point$epoch, point$loss)
  })
  output$wb_saved_loss_marker <- shiny::renderUI({
    point <- saved_inspected_point(); hover <- input$wb_saved_loss_hover
    if (is.null(point) || is.null(hover$domain) || is.null(hover$range)) return(NULL)
    x <- (point$epoch - hover$domain$left) / (hover$domain$right - hover$domain$left) *
      (hover$range$right - hover$range$left) + hover$range$left
    y <- (point$loss - hover$domain$bottom) / (hover$domain$top - hover$domain$bottom) *
      (hover$range$top - hover$range$bottom) + hover$range$bottom
    ratio <- hover$img_css_ratio
    if (is.null(ratio)) ratio <- list(x = 1, y = 1)
    shiny::div(class = "wb-loss-dot", style = sprintf("left:%.2fpx;top:%.2fpx;", x / ratio$x, y / ratio$y),
      shiny::span(class = "wb-loss-tooltip",
        style = if (x > (hover$range$left + hover$range$right) / 2)
          "left:auto;right:0;transform:none;" else "left:0;transform:none;",
        sprintf("Fit %s · epoch %s · Loss %.6g", point$fit, point$fit_epoch, point$loss)))
  })
  inspected_point <- shiny::reactive({
    h <- plot_history()
    if (is.null(input$wb_loss_hover) || !nrow(h)) return(NULL)
    point <- shiny::nearPoints(h, input$wb_loss_hover, xvar = "epoch", yvar = "loss",
      threshold = 20, maxpoints = 1)
    if (nrow(point)) point else NULL
  })
  output$wb_loss_inspection <- shiny::renderText({
    point <- inspected_point()
    if (is.null(point)) return(NULL)
    sprintf("Fit %s · epoch %s within fit · logged position %s · training loss: %.8g", point$fit, point$fit_epoch, point$epoch, point$loss)
  })
  output$wb_loss_marker <- shiny::renderUI({
    point <- inspected_point(); hover <- input$wb_loss_hover
    if (is.null(point) || is.null(hover$domain) || is.null(hover$range)) return(NULL)
    x <- (point$epoch - hover$domain$left) / (hover$domain$right - hover$domain$left) *
      (hover$range$right - hover$range$left) + hover$range$left
    y <- (point$loss - hover$domain$bottom) / (hover$domain$top - hover$domain$bottom) *
      (hover$range$top - hover$range$bottom) + hover$range$bottom
    ratio <- hover$img_css_ratio
    if (is.null(ratio)) ratio <- list(x = 1, y = 1)
    shiny::div(class = "wb-loss-dot", style = sprintf("left:%.2fpx;top:%.2fpx;", x / ratio$x, y / ratio$y),
      shiny::span(class = "wb-loss-tooltip",
        style = if (x > (hover$range$left + hover$range$right) / 2)
          "left:auto;right:0;transform:none;" else "left:0;transform:none;",
        sprintf("Fit %s · epoch %s · Loss %.6g", point$fit, point$fit_epoch, point$loss)))
  })
  output$wb_loss_values <- shiny::renderTable(plot_history(), digits = 8)
  loss_plot <- shiny::reactive({
    shiny::validate(shiny::need(nrow(plot_history()) > 0, "No recorded training loss is available."))
    r <- result()
    wb_loss_plot(plot_history(), if (!is.null(r)) r$config else NULL,
      if (!is.null(r)) r$selected_epoch else NULL)
  })
  output$wb_loss <- shiny::renderPlot(loss_plot())
  output$wb_loss_png <- shiny::downloadHandler(filename = function() "ddesonn-training-loss.png",
    content = function(file) ggplot2::ggsave(file, loss_plot(), device = "png", width = 12, height = 5, dpi = 160, bg = "white"))
  output$wb_loss_svg <- shiny::downloadHandler(filename = function() "ddesonn-training-loss.svg",
    content = function(file) ggplot2::ggsave(file, loss_plot(), device = grDevices::svg, width = 12, height = 5, bg = "white"))
  output$wb_loss_csv <- shiny::downloadHandler(filename = function() "ddesonn-training-loss.csv",
    content = function(file) utils::write.csv(plot_history(), file, row.names = FALSE))
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
  output$wb_export_filename_ui <- shiny::renderUI({
    r <- active_model(); if (is.null(r)) return(NULL)
    shiny::textInput("wb_export_filename", "Save model as", value = wb_export_filename(NULL, r$model_name), width = "100%")
  })
  output$wb_save <- shiny::downloadHandler(filename = function() wb_export_filename(input$wb_export_filename, active_model()$model_name),
    content = function(file) { shiny::req(active_model()); saveRDS(active_model(), file) })
  output$wb_save_selected <- shiny::downloadHandler(filename = function() "ddesonn-workbench-model.rds",
    content = function(file) { shiny::req(result()); saveRDS(result(), file) })
  output$wb_predictions_download <- shiny::downloadHandler(filename = function() "ddesonn-evaluated-predictions.csv",
    content = function(file) utils::write.csv(evaluation()$predictions, file, row.names = FALSE))
  output$wb_new_chart <- shiny::renderPlot({
    predictions <- new_predictions(); shiny::req(predictions, active_model())
    if (!is.null(prediction_report()$metrics)) {
      if (active_model()$prepared$mode == "regression") {
        return(ggplot2::ggplot(predictions, ggplot2::aes(x = actual, y = prediction)) +
          ggplot2::geom_point(color = "#087f8c") + ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) +
          ggplot2::labs(title = "Actual vs predicted — uploaded CSV", x = "Supplied actual value", y = "Predicted value") + ggplot2::theme_minimal(base_size = 12))
      }
      confusion <- as.data.frame(table(Actual = factor(predictions$actual, levels = active_model()$prepared$classes),
        Predicted = factor(predictions$predicted, levels = active_model()$prepared$classes)))
      return(ggplot2::ggplot(confusion, ggplot2::aes(x = Predicted, y = Actual, fill = Freq)) +
        ggplot2::geom_tile() + ggplot2::geom_text(ggplot2::aes(label = Freq)) +
        ggplot2::scale_fill_gradient(low = "#eaf5f4", high = "#087f8c") +
        ggplot2::labs(title = "Confusion matrix — uploaded CSV", subtitle = "Diagonal counts are correct predictions; off-diagonal counts are errors") +
        ggplot2::theme_minimal(base_size = 12))
    }
    if (identical(active_model()$prepared$mode, "regression")) {
      ggplot2::ggplot(predictions, ggplot2::aes(x = prediction)) +
        ggplot2::geom_histogram(bins = min(30L, nrow(predictions)), fill = "#087f8c", color = "white") +
        ggplot2::labs(title = "Predicted values", x = "Prediction", y = "Number of rows",
          subtitle = "All prediction rows; this is not a comparison with known outcomes") +
        ggplot2::theme_minimal(base_size = 12)
    } else {
      counts <- data.frame(Class = factor(active_model()$prepared$classes, levels = active_model()$prepared$classes),
        Rows = as.integer(table(factor(predictions$predicted, levels = active_model()$prepared$classes))))
      ggplot2::ggplot(counts, ggplot2::aes(x = Class, y = Rows)) +
        ggplot2::geom_col(fill = "#087f8c") +
        ggplot2::geom_text(ggplot2::aes(label = Rows), vjust = -.4) +
        ggplot2::scale_x_discrete(drop = FALSE) +
        ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, .12))) +
        ggplot2::labs(title = "Predicted class counts", x = "Predicted class", y = "Number of rows",
          subtitle = "All prediction rows; counts do not measure accuracy") +
        ggplot2::theme_minimal(base_size = 12)
    }
  })
  output$wb_new_preview <- shiny::renderTable({ shiny::req(new_predictions()); utils::head(new_predictions(), 20) }, digits = 6)
  output$wb_new_download <- shiny::downloadHandler(filename = function() "ddesonn-new-predictions.csv",
    content = function(file) { shiny::req(new_predictions()); utils::write.csv(new_predictions(), file, row.names = FALSE) })
  # Returned reactives make session state testable without exporting app internals.
  invisible(list(prepared = prepared, result = result, job = job, status = status,
    completed_runs = completed_runs, selected_run = selected_run, attempts = attempts))
}
