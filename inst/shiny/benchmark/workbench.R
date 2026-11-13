# Native disclosure controls keep help available without another JavaScript package.
wb_help <- function(topic, text, api = NULL) {
  shiny::tags$details(class = "wb-inline-help",
    shiny::tags$summary(title = text, `aria-label` = paste("Help:", topic),
      shiny::tags$svg(class = "wb-help-icon", viewBox = "0 0 20 20", width = "20", height = "20",
        `aria-hidden` = "true", focusable = "false",
        shiny::tags$circle(class = "wb-help-ring", cx = "10", cy = "10", r = "8.4"),
        shiny::tags$text(class = "wb-help-letter", x = "10", y = "14.25", `text-anchor` = "middle", "i"))),
    shiny::div(class = "wb-help-copy", tabindex = "0", role = "region",
      `aria-label` = paste(topic, "help and equivalent R"), text,
      shiny::div(class = "wb-help-code-label", "Equivalent R"),
      shiny::tags$pre(id = wb_help_key(topic), class = "shiny-text-output wb-help-code",
        tabindex = "0", `aria-label` = paste("Equivalent R code:", topic),
        if (is.null(api)) "# Open help to load the equivalent R example." else api),
      shiny::p(class = "wb-help-prerequisites", "This fragment covers only this topic. Show full R code at the bottom of the page for the complete workflow and required objects.")))
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
    shiny::div(class = "panel-body-custom",
      wb_heading("Model performance", "Performance measures how well the completed model predicts known outcomes. These results use its saved data split and preprocessing, not any new selections on Features."),
      shiny::p(class = "wb-guidance", "Finally, review results and outputs. Use validation to compare your choices, and reserve test results for your final assessment."),
      wb_field("wb_eval_split", "Evaluation split", "Training measures fit to examples the model learned from. Validation helps you assess settings. Test is held aside for a final assessment; repeatedly choosing settings from it weakens that assessment.",
        shiny::selectInput("wb_eval_split", NULL, c("Test" = "test", "Validation" = "validation", "Training" = "train"))),
      shiny::textOutput("wb_selected_model"),
      shiny::textOutput("wb_checkpoint_results"),
      shiny::tableOutput("wb_metrics"),
      shiny::tags$details(shiny::tags$summary("Training-seed validation comparison and member weights"), shiny::tableOutput("wb_seed_comparison"), shiny::verbatimTextOutput("wb_member_weights")),
      wb_code_panel("wb_evaluation_code", "Equivalent R code — displayed results"),
      wb_heading("Confusion matrix (classification)", "Rows are actual classes and columns are predicted classes. Counts on the diagonal are correct predictions; the other cells show which classes were confused.", level = 4),
      shiny::tableOutput("wb_confusion"), shiny::plotOutput("wb_diagnostics", height = "300px"),
      shiny::div(class = "wb-export-right", wb_download_button("wb_predictions_download", "Download evaluated predictions"),
      shiny::tags$details(shiny::tags$summary("Preview evaluated CSV (first 20 rows)"), shiny::div(class = "scroll-table", shiny::tableOutput("wb_predictions_preview"))))))
  models <- shiny::tabPanel("Saved models & predictions", value = "models",
    shiny::fluidRow(
      shiny::column(6, shiny::div(class = "panel-body-custom",
        wb_heading("Save or restore", "A model bundle contains the completed model plus the settings and preprocessing it needs to make consistent predictions later."),
        shiny::textOutput("wb_saved_info"), wb_download_button("wb_save", "Download model bundle (.rds)"),
        shiny::br(), shiny::br(), shiny::uiOutput("wb_completed_recipe_button"),
        wb_code_panel("wb_completed_code", "Equivalent R code — completed model"),
        shiny::actionButton("wb_preview_bundle", "Preview bundle contents"),
        shiny::fileInput("wb_load", "Load trusted workbench model (.rds, up to 50 MB)", accept = ".rds"),
        shiny::p("Load only workbench bundles from sources you trust. Other RDS model formats are not supported."),
        wb_heading("Another fit from saved weights", "This starts another training run using the saved weights, data and settings. Optimizer accumulators and epoch schedules restart, so it is a warm start rather than an exact resume."),
        wb_field("wb_more_epochs", "Additional epochs", "Run another fit from saved weights. The saved settings and preprocessing are retained; optimizer schedules restart.",
          shiny::numericInput("wb_more_epochs", NULL, 10, min = 1, max = 5000, step = 1)),
        shiny::actionButton("wb_continue", "Continue from saved weights", disabled = "disabled", `aria-disabled` = "true"),
        shiny::p("Uses the original data, preprocessing and settings, without repeating initial self-organization. Changes on Build & Train are ignored here."))),
      shiny::column(6, shiny::div(class = "panel-body-custom",
        wb_heading("Predict on new rows", "Use the completed model to predict outcomes for rows it has not seen. Include the saved predictor columns; the target column is optional. Training preprocessing is reused."),
        shiny::fileInput("wb_new_data", "New data (CSV, up to 10 MB)", accept = ".csv"),
        shiny::p("Optional packaged prediction example: customer rows without group labels. This is separate from the training, validation and test split; you do not need it to train or evaluate a model."),
        shiny::actionButton("wb_example_predict", "Predict customer groups with completed model"),
        shiny::downloadButton("wb_prediction_example_download", "Download customer prediction CSV"),
        wb_vignette_link("wb_prediction_vignettes", "Packaged-data and prediction guide in Vignettes"),
        wb_download_button("wb_new_download", "Download new predictions"),
        wb_code_panel("wb_model_code", "R example — load a saved model and predict new rows"),
        shiny::tags$details(shiny::tags$summary("Preview new-prediction CSV (first 20 rows)"), shiny::div(class = "scroll-table", shiny::tableOutput("wb_new_preview"))),
        shiny::div(class = "error", role = "alert", shiny::textOutput("wb_model_error"))))))
  list(
    shiny::tabPanel("1. Data", value = "data",
      shiny::sidebarLayout(
        shiny::sidebarPanel(
          wb_heading("Start with Data", "Choose a labeled dataset or upload a CSV with known target values. Shiny divides this one dataset into training, validation and test sets on Features. Each row is an observation and each column is a variable. Loading data does not train a model. Choose the target and predictors next, on Features."),
          shiny::selectInput("wb_example", "Packaged training example", wb_example_choices(), selected = "iris", selectize = FALSE),
          shiny::div(class = "wb-guidance", shiny::textOutput("wb_example_note")),
          shiny::actionButton("wb_demo", "Load selected example"),
          shiny::downloadButton("wb_example_download", "Download example CSV"),
          wb_vignette_link("wb_data_vignettes", "Packaged-data guide in Vignettes"),
          shiny::tags$a(href = "https://github.com/MatHatter/DDESONN/tree/main/inst/extdata", target = "_blank", rel = "noopener noreferrer", "Browse dataset source on GitHub"),
          shiny::fileInput("wb_data", "Or upload training data (CSV, up to 10 MB)", accept = ".csv"),
          shiny::p(class = "wb-guidance", "Start with Data. Inspect the rows, then choose the target and predictors in Features."),
          shiny::div(class = "wb-action-right", shiny::actionButton("wb_next_features", "Next: choose features", class = "btn-primary"))),
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
          wb_field("wb_train_fraction", "Training fraction", "Fraction of rows allocated to training. Classification random splits allocate within each class.",
            shiny::numericInput("wb_train_fraction", NULL, .6, min = .1, max = .9, step = .05)),
          wb_field("wb_val_fraction", "Validation fraction", "Fraction of rows held for validation; remaining rows become Test. Test is not used to select checkpoints.",
            shiny::numericInput("wb_val_fraction", NULL, .2, min = .05, max = .4, step = .05)),
          shiny::tags$details(class = "wb-settings-section", shiny::tags$summary("Reproducibility"),
          wb_field("wb_seed", "Data-split seed", "Controls repeatable random row splitting. Ordered splits keep row order. Basic reuses this value for its one training run; Custom uses the separate Training-run seed values on Build & Train. A value of 42 is one reproducibility seed, not 42 runs.",
            shiny::numericInput("wb_seed", NULL, 42, min = 0, max = 2147483646, step = 1)),
          shiny::p(class = "wb-guidance", "42 is a seed value, not a run count. Basic reuses it for one training run. Custom has a separate list of training-run seeds on Build & Train.")),
          wb_switch("wb_scale", "Standardize using training means and SDs",
            "Subtracts each predictor's training average and divides by its training standard deviation (SD, a measure of spread). This puts inputs on comparable scales. Validation, test and new rows reuse those same training values; the target is unchanged.", TRUE,
            api = paste("center <- colMeans(x_train)", "spread <- apply(x_train, 2, sd)",
              "spread[!is.finite(spread) | spread == 0] <- 1",
              "x_train_scaled <- sweep(sweep(x_train, 2, center, '-'), 2, spread, '/')", sep = "\n")),
          shiny::div(class = "wb-action-right", shiny::actionButton("wb_prepare", "Prepare data", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true"))),
        shiny::mainPanel(
          shiny::div(class = "panel-body-custom",
            wb_heading("Prepare for training", "Prepare data validates your choices, creates the three row splits, and fits any scaling using training rows only. The same scaling is then reused for validation, test and new data."),
            shiny::p(class = "wb-guidance", "Next, choose the target and predictors. Click Prepare data to continue to Build & Train."),
            shiny::textOutput("wb_prepared"),
            shiny::p("Target + predictors determine what the model learns from. Task determines its output and evaluation metrics. Changing any selection here requires Prepare data again."),
            wb_code_panel("wb_features_code", "Equivalent R code — prepared split and scaling")),
          shiny::div(class = "panel-body-custom",
            wb_heading("About your split", "Random classification splits keep examples from each class in each split. Ordered splitting preserves row order, which can be useful for time-based data. Neither option groups duplicate inputs or related subjects automatically."),
            shiny::p("Preprocessing is fitted only on training rows. Prepare grouped data separately when related subjects or duplicate inputs must stay together."),
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
              wb_switch("wb_diagnostics", "Detailed training diagnostics", "Off shows ordinary epoch progress and genuine warnings/errors. On also emits detailed tensor, gradient/error, optimizer and training summaries. Read the Training log disclosure under the chart: it shows the most recent 2,000 lines, retained with completed bundles. These are diagnostic logs, not extra training, a model-quality guarantee or a separate file export."))),
          shiny::conditionalPanel("input.wb_workflow === 'custom'", shiny::div(class = "wb-workload-explanation", shiny::textOutput("wb_workload"))),
          wb_vignette_link("wb_training_vignettes", "Training and plotting guides in Vignettes"),
          shiny::div(class = "wb-training-actions",
            shiny::actionButton("wb_cancel", "Cancel training", disabled = "disabled", `aria-disabled` = "true"),
            shiny::actionButton("wb_start", "Train new model", class = "btn-primary", disabled = "disabled", `aria-disabled` = "true"))),
        shiny::mainPanel(
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
            wb_heading("Current completed model", "Select a successful training run or loaded model below. Its saved settings, chart, results and downloads follow your selection. Failed or cancelled attempts are kept separately in Attempt history. Changing controls affects the next new model, not this completed model."),
            shiny::conditionalPanel("output.wb_has_model === 'yes'", shiny::p("Select a completed run to show its settings, chart and results.")),
            shiny::uiOutput("wb_run_choices"),
            shiny::textOutput("wb_settings_origin"),
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
      shiny::tabsetPanel(id = "wb_results_tabs", performance, models))
  )
}

wb_server <- function(input, output, session, helper) {
  raw <- shiny::reactiveVal(datasets::iris)
  data_source <- shiny::reactiveVal("Included iris example · 150 rows")
  data_origin <- shiny::reactiveVal(list(example = "iris"))
  dataset_preset <- shiny::reactiveVal(wb_example("iris"))
  prepared <- shiny::reactiveVal(NULL)
  result <- shiny::reactiveVal(NULL)
  completed_runs <- shiny::reactiveVal(list())
  selected_run <- shiny::reactiveVal(NULL)
  attempts <- shiny::reactiveVal(list())
  latest_attempt <- shiny::reactiveVal(NULL)
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
    new_predictions(NULL); model_error(NULL)
    sync_controls(r$config, r$prepared$mode)
    session$sendCustomMessage("wb-clear-loss-hover", list())
  }
  add_completed <- function(bundle) {
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
  # Per-session states are also enforced by server guards, not just the browser.
  action_states <- shiny::reactive({
    running <- !is.null(job()); completed <- !is.null(result())
    list(wb_start = !is.null(prepared()) && !running, wb_cancel = running,
      wb_continue = completed && !running, wb_save = completed,
      wb_review_results = completed, wb_predictions_download = completed,
      wb_new_download = !is.null(new_predictions()), wb_new_data = completed,
      wb_load = !running,
      wb_prepare = !running && length(input$wb_target) == 1 && nzchar(input$wb_target) && input$wb_target %in% names(raw()) &&
        length(input$wb_features) > 0 && !input$wb_target %in% input$wb_features,
      wb_data = !running, wb_demo = !running, wb_example = !running,
      wb_example_predict = completed && !running)
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
    preset <- dataset_preset()
    target <- if (is.null(preset)) tail(cols, 1) else if (is.null(preset$target)) "" else preset$target
    shiny::tagList(
      wb_field("wb_target", "Target", "The target is the variable you want the model to predict, such as a class label or a numeric outcome. It is kept out of the predictors.",
        shiny::selectInput("wb_target", NULL, c("Choose a target" = "", stats::setNames(cols, cols)), selected = target)),
      wb_field("wb_features", "Predictors", "Predictors are the input features used to predict the target. Select numeric columns available when you will make future predictions; do not include the target or information derived from the answer.",
        shiny::selectizeInput("wb_features", NULL, cols, selected = wb_default_features(raw(), target, preset), multiple = TRUE)))
  })
  for (id in c("wb_data_vignettes", "wb_features_vignettes", "wb_training_vignettes", "wb_prediction_vignettes", "wb_experiment_vignettes")) local({
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
      checkpoint = if (is.null(input$wb_checkpoint)) "best_validation" else input$wb_checkpoint)
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
    lines <- if (!is.null(job())) read_job_log(job()) else if (!is.null(result())) result()$training_log else if (!is.null(latest_attempt()))
      attempts()[[latest_attempt()]]$training_log else character()
    paste(tail(lines, 2000L), collapse = "\n")
  })
  output$wb_seed_comparison <- shiny::renderTable({ shiny::req(result()); result()$seed_comparison })
  output$wb_member_weights <- shiny::renderPrint({ shiny::req(result()); result()$member_weights })
  shiny::observeEvent(input$wb_target, {
    shiny::updateSelectizeInput(session, "wb_features", selected = wb_default_features(raw(), input$wb_target, dataset_preset()))
  })
  shiny::observeEvent(input$wb_data, attempt({
    busy(); candidate <- read_csv(input$wb_data)
    if (nrow(candidate) < 10 || anyDuplicated(names(candidate))) stop("Use at least 10 rows and unique column names.")
    dataset_preset(NULL); raw(candidate); prepared(NULL)
    data_origin(list(file = basename(input$wb_data$name)))
    data_source(paste("Uploaded:", basename(input$wb_data$name)))
  }))
  example_id <- shiny::reactive(if (is.null(input$wb_example)) "iris" else input$wb_example)
  output$wb_example_note <- shiny::renderText(wb_example_catalog()[[example_id()]]$note)
  shiny::observeEvent(input$wb_demo, attempt({
    busy(); example <- wb_example(example_id())
    if (isTRUE(example$prediction_only)) stop("Use unlabeled examples under Predict on new rows, after training a compatible model.")
    dataset_preset(example); raw(example$data); prepared(NULL); data_origin(example$source)
    data_source(paste0("Included ", example$label, " · ", nrow(example$data), " rows"))
    if (!is.null(example$mode)) shiny::updateSelectInput(session, "wb_mode", selected = example$mode)
    if (!is.null(example$split)) shiny::updateSelectInput(session, "wb_split", selected = example$split)
  }))
  output$wb_example_download <- shiny::downloadHandler(
    filename = function() paste0("DDESONN-", example_id(), ".csv"),
    content = function(file) wb_write_example(example_id(), file))
  output$wb_prediction_example_download <- shiny::downloadHandler(
    filename = function() "DDESONN-customer-prediction.csv",
    content = function(file) wb_write_example("customer_test", file))
  shiny::observeEvent(input$wb_example_predict, attempt({
    busy(); new_predictions(NULL)
    shiny::updateTabsetPanel(session, "workspace", selected = "results")
    shiny::updateTabsetPanel(session, "wb_results_tabs", selected = "models")
    if (is.null(result())) stop("Train or load a compatible model before predicting.")
    new_predictions(wb_new_predictions(result(), wb_example("customer_test")$data))
  }, model_error))
  # Invalidate only preparation. The immutable completed model is still usable.
  signature <- shiny::reactive(list(input$wb_target, input$wb_features, input$wb_mode,
    input$wb_seed, input$wb_train_fraction, input$wb_val_fraction, input$wb_scale, input$wb_split))
  shiny::observeEvent(signature(), prepared(NULL), ignoreInit = TRUE)
  shiny::observeEvent(input$wb_prepare, attempt({
    busy()
    p <- wb_prepare(raw(), input$wb_target, input$wb_features, input$wb_mode,
      input$wb_seed, input$wb_train_fraction, input$wb_val_fraction, input$wb_scale, input$wb_split)
    p$source <- code_source()
    prepared(p)
    if (is.null(latest_attempt())) status("Data prepared. Ready to train a new model.")
    shiny::updateTabsetPanel(session, "workspace", selected = "train")
  }))
  launch <- function(p, cfg, old = NULL) {
    busy()
    id <- begin_attempt(cfg, if (is.null(old)) "New training" else
      paste0("Warm start from Run ", if (is.null(selected_run())) "loaded model" else selected_run()))
    next_job <- wb_start(helper, p, cfg, old)
    next_job$run_id <- id
    job(next_job); history(data.frame(epoch = integer(), loss = numeric()))
    status(paste0("Run ", id, " · Starting training in a separate R process…"))
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
      history(wb_log_history(job()$log))
      finish_attempt("Cancelled")
      wb_stop(job()); job(NULL)
    }
  }))
  shiny::observe({
    shiny::invalidateLater(300, session)
    active <- shiny::isolate(job())
    if (!is.null(active)) {
      h <- wb_log_history(active$log); history(h)
      elapsed <- as.integer(difftime(Sys.time(), active$started, units = "secs"))
      status(sprintf("Run %s · Running · %s seconds · %s of %s epochs logged", active$run_id, elapsed,
        if (nrow(h)) max(h$epoch) else 0L, active$epochs))
      if (!active$process$is_alive()) {
        tryCatch({
          complete <- active$process$get_result()
          complete$history <- h
          complete$training_log <- read_job_log(active)
          complete <- wb_validate_bundle(complete)
          add_completed(complete); error(NULL)
          finish_attempt("Completed")
        }, error = function(e) {
          finish_attempt("Failed", wb_training_error(e, if (file.exists(active$error))
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
    begin_attempt(candidate$config, "Loaded saved model")
    add_completed(candidate); finish_attempt("Completed")
  }, model_error))
  shiny::observeEvent(input$wb_new_data, attempt({
    new_predictions(NULL)
    if (is.null(result())) stop("Train or load a model before predicting.")
    new_predictions(wb_new_predictions(result(), read_csv(input$wb_new_data)))
  }, model_error))
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
    paste0(if (!is.null(selected_run())) paste0("Run ", selected_run(), " · ") else "", info)
  })
  # Each disclosure is an independent, lazily rendered code output. Reading help
  # never launches a fit or mutates a completed model.
  for (topic in wb_help_topics()) local({
    selected_topic <- topic
    output[[wb_help_key(selected_topic)]] <- shiny::renderText({
      wb_help_code(selected_topic, prepared = prepared(),
        config = tryCatch(next_config(), error = function(e) NULL),
        completed = result(), source = code_source(),
        evaluation_split = if (is.null(input$wb_eval_split)) "test" else input$wb_eval_split,
        more_epochs = if (is.null(input$wb_more_epochs)) 10L else input$wb_more_epochs)
    })
  })
  full_choice <- shiny::reactiveVal("next")
  shiny::observeEvent(input$wb_full_source, full_choice(input$wb_full_source))
  full_script <- shiny::reactive({
    selected <- full_choice()
    tryCatch(wb_full_code(prepared(), tryCatch(next_config(), error = function(e) NULL), result(), selected),
      error = function(e) { shiny::validate(shiny::need(FALSE, conditionMessage(e))) })
  })
  show_full_code <- function(choice = NULL) {
    choices <- c("Next model / iris illustration" = "next")
    if (!is.null(result()$recipe)) choices <- c(choices, "Completed model's saved recipe" = "completed")
    choices <- c(choices, "R example: load a saved model and predict new rows" = "prediction")
    selected <- if (!is.null(result()$recipe)) "completed" else if (!is.null(result())) "prediction" else "next"
    if (!is.null(choice)) selected <- choice
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
    shiny::validate(shiny::need(!is.null(result()), "Train or load a completed model to preview its bundle."))
    wb_bundle_manifest(result())
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
      shiny::tagList(wb_download_info("wb_next_recipe", shiny::downloadButton("wb_next_recipe", "Download next-model R recipe")), shiny::actionButton("wb_preview_next_recipe", "Preview recipe"))
  })
  output$wb_completed_recipe_button <- shiny::renderUI({
    if (!is.null(result()$recipe)) shiny::tagList(wb_download_info("wb_completed_recipe", shiny::downloadButton("wb_completed_recipe", "Download completed-model R recipe")), shiny::actionButton("wb_preview_completed_recipe", "Preview saved recipe"))
  })
  output$wb_next_recipe <- shiny::downloadHandler(filename = function() "ddesonn-next-model.R",
    content = function(file) { shiny::req(prepared()); writeLines(wb_recipe(prepared(), next_config()), file) })
  output$wb_completed_recipe <- shiny::downloadHandler(filename = function() "ddesonn-completed-model.R",
    content = function(file) { shiny::req(result()$recipe); writeLines(result()$recipe, file) })
  output$wb_run_choices <- shiny::renderUI({
    runs <- completed_runs()
    if (!length(runs)) return(shiny::p("No completed models yet. Train a model or load a saved bundle."))
    ids <- rev(names(runs)); records <- shiny::isolate(attempts())
    shiny::radioButtons("wb_selected_run", "Completed runs", width = "100%", selected = selected_run(),
      choiceValues = ids, choiceNames = lapply(ids, function(id) shiny::tagList(
        shiny::strong(paste0("Run ", id, " · ", records[[id]]$kind)),
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
  output$wb_saved_info <- shiny::renderText(model_info())
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
  output$wb_preview <- shiny::renderTable(utils::head(raw(), 10), rownames = FALSE)
  output$wb_prepared <- shiny::renderText({
    p <- prepared()
    if (is.null(p)) return("Not prepared. Choose your columns and task, then click Prepare data.")
    counts <- vapply(p$parts, function(x) nrow(x$x), integer(1))
    paste(paste(sprintf("%s: %s rows", names(counts), counts), collapse = " · "),
      if (p$mode == "binary") paste("· positive class:", p$classes[2]) else "")
  })
  output$wb_status <- shiny::renderText(status())
  output$wb_running_status <- shiny::renderText(status())
  output$wb_attempt_title <- shiny::renderText(if (is.null(latest_attempt())) "Training status" else "Latest attempt")
  output$wb_running_flag <- shiny::renderText(if (!is.null(job())) "yes" else "no")
  output$wb_error_flag <- shiny::renderText(if (!is.null(training_error())) "yes" else "no")
  output$wb_has_model <- shiny::renderText(if (!is.null(result())) "yes" else "no")
  output$wb_chart_ready <- shiny::renderText(if (nrow(plot_history()) > 0) "yes" else "no")
  output$wb_has_logs <- shiny::renderText({
    lines <- if (!is.null(job())) TRUE else if (!is.null(result())) length(result()$training_log) > 0 else {
      id <- latest_attempt()
      !is.null(id) && length(attempts()[[id]]$training_log) > 0
    }
    if (lines) "yes" else "no"
  })
  for (flag in c("wb_running_flag", "wb_error_flag", "wb_has_model", "wb_chart_ready", "wb_has_logs"))
    shiny::outputOptions(output, flag, suspendWhenHidden = FALSE)
  output$wb_progress <- shiny::renderUI({
    active <- job(); shiny::req(active)
    h <- history(); n <- if (nrow(h)) max(h$epoch) else 0
    shiny::tags$progress(value = n, max = active$epochs, style = "width:100%;", `aria-label` = "Logged member epochs")
  })
  plot_history <- shiny::reactive({
    r <- result()
    if (!is.null(r)) {
      if (is.data.frame(r$history)) r$history else data.frame(epoch = integer(), loss = numeric())
    } else if (!is.null(job())) history() else {
      id <- latest_attempt()
      record <- if (is.null(id)) NULL else attempts()[[id]]
      if (is.data.frame(record$history)) record$history else data.frame(epoch = integer(), loss = numeric())
    }
  })
  output$wb_chart_model <- shiny::renderText({
    if (!is.null(result())) paste0("Showing selected model: Run ",
      if (is.null(selected_run())) "(loaded)" else selected_run(), " · ", wb_checkpoint_summary(result())) else
      if (!is.null(job())) "Live training loss" else if (!is.null(latest_attempt()) && nrow(plot_history()))
        paste0("Training loss from Run ", latest_attempt(), " · ", attempts()[[latest_attempt()]]$state,
          " · loss curve retained for inspection.") else NULL
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
    sprintf("Epoch %s · training loss: %.8g", point$epoch, point$loss)
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
        sprintf("Epoch %s · Loss %.6g", point$epoch, point$loss)))
  })
  output$wb_loss_values <- shiny::renderTable(plot_history(), digits = 8)
  output$wb_loss <- shiny::renderPlot({
    h <- plot_history()
    shiny::validate(shiny::need(nrow(h) > 0, if (!is.null(result()))
      "This saved model has no recorded training-loss history." else "Training loss appears as epochs are logged."))
    r <- result()
    selected <- if (!is.null(r)) r$selected_epoch else NULL
    ggplot2::ggplot(h, ggplot2::aes(x = epoch, y = loss)) +
      (if (length(selected) == 1 && is.finite(selected))
        ggplot2::geom_vline(xintercept = selected, linetype = 2, color = "#69788c")) +
      (if (nrow(h) > 1) ggplot2::geom_line(color = "#087f8c")) +
      ggplot2::geom_point(color = "#087f8c") + ggplot2::labs(x = "Logged epoch (cumulative across Custom member fits)", y = "Logged training loss",
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
  output$wb_save_selected <- shiny::downloadHandler(filename = function() "ddesonn-workbench-model.rds",
    content = function(file) { shiny::req(result()); saveRDS(result(), file) })
  output$wb_predictions_download <- shiny::downloadHandler(filename = function() "ddesonn-evaluated-predictions.csv",
    content = function(file) utils::write.csv(evaluation()$predictions, file, row.names = FALSE))
  output$wb_new_preview <- shiny::renderTable({ shiny::req(new_predictions()); utils::head(new_predictions(), 20) }, digits = 6)
  output$wb_new_download <- shiny::downloadHandler(filename = function() "ddesonn-new-predictions.csv",
    content = function(file) { shiny::req(new_predictions()); utils::write.csv(new_predictions(), file, row.names = FALSE) })
  # Returned reactives make session state testable without exporting app internals.
  invisible(list(prepared = prepared, result = result, job = job, status = status,
    completed_runs = completed_runs, selected_run = selected_run, attempts = attempts))
}
