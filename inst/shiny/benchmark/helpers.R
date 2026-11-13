# Helpers shared by the optional app and its tests.
dashboard_vignette_dir <- function(app_dir = getwd()) {
  candidates <- c(file.path(app_dir, "../../../doc"),
    file.path(app_dir, "../../doc"), system.file("doc", package = "DDESONN"))
  for (path in candidates) {
    if (nzchar(path) && dir.exists(path) && length(list.files(path, pattern = "\\.html$")))
      return(normalizePath(path))
  }
  ""
}

dashboard_vignettes <- function(doc_dir = "") {
  guides <- data.frame(
    file = c("heart-failure-standard-adagrad-1000-seeds",
      "plot-controls_scenario1-2_single-run_scenarioA",
      "plot-contols_scenario1_ensemble-runs_scenarioC-D",
      "logs_main-change-movement_ensemble_runs_scenarioD"),
    title = c("New standard-AdaGrad heart-failure benchmark: 1,000 seeds",
      "Plot controls: single-run scenarios 1 and 2",
      "Plot controls: ensemble scenarios C and D",
      "Main, change and movement logs: ensemble scenario D"),
    stringsAsFactors = FALSE)
  guides$local <- nzchar(doc_dir) & file.exists(file.path(doc_dir, paste0(guides$file, ".html")))
  guides$href <- ifelse(guides$local,
    paste0("ddesonn-vignettes/", guides$file, ".html"),
    paste0("https://github.com/MatHatter/DDESONN/blob/main/vignettes/", guides$file, ".Rmd"))
  guides
}

dashboard_vignettes_ui <- function(doc_dir = dashboard_vignette_dir()) {
  guides <- dashboard_vignettes(doc_dir)
  if (any(guides$local)) shiny::addResourcePath("ddesonn-vignettes", doc_dir)
  shiny::tagList(shiny::h3("Vignettes"),
    shiny::p("Read the package guides and research context. Links open in a new tab; online source is used when a guide is not installed."),
    shiny::tags$ul(lapply(seq_len(nrow(guides)), function(i)
      shiny::tags$li(shiny::tags$a(href = guides$href[i], guides$title[i],
        target = "_blank", rel = "noopener noreferrer")))))
}

dashboard_metrics <- c(accuracy = "Accuracy", precision = "Precision",
  recall = "Recall", f1 = "F1", roc_auc = "ROC AUC",
  log_loss = "Log loss", brier = "Brier score")

dashboard_validate <- function(data) {
  if (!is.data.frame(data) || !nrow(data))
    stop("The CSV must contain at least one seed.", call. = FALSE)
  if (anyDuplicated(names(data)))
    stop("Column names must be unique.", call. = FALSE)
  if (!"seed" %in% names(data) || !is.numeric(data$seed) ||
      any(!is.finite(data$seed)) || any(data$seed < 1 | data$seed %% 1 != 0) ||
      anyDuplicated(data$seed))
    stop("Use one row per seed, with unique positive integer seed IDs.", call. = FALSE)
  available <- character()
  for (metric in names(dashboard_metrics)) {
    columns <- paste0(metric, c("_legacy", "_corrected"))
    present <- columns %in% names(data)
    if (any(present) && !all(present))
      stop(sprintf("Include both %s and %s.", columns[1], columns[2]), call. = FALSE)
    if (all(present)) {
      for (column in columns) {
        value <- data[[column]]
        if (!is.numeric(value) || any(!is.finite(value)) || any(value < 0) ||
            (metric != "log_loss" && any(value > 1)))
          stop(sprintf("%s must contain finite %s values.", column,
            if (metric == "log_loss") "nonnegative" else "values between 0 and 1"),
            call. = FALSE)
      }
      available <- c(available, metric)
    }
  }
  if (!length(available)) stop("Include at least one supported metric pair.", call. = FALSE)
  # Recompute differences; never trust possibly stale uploaded difference columns.
  for (metric in available)
    data[[paste0(metric, "_difference")]] <-
      data[[paste0(metric, "_corrected")]] - data[[paste0(metric, "_legacy")]]
  data <- data[order(data$seed), , drop = FALSE]
  attr(data, "metrics") <- available
  data
}

dashboard_summary <- function(data) {
  metrics <- names(dashboard_metrics)[vapply(names(dashboard_metrics), function(m)
    all(paste0(m, c("_legacy", "_corrected")) %in% names(data)), logical(1))]
  do.call(rbind, lapply(metrics, function(m) {
    legacy <- data[[paste0(m, "_legacy")]]
    standard <- data[[paste0(m, "_corrected")]]
    delta <- standard - legacy
    advantage <- delta * if (m %in% c("log_loss", "brier")) -1 else 1
    tolerance <- 1e-12
    data.frame(Metric = unname(dashboard_metrics[m]), Seeds = nrow(data),
      Legacy = mean(legacy), Standard = mean(standard),
      Difference = mean(delta), Difference_SD = stats::sd(delta),
      Improved = sum(advantage > tolerance), Tied = sum(abs(advantage) <= tolerance),
      Worsened = sum(advantage < -tolerance), check.names = FALSE)
  }))
}

dashboard_server <- function(default_data) {
  force(default_data)
  function(input, output, session) {
    shiny::observeEvent(input$brain_art, {
      dialog <- shiny::modalDialog(
        title = "DDESONN brain artwork", size = "l", easyClose = TRUE,
        shiny::tags$img(src = "DDESONN_brain.png", class = "brain-full",
          alt = "DDESONN brain artwork with glowing neural connections and equations"),
        shiny::p("Project artwork; not a diagram of the selected benchmark model."),
        shiny::tags$a(href = "DDESONN_brain.png", target = "_blank", rel = "noopener",
          "Open full-resolution image"),
        footer = shiny::modalButton("Close"))
      dialog$attribs$role <- "dialog"
      dialog$attribs$`aria-label` <- "DDESONN brain artwork"
      shiny::showModal(dialog)
    })
    explain <- function() shiny::showModal(shiny::modalDialog(
      title = "Understand this comparison", easyClose = TRUE,
      shiny::p("Legacy AdaGrad is DDESONN's earlier custom implementation. Its multilayer path applied a direct weight update followed by an adaptive update to the same weights. Historical experiments reported strong results, but the update rule differed from classic AdaGrad."),
      shiny::p("Standard AdaGrad is the classic algorithm: one adaptive update per parameter tensor per step. It replaced the legacy behavior in version 7.3.0."),
      shiny::p("Mean is the average of the selected metric over the selected seed runs. The included record covers all 1,000 new standard-AdaGrad seeds compared descriptively with archived legacy observations. The historical split has repeated inputs across training and testing; correcting AdaGrad does not remove that overlap or establish a fully matched Keras comparison."),
      footer = shiny::modalButton("Close")))
    shiny::observeEvent(input$legacy_help, explain())
    shiny::observeEvent(input$standard_help, explain())
    current <- shiny::reactiveVal(default_data)
    source_name <- shiny::reactiveVal("Included heart-failure benchmark · seeds 1–1,000")
    upload_error <- shiny::reactiveVal(NULL)
    shiny::observeEvent(input$upload, {
      tryCatch({
        shiny::req(input$upload$datapath)
        if (input$upload$size > 10 * 1024^2) stop("CSV limit: 10 MB.")
        data <- utils::read.csv(input$upload$datapath, check.names = FALSE)
        data <- dashboard_validate(data)
        current(data)
        source_name(paste("Uploaded:", basename(input$upload$name)))
        upload_error(NULL)
      }, error = function(e) upload_error(conditionMessage(e)))
    })
    shiny::observeEvent(input$reset, {
      current(default_data)
      source_name("Included heart-failure benchmark · seeds 1–1,000")
      upload_error(NULL)
    })
    output$upload_error <- shiny::renderText(upload_error())
    output$source <- shiny::renderText(source_name())
    output$seed_filter <- shiny::renderUI({
      seeds <- current()$seed
      shiny::sliderInput("seeds", "Seed range", min = min(seeds),
        max = max(c(seeds, min(seeds) + 1)), value = range(seeds), step = 1,
        sep = "")
    })
    output$metric_filter <- shiny::renderUI({
      metrics <- attr(current(), "metrics")
      shiny::selectInput("metric", "Metric", choices = stats::setNames(metrics,
        unname(dashboard_metrics[metrics])), selected = metrics[1])
    })
    selected <- shiny::reactive({
      data <- current()
      if (!is.null(input$seeds)) data <- data[data$seed >= input$seeds[1] &
        data$seed <= input$seeds[2], , drop = FALSE]
      shiny::validate(shiny::need(nrow(data) > 0, "No seeds in this range."))
      data
    })
    metric <- shiny::reactive({
      available <- attr(current(), "metrics")
      if (!is.null(input$metric) && input$metric %in% available) input$metric else available[1]
    })
    summary <- shiny::reactive(dashboard_summary(selected()))
    output$cards <- shiny::renderUI({
      data <- selected(); m <- metric()
      delta <- mean(data[[paste0(m, "_difference")]])
      scale <- if (m == "accuracy") 100 else 1
      unit <- if (m == "accuracy") " percentage points" else ""
      shiny::div(class = "cards",
        shiny::div(class = "stat", shiny::span("SELECTED SEEDS"), shiny::strong(nrow(data))),
        shiny::div(class = "stat", shiny::actionLink("legacy_help", "LEGACY ADAGRAD MEAN"),
          shiny::strong(sprintf("%.6f", mean(data[[paste0(m, "_legacy")]])))),
        shiny::div(class = "stat", shiny::actionLink("standard_help", "STANDARD ADAGRAD MEAN"),
          shiny::strong(sprintf("%.6f", mean(data[[paste0(m, "_corrected")]])))),
        shiny::div(class = "stat", shiny::span("STANDARD − LEGACY"),
          shiny::strong(sprintf("%+.6f", delta * scale)), shiny::tags$small(unit)))
    })
    output$summary <- shiny::renderTable(summary(), digits = 8, striped = TRUE,
      bordered = FALSE, spacing = "m", rownames = FALSE)
    output$comparison <- shiny::renderPlot({
      data <- selected(); m <- metric()
      legacy <- data[[paste0(m, "_legacy")]]
      standard <- data[[paste0(m, "_corrected")]]
      long <- data.frame(seed = rep(data$seed, 2), value = c(legacy, standard),
        Implementation = rep(c("Legacy", "Standard"), each = nrow(data)))
      ggplot2::ggplot(long, ggplot2::aes(x = seed, y = value, color = Implementation)) +
        ggplot2::geom_point(alpha = .75, size = 2) +
        ggplot2::scale_color_manual(values = c(Legacy = "#69788c", Standard = "#087f8c")) +
        ggplot2::labs(x = "Seed", y = unname(dashboard_metrics[m]),
          title = "Results for each seed", subtitle = "Each point represents one run; metric values are fractions.") +
        ggplot2::theme_minimal(base_size = 13) +
        ggplot2::theme(legend.position = "top", panel.grid.minor = ggplot2::element_blank())
    })
    output$differences <- shiny::renderPlot({
      data <- selected(); m <- metric()
      data$delta <- data[[paste0(m, "_difference")]]
      ggplot2::ggplot(data, ggplot2::aes(x = seed, y = delta)) +
        ggplot2::geom_hline(yintercept = 0, color = "#9aa8b8", linetype = 2) +
        ggplot2::geom_segment(ggplot2::aes(xend = seed, yend = 0), color = "#087f8c") +
        ggplot2::geom_point(color = "#087f8c", size = 2) +
        ggplot2::labs(x = "Seed", y = "Standard − legacy",
          title = "Paired differences", subtitle = if (m %in% c("log_loss", "brier"))
            "Below zero favors standard AdaGrad." else "Above zero favors standard AdaGrad.") +
        ggplot2::theme_minimal(base_size = 13) +
        ggplot2::theme(panel.grid.minor = ggplot2::element_blank())
    })
    output$rows <- shiny::renderTable(utils::head(selected(), 100), digits = 8,
      striped = TRUE, rownames = FALSE)
    output$row_note <- shiny::renderText(sprintf(
      "Showing the first %s of %s selected rows. Download includes every selected row.",
      min(100, nrow(selected())), nrow(selected())))
    output$download <- shiny::downloadHandler(filename = function() "ddesonn-selected-seeds.csv",
      content = function(file) utils::write.csv(selected(), file, row.names = FALSE))
    output$download_summary <- shiny::downloadHandler(filename = function() "ddesonn-metric-summary.csv",
      content = function(file) utils::write.csv(summary(), file, row.names = FALSE))
  }
}
