# Saved research studies; viewing these records never starts training.
study_clean_pairs <- function(data, reduction) {
  stopifnot(reduction %in% c("sum", "mean"), nrow(data) == 400L,
    !anyDuplicated(data[c("seed", "framework", "reduction")]),
    all(data$test_rows == 802L), all(data$epochs == 360L),
    all(data$optimizer_updates == 78840L), all(data$checkpoint == "final"),
    all(is.finite(data$correct)), all(data$correct %% 1 == 0),
    all(data$correct >= 0 & data$correct <= 802),
    all(abs(data$accuracy - data$correct / 802) < 1e-12))
  for (r in c("sum", "mean")) for (f in c("ddesonn", "keras"))
    stopifnot(identical(sort(as.integer(data$seed[data$framework == f & data$reduction == r])), 1:100))
  a <- data[data$framework == "ddesonn" & data$reduction == reduction, ]
  b <- data[data$framework == "keras" & data$reduction == reduction, ]
  a <- a[order(a$seed), ]; b <- b[order(b$seed), ]
  data.frame(seed = a$seed, DDESONN = a$accuracy * 100,
    Keras = b$accuracy * 100, difference_pp = (a$correct - b$correct) * 100 / 802)
}

study_seed_summary <- function(pairs, cumulative = FALSE) {
  do.call(rbind, lapply(seq(10L, 100L, 10L), function(end) {
    first <- if (cumulative) 1L else end - 9L
    x <- pairs[pairs$seed >= first & pairs$seed <= end, ]
    d <- x$difference_pp
    half <- stats::qt(.975, length(d) - 1) * stats::sd(d) / sqrt(length(d))
    data.frame(first_seed = first, last_seed = end, seeds = length(d),
      DDESONN_pct = mean(x$DDESONN), Keras_pct = mean(x$Keras),
      advantage_pp = mean(d), low_pp = mean(d) - half, high_pp = mean(d) + half,
      wins = sum(d > 0), ties = sum(d == 0), losses = sum(d < 0))
  }))
}

study_link <- function(file, label) shiny::tags$a(
  href = paste0("https://github.com/MatHatter/DDESONN/blob/main/dev/research/keras-clean-reference/", file),
  label, target = "_blank", rel = "noopener noreferrer")

clean_study_context <- function()
  shiny::div(class = "context", shiny::h3("Current study: matched settings, predictor overlap removed"),
    shiny::p("100 seeds in each of four conditions: 400 fresh fits. Same original Keras training settings and shared initial tensors, with identical predictor vectors kept in one partition. All three partition pairs have zero exact overlap in raw predictors and float32 inputs. Train / validation / test: 3,500 / 698 / 802 rows."),
    shiny::p("DDESONN uses an experimental native float32 adapter for this matched protocol; these results do not describe the unchanged default trainer. Framework-native shuffle and dropout streams differ. This is one fixed split, not independent patient cohorts."))

clean_study_ui <- function() shiny::tagList(
  shiny::div(class = "study-comparison", shiny::selectInput("clean_reduction", "Matched comparison", choices = c(
    "Sum vs. sum — DDESONN FALSE / Keras sum" = "sum",
    "Mean vs. mean — DDESONN TRUE / Keras mean" = "mean"), width = "100%")),
  shiny::p("The ", shiny::code("average_gradients"), " setting controls how gradients from each training batch are combined: ",
    shiny::code("FALSE"), " sums the data-loss gradients, while ", shiny::code("TRUE"),
    " averages them. Each choice applies to every trainable parameter in this study. All four conditions use 360 epochs and batches of 16, with a smaller final batch. Standardization is fitted on training data only, and evaluation uses the final checkpoint."),
  shiny::h4("Held-out accuracy and paired advantage"),
  shiny::div(class = "scroll-table", shiny::tableOutput("clean_overall")),
  shiny::tabsetPanel(
    shiny::tabPanel("Blocks of ten", shiny::p("Disjoint blocks in numerical seed order. Positive differences favor DDESONN; units are percentage points. All ten block averages are positive for both reductions, although individual seeds lose."),
      shiny::div(class = "scroll-table", shiny::tableOutput("clean_blocks"))),
    shiny::tabPanel("Cumulative precision", shiny::p("First 10, 20, …, 100 seeds. Shading is an ordinary pointwise 95% paired-t interval across training seeds, not a sequential stopping boundary or uncertainty across datasets."),
      shiny::plotOutput("clean_convergence", height = "380px"), shiny::div(class = "scroll-table", shiny::tableOutput("clean_cumulative"))),
    shiny::tabPanel("Why 100 seeds?", shiny::h4("A retrospective precision and compute decision"),
      shiny::p("The sum advantage is +3.88 percentage points (95% interval +3.11 to +4.65); mean is +2.77 (+1.92 to +3.62). Current interval half-widths are 0.77 and 0.85 points. We retain 100 seeds per condition rather than spending more compute to refine these conditional estimates."),
      shiny::p("Agreement across blocks is a useful diagnostic, not a universal threshold. A smooth curve alone does not establish adequate precision. Unclear results may reflect a small effect rather than too few seeds. Choose a useful precision or practical-effect margin before a future study, or use valid sequential methods for repeated stopping checks."),
      shiny::p("At 250 seeds, estimated half-widths are 0.48 and 0.53 points. A ±0.5-point target would need approximately 234 and 285 seeds, assuming the observed variation persists. More seeds cannot establish generalization to other splits or datasets."),
      study_link("SEED_COUNT_RATIONALE.md", "Read the precision rationale and observed/hypothetical graphs")),
    shiny::tabPanel("Reproduce & extend", shiny::p("Four public entry points reproduce the four conditions and support additional seeds. Contribute complete four-condition bundles with environment records and validation through GitHub. Keep extensions separate from the frozen 100-seed report."),
      study_link("CONTRIBUTING_SEEDS.md", "Four scripts and contribution instructions"),
      shiny::p("Download the complete reproduction bundle first. It contains four R launch scripts, the Python Keras runners, the native adapter, frozen split, dataset, protocol, and numerical checks. The individual launch scripts below need the extracted bundle. R scripts launch Python for Keras and shared initialization; these are not replacements with different settings."),
      shiny::div(class = "study-downloads",
        shiny::downloadButton("study_bundle", "Download all four scripts + dependencies (ZIP)"),
        shiny::downloadButton("study_selected_launcher", "Download selected comparison launcher")),
      shiny::div(class = "study-downloads",
        shiny::downloadButton("study_keras_mean", "Keras mean.R"),
        shiny::downloadButton("study_ddesonn_sum", "DDESONN sum.R"),
        shiny::downloadButton("study_ddesonn_mean", "DDESONN mean.R"),
        shiny::downloadButton("study_keras_sum", "Keras sum.R")),
      shiny::h4("Proposed research: separate weight and bias reductions"),
      shiny::p("We welcome experiments that sum weight gradients and average bias gradients, or reverse those choices, for either model. These mixed settings were not tested here. The current TRUE/FALSE control applies one reduction to all parameters; mixed settings require a separate implementation and protocol, including how batch-normalization parameters and regularization are treated."),
      shiny::p("Illustrative custom-loop gradients below start with a SUM of per-example data losses. Add the dense-kernel L1 gradient once afterwards; do not divide it by batch size. These snippets are proposed modifications, not executable settings in the current trainer."),
      shiny::h5("R / native adapter proposal: reverse weight and bias choices"),
      shiny::tags$pre("# grad_W_sum and grad_b_sum: gradients of summed data loss
# n_batch: actual batch size (16, or 12 in the final batch)
# A: sum weights, average biases
grad_W <- grad_W_sum + l1_lambda * sign(W)
grad_b <- grad_b_sum / n_batch
# B: average weights, sum biases
grad_W <- grad_W_sum / n_batch + l1_lambda * sign(W)
grad_b <- grad_b_sum
# Feed these gradients to AdaGrad; do not update parameters twice.
# BN gamma/beta: choose and document a separate reduction.
# Native C++ training-loop changes are required to implement this."),
      shiny::h5("Keras custom-loop proposal: the same two choices"),
      shiny::tags$pre("# grad_kernel_sum, grad_bias_sum: SUM data-loss gradients
# Start with BinaryCrossentropy(reduction='sum') inside GradientTape.
# A: sum kernels, average biases
grad_kernel = grad_kernel_sum + l1_lambda * tf.sign(kernel)
grad_bias = grad_bias_sum / tf.cast(n_batch, grad_bias_sum.dtype)
# B: average kernels, sum biases
grad_kernel = grad_kernel_sum / tf.cast(n_batch, grad_kernel_sum.dtype) + l1_lambda * tf.sign(kernel)
grad_bias = grad_bias_sum
# Apply exactly once with optimizer.apply_gradients(...).
# Explicitly handle BN gamma/beta; exclude regularization from the
# data-loss gradients above to avoid adding L1 twice."),
      shiny::h4("How many seeds are enough?"),
      shiny::p("This study motivates a methods question rather than a newly established standard: can precision-based seed budgets save compute while maintaining reliable estimates? Evaluate candidate rules across many simulated experiments and real datasets, measuring coverage, estimation error and compute cost."))),
  shiny::downloadButton("clean_download", "Download this comparison (100 paired seeds)"))

clean_study_server <- function(data, historical = NULL, app_dir = getwd()) {
  app_dir <- normalizePath(app_dir)
  study_clean_pairs(data, "sum"); study_clean_pairs(data, "mean")
  function(input, output, session) {
    pairs <- shiny::reactive(study_clean_pairs(data,
      if (identical(input$clean_reduction, "mean")) "mean" else "sum"))
    cumulative <- shiny::reactive(study_seed_summary(pairs(), TRUE))
    output$clean_overall <- shiny::renderTable(utils::tail(cumulative(), 1), digits = 3, rownames = FALSE)
    output$clean_blocks <- shiny::renderTable(study_seed_summary(pairs()), digits = 3, rownames = FALSE)
    output$clean_cumulative <- shiny::renderTable(cumulative(), digits = 3, rownames = FALSE)
    output$clean_convergence <- shiny::renderPlot({
      ggplot2::ggplot(cumulative(), ggplot2::aes(x = seeds, y = advantage_pp)) +
        ggplot2::geom_ribbon(ggplot2::aes(ymin = low_pp, ymax = high_pp), fill = "#087f8c", alpha = .2) +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        ggplot2::geom_line(color = "#087f8c") + ggplot2::geom_point(color = "#087f8c") +
        ggplot2::labs(x = "Training seeds per condition", y = "DDESONN − Keras accuracy (percentage points)") +
        ggplot2::theme_minimal(base_size = 13)
    })
    reduction <- shiny::reactive(if (identical(input$clean_reduction, "mean")) "mean" else "sum")
    is_historical <- shiny::reactive(identical(input$experiment_study, "historical"))
    analysis_code <- shiny::reactive({
      if (is_historical()) {
        shiny::req(historical)
        study_analysis_code(historical$selected(), historical = TRUE, metric = historical$metric())
      } else study_analysis_code(data, reduction())
    })
    output$study_full_code <- shiny::renderText(analysis_code())
    shiny::observeEvent(input$study_show_full_code, {
      shiny::showModal(shiny::modalDialog(title = "Full R code — selected experiment", size = "l", easyClose = TRUE,
        shiny::p("This script reproduces the displayed saved-result analysis. It does not train models. For actual fitting, download the reproduction bundle under Reproduce & extend; Keras and shared initialization also need Python."),
        shiny::tags$pre(id = "study_full_code", class = "shiny-text-output wb-full-script", tabindex = "0"),
        shiny::downloadButton("study_analysis_download", "Download this analysis.R"),
        footer = shiny::modalButton("Close")))
    })
    output$study_analysis_download <- shiny::downloadHandler(
      filename = function() if (is_historical()) "historical-selected-analysis.R" else paste0("clean-", reduction(), "-analysis.R"),
      content = function(file) writeLines(analysis_code(), file))
    bundle <- file.path(app_dir, "reproduction.zip")
    output$study_bundle <- shiny::downloadHandler(filename = "ddesonn-clean-study-reproduction.zip",
      content = function(file) { stopifnot(file.exists(bundle)); file.copy(bundle, file, overwrite = TRUE) })
    launcher <- function(framework, reduction) paste0(framework, "_", reduction, ".R")
    copy_launcher <- function(name, file) {
      connection <- unz(bundle, name)
      on.exit(close(connection))
      writeLines(readLines(connection, warn = FALSE), file)
    }
    for (condition in list(c("keras", "mean"), c("ddesonn", "sum"), c("ddesonn", "mean"), c("keras", "sum"))) local({
      framework <- condition[1]; mode <- condition[2]
      output[[paste0("study_", framework, "_", mode)]] <- shiny::downloadHandler(
        filename = function() launcher(framework, mode),
        content = function(file) copy_launcher(launcher(framework, mode), file))
    })
    output$study_selected_launcher <- shiny::downloadHandler(
      filename = function() paste0("matched-", reduction(), ".R"),
      content = function(file) writeLines(c(
        "# Run from the extracted reproduction bundle root; requires R and Python.",
        'source("RUN_STUDY.R")',
        sprintf('run_study("ddesonn", "%s")', reduction()),
        sprintf('run_study("keras", "%s")', reduction())), file))
    output$clean_download <- shiny::downloadHandler(
      filename = function() paste0("clean-matched-", if (identical(input$clean_reduction, "mean")) "mean" else "sum", "-100.csv"),
      content = function(file) utils::write.csv(pairs(), file, row.names = FALSE))
  }
}

study_analysis_code <- function(data, reduction = "sum", historical = FALSE, metric = "accuracy") {
  literal <- function(x) paste(capture.output(dput(x)), collapse = "\n")
  if (historical) {
    paste0("# Historical overlapping-input study: saved results only; no training.\n",
      "# This reproduces the selected rows (including uploads) and selected metric.\n",
      "# Of 750 original test inputs, 738 also appear in training.\n",
      "data <- ", literal(data), "\n",
      "dashboard_metrics <- ", literal(dashboard_metrics), "\n",
      "summarize <- ", paste(deparse(dashboard_summary), collapse = "\n"), "\n",
      "print(summarize(data))\nmetric <- ", literal(metric), "\n",
      "delta <- data[[paste0(metric, '_corrected')]] - data[[paste0(metric, '_legacy')]]\n",
      "plot(data$seed, delta, xlab='Seed', ylab='Standard minus legacy', pch=16)\nabline(h=0, lty=2)\n")
  } else {
    paste0("# Current clean matched study: ", reduction, " vs. ", reduction, ".\n",
      "# Saved-result analysis, not model fitting. DDESONN uses the experimental adapter.\n",
      "# Embedded values reproduce the displayed 100 paired seeds; accuracy is percent.\n",
      "pairs <- ", literal(study_clean_pairs(data, reduction)), "\n",
      "seed_summary <- ", paste(deparse(study_seed_summary), collapse = "\n"), "\n",
      "blocks <- seed_summary(pairs)\ncumulative <- seed_summary(pairs, TRUE)\n",
      "print(tail(cumulative, 1))\nprint(blocks)\nprint(cumulative)\n",
      "with(cumulative, {\n",
      "  plot(seeds, advantage_pp, type='n', ylim=range(c(low_pp, high_pp, 0)),\n",
      "       xlab='Training seeds per condition', ylab='DDESONN minus Keras accuracy (pp)')\n",
      "  polygon(c(seeds, rev(seeds)), c(low_pp, rev(high_pp)), col='#087f8c33', border=NA)\n",
      "  lines(seeds, advantage_pp, type='b', col='#087f8c')\n  abline(h=0, lty=2)\n})\n",
      "# Intervals are pointwise 95% paired-t intervals on ONE fixed split.\n",
      "# To rerun training, extract the reproduction ZIP and source the relevant\n",
      "# ddesonn_", reduction, ".R and keras_", reduction, ".R launchers.\n")
  }
}
