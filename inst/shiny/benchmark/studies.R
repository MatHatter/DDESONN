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

clean_study_ui <- function() shiny::tagList(
  shiny::div(class = "context", shiny::h3("Current study: matched settings, predictor overlap removed"),
    shiny::p("100 seeds in each of four conditions: 400 fresh fits. Same original Keras training settings and shared initial tensors, with identical predictor vectors kept in one partition. All three partition pairs have zero exact overlap in raw predictors and float32 inputs. Train / validation / test: 3,500 / 698 / 802 rows."),
    shiny::p("DDESONN uses an experimental native float32 adapter for this matched protocol; these results do not describe the unchanged default trainer. Framework-native shuffle and dropout streams differ. This is one fixed split, not independent patient cohorts.")),
  shiny::selectInput("clean_reduction", "Matched comparison", choices = c(
    "Sum vs. sum — DDESONN FALSE and Keras sum" = "sum",
    "Mean vs. mean — DDESONN TRUE and original Keras mean" = "mean")),
  shiny::p("average_gradients = FALSE sums data-loss gradients; TRUE averages them. In this study each setting applies to all trainable parameters. Both comparisons use 360 epochs, batch size 16, train-only standardization and the final checkpoint."),
  shiny::h4("Held-out accuracy and paired advantage"),
  shiny::tableOutput("clean_overall"),
  shiny::tabsetPanel(
    shiny::tabPanel("Blocks of ten", shiny::p("Disjoint blocks in numerical seed order. Positive differences favor DDESONN; units are percentage points. All ten block averages are positive for both reductions, although individual seeds lose."),
      shiny::tableOutput("clean_blocks")),
    shiny::tabPanel("Cumulative precision", shiny::p("First 10, 20, …, 100 seeds. Shading is an ordinary pointwise 95% paired-t interval across training seeds, not a sequential stopping boundary or uncertainty across datasets."),
      shiny::plotOutput("clean_convergence", height = "380px"), shiny::tableOutput("clean_cumulative")),
    shiny::tabPanel("Why 100 seeds?", shiny::h4("A retrospective precision and compute decision"),
      shiny::p("The sum advantage is +3.88 percentage points (95% interval +3.11 to +4.65); mean is +2.77 (+1.92 to +3.62). Current interval half-widths are 0.77 and 0.85 points. We retain 100 seeds per condition rather than spending more compute to refine these conditional estimates."),
      shiny::p("Agreement across blocks is a useful diagnostic, not a universal threshold. A smooth curve alone does not establish adequate precision. Unclear results may reflect a small effect rather than too few seeds. Choose a useful precision or practical-effect margin before a future study, or use valid sequential methods for repeated stopping checks."),
      shiny::p("At 250 seeds, estimated half-widths are 0.48 and 0.53 points. A ±0.5-point target would need approximately 234 and 285 seeds, assuming the observed variation persists. More seeds cannot establish generalization to other splits or datasets."),
      study_link("SEED_COUNT_RATIONALE.md", "Read the precision rationale and observed/hypothetical graphs")),
    shiny::tabPanel("Reproduce & extend", shiny::p("Four public entry points reproduce the four conditions and support additional seeds. Contribute complete four-condition bundles with environment records and validation through GitHub. Keep extensions separate from the frozen 100-seed report."),
      study_link("CONTRIBUTING_SEEDS.md", "Four scripts and contribution instructions"),
      shiny::h4("Proposed research: separate weight and bias reductions"),
      shiny::p("We welcome experiments that sum weight gradients and average bias gradients, or reverse those choices, for either model. These mixed settings were not tested here. The current TRUE/FALSE control applies one reduction to all parameters; mixed settings require a separate implementation and protocol, including how batch-normalization parameters and regularization are treated."),
      shiny::h4("How many seeds are enough?"),
      shiny::p("This study motivates a methods question rather than a newly established standard: can precision-based seed budgets save compute while maintaining reliable estimates? Evaluate candidate rules across many simulated experiments and real datasets, measuring coverage, estimation error and compute cost."))),
  shiny::downloadButton("clean_download", "Download this comparison (100 paired seeds)"))

clean_study_server <- function(data) {
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
    output$clean_download <- shiny::downloadHandler(
      filename = function() paste0("clean-matched-", if (identical(input$clean_reduction, "mean")) "mean" else "sum", "-100.csv"),
      content = function(file) utils::write.csv(pairs(), file, row.names = FALSE))
  }
}
