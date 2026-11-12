# Run with shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN"))
if (!requireNamespace("shiny", quietly = TRUE))
  stop("Install the optional 'shiny' package to use this dashboard.")
source("helpers.R", local = TRUE)
default_data <- dashboard_validate(utils::read.csv("data/heart-failure-100.csv"))

ui <- shiny::fluidPage(
  shiny::tags$script(shiny::HTML("$(document).on('shown.bs.modal', '#shiny-modal', function() { this.focus(); });")),
  shiny::tags$head(shiny::tags$style(shiny::HTML("
    body { background:#f4f7fa; color:#233247; font-family:system-ui,sans-serif; }
    .container-fluid { max-width:1500px; margin:auto; padding:28px; }
    .masthead { background:#152b40; color:white; border-radius:16px; padding:28px 32px; margin-bottom:24px; }
    .masthead { display:flex; align-items:center; justify-content:space-between; gap:24px; }
    .brain-preview { padding:0; border:1px solid #59778e; border-radius:10px; background:#102337; color:#d8ebf4; overflow:hidden; flex-shrink:0; }
    .brain-preview img { display:block; width:220px; max-width:100%; height:auto; }
    .brain-preview span { display:block; padding:7px; font-size:12px; }
    .brain-preview:hover, .brain-preview:focus { color:white; border-color:#77d9d1; outline:2px solid #77d9d1; }
    .brain-full { display:block; width:100%; height:auto; border-radius:8px; }
    .masthead p { color:#c7d9e7; max-width:760px; margin-bottom:0; }
    .eyebrow { color:#77d9d1; letter-spacing:2px; font-size:12px; font-weight:700; }
    h1 { margin-top:10px; font-size:32px; font-weight:700; }
    h3 { font-size:20px; font-weight:650; }
    .well { background:white; border:1px solid #dde5ec; box-shadow:none; border-radius:12px; }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:22px 0; }
    .stat { padding:18px; background:white; border:1px solid #dde5ec; border-radius:12px; }
    .stat span { display:block; color:#63748a; font-size:11px; letter-spacing:1px; }
    .stat strong { display:block; font-size:25px; margin-top:8px; color:#087f8c; }
    .stat small { color:#63748a; }
    .panel-body-custom { padding:22px; background:white; border:1px solid #dde5ec; border-radius:12px; margin-top:18px; }
    .scroll-table { overflow-x:auto; }
    .context { border-left:3px solid #087f8c; padding:12px 16px; background:#eaf5f4; margin:16px 0; }
    .error { color:#a52736; }
    .nav-tabs>li.active>a { color:#087f8c; font-weight:650; }
    .btn { border-radius:7px; }
    @media(max-width:900px) { .cards { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media(max-width:700px) { .masthead { flex-direction:column; align-items:flex-start; } }
    @media(max-width:500px) { .cards { grid-template-columns:1fr; } .container-fluid { padding:12px; } }
  "))),
  shiny::div(class = "masthead",
    shiny::div(
      shiny::div(class = "eyebrow", "DDESONN 7.3.2 / EXPERIMENT EXPLORER"),
      shiny::h1("See how the runs compare"),
      shiny::p("Explore paired benchmark results across seeds. Compare accuracy, probability quality and variability before drawing a conclusion.")),
    shiny::actionButton("brain_art", label = shiny::tagList(
      shiny::tags$img(src = "DDESONN_brain.png", alt = "DDESONN brain artwork with glowing neural connections and equations"),
      shiny::span("Enlarge artwork")), class = "brain-preview",
      title = "Open enlarged brain artwork", `aria-label` = "Enlarge DDESONN brain artwork")),
  shiny::sidebarLayout(
    shiny::sidebarPanel(width = 3,
      shiny::h3("Choose your results"),
      shiny::p(shiny::textOutput("source")),
      shiny::fileInput("upload", "Upload paired results (CSV, up to 10 MB)", accept = ".csv"),
      shiny::div(class = "error", role = "alert", shiny::textOutput("upload_error")),
      shiny::actionButton("reset", "Restore included benchmark"),
      shiny::hr(), shiny::uiOutput("seed_filter"), shiny::uiOutput("metric_filter"),
      shiny::p("Accuracy, precision, recall, F1, ROC AUC and Brier score use values from 0 to 1. Lower log loss and Brier score are better."),
      shiny::hr(), shiny::downloadButton("download", "Download selected rows"),
      shiny::br(), shiny::br(), shiny::downloadButton("download_summary", "Download summary")),
    shiny::mainPanel(width = 9,
      shiny::uiOutput("cards"),
      shiny::tabsetPanel(
        shiny::tabPanel("Overview",
          shiny::div(class = "panel-body-custom", shiny::h3("Metric summary"),
            shiny::p("Means and sample SD of paired differences across the selected seeds. Differences are standard minus legacy; ties use a numerical tolerance of 1e-12. SD is unavailable for one seed."),
            shiny::div(class = "scroll-table", shiny::tableOutput("summary"))),
          shiny::div(class = "context",
            shiny::strong("Read the metrics together. "),
            "Higher accuracy can coexist with worse log loss. This explorer summarizes the supplied runs; it does not establish a causal effect or performance on new data.")),
        shiny::tabPanel("Explore seeds",
          shiny::div(class = "panel-body-custom", shiny::plotOutput("comparison", height = "350px")),
          shiny::div(class = "panel-body-custom", shiny::plotOutput("differences", height = "350px"))),
        shiny::tabPanel("Data",
          shiny::div(class = "panel-body-custom", shiny::textOutput("row_note"),
            shiny::div(class = "scroll-table", shiny::tableOutput("rows")))),
        shiny::tabPanel("About & CSV format",
          shiny::div(class = "panel-body-custom",
            shiny::h3("Bring the next benchmark"),
            shiny::p("Upload one row per seed, with a unique positive integer seed column and at least one complete metric pair. For example:"),
            shiny::tags$pre("seed,accuracy_legacy,accuracy_corrected,log_loss_legacy,log_loss_corrected\n101,0.98,0.99,0.04,0.03"),
            shiny::p("Supported prefixes: accuracy, precision, recall, f1, roc_auc, log_loss, brier. Each requires both _legacy and _corrected columns. Here, corrected means standard AdaGrad. Difference columns are recomputed. Additional columns are retained in downloads."),
            shiny::p("To explore seeds 1–1,000, upload a combined CSV containing all completed seeds. Uploads replace the current data; they do not append or start training. Invalid uploads leave the previous dataset available."),
            shiny::h3("Included research record"),
            shiny::p("The included CSV is the completed 100-seed heart-failure reconstruction documented for 7.3.1, using unchanged 7.3.0 training code. It compares archived legacy AdaGrad with standard AdaGrad. It is not a rerun of the historical 1,000-seed Keras comparison."),
            shiny::p("The same test split is evaluated across seeds. Original historical initialization is unverified. Of the 750 test inputs, 738 also occur in training; this limits conclusions about unseen cases. Seed variability is not a patient-population confidence interval."),
            shiny::tags$a(href = "https://github.com/MatHatter/DDESONN/tree/main/dev/research/heart-failure-adagrad-seeds-1-100", "Read the protocol, results and artifact provenance", target = "_blank", rel = "noopener"),
            shiny::h3("Local, native R"),
            shiny::p("This optional app uses Shiny and ggplot2. It reads CSV results without executing uploaded code. It needs no Python, external deep-learning backend, database or credentials. Uploaded data stays with the running app; use a trusted deployment if serving private results.")))))))

shiny::shinyApp(ui, dashboard_server(default_data))
