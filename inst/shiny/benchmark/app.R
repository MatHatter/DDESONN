# Optional native R model workbench; the existing launch path is preserved.
if (!requireNamespace("shiny", quietly = TRUE) || !requireNamespace("callr", quietly = TRUE))
  stop("Install optional packages 'shiny' and 'callr' to use the workbench.")
source("helpers.R", local = TRUE)
source("workbench_helpers.R", local = TRUE)
source("workbench.R", local = TRUE)
options(shiny.maxRequestSize = 50 * 1024^2)
default_data <- dashboard_validate(utils::read.csv("data/heart-failure-100.csv"))

benchmark_ui <- shiny::sidebarLayout(
    shiny::sidebarPanel(width = 3,
      shiny::h3("Choose your results"),
      shiny::tags$details(shiny::tags$summary("What are legacy and standard AdaGrad?"),
        shiny::p("Legacy means DDESONN's earlier custom AdaGrad. Its multilayer path applied a direct weight update and then an adaptive update to the same weights. Despite strong historical results, this differed from classic AdaGrad."),
        shiny::p("Standard is classic AdaGrad: one adaptive update per parameter tensor per training step. It replaced the legacy behavior in 7.3.0. The documented 100-seed follow-up had slightly higher mean accuracy; not every metric improved.")),
      shiny::p(shiny::textOutput("source")),
      shiny::fileInput("upload", "Upload paired results (CSV, up to 10 MB)", accept = ".csv"),
      shiny::div(class = "error", role = "alert", shiny::textOutput("upload_error")),
      shiny::actionButton("reset", "Restore included benchmark"),
      shiny::hr(), shiny::uiOutput("seed_filter"), shiny::uiOutput("metric_filter"),
      shiny::p("Accuracy, precision, recall, F1, ROC AUC and Brier score use values from 0 to 1. Lower log loss and Brier score are better."),
      shiny::hr(), wb_download_info("download", shiny::downloadButton("download", "Download selected rows")),
      shiny::br(), shiny::br(), wb_download_info("download_summary", shiny::downloadButton("download_summary", "Download summary"))),
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
        shiny::tabPanel("Benchmark data",
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
            shiny::p("This optional app uses Shiny, callr and ggplot2. Experiments reads saved CSV results; the training pages call the native R model API. No Python, external deep-learning backend, database or credentials are needed. Uploaded data stays with the running app; use a trusted deployment if serving private results."))))))

ui <- shiny::fluidPage(
  shiny::tags$script(shiny::HTML("$(document).on('shown.bs.modal', '#shiny-modal', function() { this.focus(); });")),
  shiny::tags$script(shiny::HTML("
    var wbActionStates = {};
    function wbApplyActionStates() {
      Object.keys(wbActionStates).forEach(function(id) {
        var element = document.getElementById(id);
        if (!element) return;
        var enabled = wbActionStates[id] && (element.tagName !== 'A' || !!element.getAttribute('href'));
        element.disabled = !enabled;
        element.setAttribute('aria-disabled', String(!enabled));
        element.classList.toggle('disabled', !enabled);
        if (element.tagName === 'A') element.setAttribute('tabindex', enabled ? '0' : '-1');
        if (element.type === 'file') {
          var button = element.closest('.btn');
          if (button) { button.classList.toggle('disabled', !enabled); button.setAttribute('aria-disabled', String(!enabled)); }
        }
      });
    }
    Shiny.addCustomMessageHandler('wb-clear-loss-hover', function(message) {
      Shiny.setInputValue('wb_loss_hover', null);
    });
    Shiny.addCustomMessageHandler('wb-action-states', function(states) {
      wbActionStates = states; wbApplyActionStates();
    });
    $(document).on('shiny:value shiny:bound', function() {
      setTimeout(function() { wbApplyActionStates(); wbPositionOpenHelp(); }, 0);
    });
    function wbPositionHelp(details) {
      var copy = details.querySelector('.wb-help-copy');
      var anchor = details.querySelector('summary').getBoundingClientRect();
      copy.style.position = 'fixed'; copy.style.right = 'auto';
      var below = window.innerHeight - anchor.bottom - 20;
      var above = anchor.top - 20;
      var useAbove = below < 160 && above > below;
      copy.style.maxHeight = Math.max(80, Math.min(560, window.innerHeight * .65, useAbove ? above : below)) + 'px';
      var box = copy.getBoundingClientRect();
      copy.style.left = Math.max(12, Math.min(anchor.right - box.width, window.innerWidth - box.width - 12)) + 'px';
      var top = useAbove ? anchor.top - box.height - 8 : anchor.bottom + 8;
      copy.style.top = Math.max(12, Math.min(top, window.innerHeight - box.height - 12)) + 'px';
    }
    document.addEventListener('toggle', function(event) {
      if (event.target.matches('.wb-inline-help') && event.target.open) wbPositionHelp(event.target);
    }, true);
    function wbPositionOpenHelp() { document.querySelectorAll('.wb-inline-help[open]').forEach(wbPositionHelp); }
    window.addEventListener('resize', wbPositionOpenHelp);
    document.addEventListener('scroll', wbPositionOpenHelp, true);
    document.addEventListener('click', function(event) {
      if (event.target.closest('[aria-disabled=true]')) { event.preventDefault(); event.stopImmediatePropagation(); }
    }, true);
    document.addEventListener('click', function(event) {
      document.querySelectorAll('.wb-inline-help[open]').forEach(function(help) {
        if (!help.contains(event.target)) help.open = false;
      });
    });
    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape') {
        var opened = Array.from(document.querySelectorAll('.wb-inline-help[open]'));
        opened.forEach(function(help) { help.open = false; });
        if (opened.length) { event.preventDefault(); opened[opened.length - 1].querySelector('summary').focus(); }
      }
    });
  ")),
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
    .tab-content > .tab-pane > .row { margin-top:18px; }
    .well { background:white; border:1px solid #dde5ec; box-shadow:none; border-radius:12px; }
    .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:0 0 22px; }
    .stat { padding:18px; background:white; border:1px solid #dde5ec; border-radius:12px; }
    .stat span { display:block; color:#63748a; font-size:11px; letter-spacing:1px; }
    .stat strong { display:block; font-size:25px; margin-top:8px; color:#087f8c; }
    .stat small { color:#63748a; }
    .panel-body-custom { padding:22px; background:white; border:1px solid #dde5ec; border-radius:12px; margin-top:18px; }
    .scroll-table { overflow-x:auto; }
    .context { border-left:3px solid #087f8c; padding:12px 16px; background:#eaf5f4; margin:16px 0; }
    .error { color:#a52736; }
    .nav-tabs>li.active>a { color:#087f8c; font-weight:650; }
    .wb-workflow-guide { color:#53677b; margin:0 0 14px; line-height:1.7; }
    .wb-workflow-guide strong { color:#155b63; margin-right:8px; }
    #workspace > li > a[data-value='data'], #workspace > li > a[data-value='features'],
    #workspace > li > a[data-value='train'], #workspace > li > a[data-value='results'] {
      background:#e8f3f2; color:#185963; border:1px solid #c9dfdc; margin-right:5px; font-weight:600;
    }
    #workspace > li.active > a[data-value='data'], #workspace > li.active > a[data-value='features'],
    #workspace > li.active > a[data-value='train'], #workspace > li.active > a[data-value='results'] {
      background:#087f8c; color:white; border-color:#087f8c;
    }
    #workspace > li > a:focus-visible { outline:3px solid #527d9b; outline-offset:2px; }
    .wb-title-row, .wb-control-title { display:flex; align-items:center; justify-content:space-between; gap:14px; }
    .wb-title-row { margin:0 0 14px; }
    .wb-title-row h3, .wb-title-row h4 { margin:0; line-height:1.4; }
    .wb-control-title { margin-bottom:5px; }
    .wb-control-title label { margin:0; }
    .wb-field .shiny-input-container { width:100%; }
    .wb-switch-row { display:flex; align-items:center; justify-content:space-between; gap:14px; margin-bottom:18px; width:100%; }
    .wb-switch-row > label { flex:1; min-width:0; margin:0; font-weight:normal; }
    .wb-switch-actions { display:flex; align-items:center; gap:12px; flex-shrink:0; }
    .wb-switch { display:inline-block; position:relative; width:34px; height:20px; margin:0; }
    .wb-switch input { position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; z-index:1; cursor:pointer; }
    .wb-switch-track { display:block; width:34px; height:20px; border-radius:12px; background:#83909b; transition:background .15s; }
    .wb-switch-track::after { content:''; position:absolute; width:14px; height:14px; left:3px; top:3px; border-radius:50%; background:white; transition:transform .15s; }
    .wb-switch input:checked + .wb-switch-track { background:#087f8c; }
    .wb-switch input:checked + .wb-switch-track::after { transform:translateX(14px); }
    .wb-switch input:focus-visible + .wb-switch-track { outline:3px solid #6f9daa; outline-offset:3px; }
    @media(prefers-reduced-motion:reduce) { .wb-switch-track, .wb-switch-track::after { transition:none; } }
    .tab-pane > .row > [class*='col-'] > .panel-body-custom:first-child { margin-top:0; }
    .wb-inline-help { position:relative; flex:0 0 auto; font-weight:normal; }
    .wb-inline-help > summary { display:inline-grid; place-items:center; list-style:none;
      cursor:pointer; width:20px; height:20px; min-width:20px; flex:none; padding:0;
      border:0; border-radius:50%; background:transparent; box-sizing:border-box; }
    .wb-help-icon { display:block; width:20px; height:20px; overflow:visible; }
    .wb-help-ring { fill:white; stroke:#6e8f9e; stroke-width:1.1; }
    .wb-help-letter { fill:#315e6b; font:normal 12px Arial,sans-serif; }
    .wb-inline-help > summary::-webkit-details-marker { display:none; }
    .wb-inline-help > summary:focus-visible { outline:3px solid #6f9daa; outline-offset:2px; }
    .wb-inline-help[open] .wb-help-ring { fill:#087f8c; stroke:#087f8c; }
    .wb-inline-help[open] .wb-help-letter { fill:white; }
    .wb-help-copy { position:absolute; display:block; right:0; top:24px; width:520px; max-width:calc(100vw - 36px);
      background:#183447; color:white; border-radius:9px; padding:13px 15px; box-shadow:0 5px 18px #18344733;
      font:normal 13px/1.6 system-ui,sans-serif; z-index:1100; max-height:min(560px,65vh); overflow:auto; overscroll-behavior:contain; }
    .wb-full-code-bar { margin-top:20px; padding:16px; background:#e8f3f2; border-radius:10px; }
    .wb-full-code-bar p { margin:8px 0 0; }
    #wb_full_notice { white-space:pre-line; }
    .wb-full-script { line-height:1.6; max-height:60vh; overflow:auto; white-space:pre-wrap; overflow-wrap:break-word; word-break:normal; margin-top:12px; }
    .wb-code-panel { margin-top:16px; }
    .wb-workflow-pills .shiny-input-container { width:100%; margin-bottom:0; }
    .wb-workflow-pills .shiny-options-group { display:flex; padding:4px; gap:4px; background:#e5ecef; border-radius:999px; }
    .wb-workflow-pills label.radio-inline { flex:1; margin:0; padding:9px 12px; text-align:center; border-radius:999px; cursor:pointer; position:relative; transition:background .15s,color .15s; }
    .wb-workflow-pills input { position:absolute; opacity:0; width:1px; height:1px; }
    .wb-workflow-pills label:has(input:checked) { color:#124452; background:white; box-shadow:0 1px 4px #18344726; font-weight:600; }
    .wb-workflow-pills label:has(input:focus-visible) { outline:3px solid #527d9b; outline-offset:2px; }
    .wb-custom-brief { font-size:13px; color:#526b7a; line-height:1.55; margin:12px 0 18px; }
    .wb-download-row { display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin:8px 0; }
    .wb-settings-section { border-top:1px solid #d8e2e8; margin-top:10px; padding-top:8px; }
    .wb-settings-section > summary { cursor:pointer; color:#155b63; font-weight:600; padding:4px 0; }
    .wb-settings-section > summary::before { content:'▸ '; }
    .wb-settings-section[open] > summary::before { content:'▾ '; }
    .wb-code-panel > summary { color:#155b63; cursor:pointer; font-weight:600; }
    .wb-code-panel pre { white-space:pre-wrap; overflow-wrap:break-word; word-break:normal; line-height:1.6; margin-top:10px; max-height:380px; overflow:auto; font-size:12px; }
    .wb-help-code-label { margin-top:12px; font-weight:600; color:#b5eee8; }
    .wb-help-prerequisites { font-size:12px; margin:10px 0 0; color:#d0e1eb; }
    .wb-help-code { line-height:1.6; color:white; background:#102b3c; border:0; margin:10px 0 0; white-space:pre-wrap; overflow-wrap:break-word; word-break:normal; overflow-x:auto; font-size:11px; }
    .wb-guidance { color:#526b7a; font-size:14px; line-height:1.6; }
    .wb-step-intro { margin:18px 0; color:#526b7a; }
    @media(max-width:700px) { #workspace > li { margin-bottom:5px; } }
    .wb-latest-attempt { background:#f0f6f8; border-left:3px solid #6f9daa; padding:10px 12px; margin-bottom:12px; }
    .wb-latest-attempt:has(.wb-training-error) { background:#fff5df; border-left-color:#d68b20; }
    #wb_selected_run .radio { padding:10px; border:1px solid #d8e5ec; border-radius:7px; margin-bottom:8px; }
    #wb_selected_run .radio:has(input:checked) { background:#e8f3f2; border-color:#087f8c; }
    #wb_selected_run input { width:19px; height:19px; margin-top:0; margin-left:-25px; }
    #wb_selected_run label { padding-left:27px; min-height:24px; }
    .wb-run-summary { display:block; margin-top:4px; }
    .wb-attempt-history { margin-top:18px; }
    .wb-attempt-row { padding:10px 0; border-bottom:1px solid #d8e5ec; }
    .wb-attempt-row pre { white-space:pre-wrap; overflow-wrap:anywhere; }
    .wb-loss-chart { position:relative; }
    #wb_loss, #wb_loss img { cursor:default !important; }
    #wb_loss_marker { position:absolute; inset:0; pointer-events:none; }
    .wb-loss-dot { position:absolute; width:13px; height:13px; border:2px solid white; border-radius:50%; background:#087f8c; box-shadow:0 0 0 1px #087f8c; transform:translate(-50%,-50%); }
    .wb-loss-tooltip { position:absolute; bottom:18px; left:0; transform:translateX(-50%); background:#183447; color:white; border-radius:4px; padding:4px 7px; white-space:nowrap; font-size:12px; }
    .wb-loss-values { max-height:300px; overflow:auto; }
    .wb-training-error { margin-top:18px; }
    .wb-training-error pre { max-height:240px; overflow:auto; white-space:pre-wrap; }
    .btn { border-radius:7px; }
    .btn:focus-visible, details > summary:focus-visible { outline:3px solid #527d9b; outline-offset:2px; }
    .btn.disabled, .btn:disabled { opacity:.55; cursor:not-allowed; }
    @media(max-width:900px) { .cards { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media(max-width:700px) { .masthead { flex-direction:column; align-items:flex-start; } }
    @media(max-width:500px) { .cards { grid-template-columns:1fr; } .container-fluid { padding:12px; }
      .tab-content > .tab-pane > .row { margin-left:-12px; margin-right:-12px; } }
  "))),
  shiny::div(class = "masthead",
    shiny::div(
      shiny::div(class = "eyebrow", "DDESONN / MODEL WORKBENCH"),
      shiny::h1("Build, train and explore"),
      shiny::p("Train a native R model, inspect its predictions and compare saved experiments.")),
    shiny::actionButton("brain_art", label = shiny::tagList(
      shiny::tags$img(src = "DDESONN_brain.png", alt = "DDESONN brain artwork with glowing neural connections and equations")), class = "brain-preview",
      title = "Open enlarged brain artwork", `aria-label` = "Enlarge DDESONN brain artwork")),
  shiny::div(class = "wb-workflow-guide", shiny::strong("Your workflow"),
    "Start with Data → choose target and predictors in Features → build and train → review Model / Results."),
  do.call(shiny::tabsetPanel, c(list(id = "workspace"), wb_ui(),
    list(shiny::tabPanel("Experiments", value = "experiments", benchmark_ui)))),
  shiny::div(class = "wb-full-code-bar",
    shiny::actionButton("wb_show_full_code", "Show full R code"),
    shiny::p("Help icons show one topic at a time. View or download the complete workflow here.")))

shiny::shinyApp(ui, function(input, output, session) {
  dashboard_server(default_data)(input, output, session)
  wb_server(input, output, session, normalizePath("workbench_helpers.R"))
})
