# Cache serialized bundles so each test gets its own mutable R6 model.
workbench_model_fixture <- local({
  bundles <- new.env(parent = emptyenv())
  function(h, mode = "multiclass", checkpoint = "final", lr = .01) {
    key <- paste(mode, checkpoint, lr)
    if (!exists(key, envir = bundles, inherits = FALSE)) {
      data <- if (mode == "multiclass") workbench_iris else workbench_mtcars
      target <- switch(mode, multiclass = "Species", binary = "am", regression = "mpg")
      features <- if (mode == "multiclass") names(iris)[1:4] else c("wt", "hp")
      p <- h$wb_prepare(data, target, features, mode)
      cfg <- h$wb_config(hidden = "2", epochs = if (mode == "multiclass") 2L else 1L,
        checkpoint = checkpoint, lr = lr)
      invisible(capture.output(bundle <- h$wb_run(p, cfg)))
      bundles[[key]] <- serialize(bundle, NULL)
    }
    unserialize(bundles[[key]])
  }
})

prediction_flow_fixture <- function() {
  h <- new.env(parent = globalenv())
  root <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(root, "workbench_helpers.R"), h)
  sys.source(file.path(root, "workbench.R"), h)
  bundle <- workbench_model_fixture(h)
  server <- h$wb_server; formals(server)$helper <- file.path(root, "workbench_helpers.R")
  list(h = h, bundle = bundle, server = server)
}
