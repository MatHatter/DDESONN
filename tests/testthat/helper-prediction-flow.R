prediction_flow_fixture <- function() {
  h <- new.env(parent = globalenv())
  root <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(root, "workbench_helpers.R"), h)
  sys.source(file.path(root, "workbench.R"), h)
  p <- h$wb_prepare(workbench_iris, "Species", names(iris)[1:4], "multiclass")
  invisible(capture.output(bundle <- h$wb_run(p, h$wb_config(hidden = "2", epochs = 2, checkpoint = "final"))))
  server <- h$wb_server; formals(server)$helper <- file.path(root, "workbench_helpers.R")
  list(h = h, bundle = bundle, server = server)
}
