# MockShinySession normally renders every output on every input change, even
# when the test only checks control state. Register outputs as tests access them.
# Once registered, the real Shiny renderer and reactive invalidation are used.
workbench_test_server <- function(app, expr) {
  session_class <- R6::R6Class("WorkbenchTestSession",
    inherit = shiny::MockShinySession, portable = FALSE,
    parent_env = asNamespace("shiny"), lock_objects = FALSE,
    public = list(
      defineOutput = function(name, func, label) {
        private$deferred[[name]] <- list(func = func, label = label)
      },
      getOutput = function(name) {
        definition <- private$deferred[[name]]
        if (!is.null(definition)) {
          private$deferred[[name]] <- NULL
          super$defineOutput(name, definition$func, definition$label)
        }
        super$getOutput(name)
      }
    ),
    private = list(deferred = list()))
  do.call(shiny::testServer,
    list(app, substitute(expr), session = session_class$new()),
    envir = parent.frame())
}
