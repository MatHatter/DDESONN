test_that("cancellation detaches jobs before waiting for worker shutdown", {
  h <- new.env(parent=globalenv())
  helper <- system.file("shiny","benchmark","workbench_helpers.R",package="DDESONN")
  sys.source(helper,h)
  sys.source(system.file("shiny","benchmark","workbench.R",package="DDESONN"),h)
  server <- h$wb_server;formals(server)$helper <- helper
  shiny::testServer(server, {
    stopped <- 0L
    h$wb_stop <- function(active) {
      if (!is.null(active)) {
        # A poll during process shutdown must see no active job.
        expect_null(job())
        stopped <<- stopped+1L
      }
    }
    fake <- function(destination) list(destination=destination, log=tempfile(),error=tempfile(),
      started=Sys.time(),run_id="1",epochs=5000,
      process=list(is_alive=function()TRUE))
    begin_attempt(h$wb_config())
    job(fake("training"));session$flushReact()
    session$setInputs(wb_cancel=1)
    expect_null(job());expect_match(status(),"cancelled")
    expect_identical(stopped,1L)
    saved_fit(list(state="Running",history=data.frame(epoch=integer(),loss=numeric()),
      log=character(),bundle=NULL))
    job(fake("saved"));session$flushReact()
    session$setInputs(wb_saved_cancel=1)
    expect_null(job());expect_identical(saved_fit()$state,"Cancelled")
    expect_identical(stopped,2L)
  })
})
