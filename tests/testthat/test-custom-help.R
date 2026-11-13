custom_help_fixture <- function() {
  h <- new.env(parent = globalenv())
  root <- system.file("shiny", "benchmark", package = "DDESONN")
  sys.source(file.path(root, "workbench_helpers.R"), h)
  sys.source(file.path(root, "workbench.R"), h)
  h
}

test_that("Custom help follows structure and combination without running a model", {
  h <- custom_help_fixture()
  for (structure in c("single", "members", "main", "temp")) {
    cfg <- h$wb_custom_config(h$wb_config(), structure, 3, "4,5", 2,
      fusion = "weighted", seed_aggregate = "median", diagnostics = TRUE)
    code <- h$wb_help_code("Run structure", config = cfg)
    env <- new.env(); eval(parse(text = code), env)
    expect_identical(env$do_ensemble, structure %in% c("main", "temp"))
    expect_equal(env$num_networks, if (structure == "single") 1 else 3)
    expect_equal(env$num_temp_iterations, if (structure == "temp") 2 else 0)
    expect_match(h$wb_help_code("Combine members within each training seed", config = cfg), 'method = "weighted"', fixed = TRUE)
    expect_match(h$wb_help_code("Combine across training seeds", config = cfg), 'seed_method <- "median"', fixed = TRUE)
    expect_match(h$wb_help_code("Detailed training diagnostics", config = cfg), "verbose <- TRUE", fixed = TRUE)
  }
  topics <- h$wb_help_topics()
  codes <- vapply(topics, h$wb_help_code, character(1))
  parsed <- vapply(codes, function(code) paste(deparse(parse(text = code)), collapse = "\n"), character(1))
  expect_false(anyDuplicated(parsed) > 0)
  expect_true(all(vapply(strsplit(codes, "\n"), function(lines) sum(nzchar(trimws(lines))) <= 14, logical(1))))
})

test_that("bundle preview describes the selected artifact without dumping tensors", {
  h <- custom_help_fixture()
  p <- h$wb_prepare(iris, "Species", names(iris)[1:4], "multiclass")
  cfg <- h$wb_custom_config(h$wb_config(), "members", 3, "4,5")
  bundle <- list(prepared = p, config = cfg, package_version = "test", completed = "test time",
    training_log = c("Epoch 1", "Epoch 2"), recipe = "library(DDESONN)\n# saved recipe")
  manifest <- h$wb_bundle_manifest(bundle)
  expect_match(manifest$Contents[manifest$Component == "Trained parameters"], "2 seeds × 3 members per model", fixed = TRUE)
  expect_match(manifest$Contents[manifest$Component == "Prepared records"], "train: 90", fixed = TRUE)
  expect_match(manifest$Contents[manifest$Component == "Settings"], "epochs = 50", fixed = TRUE)
  expect_equal(manifest$Contents[manifest$Component == "Logs"], "2 retained log lines")
  expect_equal(manifest$Contents[manifest$Component == "Recipe"], "2 lines of saved R code")
  expect_false(any(grepl("Sepal.Length =", manifest$Contents, fixed = TRUE)))
  ui <- as.character(shiny::tagList(h$wb_ui()))
  expect_match(paste(ui, collapse = ""), "wb-workflow-pills", fixed = TRUE)
  expect_match(paste(ui, collapse = ""), 'id="wb_diagnostics" type="checkbox" role="switch"', fixed = TRUE)
})
