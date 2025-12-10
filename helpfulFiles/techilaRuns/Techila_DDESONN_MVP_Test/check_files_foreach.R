#!/usr/bin/env Rscript
options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(foreach)
  library(techila)
})

# ------------------------------------------------------------
#  Purpose: verify that each R file can be eval(parse())'d
#  remotely on a Techila worker (no "cannot open connection").
# ------------------------------------------------------------

# Optional: setwd to repo root if needed
# setwd("C:/path/to/DDESONN")

read_file_text <- function(path) paste(readLines(path, warn = FALSE), collapse = "\n")

files_to_test <- c(
  "R/utils.R",
  "R/activation_functions.R",
  "R/optimizers.R",
  "R/performance_relevance_metrics.R",
  "R/update_weights_block.R",
  "R/update_biases_block.R",
  "R/api.R",
  "R/DDESONN.R",
  "Reports/EvaluatePredictionsReport.R"
)

test_code_list <- lapply(files_to_test, read_file_text)
seed_for_test <- 1L

techila::init()
techila::registerDoTechila()

cat("[FILECHECK] backend=", foreach::getDoParName(),
    " workers=", foreach::getDoParWorkers(), "\n", sep="")

filecheck_df <- foreach::foreach(
  idx = seq_along(files_to_test),
  .combine = rbind,
  .multicombine = TRUE,
  .inorder = FALSE,
  .export = c("test_code_list", "files_to_test", "seed_for_test"),
  .packages = c("methods")
) %dopar% {
  
  this_code <- test_code_list[[idx]]
  this_name <- files_to_test[[idx]]
  ok_eval <- FALSE; eval_err <- NA_character_
  
  tryCatch({
    eval(parse(text = this_code))
    ok_eval <- TRUE
  }, error = function(e) eval_err <<- conditionMessage(e))
  
  data.frame(file = this_name, ok_eval = ok_eval,
             eval_error = eval_err, stringsAsFactors = FALSE)
}

cat("[FILECHECK] rows received:", nrow(filecheck_df), "\n")
print(filecheck_df)

cat("\nInterpretation:\n",
    "- ok_eval == TRUE → file loads fine on worker\n",
    "- ok_eval == FALSE → file failed (likely missing or path issue)\n",
    "Fix any failing file before running runner_single_seed.R\n")
