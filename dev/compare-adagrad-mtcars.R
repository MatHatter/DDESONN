#!/usr/bin/env Rscript
# Matched accuracy comparison using the repository's existing small example.
# adagrad_x preserves the original custom behavior with reported good results;
# standard adagrad is evaluated separately here, without changing that default.
#
# Run from a checkout with DDESONN installed:
# Rscript dev/compare-adagrad-mtcars.R /path/outside/checkout/results
#
# Each worker is a fresh R process. It executes the original example through its
# validation/test evaluation, stopping before the larger A-D scenario runner.
# Current profile overrides optimizer selection and model-initialization seed.
# Set DDESONN_COMPARISON_PROFILE=historical to run the recovered v6.6.1
# 300-epoch script with its historical LR schedule. Also run its original RNG
# flow (no extra model-seed reset), reported separately from the five-seed mean.

args <- commandArgs(trailingOnly=TRUE)
file_arg <- grep("^--file=",commandArgs(),value=TRUE)
if (length(file_arg)!=1L) stop("Run this file with Rscript.")
script <- normalizePath(sub("^--file=","",file_arg),mustWork=TRUE)
root <- dirname(dirname(script))
example <- file.path(root,"inst","scripts","DDESONN_mtcars_A-D_examples.R")
profile <- Sys.getenv("DDESONN_COMPARISON_PROFILE","current")
stopifnot(profile %in% c("current","historical"))
# Cache identity includes the driver, example and installed package code. Never
# silently compare stale results after reinstalling a modified optimizer.
package_path <- find.package("DDESONN")
identity_files <- c(script,example,file.path(package_path,"DESCRIPTION"),
  file.path(package_path,"R","DDESONN.rdb"),file.path(package_path,"R","DDESONN.rdx"))
fingerprint <- list(profile=profile,md5=unname(tools::md5sum(identity_files)))
if (anyNA(fingerprint$md5)) stop("Cannot fingerprint comparison inputs.")

run_case <- function(method,seed,out) {
  suppressPackageStartupMessages(library(DDESONN))
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  options(DDESONN_OUTPUT_ROOT=out)
  Sys.setenv(DDESONN_ARTIFACTS_ROOT=out)
  original_model <- DDESONN::ddesonn_model
  original_fit <- DDESONN::ddesonn_fit
  initial <- NULL
  e <- new.env(parent=globalenv())
  e$ddesonn_model <- function(...) {
    if (!identical(seed,"original")) set.seed(as.integer(seed))
    model <- original_model(...)
    initial <<- lapply(model$ensemble,function(n)list(weights=n$weights,biases=n$biases))
    model
  }
  e$ddesonn_fit <- function(...) {
    fit_args <- list(...)
    fit_args$optimizer <- method
    if (profile=="historical") {
      fit_args$lr_decay_rate <- .5
      fit_args$lr_decay_epoch <- 20L
      fit_args$lr_min <- 1e-5
    }
    do.call(original_fit,fit_args)
  }
  if (profile=="historical") {
    recovered <- file.path(out,"historical-example.R")
    status <- system2("git",c("-C",shQuote(root),"show",
      "b026343b30f80ba5a23bc0cef9f77ff958b811a6:inst/scripts/DDESONN_mtcars_example.R"),
      stdout=recovered)
    if (status!=0L) stop("Cannot recover the historical example from Git.")
    expressions <- parse(recovered)
  } else expressions <- parse(example)
  boundary <- which(vapply(expressions,function(x) {
    is.call(x) && identical(x[[1L]],as.name("<-")) &&
      identical(x[[2L]],as.name("scenario_presets"))
  },logical(1L)))
  indices <- if (profile=="historical") seq_along(expressions) else {
    if (length(boundary)!=1L) stop("Could not locate the example's scenario-runner boundary.")
    seq_len(boundary-1L)
  }
  for (i in indices) {
    x <- expressions[[i]]
    # The recovered script sourced its old API. This comparison deliberately
    # uses the currently installed implementations with the recorded settings.
    if (profile=="historical" && is.call(x) && identical(x[[1L]],as.name("source"))) next
    eval(x,e)
  }
  threshold <- get("THRESHOLD_USED",envir=e,inherits=TRUE)
  valid_acc <- get("VALID_ACC",envir=e,inherits=TRUE)
  test_acc <- get("TEST_ACC",envir=e,inherits=TRUE)
  # Evaluate the original split probabilities against observed labels, retaining
  # raw probabilities rather than the example's rounded display table.
  rows <- list(); metrics <- list()
  train_majority <- as.integer(mean(e$train_y[[1L]])>=.5)
  for (split in c("train","validation","test")) {
    prefix <- if (split=="validation") "valid" else split
    x <- e[[paste0(prefix,"_x")]]
    actual <- as.integer(e[[paste0(prefix,"_y")]][[1L]])
    p <- as.numeric(DDESONN::ddesonn_predict(e$model,x,aggregate="mean")$prediction)
    stopifnot(length(p)==length(actual),all(is.finite(p)),all(p>=0 & p<=1))
    classified <- as.integer(p>=threshold)
    q <- pmin(pmax(p,1e-15),1-1e-15)
    rows[[split]] <- data.frame(optimizer=method,seed=seed,split=split,
      row=seq_along(actual),actual=actual,probability=p,predicted_class=classified,
      probability_error=actual-p,class_error=actual-classified)
    metrics[[split]] <- data.frame(optimizer=method,seed=seed,split=split,n=length(actual),
      threshold=threshold,accuracy=mean(classified==actual),
      errors=sum(classified!=actual),log_loss=-mean(actual*log(q)+(1-actual)*log(1-q)),
      brier=mean((p-actual)^2),majority_accuracy=mean(actual==train_majority))
  }
  metrics <- do.call(rbind,metrics)
  stopifnot(metrics$accuracy[metrics$split=="validation"]==valid_acc,
            metrics$accuracy[metrics$split=="test"]==test_acc)
  snapshot <- list(fingerprint=fingerprint,initial=initial,
    splits=list(train_x=e$train_x,train_y=e$train_y,valid_x=e$valid_x,
      valid_y=e$valid_y,test_x=e$test_x,test_y=e$test_y),
    controls=list(hidden_sizes=c(32L,16L),epochs=if(profile=="historical")300L else 3L,
      lr=.02,lr_decay_rate=if(profile=="historical").5 else 1,
      lr_decay_epoch=if(profile=="historical")20L else 1L,
      split_seed=42L,restore_best=TRUE,profile=profile),
    metrics=metrics,predictions=do.call(rbind,rows))
  saveRDS(snapshot,file.path(out,"result.rds"))
}

if (length(args)>0L && args[[1L]]=="--worker") {
  stopifnot(length(args)==4L,args[[2L]] %in% c("adagrad","adagrad_x"))
  run_case(args[[2L]],args[[3L]],args[[4L]])
} else {
  if (length(args)!=1L) stop("Supply an output directory outside the checkout.")
  out <- normalizePath(args[[1L]],mustWork=FALSE)
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  for (cached in list.files(out,pattern="^result[.]rds$",recursive=TRUE,full.names=TRUE)) {
    if (!identical(readRDS(cached)$fingerprint,fingerprint))
      stop("Existing results were produced by different code/settings; use a fresh output directory.")
  }
  seeds <- if(profile=="historical")c("original",as.character(1:5)) else as.character(1:5)
  tasks <- expand.grid(method=c("adagrad_x","adagrad"),seed=seeds,stringsAsFactors=FALSE)
  run_worker <- function(i) {
    method <- tasks$method[[i]]; seed <- tasks$seed[[i]]
    run_out <- file.path(out,paste0(method,"-seed-",seed))
    dir.create(run_out,recursive=TRUE,showWarnings=FALSE)
    result_file <- file.path(run_out,"result.rds")
    if (!file.exists(result_file)) {
      log <- file.path(run_out,"run.log")
      status <- system2(file.path(R.home("bin"),if (.Platform$OS.type=="windows") "Rscript.exe" else "Rscript"),
        args=c("--vanilla",shQuote(script),"--worker",method,seed,shQuote(run_out)),
        stdout=log,stderr=log)
      if (status!=0L) stop("Worker failed; inspect ",log)
    }
    result <- readRDS(result_file)
    if (!identical(result$fingerprint,fingerprint)) stop("Cached result fingerprint mismatch.")
    result
  }
  # Tiny matrices do not benefit from nested BLAS threads. Independent fits
  # use up to four CPUs, with each fit running in its own fresh R process.
  Sys.setenv(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
  cores <- parallel::detectCores()
  workers <- if (is.na(cores)) 1L else min(4L,cores)
  run_parallel <- function() {
    if (workers==1L) return(lapply(seq_len(nrow(tasks)),run_worker))
    # PSOCK also supports Windows, where fork-based mclapply cannot run.
    cluster <- parallel::makePSOCKcluster(workers)
    on.exit(parallel::stopCluster(cluster),add=TRUE)
    parallel::clusterExport(cluster,c("tasks","out","script","profile","fingerprint","run_worker"),
                            envir=environment(run_worker))
    parallel::parLapply(cluster,seq_len(nrow(tasks)),run_worker)
  }
  records <- run_parallel()
  if (any(vapply(records,inherits,logical(1L),"try-error"))) stop("A comparison worker failed.")
  for (seed in seeds) {
    pair <- records[tasks$seed==seed]
    stopifnot(identical(pair[[1]]$initial,pair[[2]]$initial),
              identical(pair[[1]]$splits,pair[[2]]$splits),
              identical(pair[[1]]$controls,pair[[2]]$controls))
  }
  metrics <- do.call(rbind,lapply(records,function(x)x$metrics))
  predictions <- do.call(rbind,lapply(records,function(x)x$predictions))
  repeated <- metrics[metrics$seed!="original",]
  summary <- do.call(rbind,lapply(split(repeated,list(repeated$optimizer,repeated$split)),function(x) {
    data.frame(optimizer=x$optimizer[[1L]],split=x$split[[1L]],seeds=nrow(x),n=x$n[[1L]],
      accuracy_mean=mean(x$accuracy),accuracy_min=min(x$accuracy),
      accuracy_max=max(x$accuracy),log_loss_mean=mean(x$log_loss),
      brier_mean=mean(x$brier),majority_accuracy=x$majority_accuracy[[1L]])
  }))
  write.csv(metrics,file.path(out,"metrics.csv"),row.names=FALSE)
  write.csv(predictions,file.path(out,"predictions.csv"),row.names=FALSE)
  write.csv(summary,file.path(out,"summary.csv"),row.names=FALSE)
  if(profile=="historical") {
    original <- metrics[metrics$seed=="original",]
    write.csv(original,file.path(out,"original-seed-flow.csv"),row.names=FALSE)
    cat("Original script RNG flow (split seed 42, no additional model seed reset):\n")
    print(original,row.names=FALSE)
  }
  cat("Five matched model-seed averages:\n")
  print(summary,row.names=FALSE)
  cat("Matched initial parameters, data splits and controls verified for all five pairs.\n")
  cat("This small seven-row test split is exploratory, not a general optimizer ranking.\n")
}
