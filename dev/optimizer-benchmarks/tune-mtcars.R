#!/usr/bin/env Rscript
# PERFORMANCE DIAGNOSIS: tune on validation labels, never test labels.
# One model seed per invocation; fresh sequential processes for every candidate.
a <- commandArgs(TRUE)
if(length(a)!=2) stop('Usage: Rscript dev/optimizer-benchmarks/tune-mtcars.R SEED NEW_OUTPUT_DIRECTORY')
seed <- as.integer(a[1]); stopifnot(!is.na(seed),seed>=0)
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1]))
benchmark <- file.path(dirname(script),'mtcars-accuracy.R')
out <- normalizePath(a[2],mustWork=FALSE)
if(dir.exists(out)) stop('Output directory already exists.')
dir.create(out,recursive=TRUE)
# Equal tuning budget for every optimizer; grid fixed before this comparison.
rates <- c(.001,.02,.1,1)
root <- normalizePath(file.path(dirname(script),'../..'))
suppressPackageStartupMessages(pkgload::load_all(root,quiet=TRUE))
methods <- ddesonn_optimizer_options()
writeLines(c('Validation-only learning-rate search; seed fixed across candidates.',
 paste('Methods:',paste(methods,collapse=','),'. Grid: .001,.02,.1,1. Epoch budget: 300.'),
 'Selection: lowest validation log loss; ties use earlier grid entry.',
 'Candidate processes do not evaluate the test split. Test evaluated only after selection.',
 'This is tuning evidence, not a formula change or general accuracy guarantee.'),file.path(out,'protocol.txt'))
write.csv(data.frame(file=c(basename(script),basename(benchmark)),md5=unname(tools::md5sum(c(script,benchmark)))),file.path(out,'scripts.csv'),row.names=FALSE)
run <- function(method,rate,dest,validation_only) {
 dir.create(dest,recursive=TRUE)
 Sys.setenv(DDESONN_BENCHMARK_LR=as.character(rate),
            DDESONN_BENCHMARK_VALIDATION_ONLY=if(validation_only)'1' else '0')
 status <- system2(file.path(R.home('bin'),'Rscript'),c('--vanilla',shQuote(benchmark),
   '--worker',seed,shQuote(dest),method),stdout=file.path(dest,'run.log'),stderr=file.path(dest,'run.log'))
 if(status!=0) {
  if(validation_only) return(NULL)
  stop('Selected run failed: ',dest)
 }
 read.csv(file.path(dest,'metrics.csv'))
}
candidates <- selected <- list(); initial <- NULL
for(method in methods) {
 scores <- list()
 for(rate in rates) {
  cat(method,'lr',rate,'validation only\n');flush.console()
  dest <- file.path(out,method,paste0('lr-',rate))
  m <- run(method,rate,dest,TRUE)
  if(is.null(m)) {
    scores[[as.character(rate)]] <- data.frame(seed=seed,optimizer=method,
      learning_rate=rate,status='failed',validation_accuracy=NA_real_,validation_log_loss=Inf)
    next
  }
  stopifnot(!'test' %in% m$split)
  snapshot <- readRDS(file.path(dest,'initial-and-split.rds'))
  if(is.null(initial)) initial <- snapshot else stopifnot(identical(initial,snapshot))
  v <- m[m$split=='validation',];stopifnot(nrow(v)==1L)
  scores[[as.character(rate)]] <- data.frame(seed=seed,optimizer=method,
    learning_rate=rate,status='completed',validation_accuracy=v$accuracy,validation_log_loss=v$log_loss)
 }
 scores <- do.call(rbind,scores); candidates[[method]] <- scores
 if(!any(is.finite(scores$validation_log_loss))) stop('No usable candidate for ',method)
 winner <- scores$learning_rate[which.min(scores$validation_log_loss)]
 cat(method,'selected lr',winner,'by validation log loss; now evaluating test\n');flush.console()
 selected[[method]] <- run(method,winner,file.path(out,method,'selected'),FALSE)
 write.csv(do.call(rbind,candidates),file.path(out,'validation-candidates.csv'),row.names=FALSE)
 write.csv(do.call(rbind,selected),file.path(out,'selected-metrics.csv'),row.names=FALSE)
}
write.csv(do.call(rbind,candidates),file.path(out,'validation-candidates.csv'),row.names=FALSE)
write.csv(do.call(rbind,selected),file.path(out,'selected-metrics.csv'),row.names=FALSE)
cat('Completed:',out,'\n')
