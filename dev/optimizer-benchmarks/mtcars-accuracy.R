#!/usr/bin/env Rscript
# ACCURACY / IMPLEMENTATION-PERFORMANCE BENCHMARK, not a structural unit test.
# One seed per invocation; optimizers run sequentially in fresh R processes.
# Fixed common settings are deliberately NOT optimizer-specific tuning.
# A single seed and seven test cars cannot establish general superiority.
args <- commandArgs(TRUE)
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1]))
root <- normalizePath(file.path(dirname(script), '../..'))
worker <- length(args)>0 && args[1]=='--worker'
if (worker) args <- args[-1]
if (length(args)<2) stop('Usage: Rscript dev/optimizer-benchmarks/mtcars-accuracy.R SEED OUTPUT_DIRECTORY [OPTIMIZER ...]')
seed <- suppressWarnings(as.integer(args[1])); stopifnot(length(seed)==1, !is.na(seed), seed>=0)
out <- normalizePath(args[2], mustWork=FALSE)
lr <- as.numeric(Sys.getenv('DDESONN_BENCHMARK_LR', '.02'))
stopifnot(length(lr)==1L, is.finite(lr), lr>0)
validation_only <- identical(Sys.getenv('DDESONN_BENCHMARK_VALIDATION_ONLY'), '1')
suppressPackageStartupMessages(pkgload::load_all(root, quiet=TRUE))
methods <- if (length(args)>2) args[-c(1,2)] else ddesonn_optimizer_options()
stopifnot(length(methods)>0, !anyDuplicated(methods), all(methods %in% ddesonn_optimizer_options()))
if (!worker) {
  if (dir.exists(out)) stop('Use a new output directory to avoid mixing runs.')
  dir.create(out, recursive=TRUE)
  writeLines(c('Purpose: accuracy and empirical implementation/performance; NOT structural correctness.',
    paste('Model seed:',seed), 'Split seed: 2026; stratified train=20 validation=5 test=7.',
    'Target: mtcars$am; features: disp,hp,wt,qsec,drat; hidden sizes: 32,16.',
    paste('Epoch budget: 300; learning rate',lr,'; fixed rate; no dropout, self-organization or regularization.'),
    'Standard AdaGrad and other supported optimizers; no legacy AdaGrad X.',
    'Validation selects snapshots. Evaluation threshold is fixed at .5; test labels never tune settings.',
    'Same initialization and split for all optimizers. Features use training-set means/SDs for all splits; automatic normalization is disabled.',
    'Shared settings are not equally optimal for every method. Report all seeds, including poor results.',
    'Accuracy, log loss, Brier score and training seconds are empirical measurements; no pass/fail accuracy threshold.'),
    file.path(out,'protocol.txt'))
  files <- c(script, list.files(file.path(root,'R'), full.names=TRUE), file.path(root,'DESCRIPTION'))
  write.csv(data.frame(file=sub(paste0(root,'/'),'',files,fixed=TRUE),md5=unname(tools::md5sum(files))),
            file.path(out,'source-fingerprint.csv'),row.names=FALSE)
  writeLines(capture.output(sessionInfo()),file.path(out,'session-info.txt'))
  failures <- character(); metrics <- predictions <- list(); initial <- NULL
  for (method in methods) {
    cat(sprintf('Seed %d: %s, 300 epoch budget\n',seed,method)); flush.console()
    dest <- file.path(out,method); dir.create(dest)
    status <- system2(file.path(R.home('bin'),'Rscript'),
      c('--vanilla',shQuote(script),'--worker',seed,shQuote(dest),method),
      stdout=file.path(dest,'run.log'),stderr=file.path(dest,'run.log'))
    if (status!=0) { failures <- c(failures,method); next }
    snapshot <- readRDS(file.path(dest,'initial-and-split.rds'))
    if (is.null(initial)) initial <- snapshot else stopifnot(identical(initial,snapshot))
    metrics[[method]] <- read.csv(file.path(dest,'metrics.csv'))
    predictions[[method]] <- read.csv(file.path(dest,'predictions.csv'))
  }
  if (length(metrics)) write.csv(do.call(rbind,metrics),file.path(out,'metrics.csv'),row.names=FALSE)
  if (length(predictions)) write.csv(do.call(rbind,predictions),file.path(out,'predictions.csv'),row.names=FALSE)
  write.csv(data.frame(optimizer=methods,status=ifelse(methods %in% failures,'failed','completed')),
            file.path(out,'status.csv'),row.names=FALSE)
  if (length(failures)) stop('Failed runs (inspect run.log): ',paste(failures,collapse=', '))
  cat('Completed all optimizers. Results: ',out,'\n')
  quit(status=0)
}
method <- methods[1]; stopifnot(length(methods)==1)
options(DDESONN_OUTPUT_ROOT=out); Sys.setenv(DDESONN_ARTIFACTS_ROOT=out)
set.seed(2026)
splits <- list(train=integer(),validation=integer(),test=integer())
for (label in 0:1) {
  rows <- sample(which(mtcars$am==label)); ntrain <- if(label==1) 8L else 12L
  nvalid <- if(label==1) 2L else 3L
  splits$train <- c(splits$train,rows[seq_len(ntrain)])
  splits$validation <- c(splits$validation,rows[ntrain+seq_len(nvalid)])
  splits$test <- c(splits$test,rows[-seq_len(ntrain+nvalid)])
}
x <- as.matrix(mtcars[,c('disp','hp','wt','qsec','drat')]); y <- matrix(mtcars$am,ncol=1)
center <- colMeans(x[splits$train,,drop=FALSE])
spread <- apply(x[splits$train,,drop=FALSE],2,sd)
stopifnot(all(is.finite(spread)),all(spread>0))
x <- scale(x,center=center,scale=spread)
set.seed(seed)
model <- ddesonn_model(5L,1L,hidden_sizes=c(32L,16L),classification_mode='binary')
saveRDS(list(splits=splits,center=center,scale=spread,weights=lapply(model$ensemble,function(n)n$weights),
             biases=lapply(model$ensemble,function(n)n$biases)),file.path(out,'initial-and-split.rds'))
tr <- splits$train; va <- splits$validation
elapsed <- system.time(ddesonn_fit(model,x[tr,,drop=FALSE],y[tr,,drop=FALSE],
  validation=list(x=x[va,,drop=FALSE],y=y[va,,drop=FALSE]),
  optimizer=method,num_epochs=300L,lr=lr,lr_decay_rate=1,self_org=FALSE,
  reg_type=NULL,numeric_columns=integer(0),dropout_rates=list(NULL,NULL,NULL),batch_normalize_data=FALSE,
  validation_metrics=TRUE,best_weights_on_latest_weights_off=TRUE,viewTables=FALSE,verbose=FALSE))[['elapsed']]
# The reporting flag alone does not replace the live baseline network weights.
# Explicitly restore the validation-selected checkpoint before any predictions.
for (net in model$ensemble) {
  if (is.null(net$best_weights) || is.null(net$best_biases))
    stop('No validation checkpoint was saved; refusing to report final-epoch weights as best.')
  net$weights <- net$best_weights
  net$biases <- net$best_biases
}
saveRDS(lapply(model$ensemble, function(net) list(weights=net$weights, biases=net$biases,
  best_weights=net$best_weights, best_biases=net$best_biases)),
  file.path(out,'evaluated-checkpoint.rds'))
metrics <- predictions <- list(); majority <- as.integer(mean(y[tr,])>=.5)
for (split in if (validation_only) c('train','validation') else names(splits)) {
  rows <- splits[[split]]; actual <- as.numeric(y[rows,])
  p <- as.numeric(ddesonn_predict(model,x[rows,,drop=FALSE],aggregate='mean')$prediction)
  stopifnot(length(p)==length(rows),all(is.finite(p)),all(p>=0 & p<=1))
  predicted <- as.integer(p>=.5); q <- pmin(pmax(p,1e-15),1-1e-15)
  metrics[[split]] <- data.frame(seed=seed,optimizer=method,split=split,n=length(rows),
    epoch_budget=300,learning_rate=lr,threshold=.5,accuracy=mean(predicted==actual),errors=sum(predicted!=actual),
    log_loss=-mean(actual*log(q)+(1-actual)*log(1-q)),brier=mean((p-actual)^2),
    majority_accuracy=mean(actual==majority),training_seconds=elapsed)
  predictions[[split]] <- data.frame(seed=seed,optimizer=method,split=split,
    row=rows,car=rownames(mtcars)[rows],actual=actual,probability=p,predicted=predicted)
}
write.csv(do.call(rbind,metrics),file.path(out,'metrics.csv'),row.names=FALSE)
write.csv(do.call(rbind,predictions),file.path(out,'predictions.csv'),row.names=FALSE)
