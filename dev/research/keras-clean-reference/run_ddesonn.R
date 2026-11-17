args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); average <- as.logical(args[2])
first <- as.integer(args[3]); last <- as.integer(args[4]); repo <- getwd()
suppressPackageStartupMessages(library(DDESONN))
source(file.path(repo, "R/keras_clean_reference.R"))
.ddesonn_gradient_reduction <- DDESONN:::.ddesonn_gradient_reduction
source(file.path(repo, "dev/research/keras-clean-reference/compile.R"))
reference_compile(repo)
x <- matrix(readBin(file.path(root,"x-float32.bin"), "numeric", 60000, size=4, endian="little"),5000,12)
y <- matrix(readBin(file.path(root,"y-float32.bin"), "numeric", 5000, size=4, endian="little"),5000,1)
unpack <- function(tensors) lapply(tensors,function(a)matrix(unlist(a$values),a$shape[[1]],a$shape[[2]]))
protocol <- jsonlite::fromJSON(file.path(root,"protocol.json"))
n_train <- protocol$rows$train; n_val <- protocol$rows$validation
stopifnot(digest::digest(file=file.path(repo,"inst/extdata/heart_failure_clinical_records.csv"),algo="sha256")==protocol$dataset_sha256,
 digest::digest(file=file.path(root,"split-rows.csv"),algo="sha256")==protocol$split_sha256,
 digest::digest(file=file.path(root,"x-float32.bin"),algo="sha256")==protocol$audit$standardized_float32_sha256,
 digest::digest(file=file.path(root,"y-float32.bin"),algo="sha256")==protocol$audit$labels_sha256,
 all(unlist(protocol$audit$raw_predictor_overlap)==0),all(unlist(protocol$audit$float32_input_overlap)==0))
mode <- if(average) "mean" else "sum"
for(seed in first:last) {
  output <- file.path(root,sprintf("ddesonn-%s-%d.json",mode,seed))
  if(file.exists(output)) next
  init <- jsonlite::fromJSON(file.path(root,sprintf("initial-%d.json",seed)),simplifyVector=FALSE)
  model <- ddesonn_model(12,1,hidden_sizes=c(64L,32L),num_networks=1L,
                         classification_mode="binary",average_gradients=average)
  cat("DDESONN",mode,"seed",seed,": starting 360 epochs\n");flush.console()
  result <- .ddesonn_fit_keras_clean_reference(model$ensemble[[1]],x,y,unpack(init$parameters),
                          unpack(init$moving),seed,average,kr_fit,n_train,n_val)
  check <- kr_predict(result$parameters,result$moving,x,n_train,n_val)
  stopifnot(identical(check,result$probability))
  parts <- list(train=seq_len(n_train),validation=seq.int(n_train+1,n_train+n_val),test=seq.int(n_train+n_val+1,5000))
  metrics <- lapply(names(parts),function(name){i<-parts[[name]]
    correct <- sum((result$probability[i]>.5)==y[i])
    list(split=name,rows=length(i),correct=correct,accuracy=correct/length(i))})
  payload <- list(seed=seed,framework="ddesonn",reduction=mode,average_gradients=average,
                   epochs=result$epochs,optimizer_updates=result$updates,checkpoint="final",metrics=metrics,
                   validation_epochs=length(result$validation_accuracy),validation_steps=result$validation_steps,
                   validation_batch_size=16L,evaluation_batch_size=32L,
                   split_sha256=protocol$split_sha256,dataset_sha256=protocol$dataset_sha256,
                   implementation="opt-in native original-Keras clean-split protocol adapter")
  tmp <- paste0(output,".tmp");jsonlite::write_json(payload,tmp,auto_unbox=TRUE,pretty=TRUE,digits=17)
  stopifnot(file.rename(tmp,output))
  cat("DDESONN",mode,"seed",seed,": test",metrics[[3]]$accuracy,"\n");flush.console()
}
