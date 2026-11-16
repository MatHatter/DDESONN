# Archived-settings benchmark. Results only: no saved models or seed RDS files.
section_three_fit <- function(split, seed, average_gradients, destination, epochs=360L) {
 suppressPackageStartupMessages(library(DDESONN))
 cfg <- split$config; p <- split$prepared; d <- split$data
 dir.create(destination,recursive=TRUE,showWarnings=FALSE)
 previous <- grDevices::dev.cur(); devices <- grDevices::dev.list()
 grDevices::pdf(file=NULL)
 on.exit({for(device in rev(setdiff(grDevices::dev.list(),devices)))grDevices::dev.off(device)
   if(previous %in% grDevices::dev.list())grDevices::dev.set(previous)},add=TRUE)
 set.seed(seed)
 acts <- ddesonn_activation_defaults("binary",c(64,32))
 model <- ddesonn_model(12,1,hidden_sizes=c(64L,32L),num_networks=1L,N=109L,
  lambda=.00028,classification_mode="binary",activation_functions=acts,
  activation_functions_predict=acts,init_method="he",custom_scale=1.04349,
  average_gradients=average_gradients)
 net <- model$ensemble[[1]]
 tr<-p$parts$train;va<-p$parts$validation
 raw_train <- d[tr$row,setdiff(names(d),"DEATH_EVENT"),drop=FALSE]
 result <- model$train(Rdata=tr$x,labels=tr$y,X_train=raw_train,
  y_train=data.frame(DEATH_EVENT=d$DEATH_EVENT[tr$row]),
  lr=.125,lr_decay_rate=.5,lr_decay_epoch=20,lr_min=1e-5,
  num_networks=1L,ensemble_number=0L,do_ensemble=FALSE,num_epochs=epochs,
  self_org=FALSE,threshold=.5,reg_type="L1",numeric_columns=names(raw_train),
  CLASSIFICATION_MODE="binary",activation_functions=acts,activation_functions_predict=acts,
  dropout_rates=list(.1),optimizer="adagrad",beta1=.9,beta2=.8,epsilon=1e-7,
  lookahead_step=5L,batch_normalize_data=TRUE,gamma_bn=.6,beta_bn=.6,
  epsilon_bn=1e-6,momentum_bn=.9,is_training_bn=TRUE,shuffle_bn=FALSE,
  loss_type="CrossEntropy",update_weights=TRUE,update_biases=TRUE,sample_weights=NULL,
  preprocessScaledData=NULL,X_validation=va$x,y_validation=va$y,
  validation_metrics=TRUE,threshold_function=DDESONN:::tune_threshold_accuracy,
  best_weights_on_latest_weights_off=TRUE,ML_NN=TRUE,train=TRUE,grouped_metrics=FALSE,
  viewTables=FALSE,verbose=FALSE,verboseLow=FALSE,output_root=destination,
  plot_controls=list(),save_per_epoch=FALSE,average_gradients=average_gradients)
 selected <- result$predicted_outputAndTime$best_val_epoch
 stopifnot(length(selected)==1L,is.finite(selected),!is.null(net$best_weights),!is.null(net$best_biases))
 net$weights<-net$best_weights;net$biases<-net$best_biases
 predictions<-lapply(p$parts,function(part)as.numeric(net$predict(part$x,
   net$weights,net$biases,acts,verbose=FALSE,debug=FALSE)$predicted_output))
 h<-new.env(parent=globalenv());sys.source(cfg$metric_runner,h)
 metrics<-do.call(rbind,lapply(names(p$parts),function(name){part<-p$parts[[name]]
   cbind(seed=seed,average_gradients=average_gradients,split=name,selected_epoch=selected,
    requested_epochs=epochs,h$clean_split_metrics(part$y,predictions[[name]],part$group))}))
 metrics
}

section_three_inputs <- function(root) {
 suppressPackageStartupMessages(library(DDESONN))
 dir.create(root,recursive=TRUE,showWarnings=FALSE)
 dataset<-system.file('extdata','heart_failure_clinical_records.csv',package='DDESONN')
 d<-read.csv(dataset)
 helper<-system.file('scripts','heart_failure_clean_split.R',package='DDESONN')
 h<-new.env(parent=globalenv());sys.source(helper,h)
 p<-h$clean_split_prepare(d,42L)
 rows<-data.frame(row=unlist(lapply(p$parts,`[[`,'row'),use.names=FALSE),
  split=rep(names(p$parts),vapply(p$parts,function(x)length(x$row),integer(1))))
 rowfile<-file.path(root,'split-rows.csv');datafile<-file.path(root,'data.csv')
 if(file.exists(rowfile))stopifnot(identical(read.csv(rowfile),rows)) else
  write.csv(rows,rowfile,row.names=FALSE)
 if(file.exists(datafile))stopifnot(identical(read.csv(datafile),d)) else file.copy(dataset,datafile)
 list(config=list(metric_runner=helper),prepared=p,data=d)
}

if(sys.nframe()==0L){
 args<-commandArgs(trailingOnly=TRUE)
 if(!length(args)%in%c(2L,4L))stop('Usage: Rscript section_three.R OUTPUT prepare|FALSE|TRUE [FIRST LAST]')
 root<-normalizePath(args[1],mustWork=FALSE);mode<-args[2]
 stopifnot(mode%in%c('prepare','FALSE','TRUE'))
 split<-section_three_inputs(root)
 if(mode!='prepare'){
  first<-if(length(args)==4L)as.integer(args[3]) else 1L
  last<-if(length(args)==4L)as.integer(args[4]) else 10L
  stopifnot(!is.na(first),!is.na(last),first>=1L,last>=first)
  output<-file.path(root,paste0('ddesonn-',tolower(mode),'-',first,'-',last))
  stopifnot(!dir.exists(output));dir.create(output)
  script<-normalizePath(sub('^--file=','',commandArgs()[grepl('^--file=',commandArgs())]))
  rows<-list()
  for(seed in seq.int(first,last)){
   destination<-file.path(output,paste0('seed-',seed))
   metric<-callr::r(function(script,split,seed,flag,destination){
    h<-new.env(parent=globalenv());sys.source(script,h)
    h$section_three_fit(split,seed,flag,destination)
   },args=list(script=script,split=split,seed=seed,flag=mode=='TRUE',destination=destination),
    libpath=.libPaths(),stdout=file.path(output,paste0('seed-',seed,'.log')),
    stderr=file.path(output,paste0('seed-',seed,'-stderr.log')))
   rows[[length(rows)+1L]]<-metric
   write.csv(do.call(rbind,rows),file.path(output,'metrics.csv'),row.names=FALSE)
   cat('Seed',seed,'test accuracy',metric$accuracy[metric$split=='test'],'\n')
  }
 }
}
