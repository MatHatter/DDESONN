library(DDESONN)
h <- new.env(parent = globalenv())
sys.source(system.file('shiny', 'benchmark', 'workbench_helpers.R', package = 'DDESONN'), h)
out <- Sys.getenv('DDESONN_MODEL_EXPORT', unset = file.path(getwd(), 'customer-segmentation-export'))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
d <- read.csv(system.file('extdata','train_multiclass_customer_segmentation.csv',package='DDESONN'), check.names=FALSE)
features <- setdiff(names(d), c('ID','Segmentation'))
pp <- h$wb_pp_config()
# The grouping helper groups raw retained predictors for any class labels.
# It does not include the target in the identity of an input.
grouping <- d[features]
grouping[] <- lapply(grouping, function(v) {
  if (is.character(v)) {v <- trimws(v); v[is.na(v)|v==''] <- NA_character_}
  v
})
identity <- match(vapply(seq_len(nrow(grouping)), function(i) digest::digest(lapply(grouping,function(v)v[i]),algo='sha256'), character(1)), unique(vapply(seq_len(nrow(grouping)), function(i) digest::digest(lapply(grouping,function(v)v[i]),algo='sha256'), character(1))))
# Merge groups if training-fitted preprocessing makes their inputs identical,
# then refit on the revised training split. No models/test accuracy select splits.
for (iteration in seq_len(20)) {
  g <- h$wb_binary_split(data.frame(identity=identity), d$Segmentation, 42L, .6, .2, 'random')
  ids <- g$ids
  fitted <- h$wb_pp_fit(d[ids$train, , drop=FALSE], features, pp)
  x <- h$wb_pp_apply(d, fitted)
  center <- colMeans(x[ids$train, , drop=FALSE])
  spread <- apply(x[ids$train, , drop=FALSE], 2, sd)
  spread[!is.finite(spread)|spread==0] <- 1
  x <- sweep(sweep(x, 2, center, '-'), 2, spread, '/')
  keys <- apply(x, 1, function(v) digest::digest(unname(v), algo='sha256'))
  sets <- lapply(ids, function(i) unique(keys[i]))
  if (!length(intersect(sets$train,sets$validation)) && !length(intersect(sets$train,sets$test)) && !length(intersect(sets$validation,sets$test))) break
  parent <- seq_len(nrow(d))
  find <- function(i) {while(parent[i]!=i)i<-parent[i];i}
  for (rows in unname(split(seq_len(nrow(d)),identity))) for (i in rows) parent[find(i)] <- find(rows[1])
  for (rows in unname(split(seq_len(nrow(d)),keys))) for (i in rows) parent[find(i)] <- find(rows[1])
  identity <- vapply(seq_len(nrow(d)),find,integer(1))
}
stopifnot(!length(intersect(sets$train,sets$validation)),!length(intersect(sets$train,sets$test)),!length(intersect(sets$validation,sets$test)))
classes <- sort(unique(as.character(d$Segmentation[ids$train])))
y <- matrix(0,nrow(d),length(classes)); y[cbind(seq_len(nrow(d)),match(d$Segmentation,classes))] <- 1
parts <- lapply(ids,function(i)list(x=x[i,,drop=FALSE],y=y[i,,drop=FALSE],actual=d$Segmentation[i],row=i))
p <- list(parts=parts, features=features, target='Segmentation', mode='multiclass',
  classes=classes, center=center, scale=spread, seed=42L, preprocessing=fitted,
  encoded_features=colnames(x), split='random',standardize=TRUE,train_fraction=.6,
  validation_fraction=.2,rows=nrow(d),split_audit=g$audit,
  source=list(package_file='train_multiclass_customer_segmentation.csv'))
cfg <- h$wb_config(hidden='32,16', epochs=100, lr=.003, optimizer='adam', checkpoint='best_validation')
logfile <- file.path(out,'training.log')
con <- file(logfile,open='wt'); sink(con)
bundle <- tryCatch(h$wb_run(p,cfg),finally={sink();close(con)})
bundle$history <- h$wb_log_history(logfile)
bundle$training_log <- readLines(logfile,warn=FALSE)
path <- file.path(out,'customer-segmentation-model-7.3.31.rds')
saveRDS(bundle,path,compress='xz')
stopifnot(file.info(path)$size < 50*1024^2)
restored <- h$wb_validate_bundle(readRDS(path))
new <- read.csv(system.file('extdata','test_multiclass_customer_segmentation.csv',package='DDESONN'),check.names=FALSE)
new <- head(new[features],20)
write.csv(new,file.path(out,'customer-rows-to-predict.csv'),row.names=FALSE,na='')
# Round-trip the exact CSV the user will upload.
new <- read.csv(file.path(out,'customer-rows-to-predict.csv'),check.names=FALSE)
a <- h$wb_new_predictions(bundle,new); b <- h$wb_new_predictions(restored,new)
stopifnot(isTRUE(all.equal(a,b)),nrow(b)==20L,all(is.finite(as.matrix(b[grep('probability_',names(b))]))))
write.csv(b,file.path(out,'expected-predictions.csv'),row.names=FALSE)
metrics <- do.call(rbind,lapply(names(bundle$evaluation),function(n)cbind(Split=n,bundle$evaluation[[n]]$metrics)))
write.csv(metrics,file.path(out,'evaluation.csv'),row.names=FALSE)
print(metrics)
cat('Selected epoch:',bundle$selected_epoch,'\nRows:',vapply(parts,function(z)nrow(z$x),integer(1)), '\nBundle bytes:',file.info(path)$size,'\n')
print(g$audit)
