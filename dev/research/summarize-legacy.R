#!/usr/bin/env Rscript
# Summarize archived observations; this does not run models or claim superiority.
a <- commandArgs(TRUE)
root <- if (length(a)) a[1] else "."
data_root <- file.path(root,"inst","extdata","heart_failure_runs")
paths <- c(file.path(data_root,"run1","SingleRun_Test_Metrics_500_seeds_20251025.rds"),
           file.path(data_root,"run2","SingleRun_Test_Metrics_500_seeds_20251026.rds"))
stopifnot(all(file.exists(paths)))
x <- lapply(paths,readRDS)
stopifnot(identical(as.integer(sort(x[[1]]$seed)),1:500),
          identical(as.integer(sort(x[[2]]$seed)),501:1000),
          all(vapply(x,nrow,integer(1))==500),!anyDuplicated(c(x[[1]]$seed,x[[2]]$seed)))
v <- c(x[[1]]$accuracy,x[[2]]$accuracy)
summary <- data.frame(stat=c("count","mean","std","min","max"),
 legacy_DDESONN=c(length(v),mean(v),sd(v),min(v),max(v)))
book <- file.path(root,"inst","scripts","vsKeras","1000SEEDSRESULTSvsKeras","1000seedsKeras.xlsx")
if (!requireNamespace("readxl",quietly=TRUE)) stop("Install readxl to inspect the archived workbook.")
k <- readxl::read_excel(book,sheet=2)
summary$Keras_workbook <- k$test_acc[match(summary$stat,k$stat)]
stopifnot(!anyNA(summary),summary$Keras_workbook[1]==1000)
print(summary,digits=12,row.names=FALSE)
cat("Historical summaries only. Original execution settings are not fully reconstructed.\n")
