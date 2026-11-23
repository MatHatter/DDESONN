# Ten rows per species keep the stratified validation and test sets populated.
workbench_iris <- iris[seq.int(1L, nrow(iris), by = 5L), ]
