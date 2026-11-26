# Ten rows per species keep the stratified validation and test sets populated.
workbench_iris <- iris[seq.int(1L, nrow(iris), by = 5L), ]

# Balanced compact rows keep binary and regression training fixtures representative.
workbench_mtcars <- local({
  class_rows <- split(seq_len(nrow(mtcars)), mtcars$am)
  rows <- unlist(lapply(class_rows, head, n = 8L), use.names = FALSE)
  mtcars[rows, , drop = FALSE]
})
