#' @keywords internal
"_PACKAGE"

#' @importFrom utils globalVariables
#' @importFrom R6 R6Class
#' @importFrom grDevices dev.list dev.off
#' @importFrom graphics abline grid legend
#' @importFrom stats aggregate dist kmeans na.omit predict rnorm sd setNames
#' @importFrom utils head object.size str tail
NULL

utils::globalVariables(c(
  "Actual","Count","ML_NN","Predicted","RUN_INDEX","SEED","Type",
  "actual_rate","bin_mid","dropout_rates","errors","fpr","label",
  "lambda","lookahead_step","prob","prob_bin","run_index","seed",
  "tpr","beta1", ".BM_DIR"
))
