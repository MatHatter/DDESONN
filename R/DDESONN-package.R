#' DDESONN: Deep Dynamic Experimental Self-Organizing Neural Network Framework
#'
#' High-level helpers for constructing, training, and evaluating
#' Deep Dynamic Experimental Self-Organizing Neural Networks.
#'
#' @docType package
#' @name DDESONN
#' @keywords internal
"_PACKAGE"
#' @importFrom utils globalVariables
NULL
utils::globalVariables(c(
  "Actual","Count","ML_NN","Predicted","RUN_INDEX","SEED","Type",
  "actual_rate","bin_mid","dropout_rates","errors","fpr","label",
  "lambda","lookahead_step","prob","prob_bin","run_index","seed",
  "tpr","beta1"
))
