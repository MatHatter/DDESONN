#' @section Optional Shiny graphical interface:
#' The package includes a guided graphical user interface (GUI) for data
#' preparation, model training/evaluation, saved-model prediction and viewing
#' research results. Install the optional app dependencies once, then launch
#' from an interactive R session:
#' \preformatted{
#' install.packages(c("shiny", "callr"))
#' shiny::runApp(system.file("shiny", "benchmark", package = "DDESONN",
#'                           mustWork = TRUE))
#' }
#' Start with Iris on Data, review Features and prepare the data, train on
#' Build & Train, then inspect Model / Results. Use the saved-model workflow
#' to load a workbench bundle and evaluate or predict rows from a CSV.
#' Information icons, Vignettes and Show full R code provide in-app guidance.
#'
#' Experiments displays saved results without starting training. Current shows
#' the clean matched DDESONN/Keras study with 100 seeds per condition, using an
#' experimental native float32 adapter on one fixed split. Historical compares
#' 1,000 standard-AdaGrad DDESONN runs with archived legacy observations, retains
#' overlapping inputs and does not rerun Keras.
#' \preformatted{
#' vignette("packaged-data-workbench", package = "DDESONN")
#' }
#' @concept Shiny
#' @keywords package
"_PACKAGE"

#' @importFrom utils globalVariables
#' @importFrom R6 R6Class
#' @importFrom grDevices dev.list dev.off
#' @importFrom graphics abline grid legend
#' @importFrom stats aggregate dist kmeans na.omit predict rnorm sd setNames
#' @importFrom utils head object.size str tail
NULL

utils::globalVariables(c(
  "Actual","Count","ML_NN","Metric","Model_Name_Outlier","Predicted",
  "RUN_INDEX","SEED","Type","Value",
  "actual_rate","bin_mid","dropout_rates","errors","fpr","label",
  "lambda","lookahead_step","prob","prob_bin","run_index","seed",
  "tpr","beta1", ".BM_DIR"
))
