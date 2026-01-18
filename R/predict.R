#' Predict method for DDESONN models
#'
#' @param object A ddesonn_model (R6) returned by ddesonn_run() / ddesonn_model().
#' @param newdata Matrix/data.frame of predictors.
#' @param ... Unused.
#' @param aggregate Aggregation strategy across ensemble members.
#' @param type Prediction type. `"response"` returns numeric predictions,
#'   while `"class"` returns class labels for classification problems.
#' @param threshold Optional threshold override when `type = "class"`.
#' @export
predict.ddesonn_model <- function(object,
                                  newdata,
                                  ...,
                                  aggregate = c("mean", "median", "none"),
                                  type = c("response", "class"),
                                  threshold = NULL) {
  if (is.null(object) || !inherits(object, "ddesonn_model")) {
    stop("'object' must be a ddesonn_model.", call. = FALSE)
  }
  if (is.null(newdata)) {
    stop("'newdata' is required.", call. = FALSE)
  }
  aggregate <- match.arg(aggregate)
  type <- match.arg(type)
  preds <- ddesonn_predict(
    model = object,
    new_data = newdata,
    aggregate = aggregate,
    type = type,
    threshold = threshold
  )
  if (type == "class") {
    return(preds$class)
  }
  if (identical(aggregate, "none")) {
    return(preds$per_model)
  }
  list(predicted_output = preds$prediction)
}
