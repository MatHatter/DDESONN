# [SSM] Joint training: pass final prediction gradients through each encoder.
.ssm_joint_context <- function(encoder, sequence_data, validation_sequence = NULL,
                               engineered_features, lr = NULL) {
  if (!is.null(lr) && (length(lr) != 1L || !is.finite(lr) || lr < 0))
    stop("ssm_lr must be one finite non-negative number.", call. = FALSE)
  prepare <- function(x) {
    .ddesonn_validate_sequence(x)
    groups <- .ssm_sequence_groups(x)
    scaled <- x[groups$first, , , drop = FALSE]
    for (j in seq_len(dim(scaled)[3L]))
      scaled[, , j] <- (scaled[, , j] - encoder$scale$center[j]) / encoder$scale$scale[j]
    list(x = scaled, group = groups$group)
  }
  context <- new.env(parent = emptyenv())
  context$encoder <- encoder
  context$train <- prepare(sequence_data)
  context$validation <- if (!is.null(validation_sequence)) prepare(validation_sequence) else NULL
  context$columns <- engineered_features + seq_len(encoder$config$state_dim)
  context$lr <- lr
  context$gradient_norm <- context$parameter_change <- numeric()
  context$best_encoder <- NULL
  context
}

.ssm_joint_inputs <- function(context, train, validation = NULL,
                              encoder = context$encoder, cache = FALSE) {
  forward <- ddesonn_ssm_forward(encoder, context$train$x, return_cache = cache)
  if (cache) context$cache <- forward$cache
  train <- as.matrix(train)
  train[, context$columns] <- forward$embedding[context$train$group, , drop = FALSE]
  if (!is.null(validation)) {
    if (is.null(context$validation)) stop("SSM validation sequences are missing.", call. = FALSE)
    embedding <- ddesonn_ssm_forward(encoder, context$validation$x)$embedding
    validation <- as.matrix(validation)
    validation[, context$columns] <- embedding[context$validation$group, , drop = FALSE]
  }
  list(train = train, validation = validation)
}

.ssm_joint_step <- function(context, input_gradient, lr) {
  input_gradient <- as.matrix(input_gradient)
  expected <- c(length(context$train$group), max(context$columns))
  .ssm_assert_shape(input_gradient, expected, "head input gradient")
  if (any(!is.finite(input_gradient)))
    stop("SSM head input gradient must be finite.", call. = FALSE)
  gradient <- input_gradient[, context$columns, drop = FALSE]
  # Sum every row's conditional prediction gradient before averaging samples.
  # Different trades sharing a market window keep their different head gradients.
  gradient <- rowsum(gradient, context$train$group, reorder = FALSE)
  gradient <- gradient * dim(context$train$x)[1L] / nrow(input_gradient)
  grads <- ddesonn_ssm_backward(context$encoder, context$cache, gradient)$params
  before <- context$encoder$params
  context$encoder <- .ddesonn_ssm_update(context$encoder, grads, context$lr %||% lr)
  context$gradient_norm <- c(context$gradient_norm, sqrt(sum(unlist(grads)^2)))
  context$parameter_change <- c(context$parameter_change,
                                 sqrt(sum((unlist(context$encoder$params) - unlist(before))^2)))
  invisible(context$encoder)
}
