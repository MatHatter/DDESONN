# ===============================================================
# DeepDynamic - DDESONN
# Deep Dynamic Experimental Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
#
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

# -------------------------
# Activation Functions
# -------------------------

#' Activation functions (DDESONN)
#'
#' A collection of activation functions used by DDESONN.
#' These functions operate on numeric vectors/matrices and preserve shape.
#'
#' @details
#' Many functions coerce inputs to matrix form and preserve dimensions.
#' Some functions are experimental and may not be suitable for training.
#'
#' @name ddesonn_activations
NULL

#' Binary step activation
#' @param x Numeric vector/matrix.
#' @return A matrix of 0/1 values with the same dimensions as `x`.
#' @export
#' @rdname ddesonn_activations
binary_activation <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0.5, 1, 0))
}
attr(binary_activation, "name") <- "binary_activation"

#' Custom binary activation (experimental)
#' @param x Numeric vector/matrix.
#' @param threshold Threshold for step.
#' @return A matrix of 0/1 values with the same dimensions as `x`.
#' @export
#' @rdname ddesonn_activations
custom_binary_activation <- function(x, threshold = -1.08) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x < threshold, 0, 1))
}
attr(custom_binary_activation, "name") <- "custom_binary_activation"

#' Custom activation (experimental)
#' @param z Numeric vector/matrix.
#' @return A matrix of 0/1 values with the same dimensions as `z`.
#' @export
#' @rdname ddesonn_activations
custom_activation <- function(z) {
  z <- as.matrix(z); dim(z) <- dim(z)
  softplus_output <- log1p(exp(z))
  return(ifelse(softplus_output > 1e-11, 1, 0))
}
attr(custom_activation, "name") <- "custom_activation"

#' Bent identity activation
#' @param x Numeric vector/matrix.
#' @return Bent identity transform.
#' @export
#' @rdname ddesonn_activations
bent_identity <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return((sqrt(x^2 + 1) - 1) / 2 + x)
}
attr(bent_identity, "name") <- "bent_identity"

#' ReLU activation
#' @param x Numeric vector/matrix.
#' @return ReLU transform.
#' @export
#' @rdname ddesonn_activations
relu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, 0))
}
attr(relu, "name") <- "relu"

#' Softplus activation
#' @param x Numeric vector/matrix.
#' @return Softplus transform.
#' @export
#' @rdname ddesonn_activations
softplus <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(log1p(exp(x)))
}
attr(softplus, "name") <- "softplus"

#' Leaky ReLU activation
#' @param x Numeric vector/matrix.
#' @param alpha Leak factor.
#' @return Leaky ReLU transform.
#' @export
#' @rdname ddesonn_activations
leaky_relu <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * x))
}
attr(leaky_relu, "name") <- "leaky_relu"

#' ELU activation
#' @param x Numeric vector/matrix.
#' @param alpha ELU alpha.
#' @return ELU transform.
#' @export
#' @rdname ddesonn_activations
elu <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}
attr(elu, "name") <- "elu"

#' Hyperbolic tangent activation
#' @param x Numeric vector/matrix.
#' @return tanh transform.
#' @export
#' @rdname ddesonn_activations
tanh <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
}
attr(tanh, "name") <- "tanh"

#' Sigmoid activation
#' @param x Numeric vector/matrix.
#' @return Sigmoid transform.
#' @export
#' @rdname ddesonn_activations
sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + exp(-x)))
}
attr(sigmoid, "name") <- "sigmoid"

#' Hard sigmoid activation
#' @param x Numeric vector/matrix.
#' @return Hard sigmoid transform.
#' @export
#' @rdname ddesonn_activations
hard_sigmoid <- function(x) {
  x <- as.matrix(x)
  out <- pmax(0, pmin(1, 0.2 * x + 0.5))
  dim(out) <- dim(x)  # Preserve shape no matter what
  return(out)
}
attr(hard_sigmoid, "name") <- "hard_sigmoid"

#' Swish activation
#' @param x Numeric vector/matrix.
#' @return Swish transform.
#' @export
#' @rdname ddesonn_activations
swish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * sigmoid(x))
}
attr(swish, "name") <- "swish"

#' Sigmoid then binary threshold
#' @param x Numeric vector/matrix.
#' @return 0/1 matrix.
#' @export
#' @rdname ddesonn_activations
sigmoid_binary <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse((1 / (1 + exp(-x))) >= 0.5, 1, 0))
}
attr(sigmoid_binary, "name") <- "sigmoid_binary"

#' GELU activation
#' @param x Numeric vector/matrix.
#' @return GELU transform.
#' @export
#' @rdname ddesonn_activations
gelu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * 0.5 * (1 + erf(x / sqrt(2))))
}
attr(gelu, "name") <- "gelu"

#' SELU activation
#' @param x Numeric vector/matrix.
#' @param lambda SELU lambda.
#' @param alpha SELU alpha.
#' @return SELU transform.
#' @export
#' @rdname ddesonn_activations
selu <- function(x, lambda = 1.0507, alpha = 1.67326) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(lambda * ifelse(x > 0, x, alpha * exp(x) - alpha))
}
attr(selu, "name") <- "selu"

#' Mish activation
#' @param x Numeric vector/matrix.
#' @return Mish transform.
#' @export
#' @rdname ddesonn_activations
mish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * tanh(log(1 + exp(x))))
}
attr(mish, "name") <- "mish"

#' PReLU activation (parametric ReLU; alpha fixed here)
#' @param x Numeric vector/matrix.
#' @param alpha Negative slope.
#' @return PReLU transform.
#' @export
#' @rdname ddesonn_activations
prelu <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * x))
}
attr(prelu, "name") <- "prelu"

#' Softmax activation (row-wise)
#' @param z Numeric vector/matrix.
#' @return Row-wise softmax probabilities.
#' @export
#' @rdname ddesonn_activations
softmax <- function(z) {
  z <- as.matrix(z); dim(z) <- dim(z)
  exp_z <- exp(z)
  return(exp_z / rowSums(exp_z))
}
attr(softmax, "name") <- "softmax"

#' Maxout activation (example)
#' @param x Numeric vector/matrix.
#' @param w1 Weight 1.
#' @param b1 Bias 1.
#' @param w2 Weight 2.
#' @param b2 Bias 2.
#' @return Elementwise max of two affine transforms.
#' @export
#' @rdname ddesonn_activations
maxout <- function(x, w1 = 0.5, b1 = 1.0, w2 = -0.5, b2 = 0.5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(pmax(w1 * x + b1, w2 * x + b2))
}
attr(maxout, "name") <- "maxout"

#' Bent ReLU activation
#' @param x Numeric vector/matrix.
#' @return Bent ReLU transform.
#' @export
#' @rdname ddesonn_activations
bent_relu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  out <- pmax(0, ((sqrt(x^2 + 1) - 1) / 2 + x))
  dim(out) <- dim(x)
  return(out)
}
attr(bent_relu, "name") <- "bent_relu"

#' Bent sigmoid activation
#' @param x Numeric vector/matrix.
#' @return Bent sigmoid transform.
#' @export
#' @rdname ddesonn_activations
bent_sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent_part <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  return(1 / (1 + exp(-bent_part)))
}
attr(bent_sigmoid, "name") <- "bent_sigmoid"

#' Arctangent activation
#' @param x Numeric vector/matrix.
#' @return atan transform.
#' @export
#' @rdname ddesonn_activations
arctangent <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(atan(x))
}
attr(arctangent, "name") <- "arctangent"

#' Sinusoid activation
#' @param x Numeric vector/matrix.
#' @return sin transform.
#' @export
#' @rdname ddesonn_activations
sinusoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(sin(x))
}
attr(sinusoid, "name") <- "sinusoid"

#' Gaussian activation
#' @param x Numeric vector/matrix.
#' @return exp(-x^2).
#' @export
#' @rdname ddesonn_activations
gaussian <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(exp(-x^2))
}
attr(gaussian, "name") <- "gaussian"

#' ISRLU activation
#' @param x Numeric vector/matrix.
#' @param alpha Alpha parameter.
#' @return ISRLU transform.
#' @export
#' @rdname ddesonn_activations
isrlu <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x >= 0, x, x / sqrt(1 + alpha * x^2)))
}
attr(isrlu, "name") <- "isrlu"

#' Bent swish activation
#' @param x Numeric vector/matrix.
#' @return Bent swish transform.
#' @export
#' @rdname ddesonn_activations
bent_swish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  return(bent * sigmoid(x))
}
attr(bent_swish, "name") <- "bent_swish"

#' Parametric bent ReLU activation
#' @param x Numeric vector/matrix.
#' @param beta Beta parameter.
#' @return Parametric bent ReLU transform.
#' @export
#' @rdname ddesonn_activations
parametric_bent_relu <- function(x, beta = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  out <- pmax(0, ((sqrt(beta * x^2 + 1) - 1) / 2 + x))
  dim(out) <- dim(x)  # <- enforce consistent dimensions
  return(out)
}
attr(parametric_bent_relu, "name") <- "parametric_bent_relu"

#' Leaky bent activation
#' @param x Numeric vector/matrix.
#' @param alpha Leak factor.
#' @return Leaky bent transform.
#' @export
#' @rdname ddesonn_activations
leaky_bent <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(((sqrt(x^2 + 1) - 1) / 2) + alpha * x)
}
attr(leaky_bent, "name") <- "leaky_bent"

#' Inverse linear unit activation
#' @param x Numeric vector/matrix.
#' @return x/(1+abs(x)).
#' @export
#' @rdname ddesonn_activations
inverse_linear_unit <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x / (1 + abs(x)))
}
attr(inverse_linear_unit, "name") <- "inverse_linear_unit"

#' Tanh-ReLU hybrid activation
#' @param x Numeric vector/matrix.
#' @return Hybrid transform.
#' @export
#' @rdname ddesonn_activations
tanh_relu_hybrid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x * tanh(x), 0))
}
attr(tanh_relu_hybrid, "name") <- "tanh_relu_hybrid"

#' Custom bent piecewise activation
#' @param x Numeric vector/matrix.
#' @param threshold Threshold for piecewise.
#' @return Piecewise transform.
#' @export
#' @rdname ddesonn_activations
custom_bent_piecewise <- function(x, threshold = 0.5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > threshold, x, (sqrt(x^2 + 1) - 1) / 2 + x))
}
attr(custom_bent_piecewise, "name") <- "custom_bent_piecewise"

#' Sharpened sigmoid activation
#' @param x Numeric vector/matrix.
#' @param temp Temperature/sharpness.
#' @return Sharpened sigmoid.
#' @export
#' @rdname ddesonn_activations
sigmoid_sharp <- function(x, temp = 5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + exp(-temp * x)))
}
attr(sigmoid_sharp, "name") <- "sigmoid_sharp"

#' Leaky SELU activation
#' @param x Numeric vector/matrix.
#' @param alpha Leak factor.
#' @param lambda SELU lambda.
#' @return Leaky SELU transform.
#' @export
#' @rdname ddesonn_activations
leaky_selu <- function(x, alpha = 0.01, lambda = 1.0507) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, lambda * x, lambda * alpha * (exp(x) - 1)))
}
attr(leaky_selu, "name") <- "leaky_selu"

#' Identity activation (linear)
#' @return Base identity function.
#' @export
#' @rdname ddesonn_activations
identity <- base::identity
attr(identity, "name") <- "identity"

###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################

# -------------------------
# Activation Derivatives
# -------------------------

#' Activation derivatives (DDESONN)
#'
#' A collection of derivative functions corresponding to DDESONN activation functions.
#'
#' @name ddesonn_activation_derivatives
NULL

#' Binary activation derivative (step; zeroed)
#' @param x Numeric vector/matrix.
#' @return Vector of zeros.
#' @export
#' @rdname ddesonn_activation_derivatives
binary_activation_derivative <- function(x) {
  return(rep(0, length(x)))  # Non-differentiable, set to 0
}

#' Custom binary activation derivative (step; zeroed)
#' @param x Numeric vector/matrix.
#' @param threshold Threshold (unused in derivative).
#' @return Vector of zeros.
#' @export
#' @rdname ddesonn_activation_derivatives
custom_binary_activation_derivative <- function(x, threshold = -1.08) {
  return(rep(0, length(x)))  # Also a step function, gradient is zero everywhere
}

#' Custom activation derivative (experimental; zeroed)
#' @param z Numeric vector/matrix.
#' @return Zero array with same dims.
#' @export
#' @rdname ddesonn_activation_derivatives
custom_activation_derivative <- function(z) {
  array(0, dim = dim(z))  # zero a.e.; undefined only where softplus == 1e-11
}
attr(custom_activation_derivative, "name") <- "custom_activation_derivative"

#' Bent identity derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
bent_identity_derivative <- function(x) {
  return(x / (2 * sqrt(x^2 + 1)) + 1)
}

#' ReLU derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
relu_derivative <- function(x) {
  return(ifelse(x > 0, 1, 0))
}

#' Softplus derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
softplus_derivative <- function(x) {
  return(1 / (1 + exp(-x)))
}

#' Leaky ReLU derivative
#' @param x Numeric vector/matrix.
#' @param alpha Leak factor.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
leaky_relu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

#' ELU derivative
#' @param x Numeric vector/matrix.
#' @param alpha ELU alpha.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
elu_derivative <- function(x, alpha = 1.0) {
  return(ifelse(x > 0, 1, alpha * exp(x)))
}

#' tanh derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
tanh_derivative <- function(x) {
  t <- tanh(x)
  return(1 - t^2)
}

#' Sigmoid derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
sigmoid_derivative <- function(x) {
  s <- 1 / (1 + exp(-x))
  return(s * (1 - s))
}

#' Hard sigmoid derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
hard_sigmoid_derivative <- function(x) {
  x <- as.matrix(x)
  if (is.null(dim(x))) dim(x) <- c(length(x), 1)  # Force matrix shape if needed
  deriv <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  deriv[which(x > -2.5 & x < 2.5, arr.ind = TRUE)] <- 0.2
  return(deriv)
}

#' Swish derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
swish_derivative <- function(x) {
  s <- 1 / (1 + exp(-x))  # sigmoid(x)
  return(s + x * s * (1 - s))
}

#' Sigmoid-binary derivative (step; zeroed)
#' @param x Numeric vector/matrix.
#' @return Vector of zeros.
#' @export
#' @rdname ddesonn_activation_derivatives
sigmoid_binary_derivative <- function(x) {
  return(rep(0, length(x)))  # Step function is not differentiable
}

#' GELU derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
gelu_derivative <- function(x) {
  phi <- 0.5 * (1 + erf(x / sqrt(2)))
  dphi <- exp(-x^2 / 2) / sqrt(2 * pi)
  return(0.5 * phi + x * dphi)
}

#' SELU derivative
#' @param x Numeric vector/matrix.
#' @param lambda SELU lambda.
#' @param alpha SELU alpha.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
selu_derivative <- function(x, lambda = 1.0507, alpha = 1.67326) {
  return(lambda * ifelse(x > 0, 1, alpha * exp(x)))
}

#' Mish derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
mish_derivative <- function(x) {
  sp <- log1p(exp(x))              # softplus
  tanh_sp <- tanh(sp)
  grad_sp <- 1 - exp(-sp)          # d(softplus) ??? sigmoid(x)
  return(tanh_sp + x * grad_sp * (1 - tanh_sp^2))
}

#' Maxout derivative (example)
#' @param x Numeric vector/matrix.
#' @param w1 Weight 1.
#' @param b1 Bias 1.
#' @param w2 Weight 2.
#' @param b2 Bias 2.
#' @return Gradient of active affine branch.
#' @export
#' @rdname ddesonn_activation_derivatives
maxout_derivative <- function(x, w1 = 0.5, b1 = 1.0, w2 = -0.5, b2 = 0.5) {
  val1 <- w1 * x + b1
  val2 <- w2 * x + b2
  return(ifelse(val1 > val2, w1, w2))  # Returns the gradient of the active unit
}

#' PReLU derivative
#' @param x Numeric vector/matrix.
#' @param alpha Negative slope.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
prelu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

#' Softmax derivative (only valid under some losses)
#' @param x Numeric vector/matrix.
#' @return Derivative matrix (approx).
#' @export
#' @rdname ddesonn_activation_derivatives
softmax_derivative <- function(x) {
  s <- softmax(x)
  return(s * (1 - s))  # Only valid when loss is MSE, not cross-entropy
}

#' Bent ReLU derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
bent_relu_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  base_deriv <- (x / (2 * sqrt(x^2 + 1))) + 1
  return(ifelse(x > 0, base_deriv, 0))
}

#' Bent sigmoid derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
bent_sigmoid_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent_part <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  sigmoid_out <- 1 / (1 + exp(-bent_part))
  dbent_dx <- (x / (2 * sqrt(x^2 + 1))) + 1
  out <- sigmoid_out * (1 - sigmoid_out) * dbent_dx
  dim(out) <- dim(x)
  return(out)
}

#' Arctangent derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
arctangent_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + x^2))
}

#' Sinusoid derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
sinusoid_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(cos(x))
}

#' Gaussian derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
gaussian_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(-2 * x * exp(-x^2))
}

#' ISRLU derivative
#' @param x Numeric vector/matrix.
#' @param alpha Alpha parameter.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
isrlu_derivative <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x >= 0, 1, (1 / sqrt(1 + alpha * x^2))^3))
}

#' Bent swish derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
bent_swish_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  s <- 1 / (1 + exp(-x))
  dbent <- (x / (2 * sqrt(x^2 + 1))) + 1
  return(dbent * s + bent * s * (1 - s))
}

#' Parametric bent ReLU derivative
#' @param x Numeric vector/matrix.
#' @param beta Beta parameter.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
parametric_bent_relu_derivative <- function(x, beta = 1.0) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  
  grad_bent <- (beta * x) / (2 * sqrt(beta * x^2 + 1)) + 1
  mask <- x > 0
  out <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  out[mask] <- grad_bent[mask]
  return(out)
}

#' Leaky bent derivative
#' @param x Numeric vector/matrix.
#' @param alpha Leak factor.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
leaky_bent_derivative <- function(x, alpha = 0.01) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  return((x / (2 * sqrt(x^2 + 1))) + alpha)
}

#' Inverse linear unit derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
inverse_linear_unit_derivative <- function(x) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + abs(x))^2)
}

#' Tanh-ReLU hybrid derivative
#' @param x Numeric vector/matrix.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
tanh_relu_hybrid_derivative <- function(x) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  result <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  pos_mask <- x > 0
  result[pos_mask] <- tanh(x[pos_mask]) + x[pos_mask] * (1 - tanh(x[pos_mask])^2)
  return(result)
}

#' Custom bent piecewise derivative
#' @param x Numeric vector/matrix.
#' @param threshold Threshold.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
custom_bent_piecewise_derivative <- function(x, threshold = 0.5) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  dbent <- (x / (2 * sqrt(x^2 + 1))) + 1
  result <- matrix(1, nrow = nrow(x), ncol = ncol(x))
  below_mask <- x <= threshold
  result[below_mask] <- dbent[below_mask]
  return(result)
}

#' Sharpened sigmoid derivative
#' @param x Numeric vector/matrix.
#' @param temp Temperature/sharpness.
#' @return Derivative matrix.
#' @export
#' @rdname ddesonn_activation_derivatives
sigmoid_sharp_derivative <- function(x, temp = 5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  s <- 1 / (1 + exp(-temp * x))
  return(temp * s * (1 - s))
}

#' Leaky SELU derivative
#' @param x Numeric vector/matrix.
#' @param alpha Leak factor.
#' @param lambda SELU lambda.
#' @return Derivative matrix/vector.
#' @export
#' @rdname ddesonn_activation_derivatives
leaky_selu_derivative <- function(x, alpha = 0.01, lambda = 1.0507) {
  ifelse(x > 0, lambda, lambda * alpha * exp(x))
}

#' Identity derivative
#' @param x Numeric vector/matrix.
#' @return Matrix of ones with same dimensions as `x`.
#' @export
#' @rdname ddesonn_activation_derivatives
identity_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  matrix(1, nrow = nrow(x), ncol = ncol(x))
}
