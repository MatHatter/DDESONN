# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Ensemble Self-Organizing Neural Network
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

binary_activation <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0.5, 1, 0))
}
attr(binary_activation, "name") <- "binary_activation"

# Fun custom derivatives (mostly zeroed out — experimental only, not for real training)
custom_binary_activation <- function(x, threshold = -1.08) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x < threshold, 0, 1))
}
attr(custom_binary_activation, "name") <- "custom_binary_activation"

custom_activation <- function(z) {
  z <- as.matrix(z); dim(z) <- dim(z)
  softplus_output <- log1p(exp(z))
  return(ifelse(softplus_output > 1e-11, 1, 0))
}
attr(custom_activation, "name") <- "custom_activation"

bent_identity <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return((sqrt(x^2 + 1) - 1) / 2 + x)
}
attr(bent_identity, "name") <- "bent_identity"

relu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, 0))
}
attr(relu, "name") <- "relu"

softplus <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(log1p(exp(x)))
}
attr(softplus, "name") <- "softplus"

leaky_relu <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * x))
}
attr(leaky_relu, "name") <- "leaky_relu"

elu <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}
attr(elu, "name") <- "elu"

tanh <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
}
attr(tanh, "name") <- "tanh"

sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + exp(-x)))
}
attr(sigmoid, "name") <- "sigmoid"

hard_sigmoid <- function(x) {
  x <- as.matrix(x)
  out <- pmax(0, pmin(1, 0.2 * x + 0.5))
  dim(out) <- dim(x)  # Preserve shape no matter what
  return(out)
}
attr(hard_sigmoid, "name") <- "hard_sigmoid"


swish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * sigmoid(x))
}
attr(swish, "name") <- "swish"

sigmoid_binary <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse((1 / (1 + exp(-x))) >= 0.5, 1, 0))
}
attr(sigmoid_binary, "name") <- "sigmoid_binary"

gelu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * 0.5 * (1 + erf(x / sqrt(2))))
}
attr(gelu, "name") <- "gelu"

selu <- function(x, lambda = 1.0507, alpha = 1.67326) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(lambda * ifelse(x > 0, x, alpha * exp(x) - alpha))
}
attr(selu, "name") <- "selu"

mish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x * tanh(log(1 + exp(x))))
}
attr(mish, "name") <- "mish"

prelu <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x, alpha * x))
}
attr(prelu, "name") <- "prelu"

softmax <- function(z) {
  z <- as.matrix(z); dim(z) <- dim(z)
  exp_z <- exp(z)
  return(exp_z / rowSums(exp_z))
}
attr(softmax, "name") <- "softmax"

# Maxout example
maxout <- function(x, w1 = 0.5, b1 = 1.0, w2 = -0.5, b2 = 0.5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(pmax(w1 * x + b1, w2 * x + b2))
}
attr(maxout, "name") <- "maxout"

bent_relu <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  out <- pmax(0, ((sqrt(x^2 + 1) - 1) / 2 + x))
  dim(out) <- dim(x)
  return(out)
}
attr(bent_relu, "name") <- "bent_relu"

bent_sigmoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent_part <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  return(1 / (1 + exp(-bent_part)))
}
attr(bent_sigmoid, "name") <- "bent_sigmoid"

arctangent <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(atan(x))
}
attr(arctangent, "name") <- "arctangent"

sinusoid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(sin(x))
}
attr(sinusoid, "name") <- "sinusoid"

gaussian <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(exp(-x^2))
}
attr(gaussian, "name") <- "gaussian"

isrlu <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x >= 0, x, x / sqrt(1 + alpha * x^2)))
}
attr(isrlu, "name") <- "isrlu"

bent_swish <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  return(bent * sigmoid(x))
}
attr(bent_swish, "name") <- "bent_swish"

parametric_bent_relu <- function(x, beta = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  out <- pmax(0, ((sqrt(beta * x^2 + 1) - 1) / 2 + x))
  dim(out) <- dim(x)  # <- enforce consistent dimensions
  return(out)
}
attr(parametric_bent_relu, "name") <- "parametric_bent_relu"


leaky_bent <- function(x, alpha = 0.01) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(((sqrt(x^2 + 1) - 1) / 2) + alpha * x)
}
attr(leaky_bent, "name") <- "leaky_bent"

inverse_linear_unit <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(x / (1 + abs(x)))
}
attr(inverse_linear_unit, "name") <- "inverse_linear_unit"

tanh_relu_hybrid <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, x * tanh(x), 0))
}
attr(tanh_relu_hybrid, "name") <- "tanh_relu_hybrid"

custom_bent_piecewise <- function(x, threshold = 0.5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > threshold, x, (sqrt(x^2 + 1) - 1) / 2 + x))
}
attr(custom_bent_piecewise, "name") <- "custom_bent_piecewise"

sigmoid_sharp <- function(x, temp = 5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + exp(-temp * x)))
}
attr(sigmoid_sharp, "name") <- "sigmoid_sharp"

leaky_selu <- function(x, alpha = 0.01, lambda = 1.0507) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x > 0, lambda * x, lambda * alpha * (exp(x) - 1)))
}
attr(leaky_selu, "name") <- "leaky_selu"

# -------- Identity (Linear) --------
identity <- base::identity
attr(identity, "name") <- "identity"

###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################

binary_activation_derivative <- function(x) {
  return(rep(0, length(x)))  # Non-differentiable, set to 0
}


custom_binary_activation_derivative <- function(x, threshold = -1.08) {
  return(rep(0, length(x)))  # Also a step function, gradient is zero everywhere
}


custom_activation_derivative <- function(z) {
  array(0, dim = dim(z))  # zero a.e.; undefined only where softplus == 1e-11
}
attr(custom_activation_derivative, "name") <- "custom_activation_derivative"


bent_identity_derivative <- function(x) {
  return(x / (2 * sqrt(x^2 + 1)) + 1)
}

relu_derivative <- function(x) {
  return(ifelse(x > 0, 1, 0))
}


softplus_derivative <- function(x) {
  return(1 / (1 + exp(-x)))
}

leaky_relu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

elu_derivative <- function(x, alpha = 1.0) {
  return(ifelse(x > 0, 1, alpha * exp(x)))
}

tanh_derivative <- function(x) {
  t <- tanh(x)
  return(1 - t^2)
}

sigmoid_derivative <- function(x) {
  s <- 1 / (1 + exp(-x))
  return(s * (1 - s))
}

hard_sigmoid_derivative <- function(x) {
  x <- as.matrix(x)
  if (is.null(dim(x))) dim(x) <- c(length(x), 1)  # Force matrix shape if needed
  deriv <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  deriv[which(x > -2.5 & x < 2.5, arr.ind = TRUE)] <- 0.2
  return(deriv)
}

swish_derivative <- function(x) {
  s <- 1 / (1 + exp(-x))  # sigmoid(x)
  return(s + x * s * (1 - s))
}

sigmoid_binary_derivative <- function(x) {
  # Approximate pseudo-gradient
  return(rep(0, length(x)))  # Step function is not differentiable
}

gelu_derivative <- function(x) {
  phi <- 0.5 * (1 + erf(x / sqrt(2)))
  dphi <- exp(-x^2 / 2) / sqrt(2 * pi)
  return(0.5 * phi + x * dphi)
}

selu_derivative <- function(x, lambda = 1.0507, alpha = 1.67326) {
  return(lambda * ifelse(x > 0, 1, alpha * exp(x)))
}

mish_derivative <- function(x) {
  sp <- log1p(exp(x))              # softplus
  tanh_sp <- tanh(sp)
  grad_sp <- 1 - exp(-sp)          # d(softplus) ≈ sigmoid(x)
  return(tanh_sp + x * grad_sp * (1 - tanh_sp^2))
}

maxout_derivative <- function(x, w1 = 0.5, b1 = 1.0, w2 = -0.5, b2 = 0.5) {
  val1 <- w1 * x + b1
  val2 <- w2 * x + b2
  return(ifelse(val1 > val2, w1, w2))  # Returns the gradient of the active unit
}

prelu_derivative <- function(x, alpha = 0.01) {
  return(ifelse(x > 0, 1, alpha))
}

softmax_derivative <- function(x) {
  s <- softmax(x)
  return(s * (1 - s))  # Only valid when loss is MSE, not cross-entropy
}

bent_relu_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  base_deriv <- (x / (2 * sqrt(x^2 + 1))) + 1
  return(ifelse(x > 0, base_deriv, 0))
}

bent_sigmoid_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent_part <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  sigmoid_out <- 1 / (1 + exp(-bent_part))
  dbent_dx <- (x / (2 * sqrt(x^2 + 1))) + 1
  out <- sigmoid_out * (1 - sigmoid_out) * dbent_dx
  dim(out) <- dim(x)
  return(out)
}

arctangent_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + x^2))
}

sinusoid_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(cos(x))
}

gaussian_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(-2 * x * exp(-x^2))
}

isrlu_derivative <- function(x, alpha = 1.0) {
  x <- as.matrix(x); dim(x) <- dim(x)
  return(ifelse(x >= 0, 1, (1 / sqrt(1 + alpha * x^2))^3))
}

bent_swish_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  bent <- ((sqrt(x^2 + 1) - 1) / 2 + x)
  s <- 1 / (1 + exp(-x))
  dbent <- (x / (2 * sqrt(x^2 + 1))) + 1
  return(dbent * s + bent * s * (1 - s))
}

parametric_bent_relu_derivative <- function(x, beta = 1.0) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  
  grad_bent <- (beta * x) / (2 * sqrt(beta * x^2 + 1)) + 1
  mask <- x > 0
  out <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  out[mask] <- grad_bent[mask]
  return(out)
}

leaky_bent_derivative <- function(x, alpha = 0.01) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  return((x / (2 * sqrt(x^2 + 1))) + alpha)
}

inverse_linear_unit_derivative <- function(x) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  return(1 / (1 + abs(x))^2)
}

tanh_relu_hybrid_derivative <- function(x) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  result <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  pos_mask <- x > 0
  result[pos_mask] <- tanh(x[pos_mask]) + x[pos_mask] * (1 - tanh(x[pos_mask])^2)
  return(result)
}

custom_bent_piecewise_derivative <- function(x, threshold = 0.5) {
  if (is.null(x) || length(x) == 0) return(x)
  x <- as.matrix(x); dim(x) <- dim(x)
  dbent <- (x / (2 * sqrt(x^2 + 1))) + 1
  result <- matrix(1, nrow = nrow(x), ncol = ncol(x))
  below_mask <- x <= threshold
  result[below_mask] <- dbent[below_mask]
  return(result)
}

sigmoid_sharp_derivative <- function(x, temp = 5) {
  x <- as.matrix(x); dim(x) <- dim(x)
  s <- 1 / (1 + exp(-temp * x))
  return(temp * s * (1 - s))
}

leaky_selu_derivative <- function(x, alpha = 0.01, lambda = 1.0507) {
  ifelse(x > 0, lambda, lambda * alpha * exp(x))
}

identity_derivative <- function(x) {
  x <- as.matrix(x); dim(x) <- dim(x)
  # derivative of f(x)=x is 1 everywhere
  matrix(1, nrow = nrow(x), ncol = ncol(x))
}
