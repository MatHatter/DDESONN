#' Update biases block (ML + SL) for DDESONN
#'
#' Apply one standard optimizer update per tensor, or dispatch to the preserved
#' custom Adagrad X implementation.
#'
#' @param self R6 model (modified in-place)
#' @param update_biases logical
#' @param optimizer character
#' @param optimizer_params_biases list (returned/updated)
#' @param bias_gradients list of per-layer bias grads (from learn())
#' @param errors list (used by the Adagrad X compatibility path)
#' @param lr numeric
#' @param reg_type NULL or one of c("L2","L1","L1_L2","Group_Lasso","Max_Norm")
#' @param beta1,beta2,epsilon numerics
#' @param epoch integer
#' @param lookahead_step integer
#' @param verbose logical
#'
#' @return list(updated_optimizer_params = optimizer_params_biases)
#' @keywords internal
#' @noRd
update_biases_block <- function(
    self, update_biases, optimizer, optimizer_params_biases, bias_gradients,
    errors, lr, reg_type, beta1, beta2, epsilon, epoch, lookahead_step,
    verbose = FALSE
) {
  if (!update_biases) return(list(updated_optimizer_params = optimizer_params_biases))
  if (identical(optimizer, "adagrad_x")) {
    return(.adagrad_x$biases(self, update_biases, "adagrad",
      optimizer_params_biases, bias_gradients, errors, lr, reg_type, beta1,
      beta2, epsilon, epoch, lookahead_step, verbose))
  }
  # Use learn()'s mean deltas in both modes, including Adagrad. Raw errors are
  # retained in the signature for compatibility, but must not replace gradients.
  .update_parameter_block(self, "biases", optimizer, optimizer_params_biases,
                          bias_gradients, lr, reg_type, beta1, beta2, epsilon,
                          epoch, lookahead_step, verbose)
}
