\name{ddesonn_ssm_init}
\alias{ddesonn_ssm_init}\alias{ddesonn_ssm_forward}\alias{ddesonn_ssm_backward}
\alias{ddesonn_ssm_encode}\alias{ddesonn_ssm_reset_state}
\title{Native-R selective state-space sequence encoder}
\usage{
ddesonn_ssm_init(sequence_features, state_dim = 16L, conv_width = 4L, seed = NULL)
ddesonn_ssm_forward(encoder, sequence_data, return_cache = FALSE)
ddesonn_ssm_backward(encoder, cache, grad_embedding)
ddesonn_ssm_encode(encoder, sequence_data, fit_scale = FALSE)
ddesonn_ssm_reset_state(encoder, state = NULL)
}
\description{Initialise, evaluate, differentiate, scale, and reset the optional
native-R causal selective state-space encoder.}
\value{An encoder, forward result, gradient list, or embedding matrix, depending
on the function called.}
