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
\arguments{
\item{sequence_features}{Number of input channels in each sequence.}
\item{state_dim}{Number of encoder state dimensions.}
\item{conv_width}{Width of the causal convolution.}
\item{seed}{Optional seed for parameter initialization.}
\item{encoder}{Encoder returned by \code{ddesonn_ssm_init()}.}
\item{sequence_data}{Numeric array with dimensions samples by timesteps by channels.}
\item{return_cache}{Whether to return intermediate values needed by the backward pass.}
\item{cache}{Forward-pass cache returned with \code{return_cache = TRUE}.}
\item{grad_embedding}{Per-sample embedding gradients before averaging over samples.}
\item{fit_scale}{Whether to fit scaling statistics on these sequences.}
\item{state}{Optional replacement state, or \code{NULL} to clear the stored state.}
}
\description{Initialise, evaluate, differentiate, scale, and reset the optional
native-R causal selective state-space encoder.}
\details{The backward pass takes per-sample embedding gradients before sample
averaging and returns parameter and input gradients averaged over samples.
For mean squared error across states, supply twice the embedding residual
divided by the number of states; do not also divide by the sample count.
Encoding and supervised encoder training reuse identical sequence
windows. Scaling statistics and the mean-loss gradient retain the original
sample multiplicities, including different labels attached to the same window.}
\value{An encoder, forward result, gradient list, or embedding matrix, depending
on the function called.}
