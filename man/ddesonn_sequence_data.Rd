\name{ddesonn_sequence_data}\alias{ddesonn_sequence_data}
\title{Build causal fixed-length sequences}
\usage{ddesonn_sequence_data(data, sequence_length = 48L, pad = FALSE)}
\arguments{\item{data}{Finite numeric observations in chronological order.}
\item{sequence_length}{Positive window length.}\item{pad}{Left-pad leading windows.}}
\description{Constructs an array of samples by time by features without using
observations after a sample's end index.}
\value{A numeric three-dimensional array with an \code{end_index} attribute.}
