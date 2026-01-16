\name{ddesonn_artifacts_root}
\alias{ddesonn_artifacts_root}
\alias{DDESONN_artifacts_root}
\alias{ddesonn_plots_dir}
\alias{DDESONN_plots_dir}
\title{Resolve artifact and plot output locations}
\usage{
ddesonn_artifacts_root(output_root = NULL)

ddesonn_plots_dir(output_root = NULL)
}
\arguments{
\item{output_root}{Optional base directory for artifacts and plots. When
\code{NULL}, a user-scoped writable directory is chosen automatically.}
}
\value{
Character vectors containing the resolved artifact or plot directory
paths (created if missing).
}
\description{
Helpers that route runtime outputs to user-writable locations outside the
installed package or source checkout. The artifact root is used as the base
for all persisted outputs, and the plots helper returns the plots
subdirectory under that root.
}
