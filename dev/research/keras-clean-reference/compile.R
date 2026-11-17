reference_compile <- function(repo) {
  Sys.setenv(PKG_LIBS="/usr/lib/x86_64-linux-gnu/libblas.so.3")
  Rcpp::sourceCpp(file.path(repo,"dev/research/keras-clean-reference/native_reference.cpp"),
    cacheDir="/workspace/keras-clean-100/rcpp-cache", rebuild=FALSE)
}
