source("dev/research/keras-original-reference/compile.R");reference_compile(getwd())
root <- "/workspace/keras-original-100"
checks <- jsonlite::fromJSON(file.path(root,"tensorflow-checks.json"),simplifyVector=FALSE)
unpack_tensor <- function(a) matrix(unlist(a$values),a$shape[[1]],a$shape[[2]])
unpack <- function(a) lapply(a,unpack_tensor)
errors <- list()
compare <- function(actual,expected,label) {
  stopifnot(identical(dim(actual),dim(expected)),all(is.finite(actual)))
  e <- max(abs(actual-expected)/(1+abs(expected)))
  errors[[length(errors)+1]] <<- list(check=label,maximum_relative_scaled_error=e)
  if(e>2e-4) stop(sprintf("%s differs from TensorFlow: %.9g",label,e))
}
for(check in checks){
 p<-unpack(check$initial$parameters);m<-unpack(check$initial$moving);a<-unpack(check$initial$accumulators)
 for(i in seq_along(check$cases)){
  expected<-check$cases[[i]]
  actual<-kr_probe(p,m,a,unpack_tensor(expected$x),unpack_tensor(expected$y),
                  unpack_tensor(expected$mask),check$average)
  prefix<-paste(check$average,i)
  compare(matrix(actual$loss),matrix(expected$loss),paste(prefix,"loss"))
  compare(actual$probability,unpack_tensor(expected$probability),paste(prefix,"probability"))
  for(name in c("gradients","parameters","moving","accumulators")) {
    wanted<-unpack(expected[[name]])
    for(j in seq_along(wanted))compare(actual[[name]][[j]],wanted[[j]],paste(prefix,name,j))
  }
  p<-actual$parameters;m<-actual$moving;a<-actual$accumulators
 }
 compare(kr_predict(p,m,unpack_tensor(check$inference_x)),unpack_tensor(check$inference),
         paste(check$average,"inference"))
}
jsonlite::write_json(list(passed=TRUE,checks=errors,tolerance=2e-4),
 file.path(root,"native-equivalence-check.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)
cat(length(errors),"TensorFlow/native checks passed\n")
