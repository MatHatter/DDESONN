.ssm_softplus <- function(x) log1p(exp(-abs(x))) + pmax(x, 0)
.ssm_sigmoid <- function(x) 1 / (1 + exp(-x))

#' Initialise a native-R selective state-space encoder
#' @param sequence_features Number of input channels.
#' @param state_dim Temporal embedding width.
#' @param conv_width Causal convolution width.
#' @param seed Optional initialization seed.
#' @return A mutable encoder list.
#' @export
ddesonn_ssm_init <- function(sequence_features, state_dim = 16L, conv_width = 4L, seed = NULL) {
  f <- as.integer(sequence_features); h <- as.integer(state_dim); k <- as.integer(conv_width)
  if (any(lengths(list(f, h, k)) != 1L) || any(!is.finite(c(f,h,k))) || any(c(f,h,k) < 1L)) stop("SSM dimensions must be positive integers.", call. = FALSE)
  if (!is.null(seed)) { old <- if (exists(".Random.seed", .GlobalEnv)) .Random.seed else NULL; on.exit(if (is.null(old)) rm(".Random.seed", envir=.GlobalEnv) else assign(".Random.seed", old, .GlobalEnv), add=TRUE); set.seed(seed) }
  s <- sqrt(2/(f+h))
  p <- list(conv=array(rnorm(k*f, sd=s), c(k,f)), conv_bias=numeric(f),
            W_dt=matrix(rnorm(f*h,sd=s),f,h), b_dt=rep(-1,h),
            W_B=matrix(rnorm(f*h,sd=s),f,h), b_B=numeric(h),
            W_C=matrix(rnorm(f*h,sd=s),f,h), b_C=numeric(h),
            A_log=rep(log(.5),h), D=matrix(rnorm(f*h,sd=s),f,h))
  structure(list(config=list(sequence_features=f,state_dim=h,conv_width=k), params=p,
                 scale=NULL, state=numeric(h), optimizer=list(step=0L,m=list(),v=list())), class="ddesonn_ssm")
}

#' Reset the streaming state of an SSM encoder
#' @param encoder Encoder returned by [ddesonn_ssm_init()].
#' @param state Optional replacement state.
#' @export
ddesonn_ssm_reset_state <- function(encoder, state = NULL) {
  h <- encoder$config$state_dim; state <- state %||% numeric(h)
  if (length(state) != h || any(!is.finite(state))) stop("state has invalid dimensions.", call.=FALSE)
  encoder$state <- as.numeric(state); encoder
}

#' Run the selective SSM forward pass
#' @param encoder SSM encoder.
#' @param sequence_data Numeric three-dimensional sequence array.
#' @param return_cache Include intermediates needed by backward propagation.
#' @export
ddesonn_ssm_forward <- function(encoder, sequence_data, return_cache = FALSE) {
  d <- .ddesonn_validate_sequence(sequence_data); cfg <- encoder$config; p <- encoder$params
  if (d[3] != cfg$sequence_features) stop("sequence feature count differs from encoder configuration.", call.=FALSE)
  n <- d[1]; T <- d[2]; f <- d[3]; h <- cfg$state_dim; k <- cfg$conv_width
  states <- array(0,c(n,T,h)); z <- array(0,c(n,T,f)); dt <- B <- C <- array(0,c(n,T,h)); out <- array(0,c(n,T,h))
  for (i in seq_len(n)) for (t in seq_len(T)) {
    for (lag in 0:(k-1L)) if (t-lag >= 1L) z[i,t,] <- z[i,t,] + sequence_data[i,t-lag,] * p$conv[lag+1L,]
    z[i,t,] <- tanh(z[i,t,] + p$conv_bias)
    dt[i,t,] <- .ssm_softplus(z[i,t,] %*% p$W_dt + p$b_dt)
    B[i,t,] <- z[i,t,] %*% p$W_B + p$b_B; C[i,t,] <- z[i,t,] %*% p$W_C + p$b_C
    prev <- if (t==1L) numeric(h) else states[i,t-1L,]
    da <- exp(dt[i,t,] * (-exp(p$A_log)))
    states[i,t,] <- da*prev + B[i,t,] * rep(z[i,t,],length.out=h)
    out[i,t,] <- C[i,t,]*states[i,t,] + z[i,t,] %*% p$D
  }
  ans <- list(embedding=matrix(out[,T,],n,h), output=out)
  if (return_cache) ans$cache <- list(x=sequence_data,z=z,dt=dt,B=B,C=C,states=states)
  ans
}

#' Encode sequences using saved training-only scaling
#' @param encoder SSM encoder.
#' @param sequence_data Sequence array.
#' @param fit_scale Fit channel means/standard deviations (training only).
#' @export
ddesonn_ssm_encode <- function(encoder, sequence_data, fit_scale = FALSE) {
  .ddesonn_validate_sequence(sequence_data)
  if (fit_scale) {
    f <- dim(sequence_data)[3]; mu <- sd <- numeric(f)
    for(j in seq_len(f)){v<-sequence_data[,,j];mu[j]<-mean(v);sd[j]<-stats::sd(as.vector(v));if(!is.finite(sd[j])||sd[j]==0)sd[j]<-1}
    encoder$scale <- list(center=mu, scale=sd)
  }
  if (is.null(encoder$scale)) stop("Encoder scaling statistics have not been fitted.", call.=FALSE)
  x <- sequence_data; for(j in seq_len(dim(x)[3])) x[,,j] <- (x[,,j]-encoder$scale$center[j])/encoder$scale$scale[j]
  ans <- ddesonn_ssm_forward(encoder,x); attr(ans$embedding,"encoder") <- encoder; ans$embedding
}

#' Back-propagate through the selective recurrence and causal convolution
#' @param encoder SSM encoder.
#' @param cache Cache returned by `ddesonn_ssm_forward(..., return_cache=TRUE)`.
#' @param grad_embedding Gradient of the loss with respect to the final embedding.
#' @return Gradients for every parameter and for the sequence input.
#' @export
ddesonn_ssm_backward <- function(encoder, cache, grad_embedding) {
  p<-encoder$params; cfg<-encoder$config; x<-cache$x; z<-cache$z; n<-dim(x)[1];T<-dim(x)[2];f<-dim(x)[3];h<-cfg$state_dim;k<-cfg$conv_width
  g<-lapply(p,function(v){array(0,dim=dim(v)%||%length(v))}); names(g)<-names(p)
  gx<-array(0,dim(x)); gs_next<-matrix(0,n,h); A <- -exp(p$A_log)
  for(t in T:1L) for(i in seq_len(n)) {
    go <- if(t==T) as.numeric(grad_embedding[i,]) else numeric(h)
    st<-cache$states[i,t,]; prev<-if(t==1L)numeric(h) else cache$states[i,t-1L,]; zz<-z[i,t,]; cc<-cache$C[i,t,]; bb<-cache$B[i,t,]; dtt<-cache$dt[i,t,]; da<-exp(dtt*A)
    g$W_C <- g$W_C + outer(zz,go*st); g$b_C<-g$b_C+go*st; gst<-go*cc+gs_next
    g$D <- g$D + outer(zz,go); gz<-as.numeric(p$D%*%go)
    gB<-gst*rep(zz,length.out=h); gxrep<-gst*bb
    g$W_B<-g$W_B+outer(zz,gB);g$b_B<-g$b_B+gB;gz<-gz+as.numeric(p$W_B%*%gB)
    gda<-gst*prev; gdt<-gda*da*A; gA<-gda*da*dtt
    rawdt<-as.numeric(zz%*%p$W_dt+p$b_dt); graw<-gdt*.ssm_sigmoid(rawdt)
    g$W_dt<-g$W_dt+outer(zz,graw);g$b_dt<-g$b_dt+graw;gz<-gz+as.numeric(p$W_dt%*%graw)
    g$A_log<-g$A_log+gA*A
    for(q in seq_len(h)) gz[(q-1L)%%f+1L]<-gz[(q-1L)%%f+1L]+gxrep[q]
    gs_next[i,]<-gst*da
    gzraw<-gz*(1-zz^2);g$conv_bias<-g$conv_bias+gzraw
    for(lag in 0:(k-1L)) if(t-lag>=1L){g$conv[lag+1L,]<-g$conv[lag+1L,]+gzraw*x[i,t-lag,];gx[i,t-lag,]<-gx[i,t-lag,]+gzraw*p$conv[lag+1L,]}
  }
  list(params=lapply(g,function(v)v/n), input=gx/n)
}

.ddesonn_ssm_update <- function(encoder, grads, lr=.001, beta1=.9, beta2=.999, epsilon=1e-8) {
  o<-encoder$optimizer;o$step<-o$step+1L
  for(nm in names(encoder$params)){if(is.null(o$m[[nm]])){o$m[[nm]]<-encoder$params[[nm]]*0;o$v[[nm]]<-encoder$params[[nm]]*0};o$m[[nm]]<-beta1*o$m[[nm]]+(1-beta1)*grads[[nm]];o$v[[nm]]<-beta2*o$v[[nm]]+(1-beta2)*grads[[nm]]^2;mh<-o$m[[nm]]/(1-beta1^o$step);vh<-o$v[[nm]]/(1-beta2^o$step);encoder$params[[nm]]<-encoder$params[[nm]]-lr*mh/(sqrt(vh)+epsilon)}
  encoder$optimizer<-o;encoder
}

.ddesonn_ssm_train <- function(encoder, sequence_data, y, epochs=3L, lr=.001) {
  x<-sequence_data; for(j in seq_len(dim(x)[3]))x[,,j]<-(x[,,j]-encoder$scale$center[j])/encoder$scale$scale[j]
  target<-matrix(as.numeric(y),nrow=dim(x)[1]);target<-target[,rep(seq_len(ncol(target)),length.out=encoder$config$state_dim),drop=FALSE]
  if(ncol(target)!=encoder$config$state_dim)target<-matrix(rep(target,length.out=nrow(target)*encoder$config$state_dim),nrow(target))
  for(ep in seq_len(as.integer(epochs))){fw<-ddesonn_ssm_forward(encoder,x,TRUE);ge<-2*(fw$embedding-target)/(nrow(target)*ncol(target));gr<-ddesonn_ssm_backward(encoder,fw$cache,ge);encoder<-.ddesonn_ssm_update(encoder,gr$params,lr)}
  encoder
}
