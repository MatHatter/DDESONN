// Opt-in float32 training kernel for DDESONN's original-Keras reference adapter.
// Defaults in the established DDESONN trainer are intentionally unchanged.
#include <Rcpp.h>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
using namespace Rcpp;
extern "C" void cblas_sgemm(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
struct M {
 int r,c; std::vector<float> v;
 M(int r_=1,int c_=1,float value=0):r(r_),c(c_),v(r_*c_,value){}
 float& operator()(int i,int j){return v[i+r*j];}
 float operator()(int i,int j) const{return v[i+r*j];}
};
M from(SEXP x){NumericVector a(x);IntegerVector d=a.attr("dim");M m(d.size()?d[0]:1,d.size()>1?d[1]:a.size());std::copy(a.begin(),a.end(),m.v.begin());return m;}
NumericMatrix to(const M& m){NumericMatrix out(m.r,m.c);std::copy(m.v.begin(),m.v.end(),out.begin());return out;}
std::vector<M> unpack(List a){std::vector<M> out;for(int i=0;i<a.size();++i)out.push_back(from(a[i]));return out;}
List pack(const std::vector<M>& a){List out(a.size());for(unsigned i=0;i<a.size();++i)out[i]=to(a[i]);return out;}
M dot(const M&a,const M&b,bool ta=false,bool tb=false){int m=ta?a.c:a.r,k=ta?a.r:a.c,n=tb?b.r:b.c;M out(m,n);cblas_sgemm(102,ta?112:111,tb?112:111,m,n,k,1,a.v.data(),a.r,b.v.data(),b.r,0,out.v.data(),m);return out;}
M sums(const M&a){M s(1,a.c);for(int j=0;j<a.c;++j)for(int i=0;i<a.r;++i)s(0,j)+=a(i,j);return s;}
void bias(M&a,const M&b){for(int j=0;j<a.c;++j)for(int i=0;i<a.r;++i)a(i,j)+=b.v[j];}
struct Pass {std::vector<M> inputs,z,rect,norm,inv;M probability;};
Pass forward(const M&x,const std::vector<M>&p,std::vector<M>&moving,const M&mask,bool training){
 Pass s;s.inputs.push_back(x);
 for(int l=0;l<2;++l){int k=4*l;M z=dot(s.inputs.back(),p[k]);bias(z,p[k+1]);s.z.push_back(z);M a=z;for(float&v:a.v)v=std::max(0.f,v);s.rect.push_back(a);M n(a.r,a.c),iv(1,a.c);
  for(int j=0;j<a.c;++j){float mean=moving[2*l](0,j),var=moving[2*l+1](0,j);
   if(training){mean=0;for(int i=0;i<a.r;++i)mean+=a(i,j);mean/=a.r;var=0;for(int i=0;i<a.r;++i){float d=a(i,j)-mean;var+=d*d;}var/=a.r;moving[2*l](0,j)=.9f*moving[2*l](0,j)+.1f*mean;moving[2*l+1](0,j)=.9f*moving[2*l+1](0,j)+.1f*var;}
   iv(0,j)=1/std::sqrt(var+1e-6f);for(int i=0;i<a.r;++i){n(i,j)=(a(i,j)-mean)*iv(0,j);a(i,j)=n(i,j)*p[k+2](0,j)+p[k+3](0,j);if(l==0&&training)a(i,j)*=mask(i,j);}}
  s.norm.push_back(n);s.inv.push_back(iv);s.inputs.push_back(a);
 }
 M z=dot(s.inputs.back(),p[8]);bias(z,p[9]);s.z.push_back(z);M prob(z.r,1);for(int i=0;i<z.r;++i)prob(i,0)=1.f/(1.f+std::exp(-z(i,0)));s.probability=prob;return s;
}
struct Step {Pass pass;std::vector<M> gradients;float loss;};
Step step(const M&x,const M&y,std::vector<M>&p,std::vector<M>&moving,std::vector<M>&acc,const M&mask,bool average,bool update){
 Pass s=forward(x,p,moving,mask,true);std::vector<M> g;for(const M&a:p)g.push_back(M(a.r,a.c));M delta(x.r,1);float loss=0;
 for(int i=0;i<x.r;++i){float z=s.z[2](i,0),label=y(i,0);loss+=std::max(z,0.f)-z*label+std::log1p(std::exp(-std::abs(z)));delta(i,0)=(s.probability(i,0)-label)/(average?x.r:1);}
 if(average)loss/=x.r;
 g[8]=dot(s.inputs[2],delta,true);g[9]=sums(delta);M back=dot(delta,p[8],false,true);
 for(int l=1;l>=0;--l){int k=4*l;
  if(l==0)for(unsigned i=0;i<back.v.size();++i)back.v[i]*=mask.v[i];
  M dx(back.r,back.c);
  for(int j=0;j<back.c;++j){float db=0,dg=0;for(int i=0;i<back.r;++i){db+=back(i,j);dg+=back(i,j)*s.norm[l](i,j);}g[k+2](0,j)=dg;g[k+3](0,j)=db;
   for(int i=0;i<back.r;++i){float v=p[k+2](0,j)*s.inv[l](0,j)*(back(i,j)-db/back.r-s.norm[l](i,j)*dg/back.r);dx(i,j)=s.z[l](i,j)>0?v:0;}}
  g[k]=dot(s.inputs[l],dx,true);g[k+1]=sums(dx);if(l)back=dot(dx,p[k],false,true);
 }
 for(int k:{0,4,8})for(unsigned i=0;i<p[k].v.size();++i){float w=p[k].v[i];loss+=.00028f*std::abs(w);g[k].v[i]+=.00028f*((w>0)-(w<0));}
 if(update)for(unsigned k=0;k<p.size();++k)for(unsigned i=0;i<p[k].v.size();++i){float v=g[k].v[i];acc[k].v[i]+=v*v;p[k].v[i]-=(.125f*v)/std::sqrt(acc[k].v[i]+1e-7f);}
 return Step{s,g,loss};
}
// Original evaluation batches: 32 rows, separately within each partition.
M predict_batches(const M& all,const std::vector<M>& p,std::vector<M>& moving){
 M out(all.r,1),mask(1,1);std::vector<int> bounds=all.r==5000?std::vector<int>{0,3500,4250,5000}:std::vector<int>{0,all.r};
 for(unsigned part=0;part+1<bounds.size();++part)for(int start=bounds[part];start<bounds[part+1];start+=32){
  int n=std::min(32,bounds[part+1]-start);M batch(n,all.c);
  for(int j=0;j<all.c;++j)for(int i=0;i<n;++i)batch(i,j)=all(start+i,j);
  Pass prediction=forward(batch,p,moving,mask,false);
  for(int i=0;i<n;++i)out(start+i,0)=prediction.probability(i,0);
 }
 return out;
}
// [[Rcpp::export]]
List kr_probe(List parameters,List moving,List accumulators,NumericMatrix x,NumericMatrix y,NumericMatrix mask,bool average,bool update=true){auto p=unpack(parameters),m=unpack(moving),a=unpack(accumulators);Step s=step(from(x),from(y),p,m,a,from(mask),average,update);return List::create(_["parameters"]=pack(p),_["moving"]=pack(m),_["accumulators"]=pack(a),_["gradients"]=pack(s.gradients),_["probability"]=to(s.pass.probability),_["loss"]=s.loss);}
// [[Rcpp::export]]
List kr_fit(List parameters,List moving,NumericMatrix x,NumericMatrix y,int seed,bool average,int epochs=360){
 if(x.nrow()!=5000||x.ncol()!=12||y.nrow()!=5000||epochs<1)stop("Reference requires the original 5000 x 12 dataset and contiguous split");
 auto p=unpack(parameters),m=unpack(moving);std::vector<M>a;for(const M&w:p)a.push_back(M(w.r,w.c,.1f));M all=from(x),labels=from(y);std::vector<int>order(3500);std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed);std::bernoulli_distribution keep(.9);int updates=0,validation_steps=0;std::vector<float> validation_loss,validation_accuracy;
 for(int epoch=0;epoch<epochs;++epoch){Rcpp::checkUserInterrupt();std::shuffle(order.begin(),order.end(),rng);
  for(int start=0;start<3500;start+=16){int n=std::min(16,3500-start);M bx(n,12),by(n,1),mask(n,64);for(int i=0;i<n;++i){by(i,0)=labels(order[start+i],0);for(int j=0;j<12;++j)bx(i,j)=all(order[start+i],j);}for(float&v:mask.v)v=keep(rng)?1.f/.9f:0.f;step(bx,by,p,m,a,mask,average,true);++updates;}
  // Original fit evaluates validation every epoch, with validation_batch_size
  // defaulting to training batch_size (16). No BN/RNG/optimizer state changes.
  float weighted_loss=0;int correct=0;float penalty=0;
  for(int k:{0,4,8})for(float w:p[k].v)penalty+=.00028f*std::abs(w);
  for(int start=3500;start<4250;start+=16){int n=std::min(16,4250-start);M vx(n,12),mask(1,1);
   for(int j=0;j<12;++j)for(int i=0;i<n;++i)vx(i,j)=all(start+i,j);
   Pass val=forward(vx,p,m,mask,false);float loss=0;
   for(int i=0;i<n;++i){float z=val.z[2](i,0),label=labels(start+i,0);loss+=std::max(z,0.f)-z*label+std::log1p(std::exp(-std::abs(z)));correct+=((val.probability(i,0)>.5f)==(label>.5f));}
   if(average)loss/=n;weighted_loss+=(loss+penalty)*n;++validation_steps;
  }
  validation_loss.push_back(weighted_loss/750);validation_accuracy.push_back(float(correct)/750);
 }
 M probability=predict_batches(all,p,m);
 return List::create(_["parameters"]=pack(p),_["moving"]=pack(m),_["accumulators"]=pack(a),_["probability"]=to(probability),_["epochs"]=epochs,_["updates"]=updates,_["validation_steps"]=validation_steps,_["validation_loss"]=validation_loss,_["validation_accuracy"]=validation_accuracy);
}
// [[Rcpp::export]]
NumericMatrix kr_predict(List parameters,List moving,NumericMatrix x){auto p=unpack(parameters),m=unpack(moving);return to(predict_batches(from(x),p,m));}
