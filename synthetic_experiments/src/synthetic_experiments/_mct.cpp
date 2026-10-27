// Wide-rectangular specialization of Todini (2007), eqs 30,40,43-46,
// with the C1 correction (doi:10.5194/hess-11-1783-2007).
// Solve continuity in storage form, avoiding coefficient-index ambiguity.
#include <cmath>
#include <algorithm>
static double storage(double i,double o,double b,double s,double n,double dx,double beta,double* derivative) {
 const double q=(i+o)*.5;
 const double h=std::pow(n*q/(b*std::sqrt(s)),.6);
 const double v=q/(b*h), c=(5./3.)*v;
 const double prism=dx*b*h, wedge=beta*q/(2*b*s*c*c);
 if(derivative) *derivative=.3*prism/q+wedge+.1*wedge*(o-i)/q;
 return prism+wedge*(o-i);
}
extern "C" int mct_cell(const double* input,double* output,long size,double b,double slope,double n,double dx,double dt,double beta,double* audit) {
 const double initial=storage(input[0],input[0],b,slope,n,dx,beta,nullptr);
 double old_s=initial, ip=input[0], op=input[0], total=0.,max_res=0.; int max_it=0;
 output[0]=op;
 for(long j=1;j<size;j++) {
  const double in=input[j], rhs=old_s+.5*dt*(ip+in-op);
  double lo=0.,hi=std::max(in,op)*2.+1.;
  auto residual=[&](double x){return storage(in,x,b,slope,n,dx,beta,nullptr)+.5*dt*x-rhs;};
  if(residual(0)>0) return 1; // no nonnegative solution; never clip flow
  int expand=0; while(residual(hi)<0 && expand++<60) hi*=2.;
  if(expand>=60) return 2;
  double x=std::max(lo,std::min(hi,op+in-ip)); int it=0;
  for(;it<80;it++) {
   double deriv=0.; const double sx=storage(in,x,b,slope,n,dx,beta,&deriv);
   const double r=sx+.5*dt*x-rhs;
   if(std::abs(r)<1e-13*std::max(1.,std::abs(rhs))) break;
   if(r>0) hi=x; else lo=x;
   double next=x-r/(deriv+.5*dt);
   if(next<=lo || next>=hi || !std::isfinite(next)) next=.5*(lo+hi);
   x=next;
  }
  if(it>=80 || !std::isfinite(x)) return 3;
  const double now_s=storage(in,x,b,slope,n,dx,beta,nullptr);
  const double balance=.5*dt*(ip+in-op-x);
  max_res=std::max(max_res,std::abs(now_s-old_s-balance));
  total+=balance; old_s=now_s; ip=in; op=x; output[j]=x;max_it=std::max(max_it,it);
 }
 audit[0]=initial; audit[1]=old_s; audit[2]=total; audit[3]=max_res; audit[4]=max_it;
 return 0;
}
