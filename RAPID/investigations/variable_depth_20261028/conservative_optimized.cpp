// Same equations and convergence criterion as conservative_v2.cpp.
// Cache hydraulic evaluations; compute monotone depth/celerity extrema once.
#include <cmath>
#include <algorithm>
#include <limits>
extern "C" int route_cells(double* signal,long size,int cells,double dt,
 double dx,double width,double slope,double roughness,double initial,
 int variable,double beta,double fixed_k,double fixed_x,double* audit) {
 const double scale=roughness/(width*std::sqrt(slope));
 auto depth=[&](double q){return std::pow(scale*q,.6);};
 struct State {double q,h,c,t,s,d,b;};
 auto evaluate=[&](double i,double o){
   State a; a.q=(i+o)/2;a.h=depth(a.q);a.c=(5./3.)*a.q/(width*a.h);
   a.t=beta*a.q/(width*slope*a.c*a.c);
   a.s=dx*width*a.h+a.t*(o-i)/2;
   a.d=.5*dx/a.c+.5*a.t+.05*a.t*(o-i)/a.q;
   a.b=.5*(dx*width*a.h/a.q+a.t);
   return a;
 };
 if(!variable)return 9;
 double qmin=std::numeric_limits<double>::infinity(),qmax=0.;
 for(int cell=0;cell<cells;cell++) {
  double ip=initial,op=initial;State prev=evaluate(ip,op);
  double sp=prev.s,sinitial=sp;
  for(long j=0;j<size;j++) {
   double ic=signal[j];if(!(ic>0)||!std::isfinite(ic))return 1;
   const double target=sp+dt*.5*(ic+ip-op);
   auto residual=[&](double o){return evaluate(ic,o).s+dt*.5*o-target;};
   double lo=0,hi=std::max({ic,op,initial,1.});
   if(residual(lo)>0)return 2;
   int expand=0;while(residual(hi)<0&&expand++<60)hi*=2;
   if(!std::isfinite(hi)||residual(hi)<0)return 3;
   double guess=op+ic-ip;if(!(guess>lo&&guess<=hi))guess=(lo+hi)/2;
   bool converged=false;State cur{};
   for(int it=0;it<80;it++) {
    cur=evaluate(ic,guess);
    const double rr=cur.s+dt*.5*guess-target,slope_r=cur.d+dt*.5;
    audit[2]=std::max(audit[2],double(it+1));
    const double ulp=std::nextafter(guess,std::numeric_limits<double>::infinity())-guess;
    const double evaluation_floor=8*std::numeric_limits<double>::epsilon()*
      (std::abs(cur.s)+std::abs(dt*.5*guess)+std::abs(target));
    const double tolerance=std::max(2e-14*std::max(1.,std::abs(target)),
      evaluation_floor+2*std::abs(slope_r)*ulp);
    if(std::abs(rr)<=tolerance){converged=true;break;}
    if(rr>0)hi=guess;else lo=guess;
    double next=guess-rr/slope_r;
    if(!(next>lo&&next<hi))next=(lo+hi)/2;
    guess=next;
   }
   if(!converged)return 4;
   if(!(guess>0)||!std::isfinite(guess))return 5;
   audit[3]=std::min(audit[3],(2*prev.b-dt)/(2*cur.b+dt));
   audit[0]=std::min(audit[0],guess);
   qmin=std::min(qmin,std::min(ic,guess));qmax=std::max(qmax,std::max(ic,guess));
   signal[j]=guess;ip=ic;op=guess;sp=cur.s;prev=cur;
  }
  audit[1]+=sp-sinitial;
 }
 const double hmin=depth(qmin),hmax=depth(qmax);
 audit[4]=std::min(audit[4],hmin);audit[5]=std::max(audit[5],hmax);
 audit[6]=std::min(audit[6],(5./3.)*qmin/(width*hmin));
 audit[7]=std::max(audit[7],(5./3.)*qmax/(width*hmax));
 return 0;
}
