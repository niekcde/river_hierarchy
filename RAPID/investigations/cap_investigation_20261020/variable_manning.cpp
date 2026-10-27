// Diagnostic four-point variable-parameter Muskingum-Cunge cell cascade.
// c and unit discharge are averaged over the four stencil points (Ponce &
// Yevjevich 1978); three-point coefficients supply the initial iterate.
#include <cmath>
#include <algorithm>
#include <vector>
extern "C" int route_cells(double* signal, long size, int cells, double dt,
 double dx, double width, double slope, double roughness, double initial,
 int variable, double fixed_k, double fixed_x, double* audit) {
 const double coeff=(5.0/3.0)*std::pow(roughness/(width*std::sqrt(slope)),0.4)*std::sqrt(slope)/roughness;
 auto celerity=[&](double q){return coeff*std::pow(q,0.4);};
 auto depth=[&](double q){return std::pow(roughness*q/(width*std::sqrt(slope)),0.6);};
 auto output=[&](double ip,double ic,double op,double c,double q){
   double k=variable?dx/c:fixed_k;
   double x=variable?0.5*(1-q/(width*slope*c*dx)):fixed_x;
   double den=dt+2*k*(1-x),a=(dt-2*k*x)/den,b=(dt+2*k*x)/den,d=(2*k*(1-x)-dt)/den;
   audit[3]=std::min(audit[3],d);
   return a*ic+b*ip+d*op;
 };
 for(int cell=0;cell<cells;cell++) {
   double ip=initial,op=initial,cip=celerity(initial),cop=cip;
   for(long j=0;j<size;j++) {
     double ic=signal[j]; if(!(ic>0) || !std::isfinite(ic))return 1;
     double cic=celerity(ic);
     double guess=output(ip,ic,op,(cip+cic+cop)/3,(ip+ic+op)/3);
     if(variable) {
       bool converged=false;
       for(int it=0;it<40;it++) {
         if(!(guess>0)||!std::isfinite(guess))return 2;
         double next=output(ip,ic,op,(cip+cic+cop+celerity(guess))/4,(ip+ic+op+guess)/4);
         audit[2]=std::max(audit[2],double(it+1));
         if(std::abs(next-guess)<=1e-10*std::max(1.,std::abs(next))){guess=next;converged=true;break;}
         guess=next;
       }
       if(!converged)return 3;
     }
     if(!(guess>0)||!std::isfinite(guess))return 4;
     signal[j]=guess;
     audit[0]=std::min(audit[0],guess);
     audit[4]=std::min(audit[4],depth((ip+ic+op+guess)/4));
     audit[5]=std::max(audit[5],depth((ip+ic+op+guess)/4));
     audit[6]=std::min(audit[6],(cip+cic+cop+celerity(guess))/4);
     audit[7]=std::max(audit[7],(cip+cic+cop+celerity(guess))/4);
     ip=ic;op=guess;cip=cic;cop=celerity(guess);
   }
   // End-state storage diagnostic: physical cross-section volume for dynamic
   // routing; exact Muskingum storage for constant coefficients.
   audit[1]+=variable?dx*width*((depth(ip)+depth(op))/2-depth(initial)):
     fixed_k*(fixed_x*ip+(1-fixed_x)*op-initial);
 }
 return 0;
}
