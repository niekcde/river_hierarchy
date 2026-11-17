// Experimental wide-rectangular Muskingum-Cunge-Todini storage update.
// Todini (2007), equations 30 and 43-46. beta below is OUR diffusion
// multiplier, not Todini's c/v correction (=5/3 for this section).
#include <cmath>
#include <algorithm>
#include <limits>
extern "C" int route_cells(double* signal,long size,int cells,double dt,
 double dx,double width,double slope,double roughness,double initial,
 int variable,double beta,double fixed_k,double fixed_x,double* audit) {
 auto depth=[&](double q){return std::pow(roughness*q/(width*std::sqrt(slope)),.6);};
 auto celerity=[&](double q){return (5./3.)*q/(width*depth(q));};
 auto storage=[&](double i,double o){double q=(i+o)/2,c=celerity(q);
   return dx*width*depth(q)+beta*q/(width*slope*c*c)*(o-i)/2;};
 auto derivative=[&](double i,double o){double q=(i+o)/2,c=celerity(q),t=beta*q/(width*slope*c*c);
   return .5*dx/c+.5*t+.05*t*(o-i)/q;};
 if(!variable)return 9; // This engine implements only the conservative formulation.
 for(int cell=0;cell<cells;cell++) {
  double ip=initial,op=initial,sp=storage(ip,op),sinitial=sp;
  for(long j=0;j<size;j++) {
   double ic=signal[j];if(!(ic>0)||!std::isfinite(ic))return 1;
   const double target=sp+dt*.5*(ic+ip-op);
   auto residual=[&](double o){return storage(ic,o)+dt*.5*o-target;};
   double lo=0,hi=std::max({ic,op,initial,1.});
   if(residual(lo)>0)return 2; // No positive solution: do not clip.
   int expand=0;while(residual(hi)<0&&expand++<60)hi*=2;
   if(!std::isfinite(hi)||residual(hi)<0)return 3;
   double guess=op+ic-ip;if(!(guess>lo&&guess<=hi))guess=(lo+hi)/2;
   bool converged=false;
   for(int it=0;it<80;it++) {
    double rr=residual(guess),slope_r=derivative(ic,guess)+dt*.5;
    audit[2]=std::max(audit[2],double(it+1));
    // Storage residual cannot resolve a discharge more finely than one ULP.
    // Include the propagated rounding floor, not an arbitrary hydraulic tolerance.
    const double ulp=std::nextafter(guess,std::numeric_limits<double>::infinity())-guess;
    const double evaluation_floor=8*std::numeric_limits<double>::epsilon()*
      (std::abs(storage(ic,guess))+std::abs(dt*.5*guess)+std::abs(target));
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
   const double qp=(ip+op)/2,qn=(ic+guess)/2,cp=celerity(qp),cn=celerity(qn);
   double bp=.5*(dx*width*depth(qp)/qp+beta*qp/(width*slope*cp*cp));
   double bn=.5*(dx*width*depth(qn)/qn+beta*qn/(width*slope*cn*cn));
   audit[3]=std::min(audit[3],(2*bp-dt)/(2*bn+dt));
   audit[0]=std::min(audit[0],guess);
   audit[4]=std::min(audit[4],std::min(depth(ic),depth(guess)));
   audit[5]=std::max(audit[5],std::max(depth(ic),depth(guess)));
   audit[6]=std::min(audit[6],std::min(celerity(ic),celerity(guess)));
   audit[7]=std::max(audit[7],std::max(celerity(ic),celerity(guess)));
   signal[j]=guess;ip=ic;op=guess;sp=storage(ip,op);
  }
  audit[1]+=sp-sinitial;
 }
 return 0;
}
