#include <cmath>
extern "C" void cell(const double* in, double* out, long size, double eq, double width, double slope, double n, double dx, double dt) {
 double ip=eq,op=eq;
 for(long j=0;j<size;j++) {
  double q=(ip+in[j]+op)/3.0;
  if(q<=0) {out[j]=NAN;continue;}
  double h=pow(n*q/(width*sqrt(slope)),0.6);
  double c=(5.0/3.0)*pow(h,2.0/3.0)*sqrt(slope)/n;
  double k=dx/c, x=0.5*(1-q/(width*slope*c*dx));
  double den=dt+2*k*(1-x);
  out[j]=((dt-2*k*x)*in[j]+(dt+2*k*x)*ip+(2*k*(1-x)-dt)*op)/den;
  ip=in[j];op=out[j];
 }
}
