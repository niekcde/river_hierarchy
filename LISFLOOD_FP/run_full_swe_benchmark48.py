"""Full SWE vs local inertia with matched physics, fields and numerical controls."""
from pathlib import Path
import json,time,os
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from shallow_water_benchmark_kernel import integrate as raw_integrate,G
SMOOTH=os.environ.get("SWE_SMOOTH", "0")=="1"
def integrate(*args,**kwargs):
    kwargs["limited"]=not SMOOTH or not kwargs.get("periodic",True)
    return raw_integrate(*args,**kwargs)
from uniform_stabilized48 import evolve
OUT=Path(__file__).resolve().parent/'results/broad_15_100m/investigation/full_swe_benchmark48';OUT.mkdir(exist_ok=True)
if SMOOTH:
    OUT=OUT/'smooth_reconstruction';OUT.mkdir(exist_ok=True)

def continuum(h,q,S,n,k,adv):
    u=q/h;alpha=2*G*n*n*q/h**(7/3);beta=(10/3)*G*S
    return np.array([[0.,-1j*k],[beta-1j*k*(G*h-(u*u if adv else 0.)),-alpha-(2j*k*u if adv else 0.)]],complex)

def verify():
    checks=[]
    for adv in [False,True]:
        h=1.0804578636021311;S=.0018810732578538286;n=.020799605931143154;q=h**(5/3)*np.sqrt(S)/n;U=np.tile([h,q],(80,1));V,_,a=integrate(U,50.,S,n,3600.,advective=adv);err=float(np.max(abs(V-U)));assert err<1e-11;checks.append(dict(test='uniform_normal_flow',advective=adv,max_error=err,audit=a.tolist()))
    U=np.tile([2.,0.],(80,1));V,_,a=integrate(U,50.,0.,0.,3600.);assert np.array_equal(U,V);checks.append(dict(test='flat_lake_at_rest',max_error=0.))
    hl=2.;hr=1.;cl=np.sqrt(G*hl)
    star=brentq(lambda h:2*(np.sqrt(G*h)-cl)+(h-hr)*np.sqrt(.5*G*(1/h+1/hr)),hr,hl);us=(star-hr)*np.sqrt(.5*G*(1/star+1/hr));cs=np.sqrt(G*star);shock=star*us/(star-hr)
    errors=[]
    for dx in [8.,4.,2.,1.]:
        x=(np.arange(int(2000/dx))+.5)*dx-1000;U=np.column_stack((np.where(x<0,hl,hr),np.zeros(len(x))));V,_,audit=integrate(U,dx,0.,0.,50.,periodic=False);xi=x/50.;exact=np.full(len(x),hr);vel=np.zeros(len(x));exact[xi<-cl]=hl;fan=(xi>=-cl)&(xi<us-cs);exact[fan]=((2*cl-xi[fan])/3)**2/G;vel[fan]=2*(cl+xi[fan])/3;middle=(xi>=us-cs)&(xi<shock);exact[middle]=star;vel[middle]=us;mask=abs(x)<400;err=float(np.mean(abs(V[mask,0]-exact[mask])));errors.append(err);checks.append(dict(test='wet_dam_break_exact_riemann',dx=dx,mean_absolute_depth_error_m=err,max_mass_residual=audit[3]));assert audit[3]<1e-8
        if dx==1:np.savez_compressed(OUT/'dam_break_check.npz',x=x,h=V[:,0],exact_h=exact,q=V[:,1],exact_q=exact*vel)
    assert all(errors[i+1]<errors[i] for i in range(3)) and errors[-1]<.005,errors
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2));print('VERIFIED',json.dumps(checks),flush=True)

def main():
    verify();n=.020799605931143154;T=3600.;L=4000.;eps=1e-5;k=2*np.pi/L;records=[];analytic=[]
    for reach,h,S in [(65,.848191594080612,.0034947845012664545),(161,1.0804578636021311,.0018810732578538286)]:
        q=h**(5/3)*np.sqrt(S)/n;A=continuum(h,q,S,n,k,False);ev,V=np.linalg.eig(A);idx=int(np.argmax(ev.real));v=V[:,idx]/V[0,idx]
        exact={adv:expm(continuum(h,q,S,n,k,adv)*T)@v for adv in [False,True]};analytic.append(dict(reach=reach,h=h,q=q,S=S,n=n,Fr=q/(h*np.sqrt(G*h)),initial_q_mode_real=float(v[1].real),initial_q_mode_imag=float(v[1].imag),full_swe_amplification=float(abs(exact[True][0])),local_inertia_amplification=float(abs(exact[False][0])),full_swe_max_eigen_real=float(max(np.linalg.eigvals(continuum(h,q,S,n,k,True)).real)),local_inertia_max_eigen_real=float(max(ev.real))))
        for dx in [100.,50.,25.,12.5,6.25]:
            x=(np.arange(int(L/dx))+.5)*dx;phase=np.exp(1j*k*x);U=np.column_stack((h+eps*np.real(phase),q+eps*np.real(v[1]*phase)))
            for adv in [True,False]:
                for cfl in ([.35,.175] if dx in [25.,6.25] else [.35]):
                    name=f'r{reach}_dx{dx:g}_'+('full' if adv else 'local_fv')+f'_cfl{cfl:g}';end,hist,audit=integrate(U,dx,S,n,T,cfl=cfl,advective=adv,wavenumber=k);amplitude=hist[-1,1]/hist[0,1];ratio=(hist[-1,2]+1j*hist[-1,3])/(hist[0,2]+1j*hist[0,3]);rec=dict(name=name,reach=reach,dx=dx,model='full_swe' if adv else 'local_inertia_same_fv',cfl=cfl,amplification=float(amplitude),analytic_amplification=float(abs(exact[adv][0])),complex_relative_error=float(abs(ratio-exact[adv][0])/abs(exact[adv][0])),steps=int(audit[0]),minimum_depth_m=float(audit[1]),maximum_depth_m=float(audit[2]),maximum_mass_residual_m2=float(audit[3]));records.append(rec);np.savez_compressed(OUT/(name+'.npz'),history=hist,x=x,initial=U,final=end);print(json.dumps(rec),flush=True)
            for theta in [1.,.7]:
                dt=min(3.,.35*dx/np.sqrt(G*h));steps=int(np.ceil(T/dt));dt=T/steps;qface=q+eps*np.real(v[1]*np.exp(1j*k*(x+.5*dx)));hist,cap,hmin,hmax=evolve(U[:,0].copy(),qface,dx,S,n,dt,steps,k*dx,theta);assert cap==0 and hmin>0;rec=dict(name=f'r{reach}_dx{dx:g}_original_theta{theta:g}',reach=reach,dx=dx,model='original_local_inertia' if theta==1 else 'stabilized_local_inertia',theta=theta,dt=dt,amplification=float(hist[-1,1]/hist[0,1]),analytic_amplification=float(abs(exact[False][0])),capped_face_steps=cap);records.append(rec);np.savez_compressed(OUT/(rec['name']+'.npz'),history=hist);print(json.dumps(rec),flush=True)
        (OUT/'results.json').write_text(json.dumps(records,indent=2));(OUT/'analytic.json').write_text(json.dumps(analytic,indent=2))
    # Required agreement with independent continuum linearization on finest grid.
    for model in ['full_swe','local_inertia_same_fv']:
        for reach in [65,161]:
            rows=[r for r in records if r['model']==model and r['reach']==reach and r['dx']==6.25];assert all(r['complex_relative_error']<.01 for r in rows),rows
    (OUT/'complete.json').write_text(json.dumps(dict(network_runs=0,benchmark_runs=len(records),finest_complex_error_below_one_percent=True),indent=2))
if __name__=='__main__':main()
