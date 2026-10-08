import sys
sys.path.insert(0,'RAPID/investigations')
from route_full_signed_validation import *
rows=[]
for ex,dt,states in [('11',69,['S000_base','S003_unit_18']),('46',35,['S000_base','S004_unit_24','S010_unit_105'])]:
 inv=pd.read_csv(OUT/ex/'inventory.csv');sel=inv[(inv.family=='indep')&inv.state.isin(states)];travel=sel.max_travel_seconds.max()
 for step in [120.,60.]:
  t,inp,sp,nfft,q0,qe,_=grid_input(ex,travel,step);metrics={}
  for r in sel.itertuples():
   f=pd.read_parquet(r.frame);q=q0+irfft(sp*response(f,dt,nfft,step),nfft)[:len(t)]
   h=pd.DataFrame(dict(time_seconds=t,outlet_cms=q));old=pd.read_parquet(BASE/f'common_caps/signed_coefficient_check/{ex}_{r.state}_dt{dt}.parquet')
   mask=(t>=old.time_seconds.min())&(t<=old.time_seconds.max())
   err=max(abs(q[mask]-np.interp(t[mask],old.time_seconds,old.outlet_cms)))/np.ptp(inp)
   peakerr=quadratic_peak(h,'outlet_cms')-quadratic_peak(old,'outlet_cms')
   rows.append(dict(example=ex,state=r.state,dt=dt,step=step,max_wave_error_fraction=err,peak_error_seconds=peakerr))
   print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'native_screen_crosscheck.csv',index=False)
