"""Independent linear advection-diffusion diagnostic at fitted parent n.

Uniform reach, radiation downstream boundary: solving D Hxx-c Hx-s H=0
gives H(s)=exp(-2*s*L/(c+sqrt(c*c+4*D*s))). Connections use the same
conservative splits. This tests the diffusion-wave approximation, not gauges
or full Saint-Venant hydraulics. No parameters are fitted here.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np,pandas as pd
from scipy.fft import rfft,irfft,next_fast_len
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.cunge import topology
OUT=ROOT/'RAPID/outputs/cunge_calibration_alpha2'


def run(ex,step=150.):
    rows=pd.read_csv(OUT/'comparison.csv',dtype={'example':str});r=rows[(rows.example==ex)&rows.method.eq('cunge')].iloc[0];p=Path(r.output_dir)
    f=pd.read_parquet(p/'cunge_prep/hydraulic_links.parquet');w=pd.read_parquet(p/'cunge_best_wave.parquet');clock=json.loads((p/'case.json').read_text())
    end=w.time_seconds.iloc[-1];n=next_fast_len(int(np.ceil(2*end/step)));time=np.arange(n)*step;alpha=20/(n*step);nt=int(end//step)+1
    ids=np.minimum((time/clock['forcing_step_s']).astype(int),len(w)-1);flow=w.q_inflow_cms.to_numpy()[ids];spectrum=rfft((flow-flow[0])*np.exp(-alpha*time))
    order,edges,root,outlet=topology(f);H=np.empty(n//2+1,complex);c=f.rapid_celerity_mps.to_numpy();D=f.hydraulic_diffusivity_m2_s.to_numpy();L=f.link_length_m.to_numpy();v=f.v.to_numpy();frac=f.split_fraction.to_numpy()
    for offset in range(0,len(H),8192):
        lap=alpha+2j*np.pi*np.arange(offset,min(offset+8192,len(H)))/(n*step);signals={root:np.ones(len(lap),complex)}
        for u in order:
            sig=signals.pop(u)
            if u==outlet:H[offset:offset+len(lap)]=sig;continue
            for i in edges[u]:
                transfer=np.exp(-2*lap*L[i]/(c[i]+np.sqrt(c[i]**2+4*D[i]*lap)))
                signals[v[i]]=signals.get(v[i],0)+sig*frac[i]*transfer
    output=flow[0]+irfft(spectrum*H,n)[:nt]*np.exp(alpha*time[:nt]);tt=time[:nt]
    # Compare to the saved forcing-interval means on their documented clock;
    # this timing convention contributes at most the input/report scale.
    mc=np.interp(tt,w.time_seconds,w.q_outlet_cms);peak=tt[output.argmax()];mcpeak=tt[mc.argmax()]
    result=dict(example=ex,report_step_s=step,cunge_peak_h=mcpeak/3600,diffusion_peak_h=peak/3600,
                diffusion_minus_cunge_peak_h=(peak-mcpeak)/3600,
                max_wave_difference_fraction=float(np.max(abs(output-mc))/np.ptp(flow[:nt])),
                diffusion_peak_cms=float(output.max()),cunge_peak_cms=float(mc.max()),
                diffusion_peak_at_boundary=bool(output.argmax() in [0,nt-1]),
                comparison_clock_uncertainty_s=float(clock['forcing_step_s']+step),
                note='Continuous uniform-reach diffusion/radiation solution; held inlet sampled on FFT grid; finite report-resolution comparison, not independent observed truth.')
    name=f'diffusion_diagnostic_step{step:g}'
    (p/(name+'.json')).write_text(json.dumps(result,indent=2));pd.DataFrame(dict(time_seconds=tt,inflow=flow[:nt],cunge=mc,diffusion=output)).to_parquet(p/(name+'.parquet'),index=False)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--examples',nargs='+');parser.add_argument('--step',type=float,default=150.);args=parser.parse_args()
    examples=args.examples or pd.read_csv(OUT/'comparison.csv',dtype={'example':str}).example.unique()
    results=[run(ex,args.step) for ex in examples];pd.DataFrame(results).to_csv(OUT/f'diffusion_diagnostic_step{args.step:g}.csv',index=False)
