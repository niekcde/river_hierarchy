let panels;const el=id=>document.getElementById(id);
function render(){
 const all=el('network').value==='all',chosen=all?panels:panels.filter(p=>p.example===el('network').value),view=el('view').value,aligned=el('gauge').value==='aligned',normalized=el('gauge').value==='normalized';
 const data=[],layout={height:all?2200:750,margin:{l:75,r:35,t:130,b:60},hovermode:'closest',legend:{orientation:'h',y:all?1.055:1.18},annotations:[],shapes:[],grid:{rows:all?5:1,columns:all?3:1,pattern:'independent',xgap:.18,ygap:.26}};
 chosen.forEach((p,i)=>{
  const suffix=i?String(i+1):'',values=[];let range=p.ranges[view];
  if(REVIEW_MODE==='calibration'&&view==='event'){let span=p.input_bounds[1]-p.input_bounds[0];range=[p.input_bounds[0]-.08*span,p.input_bounds[1]+Math.max(2,.2*span)];}
  const series=p[REVIEW_MODE+'_series'].filter(s=>REVIEW_MODE!=='calibration'||s.key!=='original');
  const peak=items=>items.reduce((maximum,s)=>s.y.reduce((m,v)=>v===null?m:Math.max(m,v),maximum),-Infinity);
  const inputPeak=peak(series.filter(s=>s.key==='forcing'));
  const downstreamPeak=peak(series.filter(s=>s.key==='downstream'));
  const factor=normalized&&Number.isFinite(inputPeak)&&downstreamPeak>0?inputPeak/downstreamPeak:1;
  series.forEach(s=>{const shift=aligned&&s.key==='downstream'?p.offset:0;const y=s.y.map(v=>v===null?null:s.key==='downstream'&&normalized?v*factor:v+shift);
   const name=REVIEW_MODE==='calibration'?({base:'Full Complexity',centerline:'single centerline',downstream:normalized?'Downstream gauge · peak normalized':s.name}[s.key]||s.name):s.name;
y.forEach((v,j)=>{if(v!==null&&s.x[j]>=range[0]&&s.x[j]<=range[1])values.push(v);});
   let line={...s.line};if(s.key==='original')line={color:'#79838c',width:1.5,dash:'dot'};
   data.push({...s,name,y,line,xaxis:'x'+suffix,yaxis:'y'+suffix,legendgroup:s.key,showlegend:i===0&&s.showlegend!==false,hovertemplate:`${p.example} · ${name}<br>%{customdata}<br>Displayed Q: %{y:.3f} m³/s<extra></extra>`});
  });
  const lo=values.reduce((a,b)=>Math.min(a,b),Infinity),hi=values.reduce((a,b)=>Math.max(a,b),-Infinity),pad=Math.max((hi-lo)*.08,1);
  layout['xaxis'+suffix]={range,title:{text:'Days from '+p.origin.slice(0,10)+' UTC'},showgrid:true};layout['yaxis'+suffix]={range:[lo-pad,hi+pad],title:{text:'Q (m³/s)'},showgrid:true};
  layout.annotations.push({xref:'x'+suffix+' domain',yref:'y'+suffix+' domain',x:.5,y:1.03,text:p.example+(REVIEW_MODE==='baseline'?' · B='+p.summary.baseline_cms.toFixed(2):''),showarrow:false,font:{size:14}});
  const vertical=(x,color,dash)=>layout.shapes.push({type:'line',xref:'x'+suffix,yref:'y'+suffix+' domain',x0:x,x1:x,y0:0,y1:1,line:{color,dash,width:1.4}});
  if(REVIEW_MODE==='baseline')p.input_bounds.forEach(x=>vertical(x,'#2580a5','dot'));
  if(REVIEW_MODE==='baseline'){
   if(p.threshold_windows){
    const colors={'0.5%':'#0072b2','1%':'#009e73','2%':'#cc79a7'};
    for(const [label,w] of Object.entries(p.threshold_windows)){
     for(const side of ['start','end']){
      const stamp=side==='start'?w.start_utc:w.end_utc;
      data.push({type:'scatter',mode:'lines',x:[w[side],w[side]],y:[lo-pad,hi+pad],xaxis:'x'+suffix,yaxis:'y'+suffix,name:label+' onset / end',legendgroup:'window_'+label,showlegend:i===0&&side==='end',line:{color:colors[label],width:2,dash:side==='start'?'dot':'dash'},hovertemplate:`${p.example} · ${label} ${side}<br>${stamp}<br>Tolerance: ${w.tolerance_cms.toFixed(3)} m³/s<extra></extra>`});
     }
    }
    const w=p.threshold_windows['1%'];
    layout.shapes.push({type:'rect',xref:'x'+suffix,yref:'y'+suffix+' domain',x0:w.start,x1:w.end,y0:0,y1:1,fillcolor:'rgba(0,158,115,.035)',line:{width:0},layer:'below'});
   }else{
    vertical(p.cutoffs['0.1pct'],'#9155b5','dash');
    layout.shapes.push({type:'rect',xref:'x'+suffix,yref:'y'+suffix+' domain',x0:p.input_bounds[0],x1:p.cutoffs['0.1pct'],y0:0,y1:1,fillcolor:'rgba(145,85,181,.035)',line:{width:0},layer:'below'});
   }
  }
 });
 el('note').textContent=all?(REVIEW_MODE==='baseline'&&chosen[0].threshold_windows?'0.5% blue · 1% green · 2% magenta. Dotted colored lines = onset; dashed = end. Click a threshold legend entry to toggle its pair; use onset/end close-ups. Input boundaries remain thin blue dotted lines.':'Blue dotted lines: input-event boundaries. '+(REVIEW_MODE==='baseline'?'Purple dashed line: proposed shared analysis end.':'Gray dotted curve: original calibrated parent; red: rerun parent.')):(REVIEW_MODE==='baseline'&&chosen[0].threshold_windows?Object.entries(chosen[0].threshold_windows).map(([label,w])=>`${label}: ${w.start_utc} → ${w.end_utc}`).join(' | '):JSON.stringify(chosen[0].summary));
 if(REVIEW_MODE==='calibration')el('note').textContent='Red: Full Complexity. Orange: single centerline. '+(normalized?'Downstream discharge scaled by input peak / downstream peak across the saved series; no time shift or baseline subtraction.':aligned?'Downstream gauge uses the original constant display offset.':'Downstream gauge shows original discharge.');
 Plotly.react(el('plot'),data,layout,{responsive:true,scrollZoom:true});
}
(async()=>{const bytes=Uint8Array.from(atob(el('payload').textContent.trim()),c=>c.charCodeAt(0));panels=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());panels.forEach(p=>el('network').add(new Option(p.example,p.example)));['network','view','gauge'].forEach(id=>el(id).onchange=render);render();})();
