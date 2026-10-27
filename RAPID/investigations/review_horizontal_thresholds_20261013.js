let panels;const el=id=>document.getElementById(id);
function render(){
 const all=el('network').value==='all',chosen=all?panels:panels.filter(p=>p.example===el('network').value),view=el('view').value,aligned=el('gauge').value==='aligned';
 const rows=all?5:1,cols=all?3:1,data=[],layout={height:all?2100:730,margin:{l:75,r:35,t:100,b:60},template:'plotly_white',hovermode:'closest',legend:{orientation:'h',y:all?1.045:1.15},annotations:[],grid:{rows,columns:cols,pattern:'independent',xgap:.18,ygap:.26}};
 chosen.forEach((p,i)=>{
  const suffix=i?String(i+1):'',range=p.ranges[view],values=[];
  p.series.forEach(s=>{const shift=aligned&&s.key==='downstream'?p.offset:0;const y=s.y.map(v=>v===null?null:v+shift);y.forEach((v,j)=>{if(v!==null&&s.x[j]>=range[0]&&s.x[j]<=range[1])values.push(v);});
   data.push({...s,y,xaxis:'x'+suffix,yaxis:'y'+suffix,legendgroup:s.key,showlegend:i===0&&s.showlegend!==false,mode:['accepted','proposed'].includes(s.key)?'lines+markers':s.mode,marker:s.marker||{size:6},customdata:s.customdata.map((t,j)=>[t,s.y[j]]),hovertemplate:`${p.example} · ${s.name}<br>%{customdata[0]}<br>Original Q: %{customdata[1]:.3f}<br>Displayed Q: %{y:.3f} m³/s<extra></extra>`});
  });
  values.push(p.threshold_summary.threshold_cms);
  const lo=values.reduce((a,b)=>Math.min(a,b),Infinity),hi=values.reduce((a,b)=>Math.max(a,b),-Infinity),pad=Math.max((hi-lo)*.08,1);
  layout['xaxis'+suffix]={range,title:{text:'Days from '+p.origin.slice(0,10)+' UTC'},showgrid:true};layout['yaxis'+suffix]={range:[lo-pad,hi+pad],title:{text:'Q (m³/s)'},showgrid:true};
  layout.annotations.push({xref:'x'+suffix+' domain',yref:'y'+suffix+' domain',x:.5,y:1.03,text:`${p.example} · Qref ${p.threshold_summary.threshold_cms.toFixed(3)} · ${p.threshold_summary.closed_states}/${p.threshold_summary.states} closed`,showarrow:false,font:{size:14}});
 });
 el('note').textContent=(all?'':JSON.stringify(chosen[0].threshold_summary.selection)+' · Base: '+chosen[0].threshold_summary.base.status+' · Centerline: '+chosen[0].threshold_summary.centerline.status+'. ')+(aligned?'Downstream gauge shifted by the documented initial-level offset; models and backgrounds remain unshifted.':'Upstream and downstream gauges shown in original discharge units. One shared horizontal threshold defines the selected pulse.');
 Plotly.react(el('plot'),data,layout,{responsive:true,scrollZoom:true});
}
(async()=>{const bytes=Uint8Array.from(atob(el('payload').textContent.trim()),c=>c.charCodeAt(0));panels=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());panels.forEach(p=>el('network').add(new Option(p.example,p.example)));['network','view','gauge'].forEach(id=>el(id).onchange=render);render();})();
