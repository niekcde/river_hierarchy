let payload;
const el=id=>document.getElementById(id), choices={};
function chosen(p){return p.background_candidates.find(c=>c.key===(choices[p.example]||p.default_background||'proposed'));}
function options(){
 const ex=el('network').value,select=el('candidate');select.innerHTML='';select.disabled=ex==='all';
 if(ex==='all'){select.add(new Option('Current per-network choices','proposed'));return;}
 const p=payload.panels.find(p=>p.example===ex);
 p.background_candidates.forEach(c=>select.add(new Option(c.label,c.key)));
 select.value=choices[ex]||p.default_background||'proposed';
}
function render(){
 const ex=el('network').value,mode=el('view').value,all=ex==='all';
 const panels=all?payload.panels:payload.panels.filter(p=>p.example===ex);
 let data=[],seen=new Set();
 panels.forEach((p,i)=>{
  const c=chosen(p),s=all&&i?String(i+1):'';
  const traces=[...p.series,
   {type:'scatter',mode:'lines+markers',x:c.line_x,y:c.line_y,name:'Selected background',legendgroup:'background',line:{color:'#9b59b6',dash:'dash',width:2},marker:{size:6},hovertemplate:'Background: %{y:.3f} m³/s<extra></extra>'},
   {type:'scatter',mode:'lines',x:c.shade_x,y:c.shade_q,name:'Deficit lower edge',legendgroup:'deficit',showlegend:false,line:{width:0},hoverinfo:'skip'},
   {type:'scatter',mode:'lines',x:c.shade_x,y:c.shade_b,name:'Background exceeds Q',legendgroup:'deficit',fill:'tonexty',fillcolor:'rgba(230,140,40,.4)',line:{width:0},hoverinfo:'skip'}];
  traces.forEach(t=>{const show=t.showlegend!==false&&!seen.has(t.legendgroup);if(show)seen.add(t.legendgroup);data.push({...t,xaxis:'x'+s,yaxis:'y'+s,showlegend:show});});
 });
 let layout;
 if(all){layout=JSON.parse(JSON.stringify(payload.layouts[mode]));layout.shapes=[];layout.annotations.forEach((a,i)=>a.text='Network '+panels[i].example);panels.forEach((p,i)=>{const s=i?String(i+1):'';layout['xaxis'+s].range=mode==='event'?chosen(p).event_x:p[mode+'_x'];layout['yaxis'+s].range=p[mode+'_y'];});}
 else{const p=panels[0];layout={height:680,margin:{l:80,r:30,t:80,b:70},xaxis:{range:mode==='event'?chosen(p).event_x:p[mode+'_x'],title:{text:'Days from '+p.origin+' (UTC)'}},yaxis:{range:p[mode+'_y'],title:{text:'Q (m³/s)'}},legend:{orientation:'h',y:1.12}};}
 layout.paper_bgcolor='white';layout.plot_bgcolor='white';el('plot').style.height=layout.height+'px';Plotly.react(el('plot'),data,layout,{responsive:true,scrollZoom:true});
 el('notes').textContent=all?'Select a network to compare the current background with alternatives.':panels[0].selection_status+'. '+chosen(panels[0]).note;
 const fmt=x=>Number(x).toPrecision(7);
 el('values').innerHTML=panels.map(p=>(all?[chosen(p)]:p.background_candidates).map(c=>`<tr><td>${p.example}</td><td>${c.label}</td><td>${c.start_utc}<br>${fmt(c.start_cms)} m³/s</td><td>${c.end_utc}<br>${fmt(c.end_cms)} m³/s</td><td>${fmt(c.minimum_excess_cms)}</td><td>${fmt(c.deficit_to_positive_volume_pct)}</td><td>${c.base_pass?'Yes':'No'}</td><td>${c.valid_pairs} / ${c.children_checked}</td></tr>`).join('')).join('');
}
(async()=>{const bytes=Uint8Array.from(atob(el('payload').textContent.trim()),c=>c.charCodeAt(0));payload=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());payload.panels.forEach(p=>el('network').add(new Option('Network '+p.example,p.example)));el('network').onchange=()=>{options();render();};el('view').onchange=render;el('candidate').onchange=()=>{choices[el('network').value]=el('candidate').value;render();};options();render();})();
