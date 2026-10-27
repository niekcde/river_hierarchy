let panels;const el=id=>document.getElementById(id);
function render(){
 const all=el('network').value==='all',chosen=all?panels:panels.filter(p=>p.example===el('network').value),view=el('view').value;
 const data=[],layout={height:all?2100:730,margin:{l:75,r:35,t:100,b:60},hovermode:'closest',legend:{orientation:'h',y:all?1.045:1.15},annotations:[],shapes:[],grid:{rows:all?5:1,columns:all?3:1,pattern:'independent',xgap:.18,ygap:.26}};
 chosen.forEach((p,i)=>{
  const suffix=i?String(i+1):'',range=p.ranges[view],values=[];
  p.series.forEach(s=>{s.y.forEach((v,j)=>{if(v!==null&&s.x[j]>=range[0]&&s.x[j]<=range[1])values.push(v);});
   data.push({...s,xaxis:'x'+suffix,yaxis:'y'+suffix,legendgroup:s.key,showlegend:i===0&&s.showlegend!==false,marker:{size:10,color:'#198754',symbol:'diamond'},hovertemplate:`${p.example} · ${s.name}<br>%{customdata}<br>Q: %{y:.3f} m³/s<extra></extra>`});
  });
  const lo=values.reduce((a,b)=>Math.min(a,b),Infinity),hi=values.reduce((a,b)=>Math.max(a,b),-Infinity),pad=Math.max((hi-lo)*.08,1);
  layout['xaxis'+suffix]={range,title:{text:'Days from '+p.origin.slice(0,10)+' UTC'},showgrid:true};layout['yaxis'+suffix]={range:[lo-pad,hi+pad],title:{text:'Q (m³/s)'},showgrid:true};
  layout.annotations.push({xref:'x'+suffix+' domain',yref:'y'+suffix+' domain',x:.5,y:1.03,text:p.example,showarrow:false,font:{size:15}});
  layout.shapes.push({type:'rect',xref:'x'+suffix,yref:'y'+suffix+' domain',x0:p.bounds[0],x1:p.bounds[1],y0:0,y1:1,fillcolor:'rgba(25,135,84,.07)',line:{width:0},layer:'below'});
  p.bounds.forEach(x=>layout.shapes.push({type:'line',xref:'x'+suffix,yref:'y'+suffix+' domain',x0:x,x1:x,y0:0,y1:1,line:{color:'#198754',dash:'dash',width:1}}));
 });
 el('note').textContent=all?'Select a network and Start/End view to inspect each proposed boundary. Detailed choices are listed below.':chosen[0].note;
 Plotly.react(el('plot'),data,layout,{responsive:true,scrollZoom:true});
}
(async()=>{const bytes=Uint8Array.from(atob(el('payload').textContent.trim()),c=>c.charCodeAt(0));panels=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());panels.forEach(p=>el('network').add(new Option(p.example,p.example)));['network','view'].forEach(id=>el(id).onchange=render);render();})();
