let panels;const el=id=>document.getElementById(id);
function panel(){return panels.find(p=>p.example===el('network').value);}
function highlights(){const p=panel(),e=el('highlight');e.innerHTML='';p.highlighted.forEach(h=>e.add(new Option(h.aliases,h.state)));e.value=p.default_highlight;}
function render(){
 const p=panel(),view=el('view').value,excess=el('units').value==='excess',selected=el('background').value,b=p[selected];
 const convert=q=>q.map((v,i)=>excess?v-b[i]:v),lower=convert(p.low),upper=convert(p.high);
 const h=p.highlighted.find(h=>h.state===el('highlight').value)||p.highlighted[0];
 const common={x:p.x,customdata:p.utc,type:'scatter',mode:'lines',hovertemplate:'%{customdata}<br>%{y:.6f} m³/s<extra>%{fullData.name}</extra>'};
 const data=[{...common,y:lower,name:'All-state minimum',line:{width:0,color:'#6599c0'}},
 {...common,y:upper,name:'All-state range',fill:'tonexty',fillcolor:'rgba(101,153,192,.3)',line:{width:0,color:'#6599c0'}},
 {...common,y:convert(p.base),name:'Base model',line:{color:'#ce3d45',width:2}},
 {...common,y:convert(h.q),name:h.state,line:{color:'#222',width:1.5}},
 {...common,y:convert(p.background),name:'Current background',line:{color:'#9b59b6',dash:'dash',width:2}},
 {...common,y:convert(p.adjusted),name:'Proposed shared background',line:{color:'#168a65',dash:'dash',width:2}}];
 const range=p.ranges[view],ys=[];
 data.forEach(d=>d.y.forEach((v,i)=>{if(p.x[i]>=range[0]&&p.x[i]<=range[1])ys.push(v);}));
 const low=ys.reduce((a,b)=>Math.min(a,b),Infinity),high=ys.reduce((a,b)=>Math.max(a,b),-Infinity),pad=Math.max((high-low)*.08,.01);
 Plotly.react(el('plot'),data,{height:670,margin:{l:90,r:35,t:75,b:70},paper_bgcolor:'white',plot_bgcolor:'white',hovermode:'closest',legend:{orientation:'h',y:1.13},
 xaxis:{range,title:{text:'Hours from '+p.info.fixed_start_utc}},yaxis:{range:[low-pad,high+pad],title:{text:excess?'Q − selected background (m³/s)':'Discharge (m³/s)'}}},{responsive:true,scrollZoom:true});
 const i=p.info;el('notes').textContent=`Network ${p.example}: ${i.passing_before}/${i.children} children currently pass; ${i.failing_at_start} fall below the line at the start and ${i.failing_at_end} at the end. Proposed shared lowering: start ${i.lower_start_by_cms.toFixed(6)}, end ${i.lower_end_by_cms.toFixed(6)} m³/s. All ${i.children} children pass the proposed line. Base excess volume changes ${i.parent_excess_volume_change_pct.toFixed(3)}%. Highlighted: ${h.aliases}.`;
}
(async()=>{const bytes=Uint8Array.from(atob(el('payload').textContent.trim()),c=>c.charCodeAt(0));panels=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());panels.forEach(p=>el('network').add(new Option('Network '+p.example,p.example)));el('network').onchange=()=>{highlights();render();};['view','units','background','highlight'].forEach(id=>el(id).onchange=render);highlights();render();})();
