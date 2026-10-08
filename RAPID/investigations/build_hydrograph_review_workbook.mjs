import fs from 'node:fs/promises';
import {FileBlob, SpreadsheetFile, Workbook} from '@oai/artifact-tool';
const root=process.env.RIVER_HIERARCHY_ROOT ?? '/Users/6256481/Code/river-hierarchy';
// Source inspected: only an empty Blad1, no cells, formulas, annotations or drawings.
const wb=Workbook.create();
const data=JSON.parse(await fs.readFile(root+'/RAPID/validation/hydrograph_match_review_20261005/workbook_evidence.json','utf8'));
const out=root+'/RAPID/reviews';
const sheets={};
for(const name of ['Review','Peak timing','Peak windows','Event starts','Tributaries','Routing','Hydrographs','Methods']){
  sheets[name]=wb.worksheets.add(name);sheets[name].showGridLines=false;
}
function col(i){let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;}
function title(s,text,description,last='H'){
 s.getRange('A2').values=[[text]];s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#17365D'};
 s.getRange(`A3:${last}3`).merge();s.getRange('A3').values=[[description]];
 s.getRange(`A3:${last}3`).format={font:{name:'Arial',size:10,color:'#475569'},wrapText:true,rowHeight:35};
}
function table(s,row,headers,values,widths){
 const end=row+values.length,last=col(headers.length-1);
 s.getRange(`A${row}:${last}${row}`).values=[headers];
 if(values.length)s.getRange(`A${row+1}:${last}${end}`).values=values;
 s.getRange(`A${row}:${last}${end}`).format={font:{name:'Arial',size:10,color:'#172B4D'},rowHeight:24,verticalAlignment:'center'};
 s.getRange(`A${row}:${last}${row}`).format={fill:'#264766',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42};
 for(let i=0;i<headers.length;i++)s.getRange(`${col(i)}${row}:${col(i)}${end}`).format.columnWidth=widths?.[i]??20;
 for(let r=row+1;r<=end;r++)if((r-row)%2===0)s.getRange(`A${r}:${last}${r}`).format.fill='#F1F5F9';
 s.freezePanes.freezeRows(row);s.freezePanes.freezeColumns(1);
 return end;
}
// Numeric Excel serials avoid an import/export Date conversion rounding edge case.
const stamp=x=>Date.parse(String(x).replace(' ','T'))/86400000+25569;
const dates=(s,range)=>s.getRange(range).setNumberFormat('yyyy-mm-dd hh:mm');
const numeric=(s,range,format='0.0')=>s.getRange(range).setNumberFormat(format);

let s=sheets['Peak timing'];
title(s,'Peak timing and discharge','UTC throughout. Positive error = model later than downstream gauge. First maxima remain the recorded reference; alternatives are diagnostics.','N');
table(s,5,['Example','Up first UTC','Up last UTC','Down first UTC','Down last UTC','Model first UTC','Up peak m³/s','Down peak m³/s','Model peak m³/s','First-max error h','95% center error h','Down / up peak','Applied forcing first UTC','Observed first lag h'],data.cases.map(r=>[r.example,stamp(r.upstream_peak_first_utc),stamp(r.upstream_peak_last_utc),stamp(r.downstream_peak_first_utc),stamp(r.downstream_peak_last_utc),stamp(r.model_peak_first_utc),r.upstream_peak_cms,r.downstream_peak_cms,r.model_peak_cms,null,r.error_95_mid_h,null,stamp(r.forcing_peak_first_utc),r.observed_first_lag_h]),[10,22,22,22,22,22,17,17,17,17,18,16,22,18]);
dates(s,'B6:F21');dates(s,'M6:M21');numeric(s,'G6:L21');numeric(s,'N6:N21');
for(let row=6;row<=21;row++){s.getRange(`J${row}`).formulas=[[`=(F${row}-D${row})*24`]];s.getRange(`L${row}`).formulas=[[`=H${row}/G${row}`]];}
s.getRange('A24').values=[['Example 48: separate candidate peaks']];s.getRange('A24').format.font={bold:true,size:13};
s.getRange('A25:H25').merge();s.getRange('A25').values=[['Four highest peaks per series; ≥18 h separation and ≥0.5% range prominence. Daily shoulders included. Rank alone does not establish a match.']];s.getRange('A25:H25').format={wrapText:true,rowHeight:30};
table(s,27,['Series','Rank','First UTC','Last UTC','Midpoint UTC','Peak m³/s','Prominence m³/s'],data.events48.map(r=>[r.series,r.chronological_rank,stamp(r.first_utc),stamp(r.last_utc),stamp(r.midpoint_utc),r.peak_cms,r.prominence_cms]),[14,10,22,22,22,22,18]);dates(s,'C28:E43');numeric(s,'F28:G43');
s.getRange('A47').values=[['Alternative observed features — retained alongside global maxima']];s.getRange('A47').format.font={bold:true,size:13};
table(s,49,['Example','Feature','Start UTC','End UTC','Reference UTC','Model minus reference h','Status'],[
 ['03','Exact maximum plateau',stamp('2023-02-17T17:45Z'),stamp('2023-02-17T23:00Z'),stamp('2023-02-17T20:22:30Z'),2.2916666667,'Model lies within plateau'],
 ['05','User-proposed shoulder',stamp('2024-01-07T19:00Z'),stamp('2024-01-08T09:00Z'),stamp('2024-01-08T02:00Z'),-2.5972222222,'Exploratory; model within window'],
 ['11','User-proposed bump',null,null,stamp('2024-05-18T09:45Z'),4.9166666667,'Exploratory point; not adopted'],
 ],[14,30,22,22,22,22,38]);dates(s,'C50:E52');numeric(s,'F50:F52');
// Table helper below the primary table must not move its original panes or narrow primary columns.
s.freezePanes.freezeRows(5);for(const c of ['B','C','D','E','F','M'])s.getRange(`${c}5:${c}52`).format.columnWidth=24;
s.getRange('B50:B52').format.wrapText=true;s.getRange('A50:G52').format.rowHeight=38;
s.getRange('A55').values=[['48: provisional chronological feature comparisons']];s.getRange('A55').format.font={bold:true,size:13};
const m48=data.events48.filter(r=>r.series==='model'),d48=data.events48.filter(r=>r.series==='downstream');
table(s,57,['Feature rank','Observed midpoint UTC','Model midpoint UTC','Model minus observed h','Pairing status'],m48.map((r,i)=>[i+1,stamp(d48[i].midpoint_utc),stamp(r.midpoint_utc),null,'Provisional daily feature pairing']),[14,24,24,24,44]);
for(let r=58;r<=61;r++)s.getRange(`D${r}`).formulas=[[`=(C${r}-B${r})*24`]];
dates(s,'B58:C61');numeric(s,'D58:D61');s.getRange('E58:E61').format.wrapText=true;
s.freezePanes.freezeRows(5);

s=sheets['Review'];s.tabColor='#17365D';
title(s,'Hydrograph match review — 5 October 2026','Recorded maxima, alternative peak windows and explanatory evidence are kept separate. No routing or primary response metrics were changed.','G');
table(s,5,['Example','First-max error h','95% center error h','Down / up peak','Evidence-based reading','Next check','Your observations (editable)'],data.cases.map(r=>[r.example,null,null,null,r.assessment,r.next_check,r.user_note]),[10,15,16,14,58,50,58]);
for(let row=6;row<=21;row++){s.getRange(`B${row}:D${row}`).formulas=[[`='Peak timing'!J${row}`,`='Peak timing'!K${row}`,`='Peak timing'!L${row}`]];}
numeric(s,'B6:D21');s.getRange('E6:G21').format.wrapText=true;s.getRange('A6:G21').format.rowHeight=95;s.getRange('G6:G21').format.fill='#FFF4CD';
s.getRange('A24:G24').merge();s.getRange('A24').values=[['Overall: peak ambiguity improves the interpretation of several cases, but sizable timing and discharge differences remain. Case 46 is a clear timing failure for the selected event.']];s.getRange('A24:G24').format={wrapText:true,rowHeight:38,font:{name:'Arial',size:11,bold:true}};
s.getRange('A26:G26').merge();s.getRange('A26').values=[['First-max error compares outlet UTC timestamps directly. The 95% comparison uses each series’ own connected ≥95%-of-maximum window midpoint. Neither is a confidence interval.']];s.getRange('A26:G26').format={wrapText:true,rowHeight:32};

s=sheets['Peak windows'];
title(s,'Connected peak windows','Thresholds are fractions of maximum discharge, not percentiles. Each row is the connected sampled block containing the FIRST maximum; gaps >1.5 median steps break a block.','H');
table(s,5,['Example','Series','Fraction of maximum','Start UTC','End UTC','Midpoint UTC','Width h','Peak m³/s'],data.windows.map(r=>[r.example,r.series,r.fraction,stamp(r.start_utc),stamp(r.end_utc),stamp(r.midpoint_utc),r.width_h,r.peak_cms]),[10,16,21,23,23,23,13,17]);
numeric(s,`C6:C${5+data.windows.length}`,'0.0%');dates(s,`D6:F${5+data.windows.length}`);numeric(s,`G6:H${5+data.windows.length}`);

s=sheets['Event starts'];
title(s,'Three different meanings of event start','Gauge review windows select observations. Legacy minima define saved rise metrics. The accepted duration uses the parent’s 1% departure and final settlement. Peak lag itself is output peak minus input peak.','G');
table(s,5,['Example','Gauge window start UTC','Gauge window end UTC','Legacy start UTC','Forcing peak UTC','Legacy start → peak min','Parent 1% departure UTC'],data.starts.map(r=>[r.example,stamp(r.gauge_window_start_utc),stamp(r.gauge_window_end_utc),stamp(r.legacy_local_min_start_utc),stamp(r.forcing_peak_utc),r.legacy_start_to_forcing_peak_min,stamp(r.parent_1pct_departure_utc)]),[10,24,24,24,24,24,25]);dates(s,'B6:E21');dates(s,'G6:G21');numeric(s,'F6:F21');
s.getRange('A24:G24').merge();s.getRange('A24').values=[['Legacy automatic start = last detected local forcing minimum before the dominant peak (no default smoothing/prominence filter); if none exists, earliest global pre-peak minimum. Small digitization wiggles can make this too late.']];s.getRange('A24:G24').format={wrapText:true,rowHeight:42};
s.getRange('A26:G26').merge();s.getRange('A26').values=[['The parent 1% departure is interpolated relative to the initial stable level, using 1% of the parent’s initial-to-peak amplitude. It refers to the modeled outlet event, not a newly selected observational start. UTC clocks were recovered from identical saved forcing where needed.']];s.getRange('A26:G26').format={wrapText:true,rowHeight:42};

s=sheets['Tributaries'];
title(s,'SWORD tributary evidence','Drainage area (facc, km²) is not discharge. Candidates enter the endpoint-projected main corridor; connected/rejoining branches are excluded from counts. Areas use complete reaches and can extend beyond clipped boundaries.','J');
table(s,5,['Example','Corridor length km','External candidates','Largest area / receiver','Inlet reach area km²','Outlet reach area km²','Reach-area growth','Unmapped-tributary flag reaches'],data.corridors.map(r=>[r.example,r.network_reference_length_km,r.external_tributary_candidates,r.largest_tributary_area_fraction,r.inlet_reach_facc_km2,r.outlet_reach_facc_km2,r.reach_area_growth_fraction,r.unmapped_tributary_flag_reaches]),[10,20,20,22,24,24,20,23]);numeric(s,'B6:B21');numeric(s,'D6:D21','0.0%');numeric(s,'E6:F21','#,##0');numeric(s,'G6:G21','0.0%');
s.getRange('A24:J24').merge();s.getRange('A24').values=[['Individual direct feeders. “Area / receiver” compares tributary accumulation with the receiving reach, not event-flow contribution. Missing area/quality remains missing; no totals are summed across possibly shared catchments.']];s.getRange('A24:J24').format={wrapText:true,rowHeight:38};
table(s,26,['Example','Tributary reach ID','Receiver reach ID','Classification','Area km²','Receiver area km²','Area / receiver','Width m','River name','facc quality'],data.tributaries.map(r=>[r.example,r.tributary_reach_id,r.receiving_reach_id,r.classification,r.facc_km2,r.receiver_facc_km2,r.area_fraction,r.width_m,r.river_name,r.facc_quality]),[10,23,23,31,24,24,20,23,22,20]);numeric(s,`E27:F${26+data.tributaries.length}`,'#,##0');numeric(s,`G27:G${26+data.tributaries.length}`,'0.0%');numeric(s,`H27:H${26+data.tributaries.length}`);s.freezePanes.freezeRows(5);

s=sheets['Routing'];
title(s,'Accumulated routing-time diagnostics','Σ p(link) × segment count × K, with width-based bifurcation allocation. This is a mean-residence diagnostic, not an observed travel time or a peak-lag prediction.','I');
table(s,5,['Example','Links','Min slope m/m','Median slope m/m','Mean residence h','Physical-length residence h','Segmentation extra h','Flow-weighted path km','Residence at S < 10⁻⁴'],data.routing.map(r=>[r.example,r.links,r.minimum_slope,r.median_slope,r.flow_weighted_mean_residence_h,r.physical_length_mean_residence_h,r.length_floor_extra_h,r.flow_weighted_path_km,r.share_residence_slope_below_1e4]),[10,12,21,21,20,26,23,24,23]);numeric(s,'C6:D21','0.00E+00');numeric(s,'E6:H21');numeric(s,'I6:I21','0.0%');
s.getRange('A24:I24').merge();s.getRange('A24').values=[['46: 339 h mean residence agrees with 339 h modeled peak lag, versus 140 h observed first-peak lag. Only 3.3 h comes from effective-length segmentation inflation. 81.8% of residence is on slopes below 10⁻⁴; 7.4% below 10⁻⁵.']];s.getRange('A24:I24').format={wrapText:true,rowHeight:40};
table(s,27,['46 distance to outlet km','Links','Residence contribution h','Median slope m/m','Median celerity m/s','Residence share'],data.bands46.map(r=>[r.band,r.links,r.residence_h,r.median_slope,r.median_celerity_mps,r.residence_fraction]),[25,12,24,21,20,26]);numeric(s,'C28:C43');numeric(s,'D28:D43','0.00E+00');numeric(s,'E28:E43','0.00');numeric(s,'F28:F43','0.0%');
table(s,47,['46 link ID','Midpoint to outlet km','Physical length m','Width m','Depth m','Slope m/m','Celerity m/s','Flow fraction','Residence contribution h'],data.links46.map(r=>[r.id_link,r.reference_mid_dist_out_km,r.link_length_m,r.width,r.depth_m,r.slope,r.rapid_celerity_mps,r.flow_fraction,r.residence_contribution_h]),[25,23,24,21,20,26,23,24,23]);numeric(s,'B48:E67','0.0');numeric(s,'F48:F67','0.00E+00');numeric(s,'G48:G67','0.00');numeric(s,'H48:H67','0.0%');numeric(s,'I48:I67');s.freezePanes.freezeRows(5);

s=sheets['Hydrographs'];title(s,'Hydrographs on their recorded UTC clocks','Equal-maximum samples are marked. Full peak spans can contain gaps. Plots are static evidence snapshots; notebook controls remain available for interactive gauge-only/model overlays.','L');
s.getRange('A1:L610').format.columnWidth=13;s.getRange('A1:L610').format.rowHeight=20;
for(let i=0;i<data.cases.length;i++){
 const r=data.cases[i],row=6+i*35;
 s.getRange(`A${row}`).values=[[`Example ${r.example}`]];s.getRange(`A${row}`).format.font={name:'Arial',size:13,bold:true};
 const bytes=await fs.readFile(root+`/RAPID/validation/peak_windows_20261005/${r.example}_peak_windows.png`);
 s.images.add({dataUrl:'data:image/png;base64,'+bytes.toString('base64'),anchor:{from:{row:row,col:0},extent:{widthPx:1100,heightPx:450}}});
 s.getRange(`A${row+25}:L${row+27}`).merge();s.getRange(`A${row+25}`).values=[[r.assessment]];s.getRange(`A${row+25}:L${row+27}`).format={font:{name:'Arial',size:11},wrapText:true};
}
const full=await fs.readFile(root+'/RAPID/validation/hydrograph_match_review_20261005/48_full_event.png');s.getRange('A568').values=[['48: full-event comparison']];s.images.add({dataUrl:'data:image/png;base64,'+full.toString('base64'),anchor:{from:{row:568,col:0},extent:{widthPx:1100,heightPx:642}}});s.freezePanes.freezeRows(4);

s=sheets['Methods'];title(s,'Definitions, provenance and review decisions','Review reference: HYDROGRAPH-MATCH-2026-10-05. Written after visual inspection showed that first-maximum timestamps can misrepresent plateau and multi-peak agreement.','D');
const methods=[
 ['First-max timing error','Model outlet first maximum UTC minus observed downstream first maximum UTC, in hours. This differs from modeled-minus-observed gauge lag when forcing and upstream maxima differ.'],
 ['Exact plateau','Report first and last equal maxima and their midpoint. A span containing separated maxima is not a continuous plateau; the midpoint can be below maximum.'],
 ['95% peak window','Connected sampled block around the first global maximum with Q ≥ 0.95 Qmax. NOT the 95th percentile, not an uncertainty interval, and not the 1% event-duration definition. No gap bridging or time realignment.'],
 ['Model inside a broad window','Descriptive overlap only. Very broad windows may admit large errors; do not equate overlap with good validation.'],
 ['Example 05 shoulder','User-specified 7 Jan 19:00 to 8 Jan 09:00 UTC. Observed Q 1089.7–1107.0 m³/s, below global maximum 1153.6. Model peak lies in the window. Tributary timing is unavailable.'],
 ['Example 11 feature','18 May 09:45 UTC is the user-proposed secondary feature, not an automatically identified replacement maximum. No date/window tuned to model agreement.'],
 ['Example 48 candidates','Four highest local maxima per series after ≥18 h separation and ≥0.5% total-series-range prominence. Low threshold retains daily shoulders. Independent lists do not prove which peaks correspond; measured tied spans remain available separately.'],
 ['Sampling resolution','Gauge tables are the repository’s processed observations. Their median time step does not establish original instrument precision. Repeated discharge values can reflect quantization.'],
 ['Lateral inputs','The modeled event uses upstream forcing only. A higher observed downstream hydrograph can reflect tributary/local inflow, rating-curve differences or other processes. SWORD drainage areas establish potential supply, not its event timing or magnitude.'],
 ['SWORD candidate rule','Trace reviewed source-to-sink main reach path. Inspect direct feeders at interior/downstream reach entrances; exclude branches reachable from the source or rejoining a common upstream channel. Exclude source-reach junctions with ambiguous clipping. Scope is the modeled corridor, not an independently redelineated gauge basin.'],
 ['facc caveat','Maximum accumulated drainage area for a reach, km². Full-reach areas can extend beyond clipped endpoints. Quality flags retained. No inferred tributary discharge and no summed-area budget. Unmapped tributary flags do not count distinct tributaries.'],
 ['Routing diagnostic','Propagate probability at each split proportional to routing width; sum probability × segment count × K. Physical-length version uses probability × L/c. Their difference isolates effective-length/segmentation inflation. No rerouting.'],
 ['Case 46 interpretation','Widespread slow transit across a long network. Ten largest individual contributions total ~16% of residence; the minimum slope alone is not the explanation. Slopes, anchored depths and Manning celerity jointly determine the modeled travel time. No evidence here authorizes increasing slopes to fit a peak.'],
 ['Response-metric implications','Changing observed peak references does not alter absolute parent–child timing, modeled relative lag, event-duration change or peak-excess change. It does alter observed-lag-normalized timing and gauge-scaled speed comparisons. Choosing another applied-forcing peak changes modeled relative-lag denominators too.'],
 ['Current assessment','Paired-collapse results remain conditional model comparisons. Better feature correspondence supports interpretation, but cannot establish absolute predictive skill or explain all residuals. Case 46 remains a material timing failure.'],
 ['Saved calculations','Numerical evidence generated by RAPID/investigations/hydrograph_match_review.py and prepare_hydrograph_review_workbook.py. Case notes are the user’s observations, paraphrased without adopting them as measured conclusions.'],
 ['Production snapshot','shared-head-anchored-depth-signed-inflow-20261003; unchanged n=0.035, depth-width exponent=0.65, celerity limits=0.05–5 m/s.'],
 ['Gauge source','SWORD_gauge_match/outputs/mask_hydro_quality.xlsx (Hydrograph_quality) and station subdaily_timeseries.parquet through the existing gauge loader.'],
 ['Timing evidence','RAPID/validation/hydrograph_match_review_20261005 and RAPID/validation/peak_windows_20261005. Original event UTC recovered only by exact saved-forcing correspondence.'],
 ['SWORD source','Local /Volumes/PhD/SWORD/v17c reach tables and frozen endpoint corridor. Field definitions: https://zenodo.org/records/15299138 ; https://github.com/SWORD-Global/SWORD'],
 ['Review workflow','Edit yellow Review observations cells; retain measured evidence. Record any later adopted reference/window and why before recalculating gauge-dependent metrics. Original empty root workbook left untouched because Excel lock file was present.'],
];
table(s,5,['Subject','Definition / evidence'],methods,[29,130]);s.getRange(`B6:B${5+methods.length}`).format.wrapText=true;s.getRange(`A6:B${5+methods.length}`).format.rowHeight=57;s.getRange(`A6:A${5+methods.length}`).format.wrapText=true;

for(const name of ['Review','Peak timing','Peak windows','Event starts','Tributaries','Routing']){
 const sh=sheets[name];sh.getRange('A6:A21').setNumberFormat('@');
}
sheets['Review'].getRange('D6:D21').setNumberFormat('0.00');
sheets['Peak timing'].getRange('L6:L21').setNumberFormat('0.00');
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Review!A5:D9',include:'values,formulas',tableMaxRows:5,tableMaxCols:4,maxChars:3500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'Formula error scan',maxChars:1500})).ndjson);
await fs.mkdir(out,{recursive:true});
for(const [name,range] of [['Review','A2:G9'],['Peak timing','A49:G61'],['Peak windows','A2:H12'],['Event starts','A2:G10'],['Tributaries','A2:H10'],['Routing','A2:I10'],['Hydrographs','A6:L34'],['Methods','A2:B11']]){
 const img=await wb.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile('/tmp/river-hydrograph-workbook/'+name.replaceAll(' ','_')+'.png',new Uint8Array(await img.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'/hydrograph_match_review.xlsx');
await fs.rm(out+'/hydrograph_match_review.xlsx.inspect.ndjson',{force:true});
console.log('EXPORTED '+out+'/hydrograph_match_review.xlsx');
