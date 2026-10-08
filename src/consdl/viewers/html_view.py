"""The cluster-consistency view page (``/view``), served by ``consdl.viewers.umap_view``."""

FOOT = (
    "<b>Views.</b> <i>seed</i>: one arch × cell, its seeds (Hungarian or NN). <i>cross</i>: one "
    "arch, seed 0 of each of its cells. <i>arch</i>: one cell, seed 0 of each architecture. Every view "
    "is ONE joint UMAP-2D (cosine) over all its panels, so the same place is the same direction; "
    "distances between clusters are not meaningful. "
    "<b>Colour</b> = the nearest centroid of the view's colour anchor (marked in its panel title; "
    "spherical k-means at 48 atoms per cluster; on a cross view the arch's largest cell): the 30 largest clusters "
    "are coloured, the rest grey; the same colour is the same anchor cluster in every panel. "
    "<b>Gauge.</b> Matches and similarities use signed decoder cos, except |cos| for any pair "
    "involving iSAE, whose atom sign is unidentified (its atoms are also sign-aligned to their "
    "centroid for the UMAP). "
    "<b>Top inputs</b>: from the precompute's pass over the input pool (ImageNet val images ranked "
    "by their peak |a| over patches; FineWeb tokens ranked by |a|). Masks dim each patch by "
    "that latent's |a|, min–max scaled per image (dim = weaker than this image's peak, not "
    "zero), encoded live; text is shaded by the signed activation (orange +, blue −). "
    "<b>fp</b> (vision): footprint cos, how alike two latents' per-image peak |a| is over the "
    "image pool; a blue chip border means above both cells' random-pair p95 (co-fires), dashed "
    "means below. "
    "<b>Similar</b>: thin rings mark every atom with cos ≥ τ to the selected latent (at most "
    "'max' per panel; fainter = lower cos), in its own panel too. Each row lists them as chips — "
    "click one to show its images in that row (dashed ring); click again to go back. 'span' = cos "
    "of the selected atom to its best NON-NEGATIVE mix of those atoms; span well above NN (bold "
    "when ≥ 0.1 higher, with ≥ 2 atoms) is a feature-splitting candidate. span rises with the "
    "number of atoms by construction — compare at equal τ and max, and confirm with the images: "
    "split children should each show a sub-case.")


PAGE_TMPL = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UMAP Latent Viewer</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d9d9de;--card:#f6f6f8;--ring:#1d1d1f;--hi:#e0003c;--hibg:rgba(224,0,60,.07)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--line:#35353b;--card:#212125;--ring:#ececf0;--hi:#ff4f7b;--hibg:rgba(255,79,123,.10)}}
:root[data-theme="dark"]{--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--line:#35353b;--card:#212125;--ring:#ececf0;--hi:#ff4f7b;--hibg:rgba(255,79,123,.10)}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:12px/1.35 system-ui,sans-serif;display:flex;flex-direction:column;overflow:hidden}
header{padding:6px 12px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap;flex:none}
header h1{font-size:14px;margin:0}
header a{color:var(--mut)}
.mut{color:var(--mut)}
details.help{position:relative}
details.help summary{cursor:pointer;color:var(--mut);list-style:none;border:1px solid var(--line);border-radius:10px;padding:0 7px}
details.help div{position:absolute;right:0;top:22px;width:min(560px,80vw);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;z-index:5;box-shadow:0 4px 18px rgba(0,0,0,.18)}
main{flex:1;min-height:0;display:flex;gap:10px;padding:8px 12px}
#left{flex:none;height:100%;min-width:0}
#panels{display:grid;gap:4px;height:100%;width:100%}
.panel{background:var(--card);border-radius:6px;position:relative;min-width:0;min-height:0;overflow:hidden}
.panel h2{position:absolute;left:6px;top:3px;margin:0;font-size:11px;font-weight:600;pointer-events:none;z-index:1}
.panel.sel h2{color:var(--hi)}
.panel canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair}
.chart{background:var(--card);border-radius:6px;position:relative;min-width:0;min-height:0;display:flex;flex-direction:column;padding:4px 6px}
.chart .hd{font-size:11px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.chart .hd b{font-weight:600}
.chart .lg{font-size:10px;display:flex;gap:8px;flex-wrap:wrap;color:var(--mut)}
.chart .lg i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:3px}
.chart .rd{font-size:10.5px;color:var(--mut);min-height:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chart .cv{position:relative;flex:1;min-height:0}
.chart canvas{position:absolute;inset:0;width:100%;height:100%;cursor:col-resize}
#right{flex:1;min-width:0;display:flex;flex-direction:column;gap:4px}
#info{flex:none;min-height:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#colhdr,.row{display:grid;grid-template-columns:158px repeat(6,minmax(0,1fr));gap:3px}
#colhdr{flex:none;color:var(--mut);font-size:11px;text-align:center}
#colhdr span.g{grid-column:span 2;border-bottom:1px solid var(--line)}
#rows{flex:1;min-height:0;display:grid;gap:3px}
.row{min-height:0;border-left:3px solid transparent;padding-left:4px;border-radius:3px}
.row.anchor{border-left-color:var(--hi);background:var(--hibg)}
.row.anchor .lab b{color:var(--hi)}
.lab{font-size:11px;overflow:hidden;min-height:0;display:flex;flex-direction:column}
.lab b{font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:2px;overflow:auto;min-height:0;margin-top:1px}
.chip{border:1px solid var(--line);border-radius:3px;padding:0 3px;cursor:pointer;font-size:10px;line-height:14px;white-space:nowrap}
.chip:hover{border-color:var(--mut)}
.chip.on{border-color:var(--fg);font-weight:700}
.chip.co{border:1.5px solid #2a78d6}
.chip.apart{border-style:dashed}
.fp{color:#2a78d6}
.gain{font-weight:700}
#simctl input[type=range]{width:90px;vertical-align:middle}
#simctl input[type=number]{width:3.2em}
#jl{width:5.5em}
header select,header input,header button{font:inherit;font-size:12px}
.cell{position:relative;min-height:0;min-width:0}
.cell img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
.cell .txt{position:absolute;inset:0 0 13px 0;overflow:hidden;font:9.5px/1.3 ui-monospace,Menlo,monospace;padding:2px 3px;background:var(--card);border-radius:4px;white-space:pre-wrap;word-break:break-word}
.cell .txt .pk{background:#ffd54f;color:#1d1d1f;border-radius:2px;padding:0 1px}
.cell .txt .tk{border-radius:2px}
.cell .txt .pk2{outline:1.5px solid currentColor;outline-offset:-1px;font-weight:700}
.cell .cap{position:absolute;left:0;right:0;bottom:0;font-size:10px;line-height:1.2;padding:1px 3px;background:rgba(0,0,0,.55);color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center}
label{cursor:pointer}
@media (max-aspect-ratio:1/1){body{overflow:auto}main{flex-direction:column}#left{height:auto;width:100%!important;aspect-ratio:1}#rows{grid-auto-rows:110px}}
</style></head><body>
<header><h1 id="title">loading…</h1>
<span id="nav"></span>
<span id="rules"></span>
<span id="jump" title="select and pin a latent by number (Enter or go)">latent: <select id="jp"></select> # <input type="number" id="jl" min="0" placeholder="id"> <button id="jgo">go</button></span>
<span id="canchor"></span>
<span id="simctl" title="Also ring every atom with decoder cos ≥ τ to the selected latent (at most 'max' per panel)">similar: cos ≥ <input type="range" id="tau" min="0.2" max="0.95" step="0.05" value="0.4"> <b id="tauv">0.40</b>
· max <input type="number" id="kk" min="1" max="50" value="8"> <label><input type="checkbox" id="simtog" checked> rings</label></span>
<span><label><input type="checkbox" id="rawtog"> unmasked images</label></span>
<span class="mut">hover = preview · click = pin · wheel/drag = zoom/pan (all panels) · dbl-click = reset</span>
<details class="help"><summary>?</summary><div>%%FOOT%%</div></details></header>
<main>
  <section id="left"><div id="panels"></div></section>
  <section id="right">
    <div id="info" class="mut">loading…</div>
    <div id="colhdr"><span></span><span class="g">top · ranks 1, 2</span><span class="g">middle · 50, 51</span><span class="g">bottom · 99, 100 of top 100</span></div>
    <div id="rows"></div>
  </section>
</main>
<script>
const PAL=["#AA0DFE","#3283FE","#85660D","#782AB6","#565656","#1C8356","#16FF32","#F7E1A0","#E2E2E2","#1CBE4F","#C4451C","#DEA0FD","#FE00FA","#325A9B","#FEAF16","#F8A19F","#90AD1C","#F6222E","#1CFFCE","#2ED9FF","#B10DA1","#C075A6","#FC1CBF","#B00068","#FBE426","#FA0087","#2E91E5","#E15F99","#1CA71C","#FB0D0D"];
let D, rule=null, pinned=null, cur=null, view=null, timer=null, COLS=1, ROWS=1;
let SIM=null, rowSel={};          // similar atoms of the selection; per-row latent picked from a chip
let HL=null, CH=null, HLpin=false; // anchor cluster lit from the chart; the chart's state; clicked-locked
let FPC={}, FPNULL={};              // footprint cos of the selection vs "panel:latent"; per-cell random-pair null
const RULE_LONG={hung:"Hungarian",nn:"nearest neighbour",fp:"footprint Hungarian (partner picked by co-firing)"};
const RULE_SHORT={hung:"Hungarian",nn:"NN",fp:"footprint"};
const cvs={}, ctxs={}, grids={}, rowEl={}, pdiv={};
const VIEW="%%VIEW%%", enc=encodeURIComponent, lbl=s=>(D.labels&&D.labels[s])||s;
fetch(`/view/data.json?id=${VIEW}`).then(r=>r.json()).then(d=>{D=d;rule=D.rules[0];init();
  fetch("/nav.json").then(r=>r.json()).then(buildNav);
  fetch("/fpnull").then(r=>r.json()).then(n=>{FPNULL=n;});});
function buildNav(N){
  const kind=D.kind, arch=D.arch||(N.arches.includes("consdl")?"consdl":N.arches[0]), cell=D.cell||N.cells[N.cells.length-1];
  const sel=(id,opts,val)=>`<select id=${id}>`+opts.map(([v,t])=>`<option value="${v}"${v==val?" selected":""}>${t}</option>`).join("")+`</select>`;
  document.getElementById("nav").innerHTML=
    sel("nk",[["seed","seeds · one cell"],["cross","cells · one arch"],["arch","archs · one cell"]],kind)+" "+
    sel("na",N.arches.map(a=>[a,N.labels[a]||a]),arch)+" "+
    sel("nc",N.cells.map(c=>[c,N.cell_labels[c]||c]),cell);
  const nk=document.getElementById("nk"),na=document.getElementById("na"),nc=document.getElementById("nc");
  const sync=()=>{na.disabled=nk.value=="arch";nc.disabled=nk.value=="cross";};sync();
  const go=()=>{sync();const k=nk.value,id=k=="seed"?`seed__${na.value}__${nc.value}`:k=="cross"?`cross__${na.value}`:`arch__${nc.value}`;
    if(id===VIEW)return;
    if(!N.views.includes(id)){document.getElementById("info").innerHTML=`<b style="color:var(--hi)">${id} is not built</b> (run viewer_precompute for it)`;return;}
    location.href="/view?id="+id;};
  nk.onchange=na.onchange=nc.onchange=go;
}
function init(){
  document.title=`UMAP · ${D.title}`;
  if(D.modality==="text"){             // no images to unmask; the column header reads "tokens"
    document.getElementById("rawtog").parentElement.style.display="none";
    const h=document.querySelectorAll("#colhdr span.g");
    if(h.length===3){h[0].textContent="top tokens · ranks 1, 2";h[1].textContent="middle · 50, 51";h[2].textContent="bottom · 99, 100 of top 100";}
  }
  document.getElementById("title").innerHTML=D.title+(D.gauge&&D.gauge!=="signed"?` <span class=mut style="font-size:11px;font-weight:normal">(${D.gauge})</span>`:"");
  document.getElementById("rules").innerHTML=D.rules.length>1
    ?`match: `+D.rules.map((r,q)=>`<label><input type="radio" name="rule" value="${r}"${q?"":" checked"}> ${RULE_LONG[r]||r}</label>`).join(" ")
    :`match: nearest neighbour <span class=mut>(not one-to-one)</span>`;
  const P=document.getElementById("panels"), n=D.seeds.length;
  COLS=Math.ceil(Math.sqrt(n)); ROWS=Math.ceil(n/COLS);
  P.style.gridTemplateColumns=`repeat(${COLS},1fr)`; P.style.gridTemplateRows=`repeat(${ROWS},1fr)`;
  let xs=[],ys=[];
  for(const s of D.seeds){const a=D.xy[s];for(let i=0;i<a.length;i+=2){xs.push(a[i]);ys.push(a[i+1]);}}
  const mn=(v)=>v.reduce((a,b)=>Math.min(a,b)), mx=(v)=>v.reduce((a,b)=>Math.max(a,b));
  view={x0:mn(xs),x1:mx(xs),y0:mn(ys),y1:mx(ys)}; view.home={...view};
  for(const s of D.seeds){
    const div=document.createElement("div");div.className="panel";pdiv[s]=div;
    div.innerHTML=`<h2></h2>`;
    const c=document.createElement("canvas");div.appendChild(c);P.appendChild(div);
    cvs[s]=c;ctxs[s]=c.getContext("2d");
    c.addEventListener("mousemove",e=>onMove(e,s));
    c.addEventListener("click",e=>{if(pan&&pan.moved)return;const h=hit(e,s);pinned=(pinned&&h!=null&&pinned.s==s&&pinned.i==h)?null:(h!=null?{s,i:h}:null);if(pinned)select(pinned.s,pinned.i,true);else if(cur)select(cur.s,cur.i,true);});
    c.addEventListener("wheel",e=>{e.preventDefault();zoom(e,s);},{passive:false});
    c.addEventListener("mousedown",e=>startPan(e,s));
    c.addEventListener("dblclick",()=>{Object.assign(view,view.home);drawAll();});
  }
  // fixed row order = panel order; the selected latent's row is marked, never moved
  const R=document.getElementById("rows");
  R.style.gridTemplateRows=`repeat(${n},minmax(0,1fr))`;
  for(const s of D.seeds){
    const row=document.createElement("div");row.className="row";
    row.innerHTML=`<div class=lab><b>${lbl(s)}</b></div>`+"<div class=cell></div>".repeat(6);
    R.appendChild(row);rowEl[s]=row;
  }
  document.querySelectorAll("input[name=rule]").forEach(r=>r.onchange=()=>{rule=r.value;if(cur)select(cur.s,cur.i,true);});
  document.getElementById("rawtog").onchange=()=>{if(cur)loadRows();};
  titles();
  setupChart();
  // jump to a latent by number
  const jp=document.getElementById("jp"),jl=document.getElementById("jl");
  jp.innerHTML=D.seeds.map(s=>`<option value="${s}">${lbl(s)}</option>`).join("");
  const setMax=()=>{jl.max=D.xy[jp.value].length/2-1;jl.placeholder=`0–${jl.max}`;};setMax();jp.onchange=setMax;
  document.getElementById("jgo").onclick=jump;
  jl.addEventListener("keydown",e=>{if(e.key==="Enter")jump();});
  // colour anchor: any seed-0 panel (only seed 0 was clustered)
  const cands=D.anchors||[];
  if(cands.length>1){
    document.getElementById("canchor").innerHTML=`colour by: <select id="cas">`+
      cands.map(s=>`<option value="${s}"${s==D.anchor?" selected":""}>${lbl(s)}</option>`).join("")+`</select>`;
    document.getElementById("cas").onchange=e=>recolour(e.target.value);
  }
  const tau=document.getElementById("tau");
  tau.oninput=()=>{document.getElementById("tauv").textContent=(+tau.value).toFixed(2);};
  tau.onchange=()=>{if(cur)loadSims();};
  document.getElementById("kk").onchange=()=>{if(cur)loadSims();};
  document.getElementById("simtog").onchange=drawAll;
  window.addEventListener("resize",layout);
  layout();
  document.getElementById("info").textContent="hover a point · click to pin";
}
function layout(){
  // left block = the panel grid at its own aspect, as tall as the window allows
  const L=document.getElementById("left"),h=L.getBoundingClientRect().height;
  const w=Math.min(h*COLS/ROWS, window.innerWidth*0.58);
  if(window.innerWidth>window.innerHeight)L.style.width=w+"px";
  drawAll();drawChart();
}
function size(c){const r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.max(1,r.width*dpr);c.height=Math.max(1,r.height*dpr);}
function tx(x,W){return (x-view.x0)/(view.x1-view.x0)*W;}
function ty(y,H){return H-(y-view.y0)/(view.y1-view.y0)*H;}
function drawAll(){for(const s of D.seeds)draw(s);}
function draw(s){
  const c=cvs[s],g=ctxs[s];size(c);const W=c.width,H=c.height,a=D.xy[s],col=D.col[s],dpr=window.devicePixelRatio||1;
  g.clearRect(0,0,W,H);
  const r=Math.max(1,1.2*dpr);
  if(HL==null){
    for(const pass of [0,1]){        // grey first, coloured on top
      for(let i=0;i<a.length/2;i++){const k=col[i];if((k<0)!==(pass==0))continue;
        g.fillStyle=k<0?"rgba(150,150,150,0.35)":PAL[k];
        g.fillRect(tx(a[2*i],W)-r/2,ty(a[2*i+1],H)-r/2,r,r);}
    }
  }else{                             // one anchor cluster lit (from the chart), everything else dimmed
    const clu=D.clu[s],R=Math.max(2,2.6*dpr);
    g.fillStyle="rgba(150,150,150,0.18)";
    for(let i=0;i<a.length/2;i++)if(clu[i]!==HL)g.fillRect(tx(a[2*i],W)-r/2,ty(a[2*i+1],H)-r/2,r,r);
    g.fillStyle=getCss("--hi");
    for(let i=0;i<a.length/2;i++)if(clu[i]===HL)g.fillRect(tx(a[2*i],W)-R/2,ty(a[2*i+1],H)-R/2,R,R);
  }
  const G=64,cells=new Map();       // spatial grid for hover
  for(let i=0;i<a.length/2;i++){const px=tx(a[2*i],W),py=ty(a[2*i+1],H);if(px<0||py<0||px>W||py>H)continue;
    const key=Math.floor(px/W*G)+","+Math.floor(py/H*G);if(!cells.has(key))cells.set(key,[]);cells.get(key).push(i);}
  grids[s]={G,cells};
  pdiv[s].classList.toggle("sel",!!cur&&cur.s==s);
  if(!cur)return;
  const P=SIM&&SIM.panels[s];
  if(P&&document.getElementById("simtog").checked){      // every similar atom: thin, fainter for lower cos
    const tau=SIM.tau;
    P.idx.forEach((j,n)=>dot(s,j,0.35+0.65*(P.cos[n]-tau)/Math.max(1e-6,1-tau)));
  }
  const m=matchesOf(cur.s,cur.i);ring(s,m[s],s==cur.s);
  if(rowSel[s]!=null&&rowSel[s]!==m[s])ring(s,rowSel[s],false,true);
}
function dot(s,i,alpha){
  const c=cvs[s],g=ctxs[s],W=c.width,H=c.height,a=D.xy[s],dpr=window.devicePixelRatio||1;
  g.globalAlpha=Math.max(0.2,Math.min(1,alpha));g.lineWidth=1.5*dpr;g.strokeStyle=getCss("--ring");
  g.beginPath();g.arc(tx(a[2*i],W),ty(a[2*i+1],H),4.5*dpr,0,7);g.stroke();g.globalAlpha=1;
}
function ring(s,i,isAnchor,dashed){
  if(i==null)return;const c=cvs[s],g=ctxs[s],W=c.width,H=c.height,a=D.xy[s],dpr=window.devicePixelRatio||1;
  const x=tx(a[2*i],W),y=ty(a[2*i+1],H),R=(isAnchor?9:7)*dpr;
  g.lineWidth=(isAnchor?3:2)*dpr;g.strokeStyle=isAnchor?getCss("--hi"):getCss("--ring");
  g.setLineDash(dashed?[3*dpr,2*dpr]:[]);
  g.beginPath();g.arc(x,y,R,0,7);g.stroke();g.setLineDash([]);
  g.beginPath();g.moveTo(x-R-7*dpr,y);g.lineTo(x-R-2*dpr,y);g.moveTo(x+R+2*dpr,y);g.lineTo(x+R+7*dpr,y);
  g.moveTo(x,y-R-7*dpr);g.lineTo(x,y-R-2*dpr);g.moveTo(x,y+R+2*dpr);g.lineTo(x,y+R+7*dpr);g.stroke();
}
function getCss(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function hit(e,s){
  const c=cvs[s],r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1,W=c.width,H=c.height;
  const px=(e.clientX-r.left)*dpr,py=(e.clientY-r.top)*dpr,{G,cells}=grids[s];
  const gx=Math.floor(px/W*G),gy=Math.floor(py/H*G);let best=null,bd=(8*dpr)**2;const a=D.xy[s];
  for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){const L=cells.get((gx+dx)+","+(gy+dy));if(!L)continue;
    for(const i of L){if(HLpin&&D.clu[s][i]!==HL)continue;   // locked cluster: pick only its atoms
      const d=(tx(a[2*i],W)-px)**2+(ty(a[2*i+1],H)-py)**2;if(d<bd){bd=d;best=i;}}}
  return best;
}
function onMove(e,s){
  if(pan||pinned)return;
  const h=hit(e,s);if(h==null)return;
  if(cur&&cur.s==s&&cur.i==h)return;
  select(s,h,false);
}
function matchesOf(s,i){
  const m={};m[s]=i;
  for(const b of D.seeds)if(b!=s)m[b]=D.match[`${rule}_${s}_${b}`][i];
  return m;
}
function titles(){
  for(const s of D.seeds)pdiv[s].querySelector("h2").textContent=lbl(s)+(s==D.anchor?" · colour anchor":"");
}
function jump(){
  const s=document.getElementById("jp").value,v=document.getElementById("jl").value,i=+v,n=D.xy[s].length/2;
  if(v===""||!Number.isInteger(i)||i<0||i>=n){
    document.getElementById("info").innerHTML=`<b style="color:var(--hi)">${lbl(s)} has latents 0–${n-1}</b>`;return;}
  // centre the shared view on it, keeping the zoom
  const x=D.xy[s][2*i],y=D.xy[s][2*i+1],hw=(view.x1-view.x0)/2,hh=(view.y1-view.y0)/2;
  view.x0=x-hw;view.x1=x+hw;view.y0=y-hh;view.y1=y+hh;
  pinned={s,i};select(s,i,true);
}
function recolour(anchor){
  document.getElementById("info").textContent="recolouring…";
  fetch(`/recolour?id=${VIEW}&anchor=${enc(anchor)}`).then(r=>r.json()).then(r=>{
    if(r.error){document.getElementById("info").textContent=r.error;return;}
    D.col=r.col;D.clu=r.clu;D.anchor=r.anchor;titles();HL=null;HLpin=false;chartData();drawAll();drawChart();
    if(cur){select(cur.s,cur.i,true);}
    else document.getElementById("info").textContent=`colours: ${lbl(r.anchor)} seed-0 clusters (K=${r.K})`;
  });
}
// ── the cluster-size chart: every panel's atoms on the colour anchor's clusters ──
// Counts come from D.clu (each atom's anchor cluster), so it needs no server
// call and follows "colour by". Shown only where the panel grid has a free slot.
const SERIES_COL={"consdl":"#2a78d6","relu":"#eb6834","topk":"#1baf7a",
  "batchtopk":"#eda100","matryoshka":"#e87ba4","mp":"#008300","isae":"#4a3aa7",
  "seed0":"#2a78d6","seed1":"#eb6834","seed2":"#1baf7a"};
function serCol(s){const p=s.split("/");return SERIES_COL[D.kind=="seed"?p[2]:p[0]]||"#888";}
function setupChart(){
  const free=COLS*ROWS-D.seeds.length;
  if(free<1||D.seeds.length>8)return;               // cross view: no slot, too many series
  const div=document.createElement("div");div.className="chart";div.style.gridColumn=`span ${free}`;
  div.innerHTML=`<div class=hd><b>Atoms on the colour anchor's clusters</b>
    <label><input type=radio name=cm value=cum checked> cumulative</label>
    <label><input type=radio name=cm value=size> per cluster</label></div>
    <div class=lg></div><div class=rd>hover the chart to light a cluster in every panel</div><div class=cv><canvas></canvas></div>`;
  document.getElementById("panels").appendChild(div);
  const c=div.querySelector("canvas");
  CH={div,c,g:c.getContext("2d"),mode:"cum"};
  div.querySelectorAll("input[name=cm]").forEach(r=>r.onchange=()=>{CH.mode=r.value;drawChart();});
  const clAt=e=>{
    const r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1,{L,W}=CH.box;
    const rank=Math.floor(((e.clientX-r.left)*dpr-L)/W*CH.K);
    return (rank>=0&&rank<CH.K)?CH.order[rank]:null;
  };
  c.addEventListener("mousemove",e=>{          // preview; a locked cluster stays put
    if(HLpin)return;const cl=clAt(e);
    if(cl===HL)return;HL=cl;drawAll();drawChart();
  });
  c.addEventListener("mouseleave",()=>{if(HLpin)return;HL=null;drawAll();drawChart();});
  c.addEventListener("click",e=>{              // lock / move the lock / release
    const cl=clAt(e);if(cl==null)return;
    if(HLpin&&cl===HL){HLpin=false;}else{HL=cl;HLpin=true;}
    drawAll();drawChart();
  });
  window.addEventListener("keydown",e=>{if(e.key==="Escape"&&HLpin){HLpin=false;HL=null;drawAll();drawChart();}});
  chartData();
}
function chartData(){
  if(!CH)return;
  let K=0;for(const s of D.seeds)for(const v of D.clu[s])if(v+1>K)K=v+1;
  const cnt={};
  for(const s of D.seeds){const a=new Array(K).fill(0);for(const v of D.clu[s])a[v]++;cnt[s]=a;}
  const order=[...Array(K).keys()].sort((p,q)=>cnt[D.anchor][q]-cnt[D.anchor][p]||p-q);
  const rankOf=new Array(K);order.forEach((cl,r)=>rankOf[cl]=r);
  CH.K=K;CH.cnt=cnt;CH.order=order;CH.rankOf=rankOf;
  const keys=[D.anchor,...D.seeds.filter(s=>s!==D.anchor)];
  CH.div.querySelector(".lg").innerHTML=keys.map(s=>`<span><i style="background:${serCol(s)};${s==D.anchor?"height:4px":""}"></i>${lbl(s)}${s==D.anchor?" (anchor)":""}</span>`).join("");
  CH.keys=keys;
}
function drawChart(){
  if(!CH)return;
  const {c,g}=CH,dpr=window.devicePixelRatio||1,r=c.getBoundingClientRect();
  c.width=Math.max(1,r.width*dpr);c.height=Math.max(1,r.height*dpr);
  const L=34*dpr,B=16*dpr,T=4*dpr,Rm=6*dpr,W=c.width-L-Rm,H=c.height-T-B;CH.box={L,W};
  g.clearRect(0,0,c.width,c.height);
  const K=CH.K,cum=CH.mode=="cum",ser={};
  let ymax=1;
  for(const s of CH.keys){
    const y=CH.order.map(cl=>CH.cnt[s][cl]);
    let v;
    if(cum){const tot=y.reduce((p,q)=>p+q,0)||1;let acc=0;v=y.map(q=>(acc+=q)/tot);}
    else{v=y.map((_,i)=>{let s2=0,n=0;for(let j=Math.max(0,i-4);j<=Math.min(K-1,i+4);j++){s2+=y[j];n++;}return s2/n;});}
    ser[s]=v;if(!cum)ymax=Math.max(ymax,...v);
  }
  const X=i=>L+(i+0.5)/K*W,Y=v=>T+H-(v/ymax)*H;
  const ink=getCss("--mut"),line=getCss("--line");
  g.strokeStyle=line;g.lineWidth=1*dpr;g.beginPath();
  for(const f of [0,0.5,1]){g.moveTo(L,Y(f*ymax));g.lineTo(L+W,Y(f*ymax));}g.stroke();
  g.fillStyle=ink;g.font=`${10*dpr}px system-ui`;g.textAlign="right";g.textBaseline="middle";
  for(const f of [0,0.5,1])g.fillText(cum?f.toFixed(1):Math.round(f*ymax),L-4*dpr,Y(f*ymax));
  g.textAlign="center";g.textBaseline="top";
  g.fillText(`cluster rank by ${lbl(D.anchor)} size (1…${K})`,L+W/2,T+H+3*dpr);
  if(!cum){                                          // the mean size, N/K
    const mean=CH.cnt[D.anchor].reduce((p,q)=>p+q,0)/K;
    g.setLineDash([4*dpr,3*dpr]);g.strokeStyle=ink;g.beginPath();g.moveTo(L,Y(mean));g.lineTo(L+W,Y(mean));g.stroke();g.setLineDash([]);
  }
  for(const s of [...CH.keys].reverse()){            // anchor drawn last, on top
    g.strokeStyle=serCol(s);g.lineWidth=(s==D.anchor?3:1.6)*dpr;g.beginPath();
    ser[s].forEach((v,i)=>{i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v));});g.stroke();
  }
  if(cur){                                           // where the selected latent's cluster ranks
    const cl=D.clu[cur.s][cur.i],x=X(CH.rankOf[cl]);
    g.strokeStyle=getCss("--hi");g.lineWidth=1.5*dpr;g.setLineDash([2*dpr,2*dpr]);
    g.beginPath();g.moveTo(x,T);g.lineTo(x,T+H);g.stroke();g.setLineDash([]);
  }
  const rd=CH.div.querySelector(".rd");
  if(HL!=null){
    const rank=CH.rankOf[HL],x=X(rank);
    g.strokeStyle=HLpin?getCss("--hi"):getCss("--fg");g.lineWidth=(HLpin?2.5:1)*dpr;
    g.beginPath();g.moveTo(x,T);g.lineTo(x,T+H);g.stroke();
    rd.innerHTML=(HLpin?`<b>locked</b> (click it again or Esc to release; UMAP hover picks only its atoms) · `:"")+
      `<b style="color:var(--hi)">cluster ${HL}</b> · rank ${rank+1} · `+
      CH.keys.map(s=>`${lbl(s)} ${CH.cnt[s][HL]}`).join(" · ");
  }else rd.textContent=cur?`selected latent's cluster: ${D.clu[cur.s][cur.i]} (dashed red) · hover the chart to light a cluster, click to lock it`
                           :"hover the chart to light a cluster in every panel · click to lock it";
}
function select(s,i,now){
  cur={s,i};drawAll();drawChart();
  const fire="";
  document.getElementById("info").innerHTML=
    `<b style="color:var(--hi)">${lbl(s)} #${i}</b> · cluster ${D.clu[s][i]}${fire} · matches: ${RULE_LONG[rule]||rule}`+
    (pinned?` · <b>pinned</b> (click it again to release)`:"");
  rowSel={};SIM=null;FPC={};
  renderLabels();                                   // labels at once; sims + images after the debounce
  clearTimeout(timer);timer=setTimeout(loadSims,now?0:220);
}
function selKey(){return cur?`${cur.s}/${cur.i}/${rule}`:"";}
function loadSims(){
  if(!cur)return;const key=selKey();
  const tau=document.getElementById("tau").value,k=document.getElementById("kk").value;
  fetch(`/sims?id=${VIEW}&key=${enc(cur.s)}&lat=${cur.i}&tau=${tau}&k=${k}`).then(r=>r.json()).then(r=>{
    if(selKey()!==key)return;
    SIM=r;
    for(const b in rowSel)if(!SIM.panels[b].idx.includes(rowSel[b]))delete rowSel[b];
    drawAll();renderLabels();loadRows();loadFp(key);
  });
}
function shown(b){                     // the latent a row shows: a picked chip, else the match
  const m=matchesOf(cur.s,cur.i);return rowSel[b]!=null?rowSel[b]:m[b];
}
function cosTo(b,lat){                 // decoder cos of `lat` in panel b to the selection
  const P=SIM&&SIM.panels[b];
  if(P){const n=P.idx.indexOf(lat);if(n>=0)return P.cos[n];if(P.nn===lat)return P.nn_cos;}
  if(b!==cur.s&&lat===matchesOf(cur.s,cur.i)[b])return D.mcos[`${rule}_${cur.s}_${b}`][cur.i];
  return null;
}
// Footprint cos (co-firing over the image pool) of the selection with every shown /
// matched / chip latent; null where a panel has no footprint (LMs, or no top-input pass).
function loadFp(key){
  if(!cur||!SIM)return;const m=matchesOf(cur.s,cur.i),want=new Set();
  for(const b of D.seeds){
    const P=SIM.panels[b];
    for(const j of [m[b],P?P.nn:null,...(P?P.idx:[])])if(j!=null&&!(b==cur.s&&j===cur.i))want.add(`${b}:${j}`);
  }
  const list=[...want];if(!list.length)return;
  fetch(`/fpcos?key=${enc(cur.s)}&lat=${cur.i}&with=${list.map(enc).join(",")}`).then(r=>r.json()).then(r=>{
    if(selKey()!==key)return;
    list.forEach((t,n)=>{if(r.vals[n]!=null)FPC[t]=r.vals[n];});
    renderLabels();
  });
}
function fpOf(b,j){const v=FPC[`${b}:${j}`];return v==null?null:v;}
function coFires(b,v){                 // above both cells' random-pair p95
  const cellOf=k=>k.split("/").slice(0,2).join("/"),a1=cellOf(cur.s),a2=cellOf(b);
  const t=Math.max((FPNULL[a1]||{}).p95||0,(FPNULL[a2]||{}).p95||0);
  return v>=t;
}
function renderLabels(){
  if(!cur)return;const m=matchesOf(cur.s,cur.i);
  for(const b of D.seeds){
    const row=rowEl[b],lat=shown(b),own=b==cur.s;row.classList.toggle("anchor",own);
    const c=own&&lat===cur.i?null:cosTo(b,lat);
    const what=own?(lat===cur.i?"selected":"sibling"):(rowSel[b]!=null&&rowSel[b]!==m[b]?"similar":(RULE_SHORT[rule]||rule));
    // .lab is a flex column: every line must be ONE element, or its pieces stack
    const fv=own&&lat===cur.i?null:fpOf(b,lat);
    let h=`<div><b>${lbl(b)}</b> #${lat}<span class=mut> · ${what}${c!=null?" "+c.toFixed(2):""}</span>`+
      (fv!=null?` <span class=fp title="footprint cos: co-firing over the image pool">fp ${fv.toFixed(2)}</span>`:"")+
      `<span class=mut> · c${D.clu[b][lat]}</span></div>`;
    const P=SIM&&SIM.panels[b];
    if(P){
      const n=P.idx.length;
      if(n){
        const gain=P.span_cos-P.nn_cos;
        h+=own?`<div class=mut>${n} sibling${n>1?"s":""} ≥ ${SIM.tau.toFixed(2)} · span ${P.span_cos.toFixed(2)}</div>`
              :`<div class="mut${n>1&&gain>=0.1?" gain":""}" title="cos of the selected atom to its best non-negative mix of these ${n} atoms, against the single best one">${n} ≥ ${SIM.tau.toFixed(2)} · span ${P.span_cos.toFixed(2)} vs NN ${P.nn_cos.toFixed(2)}</div>`;
        h+=`<div class=chips>`+P.idx.map((j,q)=>{
          const v=fpOf(b,j),cls=v==null?"":(coFires(b,v)?" co":" apart");
          return `<span class="chip${j===lat?" on":""}${cls}" data-b="${b}" data-j="${j}" title="decoder cos ${P.cos[q]} · w ${P.w[q]} in the non-negative fit${v!=null?` · footprint cos ${v} (${coFires(b,v)?"co-fires":"fires apart"})`:""}">#${j} ${P.cos[q].toFixed(2)}${v!=null?`<span class=fp>|${v.toFixed(2)}</span>`:""}</span>`;
        }).join("")+`</div>`;
      }else h+=`<div class=mut>none ≥ ${SIM.tau.toFixed(2)} (NN ${P.nn_cos.toFixed(2)})</div>`;
    }
    const lab=row.querySelector(".lab");lab.innerHTML=h;
    lab.querySelectorAll(".chip").forEach(ch=>ch.onclick=()=>{
      const bb=ch.dataset.b,j=+ch.dataset.j,def=bb==cur.s?cur.i:m[bb];
      rowSel[bb]=(shown(bb)===j)?null:j;if(rowSel[bb]===def||rowSel[bb]==null)delete rowSel[bb];
      renderLabels();drawAll();loadRow(bb);
    });
  }
}
function loadRows(){if(cur)for(const s of D.seeds)loadRow(s);}
// LMs: each cell is one top-token context (peak token highlighted)
const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
// every token of the window shaded by |a| ÷ the latent's rank-1 |a| (one scale for the
// whole row, so the bottom picks read fainter); orange positive, blue negative (iSAE's
// signed code); the peak token is boxed. Newlines show as ↵.
function tokHTML(p,ref){
  return p.toks.map((w,i)=>{
    const a=p.acts[i],f=Math.min(1,Math.abs(a)/(ref||1)),on=Math.abs(a)>1e-6;
    const bg=on?(a>0?`rgba(255,140,0,${(0.12+0.78*f).toFixed(3)})`:`rgba(42,120,214,${(0.12+0.78*f).toFixed(3)})`):"transparent";
    return `<span class="tk${p.lo+i===p.pos?" pk2":""}" style="background:${bg}" title="a = ${a.toFixed(3)}">${esc(w.replace(/\n/g,"↵"))}</span>`;
  }).join("");
}
function loadRowText(s){
  // two requests: the text at once (acts=0), then the same picks shaded
  const key=selKey(),lat=shown(s),cells=rowEl[s].querySelectorAll(".cell");
  let shaded=false;
  const draw=t=>{
    if(selKey()!==key||shown(s)!==lat)return;
    cells.forEach(c=>c.innerHTML="");
    if(!t.picks.length){cells[0].innerHTML=`<span class=mut>${t.error||"never fires in the top-input pool"}</span>`;return;}
    t.picks.forEach((p,k)=>{
      const tip=`#${p.rank} · stored |a| ${p.val.toFixed(3)} · seq ${p.seq} pos ${p.pos}`+
        (p.chk!=null?` · live/stored ${p.chk.toFixed(3)}`:"")+(t.act_error?` · no shading: ${t.act_error}`:"");
      const warn=(p.chk!=null&&Math.abs(p.chk-1)>0.02?` <span title="live peak differs from the stored value by more than 2%">⚠${p.chk.toFixed(2)}</span>`:"");
      cells[k].innerHTML=`<div class=txt title="${esc(tip)}">`+(p.acts?tokHTML(p,t.ref):
        `${esc(p.left)}<b class=pk>${esc(p.peak)}</b>${esc(p.right)}`)+`</div><div class=cap>#${p.rank} · ${p.val.toFixed(2)}${warn}</div>`;
    });
  };
  fetch(`/ctx?key=${enc(s)}&lat=${lat}&acts=0`).then(r=>r.json()).then(t=>{if(!shaded)draw(t);});
  fetch(`/ctx?key=${enc(s)}&lat=${lat}`).then(r=>r.json()).then(t=>{shaded=true;draw(t);});
}
function loadRow(s){
  if(D.modality==="text")return loadRowText(s);
  const raw=document.getElementById("rawtog").checked,key=selKey(),lat=shown(s),cells=rowEl[s].querySelectorAll(".cell");
  fetch(`/top?key=${enc(s)}&lat=${lat}`).then(r=>r.json()).then(t=>{
    if(selKey()!==key||shown(s)!==lat)return;          // a newer selection or chip won
    cells.forEach(c=>c.innerHTML="");
    if(!t.picks.length){cells[0].innerHTML=`<span class=mut>${t.error||"never fires in the top-input pool"}</span>`;return;}
    t.picks.forEach((p,k)=>{
      const src=raw?`/raw?img=${p.img}&size=160`:`/masked?img=${p.img}&key=${enc(s)}&lat=${lat}&size=160`;
      const cls=p.cls.split(",")[0];
      cells[k].innerHTML=`<img src="${src}" title="#${p.rank} · ${p.val.toFixed(3)} · ${p.cls} · val ${p.img}" onerror="this.style.outline='2px solid red';this.alt='encode failed'"><div class=cap>#${p.rank} ${cls}</div>`;
    });
  });
}
function zoom(e,s){
  const c=cvs[s],r=c.getBoundingClientRect();
  const fx=(e.clientX-r.left)/r.width,fy=1-(e.clientY-r.top)/r.height;
  const k=e.deltaY<0?0.85:1/0.85;
  const cx=view.x0+fx*(view.x1-view.x0),cy=view.y0+fy*(view.y1-view.y0);
  view.x0=cx-(cx-view.x0)*k;view.x1=cx+(view.x1-cx)*k;view.y0=cy-(cy-view.y0)*k;view.y1=cy+(view.y1-cy)*k;drawAll();
}
let pan=null;
function startPan(e,s){
  const c=cvs[s],r=c.getBoundingClientRect();pan={x:e.clientX,y:e.clientY,v:{...view},w:r.width,h:r.height,moved:false};
  const mv=ev=>{const dx=(ev.clientX-pan.x)/pan.w*(pan.v.x1-pan.v.x0),dy=(ev.clientY-pan.y)/pan.h*(pan.v.y1-pan.v.y0);
    if(Math.abs(ev.clientX-pan.x)+Math.abs(ev.clientY-pan.y)>3)pan.moved=true;
    view.x0=pan.v.x0-dx;view.x1=pan.v.x1-dx;view.y0=pan.v.y0+dy;view.y1=pan.v.y1+dy;drawAll();};
  const up=()=>{window.removeEventListener("mousemove",mv);window.removeEventListener("mouseup",up);setTimeout(()=>pan=null,0);};
  window.addEventListener("mousemove",mv);window.addEventListener("mouseup",up);
}
</script></body></html>"""
