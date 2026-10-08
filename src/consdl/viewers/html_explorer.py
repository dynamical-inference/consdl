"""The latent explorer page: one card per SAE panel = the input + that panel's UMAP.

Vanilla JS on canvases, no libraries. The UMAP colours are the view's own
anchor-cluster palette (``PAL``, the same 30 slots as the cluster-consistency
view), so a colour on the input and a colour on the UMAP are the same concept.
"""
from __future__ import annotations

FOOT = (
    "<b>What is encoded.</b> Every panel encodes the SAME crop (what the backbone saw) at "
    "<b>batch 1</b>, CLS dropped. A latent is <i>active</i> on a patch when "
    "|a| ≥ eps · max<sub>latents</sub>|a| on that patch — relative, because the reconstruction "
    "normaliser makes absolute floors scale-dependent across architectures. "
    "<b>Colour: cluster.</b> Each patch takes the anchor-cluster colour of its strongest latent "
    "(grey = a cluster outside the 30 coloured ones); opacity = that patch's peak |a| relative to "
    "the image's peak. The same colour on the UMAP is the same anchor cluster. "
    "<b>Colour: region.</b> Each panel's patches clustered ALONE (mass distance, average "
    "linkage, one tree cut at k); ids are relabelled onto the first panel for display only, and "
    "colour encodes where a region sits, so agreeing carves look alike. ARI between panels is "
    "computed on the unmatched labels. Any region claim must state its k. "
    "<b>Colour: heatmap.</b> |a| of the selected latent (or, with nothing selected, of the "
    "panel's strongest latent on this image), scaled to its own peak on this image. "
    "<b>Linking.</b> Hover a patch: its active latents are ringed in every panel (ring size = "
    "|a| / that patch's peak). Hover a latent: its matched partner (the view's Hungarian / NN "
    "decoder match) is ringed in every other panel and shown on that panel's input. A region's "
    "latents are ranked by mass-in × share-in over the support (the region ledger's rule). "
    "<b>Top images</b> come from the precompute's pass over ImageNet val, ranked by the image's "
    "peak |a| over its patches, and are masked live. "
    "<b>Text (LMs).</b> An input is a sequence of token ids resolved once and encoded by every "
    "panel: typed text (tokenised with the backbone's tokenizer), or a window of the backbone's "
    "FineWeb stream. A window is forwarded as its FULL 1024-token row and then sliced, because the "
    "LM is causal and the prefix is part of the input. Top contexts are ranked by |a| per token. "
    "Token regions: <i>free</i> = tokens "
    "decomposed alike wherever they are; <i>spans</i> = only neighbouring tokens may merge. "
    "Position 0 of a row (the attention sink) is left out of the carve. Heatmap and top contexts "
    "use the same signed shading (orange +, blue −). "
    "<b>Caveats.</b> iSAE encodes in a different space and its atom sign is unidentified "
    "(|a|, |cos|). MP runs a fixed number of rounds, the cell's target L0, so its codes are per "
    "token. FISTA codes are not bit-reproducible across GPUs.")


_CSS = r"""
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d9d9de;--card:#f6f6f8;--ring:#1d1d1f;--hi:#e0003c;--hibg:rgba(224,0,60,.07)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--line:#35353b;--card:#212125;--ring:#ececf0;--hi:#ff4f7b;--hibg:rgba(255,79,123,.10)}}
:root[data-theme="dark"]{--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--line:#35353b;--card:#212125;--ring:#ececf0;--hi:#ff4f7b;--hibg:rgba(255,79,123,.10)}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:12px/1.35 system-ui,sans-serif;display:flex;flex-direction:column;overflow:hidden}
header{padding:6px 12px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap;flex:none}
header h1{font-size:14px;margin:0}
header select,header input,header button{font:inherit;font-size:12px}
.grp{display:inline-flex;gap:4px;align-items:center;border-left:1px solid var(--line);padding-left:10px}
.mut{color:var(--mut)}
.err{color:var(--hi);font-weight:600}
details.help{position:relative}
details.help summary{cursor:pointer;color:var(--mut);list-style:none;border:1px solid var(--line);border-radius:10px;padding:0 7px}
details.help div{position:absolute;right:0;top:22px;width:min(600px,86vw);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;z-index:9;box-shadow:0 4px 18px rgba(0,0,0,.18)}
#pick{position:absolute;z-index:8;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;display:none;max-width:min(560px,90vw);max-height:60vh;overflow:auto;box-shadow:0 4px 18px rgba(0,0,0,.18)}
#pick img{width:64px;height:64px;margin:1px;cursor:pointer;border-radius:3px;object-fit:cover}
#pick img:hover{outline:2px solid var(--hi)}
#keys label{margin-right:6px;white-space:nowrap}
main{flex:1;min-height:0;display:flex;gap:10px;padding:8px 12px}
#cards{flex:1.6;min-width:0;overflow:auto;display:grid;gap:6px;align-content:start}
.card{background:var(--card);border-radius:6px;padding:4px;min-width:0;border:2px solid transparent}
.card.sel{border-color:var(--hi)}
.card h2{margin:0 0 3px 2px;font-size:11px;font-weight:600;display:flex;gap:8px;align-items:baseline;white-space:nowrap;overflow:hidden}
.card h2 .mut{font-weight:normal}
.card .cav{font-size:10px;color:var(--mut);margin:0 2px 2px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.pair .box{position:relative;aspect-ratio:1;min-width:0}
.pair canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair;border-radius:4px}
#side{flex:1;min-width:280px;display:flex;flex-direction:column;gap:6px;overflow:auto}
#info{min-height:18px}
.sec{background:var(--card);border-radius:6px;padding:5px 7px}
.sec h3{margin:0 0 4px;font-size:11px}
.lrow{display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin:2px 0}
.lrow b{min-width:80px;font-size:11px}
.chip{border:1px solid var(--line);border-radius:3px;padding:0 4px;cursor:pointer;font-size:10.5px;line-height:15px;white-space:nowrap;display:inline-flex;gap:3px;align-items:center}
.chip:hover{border-color:var(--mut)}
.chip.on{border-color:var(--fg);font-weight:700}
.chip i{display:inline-block;width:9px;height:9px;border-radius:2px}
.trow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:3px;margin:2px 0 6px}
.cell{position:relative;aspect-ratio:1;min-width:0}
.cell img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:3px}
.cell .cap{position:absolute;left:0;right:0;bottom:0;font-size:9.5px;line-height:1.2;padding:1px 2px;background:rgba(0,0,0,.55);color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center}
.tstrip{font:11.5px/1.55 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word;max-height:190px;overflow:auto;padding:3px 4px;background:var(--bg);border-radius:4px;margin-bottom:4px}
.tstrip span{border-radius:2px;cursor:crosshair}
.tstrip span.hov{outline:1.5px solid var(--fg);outline-offset:-1px}
.tstrip span.hovself{outline:2px solid var(--hi);outline-offset:-1px}
.tstrip span.ex{color:var(--mut);font-style:italic}
.tstrip span.veil{opacity:.3}
.ubox{position:relative;aspect-ratio:1.6;min-width:0}
.ubox canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair;border-radius:4px}
.trow.tx{grid-template-columns:repeat(3,minmax(0,1fr))}
.tcell{font:9.5px/1.3 ui-monospace,Menlo,monospace;background:var(--bg);border-radius:4px;padding:2px 3px;height:74px;overflow:hidden;white-space:pre-wrap;word-break:break-word;cursor:pointer;position:relative}
.tcell:hover{outline:1.5px solid var(--hi)}
.tcell .pk2{outline:1.5px solid currentColor;outline-offset:-1px;font-weight:700}
.tcell .tcap{position:absolute;right:2px;bottom:1px;font-size:9px;background:var(--card);padding:0 2px;border-radius:2px;color:var(--mut)}
#pick textarea{width:min(520px,80vw);height:140px;font:12px ui-monospace,Menlo,monospace}
.lg{display:flex;gap:6px;flex-wrap:wrap;font-size:10px;color:var(--mut);align-items:center}
.ramp{display:inline-block;width:90px;height:8px;border-radius:2px;vertical-align:middle}
@media (max-width:900px){body{overflow:auto}main{flex-direction:column}#cards{overflow:visible}}
"""

_BODY = r"""
<header><h1 id="title">latent explorer</h1>
<span id="nav"></span>
<span class="grp" id="keys"></span>
<span class="grp" id="txtin" style="display:none">input:
  <button id="btxt">type text ▾</button>
  <button id="brow" title="a random window of the backbone's FineWeb stream">random FineWeb</button>
  row <input type="number" id="iseq" min="0" style="width:5.5em"> from <input type="number" id="ilo" min="0" value="0" style="width:4em">
  len <input type="number" id="ilen" min="2" max="1024" value="96" style="width:4em"> <button id="bseq">go</button></span>
<span class="grp" id="imgin">input:
  <button id="bcls">ImageNet class ▾</button>
  # <input type="number" id="iidx" min="0" style="width:5.5em"> <button id="bgo">go</button>
  <button id="brnd" title="a random val image">random</button>
  <label title="or drop an image anywhere on the page"><input type="file" id="bup" accept="image/*" style="width:12em"></label></span>
<span class="grp">colour:
  <label><input type="radio" name="mode" value="cluster"> cluster</label>
  <label><input type="radio" name="mode" value="region"> region</label>
  <label><input type="radio" name="mode" value="heat"> heatmap</label>
  <span id="kbox">k <input type="range" id="kk" min="2" max="20" value="5" style="width:80px;vertical-align:middle"> <b id="kv">5</b>
  <label id="spbox" style="display:none" title="only neighbouring tokens may merge: every region is a contiguous span"><input type="checkbox" id="spans"> spans</label></span></span>
<span class="grp">eps <input type="number" id="eps" min="0" max="0.5" step="0.005" style="width:4.5em"></span>
<span class="grp" id="rules"></span>
<span class="grp" id="rawgrp"><label><input type="checkbox" id="rawtog"> unmasked tops</label></span>
<details class="help"><summary>?</summary><div>%%FOOT%%</div></details>
</header>
<div id="pick"></div>
<main>
  <section id="cards"></section>
  <section id="side">
    <div id="info" class="mut">loading…</div>
    <div class="sec" id="legend"></div>
    <div class="sec"><h3 id="phd">hover a patch</h3><div id="plist" class="mut">its active latents, per panel, strongest first</div></div>
    <div class="sec"><h3 id="thd">hover or click a latent</h3><div id="tops"></div></div>
  </section>
</main>
"""

_JS = r"""
const PAL=["#AA0DFE","#3283FE","#85660D","#782AB6","#565656","#1C8356","#16FF32","#F7E1A0","#E2E2E2","#1CBE4F","#C4451C","#DEA0FD","#FE00FA","#325A9B","#FEAF16","#F8A19F","#90AD1C","#F6222E","#1CFFCE","#2ED9FF","#B10DA1","#C075A6","#FC1CBF","#B00068","#FBE426","#FA0087","#2E91E5","#E15F99","#1CA71C","#FB0D0D"];
// inferno, 9 stops (matplotlib) -- the heatmap ramp
const INF=[[0,0,4],[31,12,72],[85,15,109],[136,34,106],[186,54,85],[227,89,51],[249,140,10],[249,201,50],[252,255,164]];
const RULE_LONG={hung:"Hungarian",nn:"nearest neighbour",fp:"footprint Hungarian"};
const VIEW="%%VIEW%%", BB="%%BACKBONE%%", enc=encodeURIComponent;
const Q=new URLSearchParams(location.search);
let D=null, KEYS=[], C={}, R=null, IMG=Q.get("img")||"0", IMGNAME="";
let MODE=Q.get("mode")||"cluster", K=+(Q.get("k")||5), EPS=+(Q.get("eps")||0.01), rule=null;
let hp=null, pp=null;     // hovered / pinned patch {s,p}
let hl=null, pl=null;     // hovered / pinned latent {s,i}
let view=null, pan=null, NIMG=50000, topTimer=null, gen=0;
const E={};               // per panel key: dom + caches
let TXT=null, SPANS=Q.get("spans")==="1";   // text: the input's tokens/meta; region carve = contiguous spans
const $=id=>document.getElementById(id);
const TEXT=()=>!!D&&D.modality==="text";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function rgba(hex,a){const n=parseInt(hex.slice(1),16);return `rgba(${n>>16&255},${n>>8&255},${n&255},${a.toFixed(3)})`;}
function signedBg(a,f){return a>0?`rgba(255,140,0,${(0.12+0.78*f).toFixed(3)})`:`rgba(42,120,214,${(0.12+0.78*f).toFixed(3)})`;}
const lbl=s=>(D.labels&&D.labels[s])||s;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const DPR=()=>window.devicePixelRatio||1;

// ── decoding ──────────────────────────────────────────────────────────
function b64(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u.buffer;}
function f16(buf){const h=new Uint16Array(buf),o=new Float32Array(h.length);
  for(let i=0;i<h.length;i++){const x=h[i],s=x&0x8000?-1:1,e=(x>>10)&31,m=x&1023;
    o[i]=e===0?s*m*5.960464477539063e-8:e===31?(m?NaN:s*Infinity):s*Math.pow(2,e-15)*(1+m/1024);}
  return o;}
function decode(d){
  const c={...d,ptr:new Int32Array(b64(d.ptr)),lat:new Int32Array(b64(d.lat)),val:f16(b64(d.val))};
  c.mass=new Map(); c.cols=new Map(); c.top=new Int32Array(c.P).fill(-1);
  let gmax=0;for(const v of c.rowmax)gmax=Math.max(gmax,v);c.gmax=gmax||1;
  for(let p=0;p<c.P;p++){let bi=-1,bv=0;
    for(let n=c.ptr[p];n<c.ptr[p+1];n++){const a=Math.abs(c.val[n]),l=c.lat[n];
      c.mass.set(l,(c.mass.get(l)||0)+a);if(a>bv){bv=a;bi=l;}}
    c.top[p]=bi;}
  let mm=0;for(const v of c.mass.values())mm=Math.max(mm,v);c.mmax=mm||1;
  let best=-1,bm=-1;for(const [l,v] of c.mass)if(v>bm){bm=v;best=l;}c.best=best;
  return c;}
function colOf(c,l){                 // (P,) |a| of latent l, lazily
  if(c.cols.has(l))return c.cols.get(l);
  const o=new Float32Array(c.P);
  for(let p=0;p<c.P;p++)for(let n=c.ptr[p];n<c.ptr[p+1];n++)if(c.lat[n]===l){o[p]=Math.abs(c.val[n]);break;}
  c.cols.set(l,o);return o;}
function colSigned(c,l){              // (P,) signed a of latent l (text shading keeps the sign)
  const o=new Float32Array(c.P);
  for(let p=0;p<c.P;p++)for(let n=c.ptr[p];n<c.ptr[p+1];n++)if(c.lat[n]===l){o[p]=c.val[n];break;}
  return o;}
function rowTop(c,p,n){const r=[];for(let q=c.ptr[p];q<c.ptr[p+1];q++)r.push([c.lat[q],c.val[q]]);
  r.sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));return n?r.slice(0,n):r;}
function regionTop(c,members,n){      // mass_in * share_in over the support
  const inside=new Map(),mem=new Set(members);
  for(const p of mem)for(let q=c.ptr[p];q<c.ptr[p+1];q++){const l=c.lat[q];inside.set(l,(inside.get(l)||0)+Math.abs(c.val[q]));}
  const r=[];for(const [l,m] of inside){const t=c.mass.get(l)||m;r.push([l,m*(m/t),m/t]);}
  r.sort((a,b)=>b[1]-a[1]);return r.slice(0,n);}

// ── boot ──────────────────────────────────────────────────────────────
fetch(`/view/data.json?id=${VIEW}`).then(r=>r.json()).then(d=>{
  if(d.error){$("info").innerHTML=`<span class=err>${d.error}</span>`;return;}
  D=d;rule=D.rules[0];
  const want=(Q.get("keys")||"").split(",").filter(k=>D.seeds.includes(k));
  KEYS=want.length?want:D.seeds.slice();
  fetch("/nav.json").then(r=>r.json()).then(buildNav);
  if(TEXT()){if(!Q.get("img"))IMG="t_s0_0_96";}
  else fetch("/api/n_images").then(r=>r.json()).then(r=>{if(r.n)NIMG=r.n;$("iidx").max=NIMG-1;});
  setupControls();buildCards();loadInput();
});

function buildNav(N){
  const kind=D.kind, arch=D.arch||(N.arches.includes("consdl")?"consdl":N.arches[0]), cell=D.cell||N.cells[N.cells.length-1];
  const sel=(id,opts,val)=>`<select id=${id}>`+opts.map(([v,t])=>`<option value="${v}"${v==val?" selected":""}>${t}</option>`).join("")+`</select>`;
  $("nav").innerHTML=
    sel("nk",[["seed","seeds · one cell"],["cross","cells · one arch"],["arch","archs · one cell"]],kind)+" "+
    sel("na",N.arches.map(a=>[a,N.labels[a]||a]),arch)+" "+
    sel("nc",N.cells.map(c=>[c,N.cell_labels[c]||c]),cell);
  const nk=$("nk"),na=$("na"),nc=$("nc");
  const sync=()=>{na.disabled=nk.value=="arch";nc.disabled=nk.value=="cross";};sync();
  nk.onchange=na.onchange=nc.onchange=()=>{sync();
    const k=nk.value,id=k=="seed"?`seed__${na.value}__${nc.value}`:k=="cross"?`cross__${na.value}`:`arch__${nc.value}`;
    if(id===VIEW)return;
    if(!N.views.includes(id)){$("info").innerHTML=`<span class=err>${id} is not built</span>`;return;}
    location.href=`/explore?id=${id}&img=${enc(IMG)}&mode=${MODE}&k=${K}&eps=${EPS}`;};
}
function setUrl(){
  const u=new URLSearchParams({id:VIEW,img:IMG,mode:MODE,k:K,eps:EPS});
  if(TEXT()&&SPANS)u.set("spans","1");
  if(KEYS.length!==D.seeds.length)u.set("keys",KEYS.join(","));
  history.replaceState(null,"","/explore?"+u.toString());}

function setupControls(){
  document.title=`explore · ${D.title}`;
  $("title").innerHTML=`${D.title} <span class=mut style="font-size:11px;font-weight:normal">${BB}${D.gauge&&D.gauge!=="signed"?" · "+D.gauge:""}</span>`;
  $("keys").innerHTML="panels: "+D.seeds.map(s=>`<label><input type=checkbox value="${s}"${KEYS.includes(s)?" checked":""}> ${lbl(s)}</label>`).join("")
    +` <button id="kone" title="only the first ticked panel">single</button> <button id="kall">all</button>`;
  $("keys").querySelectorAll("input").forEach(b=>b.onchange=()=>{
    const on=[...$("keys").querySelectorAll("input:checked")].map(x=>x.value);
    if(!on.length){b.checked=true;return;}
    KEYS=D.seeds.filter(s=>on.includes(s));panelsChanged();});
  $("kone").onclick=()=>{KEYS=[KEYS[0]];$("keys").querySelectorAll("input").forEach(x=>x.checked=x.value===KEYS[0]);panelsChanged();};
  $("kall").onclick=()=>{KEYS=D.seeds.slice();$("keys").querySelectorAll("input").forEach(x=>x.checked=true);panelsChanged();};
  $("rules").innerHTML=D.rules.length>1
    ?"match: "+D.rules.map((r,q)=>`<label><input type=radio name=rule value="${r}"${q?"":" checked"}> ${RULE_LONG[r]||r}</label>`).join(" ")
    :`match: ${RULE_LONG[D.rules[0]]||D.rules[0]}`;
  document.querySelectorAll("input[name=rule]").forEach(r=>r.onchange=()=>{rule=r.value;redrawAll();loadTops(true);});
  document.querySelectorAll("input[name=mode]").forEach(r=>{r.checked=r.value===MODE;r.onchange=()=>{MODE=r.value;modeChanged();};});
  $("kk").value=K;$("kv").textContent=K;
  $("kk").oninput=()=>{K=+$("kk").value;$("kv").textContent=K;setUrl();drawInputs();legend();};
  $("eps").value=EPS;
  $("eps").onchange=()=>{const v=+$("eps").value;if(!(v>=0&&v<0.5)){$("eps").value=EPS;return;}EPS=v;setUrl();loadCodes();};
  $("rawtog").onchange=()=>loadTops(true);
  $("bgo").onclick=()=>{const v=$("iidx").value;if(v!=="")setImage(String(+v),"");};
  $("iidx").addEventListener("keydown",e=>{if(e.key==="Enter")$("bgo").onclick();});
  $("brnd").onclick=()=>setImage(String(Math.floor(Math.random()*NIMG)),"");
  $("bup").onchange=e=>{if(e.target.files[0])upload(e.target.files[0]);};
  document.body.addEventListener("dragover",e=>e.preventDefault());
  document.body.addEventListener("drop",e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f)upload(f);});
  $("bcls").onclick=openPicker;
  document.addEventListener("click",e=>{if(!$("pick").contains(e.target)&&e.target!==$("bcls")&&e.target!==$("btxt"))$("pick").style.display="none";});
  window.addEventListener("resize",()=>{layout();});
  window.addEventListener("keydown",e=>{if(e.key==="Escape"){pl=null;pp=null;hl=null;hp=null;redrawAll();sidePatch();loadTops(true);}});
  $("kbox").style.display=MODE==="region"?"":"none";
  if(TEXT()){
    $("imgin").style.display="none";$("rawgrp").style.display="none";
    $("txtin").style.display="";$("spbox").style.display="";
    $("spans").checked=SPANS;
    $("spans").onchange=()=>{SPANS=$("spans").checked;R=null;setUrl();if(MODE==="region")loadRegions();};
    $("btxt").onclick=openTextBox;
    $("brow").onclick=()=>textFrom(`/api/random_ctx?key=${enc(KEYS[0])}&len=${+$("ilen").value||96}`);
    $("bseq").onclick=()=>{const s=$("iseq").value;if(s==="")return;const lo=+$("ilo").value||0,n=+$("ilen").value||96;
      textFrom(`/api/stats_ctx?key=${enc(KEYS[0])}&seq=${+s}&lo=${lo}&hi=${lo+n}`);};
  }
}
function textFrom(url){
  $("info").textContent="fetching the window…";
  fetch(url).then(r=>r.json()).then(r=>{if(r.error){$("info").innerHTML=`<span class=err>${esc(r.error)}</span>`;return;}setImage(r.token,"");});
}
function openTextBox(){
  const P=$("pick"),b=$("btxt").getBoundingClientRect();
  P.style.left=b.left+"px";P.style.top=(b.bottom+4)+"px";P.style.display="block";
  P.innerHTML=`<textarea id=ttx placeholder="type or paste text (≤ 1024 tokens), then encode"></textarea><br>`+
    `<button id=tgo>encode</button> <span class=mut>tokenised with ${esc(BB)}'s tokenizer; every panel encodes the same ids</span>`;
  $("ttx").focus();
  $("tgo").onclick=()=>{const t=$("ttx").value;if(!t.trim())return;P.style.display="none";
    $("info").textContent="tokenising…";
    fetch("/api/text",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t,key:KEYS[0]})})
      .then(r=>r.json()).then(r=>{if(r.error){$("info").innerHTML=`<span class=err>${esc(r.error)}</span>`;return;}setImage(r.token,"");});};
}
function modeChanged(){
  $("kbox").style.display=MODE==="region"?"":"none";setUrl();
  if(MODE==="region"&&!R)loadRegions();else{drawInputs();legend();sidePatch();}
}
function panelsChanged(){setUrl();R=null;buildCards();loadCodes();}

// ── picker / upload ───────────────────────────────────────────────────
let CLASSES=null;
function openPicker(){
  const P=$("pick"),b=$("bcls").getBoundingClientRect();
  P.style.left=b.left+"px";P.style.top=(b.bottom+4)+"px";P.style.display="block";
  const show=()=>{P.innerHTML=`<input id=pq placeholder="filter classes" style="width:95%;margin-bottom:4px"><div id=pl2></div><div id=pim></div>`;
    const list=()=>{const q=$("pq").value.toLowerCase();
      $("pl2").innerHTML=CLASSES.filter(c=>!q||c.name.toLowerCase().includes(q)).slice(0,60)
        .map(c=>`<span class=chip data-c="${c.cls}">${c.name}</span>`).join(" ");
      $("pl2").querySelectorAll(".chip").forEach(ch=>ch.onclick=ev=>{ev.stopPropagation();
        fetch(`/api/images?cls=${ch.dataset.c}`).then(r=>r.json()).then(r=>{
          $("pim").innerHTML=r.images.map(i=>`<img src="/api/crop?img=${i}&s=64" data-i="${i}" title="#${i}">`).join("");
          $("pim").querySelectorAll("img").forEach(im=>im.onclick=()=>{P.style.display="none";setImage(im.dataset.i,ch.textContent);});});});};
    $("pq").oninput=list;list();$("pq").focus();};
  if(CLASSES)show();else fetch("/api/classes").then(r=>r.json()).then(r=>{CLASSES=r.classes;show();});
}
function upload(f){
  if(f.size>12*1024*1024){$("info").innerHTML=`<span class=err>${f.name} is over 12 MB</span>`;return;}
  $("info").textContent=`uploading ${f.name}…`;
  const rd=new FileReader();
  rd.onload=()=>fetch("/api/upload",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:rd.result,name:f.name})}).then(r=>r.json()).then(r=>{
    if(r.error){$("info").innerHTML=`<span class=err>${r.error}</span>`;return;}
    setImage(r.token,`${r.name} (${r.size}, uploaded)`);});
  rd.readAsDataURL(f);
}
function setImage(img,name){IMG=img;IMGNAME=name;pp=null;hp=null;R=null;setUrl();loadInput();}

// ── cards ─────────────────────────────────────────────────────────────
function buildCards(){
  const W=$("cards");W.innerHTML="";
  for(const k in E)delete E[k];
  W.style.gridTemplateColumns=KEYS.length===1?"minmax(0,900px)":KEYS.length<=3?"1fr":"repeat(2,minmax(0,1fr))";
  let xs=[],ys=[];
  for(const s of KEYS){const a=D.xy[s];for(let i=0;i<a.length;i+=2){xs.push(a[i]);ys.push(a[i+1]);}}
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(const v of xs){if(v<x0)x0=v;if(v>x1)x1=v;}for(const v of ys){if(v<y0)y0=v;if(v>y1)y1=v;}
  const px=(x1-x0)*0.02,py=(y1-y0)*0.02;
  view={x0:x0-px,x1:x1+px,y0:y0-py,y1:y1+py};view.home={...view};
  for(const s of KEYS){
    const card=document.createElement("div");card.className="card";
    card.innerHTML=`<h2><span>${lbl(s)}</span><span class=mut data-r=stat></span></h2><div class=cav data-r=cav></div>`+
      (TEXT()?`<div class=tstrip data-r=tin></div><div class=ubox><canvas data-r=cum></canvas></div>`
             :`<div class=pair><div class=box><canvas data-r=cin></canvas></div><div class=box><canvas data-r=cum></canvas></div></div>`);
    W.appendChild(card);
    const e={card,cin:card.querySelector("[data-r=cin]"),tin:card.querySelector("[data-r=tin]"),cum:card.querySelector("[data-r=cum]"),
             stat:card.querySelector("[data-r=stat]"),cav:card.querySelector("[data-r=cav]"),base:null,grid:null};
    E[s]=e;
    if(e.cin){
      e.cin.addEventListener("mousemove",ev=>{const p=patchAt(ev,s);if(p==null)return;
        if(!hp||hp.p!==p||hp.s!==s){hp={s,p};onPatch();}});
      e.cin.addEventListener("mouseleave",()=>{hp=null;onPatch();});
      e.cin.addEventListener("click",ev=>{const p=patchAt(ev,s);pp=(pp&&pp.p===p&&pp.s===s)?null:{s,p};onPatch();});
    }else{                              // text: the token spans carry data-p
      e.tin.addEventListener("mouseover",ev=>{const t=ev.target.dataset&&ev.target.dataset.p;if(t==null)return;
        const p=+t;if(!hp||hp.p!==p||hp.s!==s){hp={s,p};onPatch();}});
      e.tin.addEventListener("mouseleave",()=>{hp=null;onPatch();});
      e.tin.addEventListener("click",ev=>{const t=ev.target.dataset&&ev.target.dataset.p;if(t==null)return;
        const p=+t;pp=(pp&&pp.p===p&&pp.s===s)?null:{s,p};onPatch();});
    }
    e.cum.addEventListener("mousemove",ev=>{if(pan)return;const i=hit(ev,s);
      if(i==null){if(hl){hl=null;onLatent();}return;}
      if(!hl||hl.s!==s||hl.i!==i){hl={s,i};onLatent();}});
    e.cum.addEventListener("mouseleave",()=>{if(hl){hl=null;onLatent();}});
    e.cum.addEventListener("click",ev=>{if(pan&&pan.moved)return;const i=hit(ev,s);
      pl=(i==null||(pl&&pl.s===s&&pl.i===i))?null:{s,i};onLatent(true);});
    e.cum.addEventListener("wheel",ev=>{ev.preventDefault();zoom(ev,s);},{passive:false});
    e.cum.addEventListener("mousedown",ev=>startPan(ev,s));
    e.cum.addEventListener("dblclick",()=>{Object.assign(view,view.home);invalidate();redrawAll();});
  }
  layout();
}
function layout(){for(const s of KEYS){if(E[s].cin)size(E[s].cin);size(E[s].cum);E[s].base=null;}redrawAll();}
function size(c){const r=c.getBoundingClientRect(),d=DPR();c.width=Math.max(1,Math.round(r.width*d));c.height=Math.max(1,Math.round(r.height*d));}
function invalidate(){for(const s of KEYS)if(E[s])E[s].base=null;}

// ── data loads ────────────────────────────────────────────────────────
let CROP=null;
function loadInput(){
  const g=++gen;
  $("info").textContent=`encoding ${IMGNAME||"#"+IMG}…`;
  if(TEXT()){loadCodes();return;}
  CROP=new Image();CROP.onload=()=>{if(g===gen)drawInputs();};
  CROP.onerror=()=>{$("info").innerHTML=`<span class=err>image ${IMG} could not be loaded${IMG.startsWith("u_")?" — the upload may have been evicted; upload again":""}</span>`;};
  CROP.src=`/api/crop?img=${enc(IMG)}&s=448`;
  loadCodes();
}
function loadCodes(){
  const g=gen,t0=performance.now();
  fetch(`/api/codes?id=${VIEW}&keys=${enc(KEYS.join(","))}&img=${enc(IMG)}&eps=${EPS}`).then(r=>r.json()).then(r=>{
    if(g!==gen)return;
    if(r.error){$("info").innerHTML=`<span class=err>${r.error}</span>`;return;}
    C={};for(const s of KEYS)C[s]=decode(r.panels[s]);
    if(TEXT()){
      TXT={tokens:r.tokens,input:r.input,ex:r.exclude_first};
      const strip=r.tokens.map((w,p)=>`<span data-p="${p}"${TXT.ex&&p===0?' class=ex title="position 0 (attention sink): left out of the region carve"':""}>${esc(w.replace(/\n/g,"↵\n"))}</span>`).join("");
      for(const s of KEYS)E[s].tin.innerHTML=strip;
    }
    const unit=TEXT()?"token":"patch";
    for(const s of KEYS){const c=C[s];
      E[s].stat.textContent=`d=${c.d_sae} · `+(TEXT()?`${c.P} tokens`:`${c.side}×${c.side}`)+` · ${c.mass.size} latents active · mean ${(c.nnz/c.P).toFixed(1)}/${unit}`;
      E[s].cav.textContent=c.caveat||"";}
    const what=TEXT()?(r.input.kind==="stats"?esc(r.input.name):`typed text · ${r.tokens.length} tokens`)
                     :(IMGNAME||(IMG.startsWith("u_")?"upload "+IMG:"val #"+IMG));
    $("info").innerHTML=`<b>${what}</b> · ${TEXT()?"":"batch 1 · "}support |a| ≥ ${EPS}·max per ${unit} · <span class=mut>${Math.round(performance.now()-t0)} ms</span>`;
    invalidate();
    if(MODE==="region")loadRegions();else{redrawAll();legend();}
    sidePatch();loadTops(true);
  }).catch(e=>{$("info").innerHTML=`<span class=err>${e}</span>`;});
}
function loadRegions(){
  const g=gen;$("info").insertAdjacentHTML("beforeend"," · <span id=rgn class=mut>clustering regions…</span>");
  fetch(`/api/regions?id=${VIEW}&keys=${enc(KEYS.join(","))}&img=${enc(IMG)}&contiguous=${TEXT()&&SPANS?1:0}`).then(r=>r.json()).then(r=>{
    if(g!==gen)return;const el=$("rgn");if(el)el.remove();
    if(r.error){$("info").insertAdjacentHTML("beforeend",` <span class=err>${r.error}</span>`);return;}
    R=r;drawInputs();legend();sidePatch();});
}

// ── input canvas ──────────────────────────────────────────────────────
function patchAt(ev,s){const c=C[s];if(!c)return null;const r=E[s].cin.getBoundingClientRect();
  const x=Math.floor((ev.clientX-r.left)/r.width*c.side),y=Math.floor((ev.clientY-r.top)/r.height*c.side);
  return (x<0||y<0||x>=c.side||y>=c.side)?null:y*c.side+x;}
function inferno(t){t=Math.max(0,Math.min(1,t))*(INF.length-1);const i=Math.min(INF.length-2,Math.floor(t)),f=t-i,a=INF[i],b=INF[i+1];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;}
function selLat(){return pl||hl;}
function latFor(s){                  // the latent shown on panel s: the selection or its match
  const L=selLat();if(!L)return null;if(L.s===s)return L.i;
  const m=D.match[`${rule}_${L.s}_${s}`];return m?m[L.i]:null;}
function activePatch(){return pp||hp;}
function regionMembers(s,p){if(!R||!R.panels[s])return null;const lab=R.panels[s].labels[K];const r=lab[p];
  const out=[];for(let q=0;q<lab.length;q++)if(lab[q]===r)out.push(q);return out;}
function drawInputs(){for(const s of KEYS)drawInput(s);}
function drawText(s){
  // the token strip: same three modes as the patch canvas, as span backgrounds
  const e=E[s],c=C[s];e.card.classList.toggle("sel",!!selLat()&&selLat().s===s);
  if(!c||!e.tin)return;
  const sp=e.tin.children;if(sp.length!==c.P)return;
  const lat=latFor(s),heat=lat!=null||MODE==="heat",A=activePatch();
  let bg=new Array(c.P).fill("transparent"),tip=new Array(c.P).fill("");
  if(heat){
    const l=lat!=null?lat:c.best,v=colSigned(c,l);let mx=0;for(const x of v)mx=Math.max(mx,Math.abs(x));
    for(let p=0;p<c.P;p++){if(v[p]!==0)bg[p]=signedBg(v[p],mx>0?Math.abs(v[p])/mx:0);tip[p]=`#${l}: a = ${v[p].toFixed(3)}`;}
  }else if(MODE==="region"&&R&&R.panels[s]){
    const lab=R.panels[s].labels[K],cols=R.panels[s].colors[K];
    for(let p=0;p<c.P;p++){if(lab[p]>=0)bg[p]=rgba(cols[lab[p]]||"#888888",0.55);tip[p]=lab[p]>=0?`region ${lab[p]}`:"unassigned";}
  }else{
    for(let p=0;p<c.P;p++){const l=c.top[p];if(l<0)continue;const k=D.col[s][l];
      bg[p]=rgba(k<0?"#9a9a9a":PAL[k],0.15+0.7*c.rowmax[p]/c.gmax);tip[p]=`strongest #${l} (${c.rowmax[p].toFixed(3)})`;}
  }
  const veil=A!=null&&MODE==="region"&&R&&!heat?new Set(regionMembers(s,A.p)||[]):null;
  for(let p=0;p<c.P;p++){const el=sp[p];el.style.background=bg[p];el.title=tip[p];
    el.classList.toggle("hov",!!A&&A.p===p&&A.s!==s);el.classList.toggle("hovself",!!A&&A.p===p&&A.s===s);
    el.classList.toggle("veil",!!veil&&!veil.has(p));}
}
function drawInput(s){
  if(TEXT())return drawText(s);
  const e=E[s],c=C[s],cv=e.cin,g=cv.getContext("2d"),W=cv.width,H=cv.height;
  g.clearRect(0,0,W,H);
  if(CROP&&CROP.complete&&CROP.naturalWidth){g.filter="grayscale(0.75) contrast(0.8)";g.drawImage(CROP,0,0,W,H);g.filter="none";}
  if(!c)return;
  const S=c.side,cw=W/S,ch=H/S,lat=latFor(s);
  const heat=lat!=null||MODE==="heat";
  if(heat){
    const l=lat!=null?lat:c.best,col=colOf(c,l);let mx=0;for(const v of col)mx=Math.max(mx,v);
    for(let p=0;p<c.P;p++){const v=mx>0?col[p]/mx:0;if(v<=0)continue;
      g.globalAlpha=0.25+0.6*v;g.fillStyle=inferno(v);g.fillRect((p%S)*cw,Math.floor(p/S)*ch,cw+0.5,ch+0.5);}
    g.globalAlpha=1;
  }else if(MODE==="region"&&R&&R.panels[s]){
    const lab=R.panels[s].labels[K],cols=R.panels[s].colors[K];
    g.globalAlpha=0.5;
    for(let p=0;p<c.P;p++){g.fillStyle=cols[lab[p]]||"#888";g.fillRect((p%S)*cw,Math.floor(p/S)*ch,cw+0.5,ch+0.5);}
    g.globalAlpha=1;g.strokeStyle="rgba(0,0,0,.8)";g.lineWidth=1.2*DPR();g.beginPath();
    for(let p=0;p<c.P;p++){const x=p%S,y=Math.floor(p/S);
      if(x<S-1&&lab[p]!==lab[p+1]){g.moveTo((x+1)*cw,y*ch);g.lineTo((x+1)*cw,(y+1)*ch);}
      if(y<S-1&&lab[p]!==lab[p+S]){g.moveTo(x*cw,(y+1)*ch);g.lineTo((x+1)*cw,(y+1)*ch);}}
    g.stroke();
  }else{                              // cluster hue of the strongest latent, alpha = peak strength
    for(let p=0;p<c.P;p++){const l=c.top[p];if(l<0)continue;const k=D.col[s][l];
      g.globalAlpha=0.15+0.7*c.rowmax[p]/c.gmax;g.fillStyle=k<0?"#9a9a9a":PAL[k];
      g.fillRect((p%S)*cw,Math.floor(p/S)*ch,cw+0.5,ch+0.5);}
    g.globalAlpha=1;
  }
  const A=activePatch();
  if(A!=null){
    if(MODE==="region"&&R&&!heat){    // veil outside the hovered patch's region
      const mem=new Set(regionMembers(s,A.p)||[]);g.fillStyle="rgba(0,0,0,.45)";
      for(let p=0;p<c.P;p++)if(!mem.has(p))g.fillRect((p%S)*cw,Math.floor(p/S)*ch,cw+0.5,ch+0.5);}
    g.lineWidth=2*DPR();g.strokeStyle=A.s===s?css("--hi"):"#fff";
    g.strokeRect((A.p%S)*cw+1,Math.floor(A.p/S)*ch+1,cw-2,ch-2);}
  e.card.classList.toggle("sel",!!selLat()&&selLat().s===s);
}

// ── UMAP canvas ───────────────────────────────────────────────────────
function tx(x,W){return (x-view.x0)/(view.x1-view.x0)*W;}
function ty(y,H){return H-(y-view.y0)/(view.y1-view.y0)*H;}
function baseLayer(s){                // dimmed dictionary + this image's active set, cached
  const e=E[s];if(e.base)return e.base;
  const cv=e.cum,W=cv.width,H=cv.height,a=D.xy[s],col=D.col[s],d=DPR(),c=C[s];
  const off=document.createElement("canvas");off.width=W;off.height=H;const g=off.getContext("2d");
  const r=Math.max(1,1.1*d);
  g.fillStyle=c?"rgba(150,150,150,0.18)":"rgba(150,150,150,0.45)";
  for(let i=0;i<a.length/2;i++)if(!c||!c.mass.has(i))g.fillRect(tx(a[2*i],W)-r/2,ty(a[2*i+1],H)-r/2,r,r);
  if(c){for(const [i,m] of c.mass){const k=col[i],R2=Math.max(1.6*d,(1.4+3.2*Math.sqrt(m/c.mmax))*d);
      g.fillStyle=k<0?"#8a8a8a":PAL[k];g.fillRect(tx(a[2*i],W)-R2/2,ty(a[2*i+1],H)-R2/2,R2,R2);}}
  const G=64,cells=new Map();
  for(let i=0;i<a.length/2;i++){const px=tx(a[2*i],W),py=ty(a[2*i+1],H);if(px<0||py<0||px>W||py>H)continue;
    const key=Math.floor(px/W*G)+","+Math.floor(py/H*G);if(!cells.has(key))cells.set(key,[]);cells.get(key).push(i);}
  e.grid={G,cells};e.base=off;return off;}
function ringAt(g,s,i,R,lw,color,alpha,dash){
  const cv=E[s].cum,a=D.xy[s],x=tx(a[2*i],cv.width),y=ty(a[2*i+1],cv.height),d=DPR();
  g.globalAlpha=alpha;g.lineWidth=lw*d;g.strokeStyle=color;g.setLineDash(dash?[3*d,2*d]:[]);
  g.beginPath();g.arc(x,y,R*d,0,7);g.stroke();g.setLineDash([]);g.globalAlpha=1;return [x,y];}
function cross(g,x,y,R,color){const d=DPR();g.strokeStyle=color;g.lineWidth=2*d;g.beginPath();
  g.moveTo(x-R*d-7*d,y);g.lineTo(x-R*d-2*d,y);g.moveTo(x+R*d+2*d,y);g.lineTo(x+R*d+7*d,y);
  g.moveTo(x,y-R*d-7*d);g.lineTo(x,y-R*d-2*d);g.moveTo(x,y+R*d+2*d);g.lineTo(x,y+R*d+7*d);g.stroke();}
function drawUmap(s){
  const e=E[s],cv=e.cum,g=cv.getContext("2d");g.clearRect(0,0,cv.width,cv.height);
  g.drawImage(baseLayer(s),0,0);const c=C[s];if(!c)return;
  const A=activePatch(),ring=css("--ring");
  if(A!=null){
    if(MODE==="region"&&R&&!selLat()){const mem=regionMembers(s,A.p);
      if(mem){const top=regionTop(c,mem,12);const t0=top.length?top[0][1]:1;
        for(const [l,sc] of top)ringAt(g,s,l,3+6*Math.sqrt(sc/t0),1.6,ring,0.5+0.5*sc/t0);}}
    else{const rm=c.rowmax[A.p]||1;
      for(const [l,v] of rowTop(c,A.p,24))ringAt(g,s,l,3+7*Math.abs(v)/rm,1.5,ring,0.35+0.65*Math.abs(v)/rm);}
  }
  const L=selLat();
  if(L){const l=latFor(s);if(l!=null){const own=L.s===s,[x,y]=ringAt(g,s,l,own?9:7,own?3:2,own?css("--hi"):ring,1);cross(g,x,y,own?9:7,own?css("--hi"):ring);}}
}
function redrawAll(){if(!D)return;for(const s of KEYS){drawInput(s);drawUmap(s);}}
function hit(ev,s){
  const e=E[s];if(!e.grid)baseLayer(s);const cv=e.cum,r=cv.getBoundingClientRect(),d=DPR(),W=cv.width,H=cv.height;
  const px=(ev.clientX-r.left)*d,py=(ev.clientY-r.top)*d,{G,cells}=e.grid,a=D.xy[s];
  const gx=Math.floor(px/W*G),gy=Math.floor(py/H*G);let best=null,bd=(8*d)**2;
  for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){const Lc=cells.get((gx+dx)+","+(gy+dy));if(!Lc)continue;
    for(const i of Lc){const dd=(tx(a[2*i],W)-px)**2+(ty(a[2*i+1],H)-py)**2;
      // prefer atoms active on this image: they win ties within 2x the distance
      const w=C[s]&&C[s].mass.has(i)?dd/4:dd;if(w<bd){bd=w;best=i;}}}
  return best;}
function zoom(ev,s){
  const r=E[s].cum.getBoundingClientRect();
  const fx=(ev.clientX-r.left)/r.width,fy=1-(ev.clientY-r.top)/r.height,k=ev.deltaY<0?0.85:1/0.85;
  const cx=view.x0+fx*(view.x1-view.x0),cy=view.y0+fy*(view.y1-view.y0);
  view.x0=cx-(cx-view.x0)*k;view.x1=cx+(view.x1-cx)*k;view.y0=cy-(cy-view.y0)*k;view.y1=cy+(view.y1-cy)*k;
  invalidate();redrawAll();}
function startPan(ev,s){
  const r=E[s].cum.getBoundingClientRect();pan={x:ev.clientX,y:ev.clientY,v:{...view},w:r.width,h:r.height,moved:false};
  const mv=e2=>{const dx=(e2.clientX-pan.x)/pan.w*(pan.v.x1-pan.v.x0),dy=(e2.clientY-pan.y)/pan.h*(pan.v.y1-pan.v.y0);
    if(Math.abs(e2.clientX-pan.x)+Math.abs(e2.clientY-pan.y)>3)pan.moved=true;
    view.x0=pan.v.x0-dx;view.x1=pan.v.x1-dx;view.y0=pan.v.y0+dy;view.y1=pan.v.y1+dy;invalidate();redrawAll();};
  const up=()=>{window.removeEventListener("mousemove",mv);window.removeEventListener("mouseup",up);setTimeout(()=>pan=null,0);};
  window.addEventListener("mousemove",mv);window.addEventListener("mouseup",up);}

// ── linking ───────────────────────────────────────────────────────────
function onPatch(){redrawAll();sidePatch();}
function onLatent(now){redrawAll();sidePatch();loadTops(now);}
function sw(s,l){const k=D.col[s][l];return `<i style="background:${k<0?"#9a9a9a":PAL[k]}"></i>`;}
function sidePatch(){
  const A=activePatch();
  if(!A||!C[KEYS[0]]){const u=TEXT()?"token":"patch";$("phd").textContent=`hover a ${u}`;$("plist").innerHTML=`<span class=mut>its active latents, per panel, strongest first · click a ${u} to pin it</span>`;return;}
  const c0=C[A.s]||C[KEYS[0]];
  const xy=TEXT()?`token “${esc((TXT.tokens[A.p]||"").replace(/\n/g,"↵"))}” at position ${(TXT.input.first||0)+A.p}`
                 :`patch (${A.p%c0.side}, ${Math.floor(A.p/c0.side)})`;
  const reg=MODE==="region"&&R&&!selLat();
  $("phd").innerHTML=reg?`region of ${xy} at k=${K}${TEXT()?(SPANS?" (spans)":" (free)"):""} — latents by mass-in × share-in${pp?" · pinned":""}`
                        :`${xy}${pp?" · pinned (click again to release)":""}`;
  let h="";
  for(const s of KEYS){const c=C[s];if(!c)continue;
    let items;
    if(reg){const mem=regionMembers(s,A.p)||[];items=regionTop(c,mem,8).map(([l,sc,sh])=>[l,`${sc.toFixed(2)} · ${Math.round(100*sh)}% in`]);}
    else items=rowTop(c,A.p,8).map(([l,v])=>[l,v.toFixed(3)]);
    const on=latFor(s);
    h+=`<div class=lrow><b>${lbl(s)}</b>`+(items.length?items.map(([l,t])=>`<span class="chip${l===on?" on":""}" data-s="${s}" data-l="${l}">${sw(s,l)}#${l} <span class=mut>${t}</span></span>`).join(""):`<span class=mut>no active latent</span>`)+`</div>`;}
  $("plist").innerHTML=h;
  $("plist").querySelectorAll(".chip").forEach(ch=>{
    ch.onmouseenter=()=>{if(!pl){hl={s:ch.dataset.s,i:+ch.dataset.l};redrawAll();loadTops(false);}};
    ch.onclick=()=>{const L={s:ch.dataset.s,i:+ch.dataset.l};pl=(pl&&pl.s===L.s&&pl.i===L.i)?null:L;onLatent(true);};});
}
function legend(){
  let h=`<div class=lg>`;
  const L=selLat(),T=TEXT(),on=T?"this text":"this image";
  if(L||MODE==="heat"){
    h+=`<span>heatmap of ${L?"the selected latent (and its match per panel)":`each panel's strongest latent on ${on}`}</span>`;
    h+=T?`<span class=ramp style="background:linear-gradient(90deg,${signedBg(-1,1)},transparent,${signedBg(1,1)})"></span><span>negative ← 0 → positive, scaled to its peak |a| on this text (the top contexts use the same colours)</span>`
        :`<span class=ramp style="background:linear-gradient(90deg,${[0,.25,.5,.75,1].map(inferno).join(",")})"></span><span>0 → its peak on this image</span>`;}
  else if(MODE==="region"&&R){
    h+=T?`<span>token regions at k=${K}, ${R.contiguous?"contiguous spans":"free clusters"}; colour = where the region sits in the text${R.exclude_first?"; position 0 (attention sink) left out":""}</span>`
        :`<span>regions at k=${K}; colour = where the region sits</span>`;
    const ari=R.ari[K]||{},ks=Object.keys(ari);
    if(ks.length)h+=`<span>ARI: `+ks.map(p=>{const [a,b]=p.split("|");return `${lbl(a)}–${lbl(b)} <b>${ari[p].toFixed(2)}</b>`;}).join(" · ")+`</span>`;}
  else h+=`<span>${T?"token":"patch"} = anchor-cluster colour of its strongest latent; opacity = its peak |a|. UMAP: coloured = active on ${on} (size ∝ √mass), grey = inactive</span>`;
  $("legend").innerHTML=h+`</div>`;
}

// ── top images ────────────────────────────────────────────────────────
function loadTops(now){
  clearTimeout(topTimer);
  const go=()=>{
    const L=selLat();
    if(!L){$("thd").textContent="hover or click a latent";$("tops").innerHTML=`<span class=mut>in a UMAP, or a chip above · Esc clears</span>`;return;}
    const raw=$("rawtog").checked,sel=`${L.s}:${L.i}:${rule}:${raw}`;
    $("thd").innerHTML=`latent #${L.i} of ${lbl(L.s)}${pl?" · pinned":""} — top ${TEXT()?"tokens (in context)":"images"} per panel (ranks 1,2 · 50,51 · 99,100)`;
    const T=$("tops");T.innerHTML="";T.dataset.sel=sel;
    const shadeItems=[];
    for(const s of KEYS){
      const l=latFor(s);const div=document.createElement("div");
      const mc=s===L.s?"":(()=>{const m=D.mcos[`${rule}_${L.s}_${s}`];return m?` · cos ${m[L.i].toFixed(2)}`:"";})();
      const onImg=C[s]&&l!=null?(()=>{const v=colOf(C[s],l);let mx=0;for(const x of v)mx=Math.max(mx,x);return ` · peak here ${mx.toFixed(3)}`;})():"";
      div.innerHTML=`<div class=lrow><b>${lbl(s)}</b><span class=chip>${l!=null?sw(s,l)+"#"+l:"—"}</span><span class=mut>${s===L.s?"selected":"match"+mc}${onImg}</span></div><div class="trow${TEXT()?" tx":""}"></div>`;
      T.appendChild(div);
      if(l==null)continue;
      if(TEXT()){const live=()=>T.dataset.sel===sel,draw=topContexts(s,l,div.querySelector(".trow"),live);
        shadeItems.push({s,l,draw});continue;}
      fetch(`/top?key=${enc(s)}&lat=${l}`).then(r=>r.json()).then(r=>{
        if(T.dataset.sel!==sel)return;
        const row=div.querySelector(".trow");
        if(r.error||!r.picks.length){row.innerHTML=`<span class=mut>${r.error||"never fires in the top-input pool"}</span>`;return;}
        row.innerHTML=r.picks.map(p=>{const src=raw?`/raw?img=${p.img}&size=128`:`/masked?img=${p.img}&key=${enc(s)}&lat=${l}&size=128`;
          return `<div class=cell title="#${p.img} · ${p.cls} · ${p.val.toFixed(3)}"><img src="${src}" data-i="${p.img}" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'mut',textContent:'image failed'}))"><div class=cap>${p.rank} · ${p.cls}</div></div>`;}).join("");
        row.querySelectorAll("img").forEach(im=>{im.style.cursor="pointer";im.title+=" — click to explore this image";
          im.onclick=()=>setImage(im.dataset.i,"");});
      });
    }
    // shading costs GPU (a live LM + SAE pass per pick): only once the selection has
    // settled, and through the one-in-flight queue, so a mouse sweep cannot pile work up
    clearTimeout(shadeTimer);
    if(shadeItems.length){const job={live:()=>T.dataset.sel===sel,items:shadeItems};
      shadeTimer=setTimeout(()=>shadeJob(job),pl?0:SHADE_DWELL);}
  };
  if(now)go();else topTimer=setTimeout(go,200);
}
// ── shading queue: at most one batch in flight, and the newest request replaces a queued one ──
const SHADE_DWELL=500;
let shadeTimer=null;const SHQ={busy:false,next:null};
function shadeJob(job){SHQ.next=job;pumpShade();}
function pumpShade(){
  if(SHQ.busy||!SHQ.next)return;
  const j=SHQ.next;SHQ.next=null;
  if(!j.live()){pumpShade();return;}         // superseded while queued: never sent
  SHQ.busy=true;
  Promise.all(j.items.map(it=>fetch(`/ctx?key=${enc(it.s)}&lat=${it.l}`).then(r=>r.json())
      .then(t=>{if(j.live())it.draw(t,true);}).catch(()=>{})))
    .finally(()=>{SHQ.busy=false;pumpShade();});
}
function tokHTML(p,ref){               // the UMAP viewer's shading, same colours as the text heatmap
  return p.toks.map((w,i)=>{
    const a=p.acts?p.acts[i]:0,f=Math.min(1,Math.abs(a)/(ref||1)),bg=Math.abs(a)>1e-6?signedBg(a,f):"transparent";
    return `<span class="${p.lo+i===p.pos?"pk2":""}" style="background:${bg}" title="a = ${a.toFixed(3)}">${esc(w.replace(/\n/g,"↵"))}</span>`;
  }).join("");}
function topContexts(s,l,row,live){
  // the text at once (acts=0: no GPU); shading arrives later through the queue
  let shaded=false;
  const draw=(t,sh)=>{
    if(!live()||(shaded&&!sh))return;
    if(sh)shaded=true;
    if(t.error||!t.picks.length){row.innerHTML=`<span class=mut>${esc(t.error||"never fires in the top-input pool")}</span>`;return;}
    row.innerHTML=t.picks.map((p,k)=>{
      const warn=p.chk!=null&&Math.abs(p.chk-1)>0.02?` ⚠${p.chk.toFixed(2)}`:"";
      const tip=`#${p.rank} · |a| ${p.val.toFixed(3)} · row ${p.seq} pos ${p.pos}`+(p.chk!=null?` · live/stored ${p.chk.toFixed(3)}`:"")+" — click to explore this context";
      return `<div class=tcell data-k="${k}" title="${esc(tip)}">`+(p.acts?tokHTML(p,t.ref):`${esc(p.left)}<b class=pk2>${esc(p.peak)}</b>${esc(p.right)}`)+
        `<span class=tcap>#${p.rank} · ${p.val.toFixed(2)}${warn}</span></div>`;}).join("");
    row.querySelectorAll(".tcell").forEach(el=>el.onclick=()=>{const p=t.picks[+el.dataset.k];
      textFrom(`/api/stats_ctx?key=${enc(s)}&seq=${p.seq}&lo=${Math.max(0,p.pos-48)}&hi=${p.pos+49}`);});
  };
  fetch(`/ctx?key=${enc(s)}&lat=${l}&acts=0`).then(r=>r.json()).then(t=>draw(t,false));
  return draw;
}
"""


def page(view_id: str, *, backbone: str) -> str:
    js = _JS.replace("%%VIEW%%", view_id).replace("%%BACKBONE%%", backbone)
    return ("<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Latent Explorer</title><style>" + _CSS + "</style></head><body>"
            + _BODY.replace("%%FOOT%%", FOOT) + "<script>" + js + "</script></body></html>")
