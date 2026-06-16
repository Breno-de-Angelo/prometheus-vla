#!/usr/bin/env python3
# Monta o player/analisador interativo de 1 episódio a partir de replay(s)
# gerados por gen_episode_replay.py. Uso:
#   build_replay_html.py OUT.html "valfix:/tmp/replays/valfix" ["asis:/tmp/replays/asis"]
import sys, json, base64
from pathlib import Path

OUT = Path(sys.argv[1])
PAIRS = sys.argv[2:]


def b64(p):
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


MODELS = {}
for pair in PAIRS:
    label, d = pair.split(":", 1)
    d = Path(d)
    man = json.loads((d / "manifest.json").read_text())
    n = man["n_frames"]
    rgb, dep, att = [], [], []
    for i in range(n):
        rgb.append(b64(d / "f" / f"rgb_{i:03d}.jpg"))
        dep.append(b64(d / "f" / f"dep_{i:03d}.jpg"))
        att.append(b64(d / "f" / f"att_{i:03d}.jpg"))
    MODELS[label] = {"fps": man.get("fps", 30), "n": n, "ckpt": man.get("ckpt", ""),
                     "episode": man.get("episode"), "cfg": man.get("cfg", {}),
                     "wandb_url": man.get("wandb_url", ""), "run_name": man.get("run_name", ""),
                     "frames": man["frames"], "rgb": rgb, "dep": dep, "att": att}

WHEN = json.dumps(None)  # placeholder, stamped by caller if needed
DATA_JS = json.dumps(MODELS, separators=(",", ":"))
LABELS = list(MODELS.keys())

HTML = """<!doctype html>
<!-- Analisador interativo de 1 episódio (replay gravado 1x; toca/controla sem re-rodar o modelo).
     RGB + depth + atenção real + juntas (rad, pred x GT) + chunks completos + trajetória real x calculada. -->
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replay interativo — pega do copo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#070b11;--bg2:#0a0f17;--card:#0d131c;--card2:#111925;--line:#1b2532;--line2:#26344a;
--ink:#dce6f1;--ink2:#8ea0b6;--ink3:#5c6c80;--cyan:#22d3ee;--amber:#ff9e3d;--green:#34d399;--red:#f2607a;--violet:#a78bfa;
--disp:'Chakra Petch',sans-serif;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;
background-image:radial-gradient(900px 500px at 85% -10%,rgba(34,211,238,.06),transparent 60%);}
.wrap{max-width:1240px;margin:0 auto;padding:20px 18px 70px}
h1{font-family:var(--disp);font-weight:700;font-size:25px;letter-spacing:.01em}
h1 .c{color:var(--cyan)}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:14px 0}
.tag{font-family:var(--mono);font-size:11px;color:var(--ink2);background:var(--bg2);border:1px solid var(--line);border-radius:7px;padding:5px 9px}
.seg{display:inline-flex;border:1px solid var(--line2);border-radius:8px;overflow:hidden}
.seg button{background:var(--bg2);color:var(--ink2);border:0;padding:6px 14px;font-family:var(--mono);font-size:12px;cursor:pointer}
.seg button.on{background:var(--cyan);color:#06222b;font-weight:600}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
@media(max-width:760px){.panels{grid-template-columns:1fr}}
figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#000}
figure img{width:100%;display:block;aspect-ratio:848/480;object-fit:cover}
figcaption{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;color:var(--ink2);text-align:center;padding:5px 0;background:var(--bg2)}
.ctrl{display:flex;align-items:center;gap:12px;margin:14px 0;background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:11px;padding:11px 14px}
.ctrl button{background:var(--bg2);color:var(--ink);border:1px solid var(--line2);border-radius:8px;padding:7px 12px;font-family:var(--mono);font-size:13px;cursor:pointer}
.ctrl button:hover{border-color:var(--cyan)}
.ctrl .big{font-size:15px;padding:7px 16px}
.ctrl input[type=range]{flex:1;accent-color:var(--cyan)}
.ctrl .fr{font-family:var(--mono);font-size:13px;color:var(--cyan);min-width:96px;text-align:right}
.grid2{display:grid;grid-template-columns:1.1fr 1fr;gap:14px;margin-top:6px}
@media(max-width:880px){.grid2{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:12px;padding:14px 15px}
.card h3{font-family:var(--disp);font-weight:700;font-size:13px;letter-spacing:.05em;text-transform:uppercase;color:var(--cyan);margin-bottom:10px}
.desc{border-left:3px solid var(--amber);background:rgba(255,158,61,.05);border-radius:8px;padding:11px 13px;font-size:13.5px;color:var(--ink);min-height:64px}
.desc b{color:var(--amber)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
th,td{padding:4px 7px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;color:var(--ink2)}
thead th{color:var(--ink3);font-size:10px;text-transform:uppercase}
td.d{color:var(--ink3)}
.force{display:flex;align-items:center;gap:10px;margin-top:4px}
.fbar{position:relative;flex:1;height:22px;background:var(--bg2);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.fbar i{position:absolute;left:0;top:0;height:100%;border-radius:5px}
.fbar .gt{height:6px;top:auto;bottom:0;opacity:.9}
.chunkbox{margin-top:6px}
.chunkbox .row{display:flex;align-items:center;gap:10px}
.chunkbox input[type=range]{flex:1;accent-color:var(--violet)}
.chunkbox .k{font-family:var(--mono);font-size:12px;color:var(--violet);min-width:96px;text-align:right}
svg{width:100%;height:auto;display:block;background:var(--bg2);border-radius:8px;border:1px solid var(--line)}
.dimsel{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
.dimsel button{background:var(--bg2);color:var(--ink2);border:1px solid var(--line);border-radius:6px;padding:3px 9px;font-family:var(--mono);font-size:11px;cursor:pointer}
.dimsel button.on{background:var(--violet);color:#1a1033;border-color:var(--violet)}
.leg{font-family:var(--mono);font-size:11px;color:var(--ink2);display:flex;gap:14px;margin-top:7px;flex-wrap:wrap}
.leg i{display:inline-block;width:18px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.foot{margin-top:30px;color:var(--ink3);font-family:var(--mono);font-size:11px;border-top:1px solid var(--line);padding-top:14px}
</style></head><body><div class="wrap">

<h1>Replay interativo — <span class="c">pega do copo</span></h1>
<div class="bar">
  <span id="seg" class="seg"></span>
  <span class="tag" id="meta"></span>
  <span class="tag">grava 1x · toca/controla sem re-rodar o modelo</span>
</div>
<div class="bar"><span class="tag" id="modelcfg" style="white-space:normal;line-height:1.8;max-width:100%"></span></div>

<div class="panels">
  <figure><img id="imRgb" alt="RGB"><figcaption>RGB (head_camera)</figcaption></figure>
  <figure><img id="imDep" alt="DEPTH"><figcaption>DEPTH</figcaption></figure>
  <figure><img id="imAtt" alt="ATT"><figcaption>ATENÇÃO (real, action expert → imagem)</figcaption></figure>
</div>

<div class="ctrl">
  <button id="play" class="big">▶︎</button>
  <button id="prev">⏮ frame</button>
  <button id="next">frame ⏭</button>
  <button id="loop" class="on">loop: on</button>
  <input id="fslider" type="range" min="0" value="0">
  <span class="fr" id="frnum"></span>
</div>

<div class="card"><h3>O que está acontecendo</h3><div id="desc" class="desc"></div></div>

<div class="grid2">
  <div class="card">
    <h3>Juntas + força — neste frame (previsto × real)</h3>
    <div class="force">
      <span class="tag" style="min-width:64px">força</span>
      <div class="fbar"><i id="fPred" style="background:var(--green)"></i><i id="fGt" class="gt" style="background:var(--ink2)"></i></div>
      <span class="fr" id="fTxt" style="min-width:140px"></span>
    </div>
    <table style="margin-top:10px"><thead><tr><th>junta</th><th>previsto (rad)</th><th>real (rad)</th><th>Δ</th></tr></thead><tbody id="jtab"></tbody></table>
  </div>
  <div class="card">
    <h3>Chunk calculado — 50 ações planejadas neste frame</h3>
    <div class="chunkbox">
      <div class="row"><button id="cprev">⏮</button><input id="cslider" type="range" min="0" max="49" value="0"><button id="cnext">⏭</button><span class="k" id="cnum"></span></div>
      <table style="margin-top:8px"><thead><tr><th>dim</th><th>passo k (rad/sq)</th><th>k=0 (exec)</th></tr></thead><tbody id="ctab"></tbody></table>
    </div>
  </div>
</div>

<div class="card" style="margin-top:14px">
  <h3>Trajetória — real (GT) × calculada (executada) + plano do chunk</h3>
  <div class="dimsel" id="dimsel"></div>
  <svg id="traj" viewBox="0 0 1180 260" preserveAspectRatio="none"></svg>
  <div class="leg"><span><i style="background:var(--ink2)"></i>real (GT)</span><span><i style="background:var(--green)"></i>calculada (1ª ação de cada chunk)</span><span><i style="background:var(--violet)"></i>plano de 50 passos (frame atual)</span><span><i style="background:var(--cyan)"></i>frame atual</span></div>
</div>

<div class="foot">replay gravado por gen_episode_replay.py · atenção via attn_recorder · 1 episódio de val · juntas em radianos · squeeze 0–1</div>
</div>

<script>
const MODELS = __DATA__;
const LABELS = __LABELS__;
const JN = ["0 shoulder_pitch","1 shoulder_roll","2 shoulder_yaw","3 elbow","4 wrist_roll","5 wrist_pitch","6 wrist_yaw"];
let M=LABELS[0], i=0, ck=0, dim=7, playing=false, loop=true, timer=null;

function cur(){ return MODELS[M]; }
function frame(){ return cur().frames[i]; }

function buildSeg(){
  const s=document.getElementById('seg');
  s.innerHTML = LABELS.map(l=>`<button data-l="${l}" class="${l===M?'on':''}">${l}</button>`).join('');
  s.querySelectorAll('button').forEach(b=>b.onclick=()=>{M=b.dataset.l; i=Math.min(i,cur().n-1); ck=0; buildSeg(); buildDims(); refresh();});
}
function buildDims(){
  const ds=document.getElementById('dimsel');
  const dims=[{k:7,n:'squeeze'},...JN.map((n,k)=>({k,n:n.split(' ')[1]}))];
  ds.innerHTML = dims.map(d=>`<button data-d="${d.k}" class="${d.k===dim?'on':''}">${d.n}</button>`).join('');
  ds.querySelectorAll('button').forEach(b=>b.onclick=()=>{dim=+b.dataset.d; buildDims(); drawTraj();});
}

function setFrame(n){ i=Math.max(0,Math.min(cur().n-1,n)); ck=Math.min(ck,49); refresh(); }

function refresh(){
  const c=cur(), f=frame();
  document.getElementById('imRgb').src=c.rgb[i];
  document.getElementById('imDep').src=c.dep[i]||'';
  document.getElementById('imAtt').src=c.att[i]||'';
  document.getElementById('meta').textContent=`${M} · ep ${c.episode} · ${c.n} frames @ ${c.fps}fps`;
  const cf=c.cfg||{}; const ce=Object.keys(cf).filter(k=>k!=='checkpoint'&&k!=='task').map(k=>k+'='+cf[k]).join(' · ');
  const wl=c.wandb_url?' &nbsp;·&nbsp; <b>wandb:</b> <a href="'+c.wandb_url+'" target="_blank" style="color:var(--cyan)">'+(c.wandb_url.split('/runs/')[1]||'link')+'</a>':'';
  document.getElementById('modelcfg').innerHTML='<b>run:</b> '+(c.run_name||M)+wl+'<br><b>modelo:</b> <code>'+(c.ckpt||'?')+'</code>'+(ce?'<br><b>config:</b> '+ce:'')+(cf.task?'<br><b>task:</b> "'+cf.task+'"':'');
  document.getElementById('fslider').max=c.n-1;
  document.getElementById('fslider').value=i;
  document.getElementById('frnum').textContent=`frame ${i}/${c.n-1}`;
  // força
  const fp=f.pred_sq, fg=f.gt_sq;
  document.getElementById('fPred').style.width=(fp*100)+'%';
  document.getElementById('fGt').style.width=(fg*100)+'%';
  document.getElementById('fTxt').textContent=`pred ${fp.toFixed(2)} · GT ${fg.toFixed(2)}`;
  // juntas
  let rows='';
  for(let k=0;k<7;k++){ const p=f.pred_j[k],g=f.gt_j[k],d=p-g;
    const col=Math.abs(d)>0.2?'var(--red)':Math.abs(d)>0.08?'var(--amber)':'var(--ink3)';
    rows+=`<tr><td>${JN[k]}</td><td>${p.toFixed(3)}</td><td>${g.toFixed(3)}</td><td style="color:${col}">${d>=0?'+':''}${d.toFixed(3)}</td></tr>`; }
  document.getElementById('jtab').innerHTML=rows;
  renderChunk();
  drawTraj();
  document.getElementById('desc').innerHTML=describe();
}

function renderChunk(){
  const f=frame();
  document.getElementById('cslider').value=ck;
  document.getElementById('cnum').textContent=`passo ${ck}/49`;
  const step=f.chunk[ck], step0=f.chunk[0];
  let rows='';
  for(let k=0;k<7;k++){ rows+=`<tr><td>${JN[k]}</td><td>${step[k].toFixed(3)}</td><td class="d">${step0[k].toFixed(3)}</td></tr>`; }
  rows+=`<tr><td>squeeze</td><td>${step[7].toFixed(2)}</td><td class="d">${step0[7].toFixed(2)}</td></tr>`;
  document.getElementById('ctab').innerHTML=rows;
}

function describe(){
  const c=cur(), f=frame(), p=i>0?c.frames[i-1]:f;
  const sq=f.pred_sq, dsq=sq-p.pred_sq;
  let hand = sq<0.15?'<b>mão aberta</b>':sq>0.85?'<b>mão fechada</b>':`mão em transição (${(sq*100).toFixed(0)}%)`;
  let mov = Math.abs(dsq)>0.05?(dsq>0?' — <b>fechando</b>':' — <b>abrindo</b>'):'';
  // junta que mais mexeu
  let mj=0,md=0; for(let k=0;k<7;k++){const d=Math.abs(f.pred_j[k]-p.pred_j[k]); if(d>md){md=d;mj=k;}}
  // maior erro vs GT
  let ej=0,me=0; for(let k=0;k<7;k++){const d=Math.abs(f.pred_j[k]-f.gt_j[k]); if(d>me){me=d;ej=k;}}
  let arm = md>0.01?` · braço movendo <b>${JN[mj]}</b> (${md.toFixed(3)} rad/frame)`:' · braço ~parado';
  let err = ` · maior erro vs real: <b>${JN[ej]}</b> ${me.toFixed(3)} rad`;
  let grasp = (Math.min(sq,p.pred_sq)<0.5 && Math.max(sq,p.pred_sq)>=0.5)?' &nbsp;⟵ <b style="color:var(--red)">momento da pega</b>':'';
  return `<b>frame ${i}</b>: ${hand}${mov}${arm}${err}${grasp}`;
}

function valOf(f,d){ return d===7? f.pred_sq : f.pred_j[d]; }
function gtOf(f,d){ return d===7? f.gt_sq : f.gt_j[d]; }
function chunkOf(f,k,d){ return f.chunk[k][d]; }

function drawTraj(){
  const c=cur(), N=c.n, svg=document.getElementById('traj');
  const W=1180,H=260,mL=44,mR=10,mT=10,mB=22;
  // y-range
  let lo=1e9,hi=-1e9;
  for(const f of c.frames){ const a=valOf(f,dim),b=gtOf(f,dim); lo=Math.min(lo,a,b); hi=Math.max(hi,a,b); }
  if(hi-lo<1e-6){hi=lo+1;}
  const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
  const sx=x=>mL+(x/(N-1))*(W-mL-mR), sy=v=>H-mB-((v-lo)/(hi-lo))*(H-mT-mB);
  const NS='http://www.w3.org/2000/svg'; while(svg.firstChild)svg.removeChild(svg.firstChild);
  const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
  // grid y
  for(let g=0;g<=4;g++){const v=lo+(hi-lo)*g/4,y=sy(v);
    svg.appendChild(el('line',{x1:mL,y1:y,x2:W-mR,y2:y,stroke:'#1b2532'}));
    const t=el('text',{x:mL-5,y:y+3,fill:'#5c6c80','font-size':10,'font-family':'IBM Plex Mono','text-anchor':'end'});t.textContent=v.toFixed(2);svg.appendChild(t);}
  function line(fn,color,w){let d='';for(let k=0;k<N;k++){d+=(k?'L':'M')+sx(k).toFixed(1)+' '+sy(fn(c.frames[k])).toFixed(1)+' ';}
    svg.appendChild(el('path',{d,fill:'none',stroke:color,'stroke-width':w,'stroke-linejoin':'round'}));}
  line(f=>gtOf(f,dim),'#8ea0b6',2);        // real
  line(f=>valOf(f,dim),'#34d399',2);       // calculada (executada)
  // plano do chunk do frame atual (50 passos a partir de i)
  const f=c.frames[i]; let dp='';
  for(let k=0;k<f.chunk.length;k++){const x=Math.min(N-1,i+k); dp+=(k?'L':'M')+sx(x).toFixed(1)+' '+sy(chunkOf(f,k,dim)).toFixed(1)+' ';}
  svg.appendChild(el('path',{d:dp,fill:'none',stroke:'#a78bfa','stroke-width':2,'stroke-dasharray':'4 3',opacity:.9}));
  // playhead
  svg.appendChild(el('line',{x1:sx(i),y1:mT,x2:sx(i),y2:H-mB,stroke:'#22d3ee','stroke-width':1.5}));
}

// controles
function play(){ playing=!playing; document.getElementById('play').textContent=playing?'❚❚':'▶︎';
  if(playing){ const dt=1000/(cur().fps||30); timer=setInterval(()=>{ if(i>=cur().n-1){ if(loop) i=0; else {play();return;} } else i++; refresh(); }, Math.max(33,dt)); }
  else { clearInterval(timer); } }
document.getElementById('play').onclick=play;
document.getElementById('prev').onclick=()=>setFrame(i-1);
document.getElementById('next').onclick=()=>setFrame(i+1);
document.getElementById('loop').onclick=function(){ loop=!loop; this.textContent='loop: '+(loop?'on':'off'); this.classList.toggle('on',loop); };
document.getElementById('fslider').oninput=e=>setFrame(+e.target.value);
document.getElementById('cslider').oninput=e=>{ck=+e.target.value; renderChunk();};
document.getElementById('cprev').onclick=()=>{ck=Math.max(0,ck-1);renderChunk();};
document.getElementById('cnext').onclick=()=>{ck=Math.min(49,ck+1);renderChunk();};
document.addEventListener('keydown',e=>{ if(e.key==='ArrowRight')setFrame(i+1); else if(e.key==='ArrowLeft')setFrame(i-1); else if(e.key===' '){e.preventDefault();play();} });

buildSeg(); buildDims(); refresh();
</script></body></html>"""

HTML = HTML.replace("__DATA__", DATA_JS).replace("__LABELS__", json.dumps(LABELS))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"[ok] {OUT}  ({len(HTML)//1024} KB)  models={LABELS}")
