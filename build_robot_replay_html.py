#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay HTML de uma run de inferência no robô real, a partir da pasta --record
(run_<ts>/ com rgb/*.jpg, attn/*.jpg, chunks.jsonl). Player navegável: RGB + mapa de
atenção da VLA lado a lado + curva do squeeze e das juntas do braço, sincronizados.
Uso: python build_robot_replay_html.py <run_dir> <out.html>
"""
import json, base64, sys, os

RUN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/robotrun2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/probs/robot_run4a_replay_20260616.html"
GEN_TS = "2026-06-16 18:45 (-03)"

meta = json.load(open(os.path.join(RUN, "meta.json")))
ckpt = meta.get("checkpoint", "?").split("/")
ckpt_name = "/".join(ckpt[-3:]) if len(ckpt) >= 3 else meta.get("checkpoint", "?")

ARM = ['Sh·Pitch', 'Sh·Roll', 'Sh·Yaw', 'Elbow', 'Wr·Roll', 'Wr·Pitch', 'Wr·Yaw']
COLS = ['#38bdf8', '#34d399', '#a78bfa', '#fb923c', '#f472b6', '#facc15', '#f87171']

import cv2, numpy as np

def b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def b64_overlay(rgb_path, attn_path):
    """Heatmap 16x16 da VLA -> resize cúbico + JET + sobreposto no RGB (igual ao dashboard)."""
    rgb = cv2.imread(rgb_path)  # BGR (imread)
    hm = cv2.imread(attn_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None or hm is None:
        return b64(rgb_path)
    hm = cv2.resize(hm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    # cena dessaturada (cinza) por baixo + heatmap JET por cima (alpha ∝ intensidade da atenção):
    # a cena fica sempre visível e o foco da VLA se destaca, mesmo com atenção espalhada.
    base = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET).astype(np.float32)
    a = (np.clip(hm.astype(np.float32) / 255.0, 0, 1) * 0.62)[..., None]
    over = (base * (1 - a) + hm_color * a).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", over, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else b64(rgb_path)

rgb, attn, sq, arm, infer = [], [], [], [], []
for line in open(os.path.join(RUN, "chunks.jsonl")):
    d = json.loads(line)
    rp = os.path.join(RUN, d["rgb"]); ap = os.path.join(RUN, d.get("attn", ""))
    if not os.path.exists(rp):
        continue
    rgb.append(b64(rp))
    attn.append(b64_overlay(rp, ap) if d.get("attn") and os.path.exists(ap) else "")
    a0 = d["actions"][0]
    sq.append(round(max(0.0, min(1.0, (a0[7] + 1) / 2)), 3))   # squeeze norm[-1,1] -> [0,1]
    arm.append([round(x, 3) for x in d["state_raw"][:7]])       # braço físico (rad)
    infer.append(round(d.get("infer_ms", 0)))

N = len(rgb)
payload = {"sq": sq, "arm": arm, "names": ARM, "cols": COLS}

# stats
closed = sum(1 for s in sq if s > 0.75); openn = sum(1 for s in sq if s < 0.25)
first_close = next((i for i, s in enumerate(sq) if s > 0.5), None)
lat_avg = round(sum(infer) / N) if N else 0

# embute imagens num array JS
def jsarr(lst):
    return "[" + ",".join('"' + x + '"' for x in lst) + "]"

HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replay robô — run4a best@9500</title>
<style>
:root{--bg:#0b1020;--card:#121a2e;--ink:#e8eefc;--mut:#8aa0c6;--line:#23304d;--cyan:#38bdf8;--amber:#f59e0b;}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0b1020,#0a0e1c);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 70px}
h1{font-size:25px;margin:0 0 4px}
.sub{color:var(--mut);font-size:12.5px;margin:0 0 2px}
.tag{display:inline-block;font-size:11.5px;color:var(--mut);border:1px solid var(--line);border-radius:999px;padding:2px 10px;margin:8px 6px 0 0}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:10px 15px}
.kpi b{font-size:19px;display:block} .kpi span{font-size:11.5px;color:var(--mut)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
@media(max-width:760px){.imgs{grid-template-columns:1fr}}
figure{margin:0;background:#05080d;border:1px solid var(--line);border-radius:12px;overflow:hidden;position:relative}
figure img{width:100%;display:block;aspect-ratio:848/480;object-fit:cover;background:#000}
figcaption{position:absolute;top:8px;left:10px;font-size:11px;color:#cdd9f2;background:rgba(8,16,28,.7);padding:3px 8px;border-radius:6px}
.ctl{display:flex;align-items:center;gap:12px;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:10px 15px;margin:8px 0}
.ctl button{background:var(--cyan);color:#06121f;border:none;border-radius:7px;padding:7px 15px;font-weight:700;cursor:pointer;font-size:13px}
.ctl input[type=range]{flex:1}
.ctl .fnum{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--mut);min-width:120px}
canvas{width:100%;display:block;background:var(--card);border:1px solid var(--line);border-radius:11px;margin-top:10px}
h3{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin:18px 0 4px}
footer{margin-top:34px;border-top:1px solid var(--line);padding-top:14px;color:#5b6b89;font-size:11.5px}
</style></head><body><div class="wrap">

<h1>Replay no robô real — run4a best@9500</h1>
<p class="sub">Gerado em __GEN__ · gravação <code>__RUN__</code> · ckpt <code>__CKPT__</code> · tarefa "__TASK__"</p>
<p><span class="tag">__N__ decisões</span><span class="tag">~__LAT__ms/inferência</span><span class="tag">π0.5 right8/armstate7</span><span class="tag">EMA · state-dropout</span></p>

<div class="kpis">
  <div class="kpi"><b>__N__</b><span>decisões (1 img/chunk)</span></div>
  <div class="kpi"><b>__FC__</b><span>1º fechamento (chunk)</span></div>
  <div class="kpi"><b>__CL__%</b><span>tempo mão fechada</span></div>
  <div class="kpi"><b>__LAT__ms</b><span>latência média</span></div>
</div>

<div class="imgs">
  <figure><img id="imRgb"><figcaption>RGB · head cam</figcaption></figure>
  <figure><img id="imAtt"><figcaption>Atenção · VLA (onde olha)</figcaption></figure>
</div>

<div class="ctl">
  <button id="play">▶ Play</button>
  <input id="slider" type="range" min="0" max="__MAX__" value="0">
  <span class="fnum" id="fnum">chunk 0 / __MAX__</span>
</div>

<h3>Squeeze — mão aberta (0) ↔ fechada (1)</h3>
<canvas id="cSq" height="120"></canvas>
<h3>Braço — juntas medidas (rad)</h3>
<canvas id="cArm" height="170"></canvas>

<footer>
1 frame por decisão (chunk) — a imagem + atenção que a VLA usou pra prever aquele chunk de 50 ações.
Squeeze derivado da 1ª ação do chunk (normalizado→0–1); braço = estado medido (rad). Sem áudio/vídeo: é um player de frames.
Gravado por <code>--record</code> (run_recorder) · gerado por <code>build_robot_replay_html.py</code>.
</footer>
</div>
<script>
const RGB = __RGB__, ATT = __ATT__, P = __PAYLOAD__;
const N = RGB.length;
const imRgb = document.getElementById('imRgb'), imAtt = document.getElementById('imAtt');
const slider = document.getElementById('slider'), fnum = document.getElementById('fnum'), playBtn = document.getElementById('play');
let cur = 0, playing = false, timer = null;
function show(i){
  cur = Math.max(0, Math.min(N-1, i));
  imRgb.src = RGB[cur]; imAtt.src = ATT[cur] || RGB[cur];
  slider.value = cur; fnum.textContent = 'chunk ' + cur + ' / ' + (N-1) + '  ·  squeeze ' + P.sq[cur].toFixed(2);
  drawAll();
}
slider.oninput = () => show(+slider.value);
playBtn.onclick = () => {
  playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) timer = setInterval(() => { if (cur >= N-1) show(0); else show(cur+1); }, 120);
  else clearInterval(timer);
};
function line(ctx, data, W, H, pT, pB, ymin, ymax, col, w){
  const X = i => 40 + (i/(N-1))*(W-52), Y = v => pT + (1-(v-ymin)/(ymax-ymin))*(H-pT-pB);
  ctx.strokeStyle=col; ctx.lineWidth=w||2; ctx.beginPath();
  data.forEach((v,i)=>{const x=X(i),y=Y(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  return {X,Y};
}
function grid(ctx,W,H,pT,pB,ymin,ymax){
  ctx.strokeStyle='#1c2940';ctx.fillStyle='#6b7c9c';ctx.font='10px ui-monospace,monospace';ctx.lineWidth=1;
  for(let k=0;k<=3;k++){const v=ymin+(ymax-ymin)*k/3,y=pT+(1-k/3)*(H-pT-pB);ctx.beginPath();ctx.moveTo(40,y);ctx.lineTo(W-12,y);ctx.stroke();ctx.fillText(v.toFixed(2),4,y+3);}
}
function playhead(ctx,W,H,pT,pB){
  const x=40+(cur/(N-1))*(W-52); ctx.strokeStyle='#38bdf8';ctx.globalAlpha=.8;ctx.lineWidth=1.4;
  ctx.beginPath();ctx.moveTo(x,pT);ctx.lineTo(x,H-pB);ctx.stroke();ctx.globalAlpha=1;
}
function drawAll(){
  // squeeze
  let c=document.getElementById('cSq'); fit(c); let ctx=c.getContext('2d'); const W=c.clientWidth,H=c.clientHeight;
  ctx.clearRect(0,0,W,H); grid(ctx,W,H,12,18,0,1); line(ctx,P.sq,W,H,12,18,0,1,'#38bdf8',2.2); playhead(ctx,W,H,12,18);
  // arm
  c=document.getElementById('cArm'); fit(c); ctx=c.getContext('2d'); const W2=c.clientWidth,H2=c.clientHeight;
  ctx.clearRect(0,0,W2,H2);
  let mn=1e9,mx=-1e9; P.arm.forEach(r=>r.forEach(v=>{if(v<mn)mn=v;if(v>mx)mx=v;})); const pd=(mx-mn)*.1||.1; mn-=pd;mx+=pd;
  grid(ctx,W2,H2,12,18,mn,mx);
  for(let j=0;j<7;j++){ line(ctx, P.arm.map(r=>r[j]), W2,H2,12,18,mn,mx, P.cols[j],1.6); }
  playhead(ctx,W2,H2,12,18);
  // legenda
  ctx.font='10px ui-monospace,monospace'; let lx=44;
  for(let j=0;j<7;j++){ctx.fillStyle=P.cols[j];ctx.fillRect(lx,2,9,3);ctx.fillStyle='#9fb2d4';ctx.fillText(P.names[j],lx+12,7);lx+=12+ctx.measureText(P.names[j]).width+12;}
}
function fit(c){const dpr=Math.min(2,window.devicePixelRatio||1);const w=c.clientWidth,h=c.clientHeight;if(c.width!==w*dpr||c.height!==h*dpr){c.width=w*dpr;c.height=h*dpr;}c.getContext('2d').setTransform(dpr,0,0,dpr,0,0);}
window.addEventListener('resize',drawAll);
show(0);
</script>
</body></html>"""

HTML = (HTML
  .replace("__GEN__", GEN_TS).replace("__RUN__", os.path.basename(RUN.rstrip("/")))
  .replace("__CKPT__", ckpt_name).replace("__TASK__", meta.get("task", "?"))
  .replace("__N__", str(N)).replace("__MAX__", str(N - 1))
  .replace("__FC__", str(first_close) if first_close is not None else "—")
  .replace("__CL__", str(round(100 * closed / N)) if N else "0")
  .replace("__LAT__", str(lat_avg))
  .replace("__RGB__", jsarr(rgb)).replace("__ATT__", jsarr(attn))
  .replace("__PAYLOAD__", json.dumps(payload)))

os.makedirs(os.path.dirname(OUT), exist_ok=True) if os.path.dirname(OUT) else None
open(OUT, "w").write(HTML)
print("escrito:", OUT, "(%.1f MB)" % (len(HTML) / 1048576))
print("stats: N=%d  1stclose=%s  closed=%d open=%d  lat=%dms" % (N, first_close, closed, openn, lat_avg))
