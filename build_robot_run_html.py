#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera um HTML de analise temporal de uma run de inferencia no robo real (a partir do LOG).
A run nao gravou frames (--live sem --record) -> sem imagem; analisa a telemetria do log:
squeeze (mao aberta/fechada), juntas do braco e latencia por chunk.
Uso: python build_robot_run_html.py <run.log> <out.html>
"""
import re, json, sys, ast

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/robotrun/run.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/probs/robot_run4a_best7500_20260616.html"

GEN_TS = "2026-06-16 16:10 (-03)"
RUN_ID = "infer_right14_20260616_154906"

txt = open(LOG, encoding="utf-8", errors="replace").read().splitlines()

def ts_to_s(line):
    m = re.match(r"(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d),(\d\d\d)", line)
    if not m: return None
    h, mi, s, ms = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    return h*3600 + mi*60 + s + ms/1000.0

chunks = []
i = 0
cur = None
while i < len(txt):
    line = txt[i]
    mlat = re.search(r"chunk (\d+): previsto em (\d+)ms", line)
    if mlat:
        cur = {"n": int(mlat.group(1)), "lat": int(mlat.group(2)), "t": ts_to_s(line)}
    elif "braço:" in line and cur is not None:
        d = line.split("braço:", 1)[1].strip()
        try: cur["arm"] = ast.literal_eval(d)
        except Exception: cur["arm"] = {}
    elif "mão:" in line and cur is not None:
        d = line.split("mão:", 1)[1].strip()
        try:
            cur["hand"] = ast.literal_eval(d)
            chunks.append(cur); cur = None
        except Exception:
            cur = None
    i += 1

if not chunks:
    print("nenhum chunk parseado"); sys.exit(1)

t0 = chunks[0]["t"]
for c in chunks:
    c["rt"] = (c["t"] - t0) if c["t"] is not None else 0.0
    h = c.get("hand", {})
    # squeeze proxy: index_0 vai de 0 (aberta) a ~1.57 (fechada)
    idx0 = h.get("right_hand_index_0_joint.q", 0.0)
    c["sq"] = max(0.0, min(1.0, idx0 / 1.57))

# series
T   = [round(c["rt"], 2) for c in chunks]
SQ  = [round(c["sq"], 3) for c in chunks]
LAT = [c["lat"] for c in chunks]
ARM_KEYS = ["kRightShoulderPitch.q", "kRightElbow.q", "kRightWristPitch.q"]
ARM = {k: [round(c.get("arm", {}).get(k, 0.0), 3) for c in chunks] for k in ARM_KEYS}

# stats
dur = T[-1] if T else 0
n = len(chunks)
lat_avg = sum(LAT)/n
open_n = sum(1 for s in SQ if s < 0.25)
closed_n = sum(1 for s in SQ if s > 0.75)
mid_n = n - open_n - closed_n
# primeiro fechamento (sq cruza 0.5)
first_close = next((c for c in chunks if c["sq"] > 0.5), None)
fc_t = round(first_close["rt"], 1) if first_close else None
fc_n = first_close["n"] if first_close else None
# reaberturas (sq volta < 0.25 depois de ter fechado)
reopened = False
seen_closed = False
for c in chunks:
    if c["sq"] > 0.75: seen_closed = True
    if seen_closed and c["sq"] < 0.25: reopened = True; break

payload = {"T": T, "SQ": SQ, "LAT": LAT, "ARM": ARM, "ARM_KEYS": ARM_KEYS}

verdict_close = ("a mão fecha em ~%.0fs (chunk %d) e %sreabre depois"
                 % (fc_t, fc_n, "" if reopened else "NÃO ")) if fc_t is not None else "a mão nunca fecha"

HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run no robô — run4a best@7500</title>
<style>
:root{--bg:#0b1020;--card:#121a2e;--ink:#e8eefc;--mut:#8aa0c6;--line:#23304d;--cyan:#38bdf8;--amber:#f59e0b;--green:#34d399;--red:#fb7185;}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0b1020,#0a0e1c);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:27px;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 3px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:14px 0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.kpi b{display:block;font-size:23px;letter-spacing:-.5px}
.kpi span{font-size:12px;color:var(--mut)}
.lede{font-size:15.5px;background:linear-gradient(135deg,#13203c,#101830);border-left:3px solid var(--amber);border-radius:12px;padding:15px 18px}
.lede b{color:#fff}
h3{font-size:15px;color:#cdd9f2;margin:22px 0 6px}
.hint{color:var(--mut);font-size:12.5px;margin:0 0 8px}
canvas{width:100%;height:180px;display:block}
.tag{display:inline-block;font-size:11.5px;color:var(--mut);border:1px solid var(--line);border-radius:999px;padding:2px 10px;margin-right:6px}
.warn{color:var(--amber)} .ok{color:var(--green)}
code{color:#cbd5e1;font-size:12px}
footer{margin-top:40px;border-top:1px solid var(--line);padding-top:16px;color:#5b6b89;font-size:12px}
</style></head><body><div class="wrap">

<h1>Run de inferência no robô real — run4a best@7500</h1>
<p class="sub">Gerado em __GEN__ · log <code>__RUNID__</code> · checkpoint EMA (state-dropout)</p>
<p><span class="tag">π0.5 right8/armstate7</span><span class="tag">__N__ chunks</span><span class="tag">__DUR__ s</span><span class="tag">deploy robô 10.9.8.73</span></p>

<div class="kpis">
  <div class="kpi"><b>__N__</b><span>chunks (50 ações cada)</span></div>
  <div class="kpi"><b>__LAT__ms</b><span>latência média / chunk</span></div>
  <div class="kpi"><b class="__SQCLS__">__CLOSEDPCT__%</b><span>tempo com mão fechada</span></div>
  <div class="kpi"><b>__FCT__s</b><span>1º fechamento</span></div>
</div>

<div class="card lede">
<b>Resumo:</b> a run controlou o braço direito por __DUR__ s (__N__ chunks, ~__LAT__ms cada). __VERDICT__.
A mão ficou <b>aberta __OPEN__</b> · <b>fechada __CLOSED__</b> · intermediária __MID__ chunks.
Sem frames gravados (rodou <code>--live</code> sem <code>--record</code>) → esta é a telemetria do log; pra um replay com imagem+atenção, rodar com <code>--record</code>.
</div>

<h3>Squeeze — mão aberta (0) ↔ fechada (1)</h3>
<p class="hint">proxy = index_0 / 1.57. É a curva-chave do run4a: idealmente a mão abre, aproxima do copo e fecha NA HORA.</p>
<div class="card"><canvas id="cSq"></canvas></div>

<h3>Juntas do braço (rad) ao longo da run</h3>
<p class="hint">ombro-pitch, cotovelo e punho-pitch — o movimento de alcance.</p>
<div class="card"><canvas id="cArm"></canvas></div>

<h3>Latência de predição (ms/chunk)</h3>
<div class="card"><canvas id="cLat"></canvas></div>

<footer>
Fonte: <code>train/log/__RUNID__.log</code> (1ª ação clamp de cada chunk). Squeeze é proxy do index_0 (não o squeeze cru).
Gerado por <code>build_robot_run_html.py</code>. Pra replay com vídeo/atenção: <code>DRY_RUN=0 RECORD=1 ./start_inference.sh</code> (quando o --record estiver no script).
</footer>
</div>
<script>
const P = __PAYLOAD__;
function draw(id, series, opts){
  const c=document.getElementById(id), dpr=window.devicePixelRatio||1;
  const W=c.clientWidth,H=c.clientHeight; c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d"); x.scale(dpr,dpr);
  const pL=44,pR=12,pT=10,pB=22;
  const T=P.T, xmax=T[T.length-1]||1;
  let all=[]; series.forEach(s=>all=all.concat(s.data));
  let ymin=opts.ymin!==undefined?opts.ymin:Math.min(...all);
  let ymax=opts.ymax!==undefined?opts.ymax:Math.max(...all);
  const pad=(ymax-ymin)*0.1||0.1; if(opts.ymin===undefined)ymin-=pad; if(opts.ymax===undefined)ymax+=pad;
  const X=v=>pL+v/xmax*(W-pL-pR), Y=v=>pT+(1-(v-ymin)/(ymax-ymin))*(H-pT-pB);
  x.strokeStyle="#1c2940";x.fillStyle="#6b7c9c";x.font="10px system-ui";x.lineWidth=1;
  for(let k=0;k<=4;k++){const v=ymin+(ymax-ymin)*k/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(W-pR,y);x.stroke();x.fillText(v.toFixed(2),4,y+3);}
  for(let k=0;k<=4;k++){const tv=xmax*k/4;x.fillText(tv.toFixed(0)+"s",X(tv)-6,H-7);}
  series.forEach(s=>{
    x.strokeStyle=s.color;x.lineWidth=s.w||1.8;x.beginPath();
    s.data.forEach((v,j)=>{const px=X(T[j]),py=Y(v);j?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();
  });
  if(opts.legend){x.font="11px system-ui";let lx=pL+6;series.forEach(s=>{x.fillStyle=s.color;x.fillRect(lx,pT+2,10,3);x.fillStyle="#9fb2d4";x.fillText(s.name,lx+14,pT+8);lx+=14+x.measureText(s.name).width+18;});}
}
draw("cSq",[{data:P.SQ,color:"#38bdf8",name:"squeeze",w:2}],{ymin:0,ymax:1});
const colors={"kRightShoulderPitch.q":"#f59e0b","kRightElbow.q":"#34d399","kRightWristPitch.q":"#a78bfa"};
draw("cArm",P.ARM_KEYS.map(k=>({data:P.ARM[k],color:colors[k],name:k.replace("kRight","").replace(".q",""),w:1.6})),{legend:true});
draw("cLat",[{data:P.LAT,color:"#fb7185",name:"ms",w:1.4}],{ymin:0});
window.addEventListener("resize",()=>{draw("cSq",[{data:P.SQ,color:"#38bdf8",name:"squeeze",w:2}],{ymin:0,ymax:1});draw("cArm",P.ARM_KEYS.map(k=>({data:P.ARM[k],color:colors[k],name:k.replace("kRight","").replace(".q",""),w:1.6})),{legend:true});draw("cLat",[{data:P.LAT,color:"#fb7185",name:"ms",w:1.4}],{ymin:0});});
</script>
</body></html>"""

def pct(x): return round(100*x/n)
HTML = (HTML
  .replace("__GEN__", GEN_TS).replace("__RUNID__", RUN_ID)
  .replace("__N__", str(n)).replace("__DUR__", "%.0f"%dur)
  .replace("__LAT__", "%.0f"%lat_avg)
  .replace("__CLOSEDPCT__", str(pct(closed_n)))
  .replace("__SQCLS__", "warn" if pct(closed_n) > 70 else "ok")
  .replace("__FCT__", ("%.1f"%fc_t) if fc_t is not None else "—")
  .replace("__OPEN__", str(open_n)).replace("__CLOSED__", str(closed_n)).replace("__MID__", str(mid_n))
  .replace("__VERDICT__", verdict_close)
  .replace("__PAYLOAD__", json.dumps(payload)))

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True) if os.path.dirname(OUT) else None
open(OUT, "w").write(HTML)
print("escrito:", OUT, "(%.0f KB)" % (len(HTML)/1024))
print("stats: n=%d dur=%.0fs lat=%.0fms  open=%d closed=%d mid=%d  1stclose=%ss reopened=%s"
      % (n, dur, lat_avg, open_n, closed_n, mid_n, fc_t, reopened))
