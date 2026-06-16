#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera COMPARACAO_3RUNS.html — dashboard self-contained do A/B armstate7 com as 3 runs
(as-is 6kr7d8nz / valfix y32omum0 / ema 6ivtoov9) + a galeria de evidências do problema
do grasp (frames do deploy real, probes, replays interativos).

Fonte dos números: /tmp/three_runs.json (extraído do wandb prometheus-lcad/prometheus_g1).
Imagens pequenas vão embutidas em base64; vídeos e HTMLs pesados são LINKADOS (caminho relativo).

Uso: conda run -n g1_env python3 build_3runs_html.py
"""
import json, base64, os, statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "docs/diagnostico_treino_grasp/COMPARACAO_3RUNS.html")
DATA = json.load(open("/tmp/three_runs.json"))

GEN_TS = "2026-06-16 11:15 (-03)"   # passado fixo (Date.now indisponível no harness)

def b64img(relpath):
    p = os.path.join(ROOT, relpath)
    ext = relpath.rsplit(".",1)[-1].lower()
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}[ext]
    with open(p, "rb") as f:
        return "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode())

# caminho relativo do HTML (em docs/diagnostico_treino_grasp/) para os artefatos pesados
def rel(relpath_from_root):
    return os.path.relpath(os.path.join(ROOT, relpath_from_root), os.path.dirname(OUT))

# ---------- estatísticas ----------
def window(steps, vals, lo, hi):
    return [v for s,v in zip(steps,vals) if v is not None and lo<=s<=hi]
def cv(xs):
    m=st.mean(xs); return 100*st.pstdev(xs)/m if m else 0
def maxjump(steps,vals,lo,hi):
    pts=[(s,v) for s,v in zip(steps,vals) if v is not None and lo<=s<=hi]
    return max(abs(pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1)) if len(pts)>1 else 0
def best(steps,vals):
    pts=[(s,v) for s,v in zip(steps,vals) if v is not None]
    return min(pts,key=lambda x:x[1])

LO, HI = 3000, 9000   # janela comum de regime (EMA só vai a 9.3k)

def metric_series(tag, base_key):
    r=DATA[tag]; S=r["steps"]
    key = base_key + ("_ema" if tag=="ema" else "")
    return S, r["data"].get(key)

RUNMETA = {
  "asis":   {"label":"RUN 1 · as-is",   "color":"#94a3b8", "rid":"6kr7d8nz",
             "sub":"régua antiga · Atena GPU2", "state":"finished 20k"},
  "valfix": {"label":"RUN 2 · val-fixes","color":"#38bdf8", "rid":"y32omum0",
             "sub":"régua nova · TRX50 GPU0", "state":"finished 20k"},
  "ema":    {"label":"RUN 3 · EMA",      "color":"#f59e0b", "rid":"6ivtoov9",
             "sub":"val-fixes + EMA · TRX50 GPU0", "state":"crashed 9.3k"},
}

METRICS = [
  ("eval/val_loss",              "val_loss",              "proxy de flow-matching (não decide o best)"),
  ("eval/val_action_mse",        "val_action_mse (geral)","métrica que DECIDE o best"),
  ("eval/val_action_mse_arm",    "braço (dims 0–6)",      "o braço APRENDE — cai e generaliza"),
  ("eval/val_action_mse_grasp",  "grasp (dim 7)",         "aqui mora o problema (escalar quase-binário)"),
]

# ---------- monta dados p/ os gráficos (JS) ----------
chart_payload = {}
for base_key,_,_ in METRICS:
    series={}
    for tag in ("asis","valfix","ema"):
        S,V = metric_series(tag, base_key)
        if V is None: continue
        series[tag]=[[s,v] for s,v in zip(S,V) if v is not None]
    chart_payload[base_key]=series

# ---------- tabela de estabilidade (régua) ----------
def stab_row(base_key):
    cells={}
    for tag in ("asis","valfix","ema"):
        S,V=metric_series(tag,base_key)
        if V is None: cells[tag]=None; continue
        bs,bv=best(S,V)
        cells[tag]={"best_s":bs,"best_v":bv,"cv":cv(window(S,V,LO,HI)),"jump":maxjump(S,V,LO,HI)}
    return cells

# ====================================================================
# HTML
# ====================================================================
def fmt(x,d=4): return ("%."+str(d)+"f")%x if x is not None else "—"

# galeria de frames do robô
ROBO = [
 ("14-dim best@5500 — a mão CICLA e nunca agarra", [
    ("run1_14dim_inicio.jpg","Início: copo na mesa, mão aberta no HOME"),
    ("run1_14dim_humano_apresenta_copo.jpg","Humano teve que OFERECER o copo (a mão cicla, não agarra)"),
    ("run1_14dim_fim_braco_recolhido.jpg","Fim: braço recolhido, sem pega"),
 ]),
 ("8-dim best@8000 — a mão fecha, mas FORA do alvo", [
    ("run2_8dim_inicio.jpg","Início: reach já mira o canto errado"),
    ("run2_8dim_mao_fecha_fora_do_alvo.jpg","Mão fecha no canto inf-esq, ~15–20 cm do copo"),
    ("run2_8dim_fim.jpg","Fim: braço recolhido, copo na mesa"),
 ]),
]
robo_html=""
for titulo, frames in ROBO:
    cards="".join(
      '<figure><img src="%s"><figcaption>%s</figcaption></figure>'%(b64img("docs/investigacao_deploy_robo/midia/"+fn),cap)
      for fn,cap in frames)
    robo_html+='<div class="robo-block"><h4>%s</h4><div class="strip">%s</div></div>'%(titulo,cards)

# analysis PNGs das 3 runs (deploy offline)
ANALYSIS=[
 ("asis","docs/probs/deploy_asis_run20260615_150001_analysis.png","docs/probs/deploy_asis_run20260615_150001.html"),
 ("valfix","docs/probs/deploy_valfix_run20260615_165119_analysis.png","docs/probs/deploy_valfix_run20260615_165119.html"),
 ("ema","docs/probs/deploy_ema_run20260615_160710_analysis.png","docs/probs/deploy_ema_run20260615_160710.html"),
]
analysis_html=""
for tag,png,htm in ANALYSIS:
    m=RUNMETA[tag]
    analysis_html+='<figure class="ana"><a href="%s" target="_blank"><img src="%s"></a><figcaption><b style="color:%s">%s</b> — análise do deploy no robô real · <a href="%s" target="_blank">replay interativo ↗</a></figcaption></figure>'%(
        rel(htm), b64img(png), m["color"], m["label"], rel(htm))

# vídeos MP4 do deploy real das 3 runs (gerados de /tmp/frames2mp4.py)
VIDS=[
 ("asis",  "docs/investigacao_deploy_robo/videos/run_asis_rollout.mp4",  "41 s · 327 frames @8fps"),
 ("valfix","docs/investigacao_deploy_robo/videos/run_valfix_rollout.mp4","33 s · 267 frames @8fps"),
 ("ema",   "docs/investigacao_deploy_robo/videos/run_ema_rollout.mp4",   "84 s · 676 frames @8fps"),
]
vids_html=""
for tag,mp4,meta in VIDS:
    m=RUNMETA[tag]
    vids_html+='<figure class="vid"><video src="%s" controls preload="metadata"></video><figcaption><b style="color:%s">%s</b> — head_camera do robô · %s</figcaption></figure>'%(
        rel(mp4), m["color"], m["label"], meta)

# links de evidência (replays/saliência/vídeos)
LINKS=[
 ("Pôster de saliência (a imagem é usada? onde?)", "docs/diagnostico_treino_grasp/RESULTADO_VISUAL_SALIENCIA.html",
  "Sensibilidade à imagem 0,014→0,693 ao tirar os dedos do state; foco passa a ser o COPO."),
 ("Replay ep214 — SEEDADO vs NÃO-seedado", "docs/probs/replay_ep214_valfix_SEEDADO.html",
  "O jitter da traj predita cai 89% (0,043→0,005 rad) só fixando o ruído do amostrador — é o que a régua conserta."),
 ("Replay ep214 — NÃO-seedado (ruído solto)", "docs/probs/replay_ep214_valfix_NAOseedado.html",
  "A versão antiga da régua: ruído re-sorteado a cada eval inflava a oscilação do grasp."),
 ("Replay ep220 — falso disparo do grasp", "docs/probs/replay_ep220_falso_disparo.html",
  "Caso de squeeze predito que não bate com o GT."),
 ("RUN3: raw × EMA (pareado, mesma trajetória)", "docs/diagnostico_treino_grasp/run3_ema/RUN3_EMA.html",
  "A comparação mais limpa do efeito da EMA — sem confound de máquina/run-a-run."),
 ("Vídeo robô — modelo ANTIGO 14-dim cicla (MP4)", "docs/investigacao_deploy_robo/midia/run1_14dim_best5500_robo.mp4",
  "Pré-armstate7 (não é nenhuma das 3 runs): a mão abre/fecha 7× e nunca agarra."),
 ("Vídeo robô — modelo ANTIGO 8-dim fecha fora do alvo (MP4)", "docs/investigacao_deploy_robo/midia/run2_8dim_best_robo.mp4",
  "Pré-armstate7: fecha uma vez e segura, mas o braço não chegou ao copo."),
]
links_html="".join(
  '<a class="evlink" href="%s" target="_blank"><span class="evt">%s</span><span class="evd">%s</span></a>'%(rel(p),t,d)
  for t,p,d in LINKS)

# ---------- tabela das runs ----------
def best_of(tag,base_key):
    S,V=metric_series(tag,base_key)
    if V is None: return "—"
    bs,bv=best(S,V); return "%.4f <small>@%d</small>"%(bv,bs)

runtable=""
for tag in ("asis","valfix","ema"):
    m=RUNMETA[tag]; r=DATA[tag]
    runtable+="""<tr>
      <td><span class="dot" style="background:%s"></span><b>%s</b><br><small>%s</small></td>
      <td><code>%s</code></td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
      <td>%s</td>
    </tr>"""%(m["color"],m["label"],m["sub"],m["rid"],
             ('<span class="ok">finished 20k</span>' if r["state"]=="finished" else '<span class="warn">crashed @%d</span>'%r["last_step"]),
             best_of(tag,"eval/val_action_mse"),
             best_of(tag,"eval/val_action_mse_arm"),
             best_of(tag,"eval/val_action_mse_grasp"))

# ---------- tabela régua (estabilidade) ----------
def stabtable():
    rows=""
    for base_key,label,_ in METRICS:
        c=stab_row(base_key)
        def cell(tag):
            v=c.get(tag)
            if not v: return '<td class="na">—</td>'
            return '<td>CV <b>%.1f%%</b><br><small>salto %.3f</small></td>'%(v["cv"],v["jump"])
        rows+='<tr><td class="ml">%s</td>%s%s%s</tr>'%(label,cell("asis"),cell("valfix"),cell("ema"))
    return rows

HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A/B armstate7 — 3 runs + evidências do grasp</title>
<style>
:root{--bg:#0b1020;--card:#121a2e;--ink:#e8eefc;--mut:#8aa0c6;--line:#23304d;
 --asis:#94a3b8;--valfix:#38bdf8;--ema:#f59e0b;--ok:#34d399;--warn:#fb7185;}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0b1020,#0a0e1c);color:var(--ink);
 font:15px/1.55 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:21px;margin:46px 0 14px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;color:var(--mut);font-weight:600;margin:24px 0 10px;text-transform:uppercase;letter-spacing:.6px}
.sub{color:var(--mut);margin:0 0 4px}
.tag{display:inline-block;font-size:12px;color:var(--mut);border:1px solid var(--line);
 border-radius:999px;padding:2px 10px;margin-right:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:14px 0}
.lede{font-size:16px;background:linear-gradient(135deg,#13203c,#101830);border-left:3px solid var(--valfix)}
.lede b{color:#fff}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
td small{color:var(--mut)} code{color:#cbd5e1;font-size:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
.ok{color:var(--ok)} .warn{color:var(--warn)} .na{color:#54627e}
td.ml,td.ml b{color:var(--ink)} .ml{font-weight:600}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.chart-card h4{margin:0 0 2px;font-size:15px}.chart-card .hint{color:var(--mut);font-size:12.5px;margin:0 0 8px}
canvas{width:100%;height:210px;display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);margin:10px 0 26px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
.delta{font-size:13px;background:#0e1830;border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-top:10px;color:var(--mut)}
.delta b{color:#fff}
.robo-block{margin:18px 0}.robo-block h4{margin:0 0 8px;font-size:15px;color:#dbe6ff}
.strip{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
@media(max-width:720px){.strip{grid-template-columns:1fr}}
figure{margin:0;background:#0e1526;border:1px solid var(--line);border-radius:10px;overflow:hidden}
figure img{width:100%;display:block}
figcaption{font-size:12px;color:var(--mut);padding:8px 10px;line-height:1.4}
.ana img{cursor:zoom-in}
.vids{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:8px}
@media(max-width:720px){.vids{grid-template-columns:1fr}}
.vid video{width:100%;display:block;background:#000;border-radius:8px 8px 0 0}
.evlinks{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:720px){.evlinks{grid-template-columns:1fr}}
.evlink{display:block;background:#0e1526;border:1px solid var(--line);border-radius:10px;padding:12px 14px;
 text-decoration:none;color:var(--ink);transition:.15s}
.evlink:hover{border-color:var(--valfix);background:#11203a}
.evt{display:block;font-weight:600;font-size:14px;margin-bottom:3px}
.evd{display:block;font-size:12.5px;color:var(--mut)}
footer{margin-top:50px;color:#5b6b89;font-size:12.5px;border-top:1px solid var(--line);padding-top:16px}
a{color:#7cc4ff}
</style></head><body><div class="wrap">

<h1>A/B armstate7 — as 3 runs &amp; as evidências do grasp</h1>
<p class="sub">Gerado em __GEN__ · fonte: wandb <code>prometheus-lcad/prometheus_g1</code> (números reais, scan_history)</p>
<p><span class="tag">RUN 1 · as-is · 6kr7d8nz</span><span class="tag">RUN 2 · val-fixes · y32omum0</span><span class="tag">RUN 3 · EMA · 6ivtoov9</span></p>

<div class="card lede">
<b>Em uma frase:</b> a régua nova (RUN 2) e a EMA (RUN 3) deixam a curva do grasp
<b>muito mais estável</b> (CV 25%→7%→4%), mas <b>não melhoram o valor</b> do grasp
(~0,31 nas três) — porque o problema não é de medição: <b>a mão fecha pela propriocepção, cega à imagem</b>.
As três runs treinam o mesmo modelo; o que muda é como medimos e (na RUN 3) a média EMA dos pesos.
</div>

<h2>1 · As 3 runs lado a lado</h2>
<div class="card"><table>
<tr><th>run</th><th>wandb</th><th>estado</th><th>best val_mse</th><th>best braço</th><th>best grasp</th></tr>
__RUNTABLE__
</table>
<p class="sub" style="margin-top:12px;font-size:13px">⚠️ A <b>RUN 3 (EMA) crashou em 9.300 steps</b> — não fechou os 20k. As comparações abaixo usam a janela comum <b>3k–9k</b> pra ser justa com a EMA. <code>best @step</code> é o mínimo de cada série.</p>
</div>

<h2>2 · As curvas (números reais do wandb)</h2>
<div class="legend">
 <span><i style="background:var(--asis)"></i>RUN 1 as-is</span>
 <span><i style="background:var(--valfix)"></i>RUN 2 val-fixes</span>
 <span><i style="background:var(--ema)"></i>RUN 3 EMA (até 9.3k)</span>
</div>
<div class="grid2">__CHARTS__</div>
<div class="delta">
<b>Como ler:</b> o <b>val_loss</b> é gêmeo nas três (os fixes não mudam o que a rede aprende) e o
<b>braço</b> cai e generaliza em todas. O <b>grasp</b> é o que oscilava — e é exatamente onde a régua
e a EMA agem: a linha fica lisa, mas o patamar não baixa.
</div>

<h2>3 · A régua: estabilidade na janela 3k–9k</h2>
<div class="card"><table>
<tr><th>métrica</th><th style="color:var(--asis)">RUN 1 as-is</th><th style="color:var(--valfix)">RUN 2 val-fixes</th><th style="color:var(--ema)">RUN 3 EMA</th></tr>
__STABTABLE__
</table>
<div class="delta" style="margin-top:14px">
<b>O ganho é confiabilidade, não valor.</b> No <b>grasp</b>: CV <b>25,0% → 7,4% → 4,2%</b> e salto máximo
<b>0,225 → 0,081 → 0,030</b>. O <b>val_loss</b> já era estável (média de muitas dims contínuas) e quase
não muda — o ruído só sacaneava o grasp (1 escalar quase-binário). A régua conserta exatamente a parte quebrada;
a EMA aperta mais ainda. O número do grasp <b>sobe</b> de as-is→valfix (0,18→0,31) porque a régua curta
de 4 batches pescava um vale sortudo — o 0,31 é o honesto.
</div></div>

<h2>4 · Por que o grasp não melhora — as evidências</h2>
<p class="sub">As probes abaixo demonstram a causa-raiz: <b>o fechamento da mão é ditado pela propriocepção (sobretudo os dedos medidos), não pela visão.</b></p>

<div class="card">
<h3>Probe de propriocepção (drop-proprio)</h3>
<p style="margin:0 0 6px">Trocar o <code>state</code> de um frame fechado pelo de um aberto <b>inverte</b> a decisão da mão (separação +0,95→−0,88 no 8-dim; +0,99→−0,99 no 14-dim) — com a imagem real intacta. Zerar <b>só os dedos medidos</b> colapsa a separação (+0,95→+0,11); zerar <b>só o braço</b> quase não muda (+0,74). → a mão <b>ecoa o que os dedos já fazem</b>.</p>
</div>
<div class="card">
<h3>Probe de grounding (ablação de imagem)</h3>
<p style="margin:0 0 6px">Zerar ou trocar a imagem quase não move o squeeze: sensibilidade à imagem = <b>0,145</b> (8-dim) e <b>0,010</b> (14-dim, ≈ cego). A correção <b>armstate7</b> (tirar os dedos do state) elevou a sensibilidade à imagem de <b>0,014 → 0,693</b> — é o que estas 3 runs testam.</p>
</div>

<h3>Deploy real no robô — o sintoma</h3>
__ROBO__

<h3>Vídeo do deploy no robô real — as 3 runs</h3>
<p class="sub">Head-camera durante a inferência no robô (frames extraídos dos replays). A run3/EMA <b>foi</b> deployada (checkpoint anterior ao crash @9.3k).</p>
<div class="vids">__VIDS__</div>

<h3>Análise do rollout das 3 runs (clique p/ replay interativo)</h3>
<div class="grid2">__ANALYSIS__</div>

<h3>Mais evidências interativas</h3>
<div class="evlinks">__LINKS__</div>

<footer>
Self-contained: frames e PNGs de análise estão embutidos (base64); vídeos MP4, replays e o pôster de saliência
abrem por link relativo (mantenha a árvore <code>docs/</code> ao mover o arquivo).<br>
Reprodução das runs: <code>docs/diagnostico_treino_grasp/REPRODUTIBILIDADE_AB_3RUNS.md</code> ·
diagnóstico completo: <code>docs/diagnostico_treino_grasp/00_INDEX.md</code>.<br>
Régua = protocolo de validação (policy.eval, ruído+timestep fixos, val_action_mse em 16 batches). EMA decay 0.999.
</footer>
</div>

<script>
const PAYLOAD = __PAYLOAD__;
const COLORS = {asis:"#94a3b8", valfix:"#38bdf8", ema:"#f59e0b"};
const ORDER = ["asis","valfix","ema"];
function draw(canvas, series){
  const dpr = window.devicePixelRatio||1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width=W*dpr; canvas.height=H*dpr;
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
  const padL=46,padR=12,padT=12,padB=24;
  let xs=[],ys=[];
  ORDER.forEach(t=>{(series[t]||[]).forEach(p=>{xs.push(p[0]);ys.push(p[1]);});});
  if(!xs.length) return;
  const xmin=0, xmax=20000;
  let ymin=Math.min(...ys), ymax=Math.max(...ys);
  const pad=(ymax-ymin)*0.12||0.01; ymin-=pad; ymax+=pad; if(ymin<0)ymin=0;
  const X=v=>padL+(v-xmin)/(xmax-xmin)*(W-padL-padR);
  const Y=v=>padT+(1-(v-ymin)/(ymax-ymin))*(H-padT-padB);
  // grid + y labels
  ctx.strokeStyle="#1c2940"; ctx.fillStyle="#6b7c9c"; ctx.font="10px system-ui"; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4, y=Y(v);
    ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke();
    ctx.fillText(v.toFixed(3),4,y+3);}
  // x labels
  [0,5000,10000,15000,20000].forEach(xv=>{ctx.fillText((xv/1000)+"k",X(xv)-6,H-8);});
  // 3k-9k window shade
  ctx.fillStyle="rgba(56,189,248,.05)";
  ctx.fillRect(X(3000),padT,X(9000)-X(3000),H-padT-padB);
  // lines
  ORDER.forEach(t=>{
    const pts=series[t]; if(!pts||!pts.length)return;
    ctx.strokeStyle=COLORS[t]; ctx.lineWidth=t==="valfix"?2.2:1.8;
    ctx.beginPath();
    pts.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.stroke();
    ctx.fillStyle=COLORS[t];
    pts.forEach(p=>{ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),1.6,0,7);ctx.fill();});
  });
}
function renderAll(){
  document.querySelectorAll("canvas[data-key]").forEach(c=>{
    draw(c, PAYLOAD[c.dataset.key]||{});
  });
}
window.addEventListener("resize",renderAll);
renderAll();
</script>
</body></html>"""

# charts blocks
charts=""
for base_key,label,hint in METRICS:
    charts+='<div class="card chart-card"><h4>%s</h4><p class="hint">%s</p><canvas data-key="%s"></canvas></div>'%(
        label,hint,base_key)

HTML = (HTML
  .replace("__GEN__",GEN_TS)
  .replace("__RUNTABLE__",runtable)
  .replace("__CHARTS__",charts)
  .replace("__STABTABLE__",stabtable())
  .replace("__ROBO__",robo_html)
  .replace("__VIDS__",vids_html)
  .replace("__ANALYSIS__",analysis_html)
  .replace("__LINKS__",links_html)
  .replace("__PAYLOAD__",json.dumps(chart_payload)))

with open(OUT,"w") as f:
    f.write(HTML)
print("escrito:",OUT,"(%.1f KB)"%(len(HTML)/1024))
