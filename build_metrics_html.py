#!/usr/bin/env python3
# Gera um HTML explicando cada bloco de TRAIN e EVAL (o que significam) com os DADOS
# REAIS do run 6kr7d8nz (armstate7-8k), puxados do wandb em /tmp/run_data.json.
import json, html
from pathlib import Path

D = json.load(open("/tmp/run_data.json"))
S, DIMS, M = D["series"], D["dims"], D["meta"]
OUT = Path("docs/diagnostico_treino_grasp/METRICAS_EXPLICADAS.html")

def js(x): return json.dumps(x)
ARM = ['Sh·Pitch','Sh·Roll','Sh·Yaw','Elbow','Wr·Roll','Wr·Pitch','Wr·Yaw']

HTML = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Métricas de Treino & Eval — o que cada bloco significa</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#070b11;--bg2:#0a0f17;--card:#0d131c;--card2:#111925;--line:#1b2532;--line2:#26344a;
 --ink:#dce6f1;--ink2:#8ea0b6;--ink3:#5c6c80;--cyan:#22d3ee;--amber:#ff9e3d;--green:#34d399;--red:#f2607a;
 --filt:#f59e0b;--disp:'Chakra Petch',sans-serif;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;font-size:15px;
 background-image:radial-gradient(900px 500px at 80% -8%,rgba(34,211,238,.06),transparent 60%),radial-gradient(800px 500px at 0% 108%,rgba(255,158,61,.05),transparent 55%);}}
.wrap{{max-width:1080px;margin:0 auto;padding:34px 22px 80px}}
h1{{font-family:var(--disp);font-weight:700;font-size:32px;letter-spacing:.01em;line-height:1.1;margin-bottom:6px}}
h1 .b{{color:var(--cyan)}} h1 .a{{color:var(--amber)}}
.lead{{color:var(--ink2);font-size:15.5px;max-width:760px;margin-bottom:6px}}
.runbar{{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 10px}}
.tag{{font-family:var(--mono);font-size:11.5px;color:var(--ink2);background:var(--bg2);border:1px solid var(--line);border-radius:7px;padding:5px 10px}}
.tag b{{color:var(--ink)}} .tag.warn{{color:var(--amber);border-color:rgba(255,158,61,.35)}}
h2{{font-family:var(--disp);font-weight:700;font-size:21px;letter-spacing:.06em;text-transform:uppercase;margin:38px 0 4px;
 display:flex;align-items:center;gap:11px}}
h2 i{{width:4px;height:20px;border-radius:2px;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}}
h2.ev i{{background:var(--amber);box-shadow:0 0 10px var(--amber)}}
.sub{{color:var(--ink3);font-family:var(--mono);font-size:12px;margin-bottom:16px;letter-spacing:.02em}}
.card{{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:13px;
 padding:16px 18px;margin:12px 0;box-shadow:0 1px 0 rgba(255,255,255,.025) inset,0 12px 30px rgba(0,0,0,.28)}}
.card h3{{font-family:var(--mono);font-size:14px;font-weight:600;color:var(--ink);margin-bottom:3px;letter-spacing:.01em}}
.card h3 .k{{color:var(--cyan)}} .card.amber h3 .k{{color:var(--amber)}}
.card .val{{float:right;font-family:var(--mono);font-weight:600;font-size:15px;color:var(--ink)}}
.card p{{color:var(--ink2);font-size:14px;margin-top:6px}}
.card p b{{color:var(--ink)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.chartbox{{margin-top:12px;background:#05080d;border:1px solid var(--line);border-radius:9px;padding:10px 10px 4px}}
.chartttl{{font-family:var(--mono);font-size:11px;color:var(--ink3);margin-bottom:4px;display:flex;justify-content:space-between}}
.leg{{display:flex;gap:14px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;color:var(--ink2);margin:6px 0 2px}}
.leg i{{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}}
svg{{display:block;width:100%}}
.call{{border-left:3px solid var(--cyan);background:rgba(34,211,238,.05);padding:12px 15px;border-radius:0 9px 9px 0;margin:14px 0;font-size:14.5px}}
.call.amber{{border-color:var(--amber);background:rgba(255,158,61,.05)}}
.call b{{color:var(--ink)}}
.foot{{color:var(--ink3);font-family:var(--mono);font-size:11.5px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}}
.hl{{color:var(--cyan);font-weight:600}} .hla{{color:var(--amber);font-weight:600}} .hlg{{color:var(--green);font-weight:600}} .hlr{{color:var(--red);font-weight:600}}
</style></head><body><div class="wrap">

<h1>Métricas de <span class="b">treino</span> &amp; <span class="a">eval</span> — o que cada bloco significa</h1>
<p class="lead">Lendo cada bloco que o wandb registra, com os <b>números reais</b> do run desta noite. O π0.5 aprende por <b>flow-matching</b> (um campo de velocidade que leva ruído → ação); quase tudo gira em torno disso.</p>
<div class="runbar">
 <span class="tag">run <b>cup_pi05_right8_armstate7_lf</b> · 6kr7d8nz</span>
 <span class="tag">batch <b>{M['batch']}</b> · lr₀ <b>1e-4</b></span>
 <span class="tag">treino 214 eps · val 24 eps</span>
 <span class="tag warn">crashou no step <b>{M['last_step']}</b>/{M['steps_planned']} (queda do lab)</span>
 <span class="tag">best @ step <b>{M['best_step']}</b> · val_action_mse <b>{M['best_vmse']:.3f}</b></span>
</div>

<h2><i></i>Bloco TRAIN — o aprendizado a cada passo</h2>
<div class="sub">prefixo <b>train/</b> · medido no batch de treino, a cada passo do otimizador</div>

<div class="card">
 <h3><span class="k">train/loss</span><span class="val">{S['train/loss'][-1][1]:.4f}</span></h3>
 <p>A <b>perda de flow-matching</b>: o MSE entre a velocidade que o modelo prevê e a velocidade-alvo (que leva ruído→ação do dataset). É o que o otimizador minimiza. <b>Cair e estabilizar baixo</b> = o modelo está aprendendo a reproduzir as ações demonstradas. Aqui caiu rápido e estabilizou em ~<span class="hl">0,013</span>.</p>
 <div class="chartbox"><div class="chartttl"><span>train/loss × step</span><span id="t_loss_last"></span></div><svg id="c_loss" viewBox="0 0 600 150"></svg></div>
</div>

<div class="grid2">
 <div class="card"><h3><span class="k">train/lr</span><span class="val">{S['train/lr'][-1][1]:.2g}</span></h3>
  <p>Taxa de aprendizado — agenda <b>cosseno</b> decaindo de 1e-4. Passo grande no começo, fino no fim pra assentar.</p>
  <div class="chartbox"><div class="chartttl"><span>train/lr × step</span></div><svg id="c_lr" viewBox="0 0 600 110"></svg></div></div>
 <div class="card"><h3><span class="k">train/grad_norm</span><span class="val">{S['train/grad_norm'][-1][1]:.3f}</span></h3>
  <p>Norma do gradiente (clipada em 1,0). Mede <b>estabilidade</b>; picos = instabilidade. Aqui ~<span class="hl">0,17</span>, bem comportado.</p>
  <div class="chartbox"><div class="chartttl"><span>train/grad_norm × step</span></div><svg id="c_gn" viewBox="0 0 600 110"></svg></div></div>
</div>

<div class="card">
 <h3>train/epochs · samples · update_s</h3>
 <p>Contabilidade do treino: passou <b>~8,3 épocas</b> pelo dataset (cada época = ver todas as 214 demos), ~<b>397k amostras</b>, a <b>~1,85 s/passo</b> (≈ por isso 20k steps ≈ 10-17h). Não são métricas de qualidade — são de progresso/custo.</p>
</div>

<div class="card">
 <h3><span class="k">dim_train/loss_dim_XX</span></h3>
 <p>A <b>mesma loss decomposta por dimensão da ação</b> (0-6 = juntas do braço; 7 = squeeze da mão). Mostra <b>qual saída é mais difícil de prever</b>. Aqui todas baixas (~0,010-0,014); o <span class="hla">squeeze (dim 7) = {DIMS['vloss_07'] and ''}{S['dim_eval/val_loss_dim_07'] and ''}~0,018</span>, levemente maior — já é a dimensão mais difícil mesmo no treino.</p>
</div>

<h2 class="ev"><i></i>Bloco EVAL — generalização (e a sutileza dos 2 tipos)</h2>
<div class="sub">prefixo <b>eval/</b> · medido em 24 episódios NÃO vistos no treino, a cada eval</div>

<div class="call amber"><b>⚠️ O ponto que mais engana:</b> há <b>DOIS tipos</b> de métrica de eval. Uma <b>SOBE</b> (e parece que o modelo piora), a outra é a que <b>realmente importa</b>. Confundir as duas leva a conclusões erradas.</div>

<div class="card amber">
 <h3><span class="k">eval/val_loss</span><span class="val">{S['eval/val_loss'][-1][1]:.3f}</span></h3>
 <p>A <b>mesma loss de flow-matching, mas nos dados de validação</b>. É uma <b>PROXY frouxa</b>: ela mede o ajuste do campo de velocidade num ponto aleatório do caminho ruído→ação — não a ação final. <b>Pode SUBIR</b> (leve overfit do campo) <b>enquanto o modelo continua melhorando na ação que importa</b>. Não entre em pânico quando ela sobe. Aqui: ~<span class="hla">0,15, subindo</span>.</p>
</div>

<div class="card">
 <h3><span class="k">eval/val_action_mse</span><span class="val">{S['eval/val_action_mse'][-1][1]:.3f}</span></h3>
 <p>O erro da <b>AÇÃO DE FATO GERADA</b>: roda o <code>predict_action_chunk</code> inteiro (flow-matching até o fim) e compara com a ação real, em espaço normalizado. <b>É isto que o robô faria</b> → a métrica que vale. Cair = melhor. Aqui ~<span class="hl">0,064</span> (média de todas as 8 dims). O <b>best</b> do run foi por esta métrica (step 8000, 0,052).</p>
 <div class="leg"><span><i style="background:var(--amber)"></i>val_loss (proxy, sobe)</span><span><i style="background:var(--cyan)"></i>val_action_mse (média)</span><span><i style="background:var(--green)"></i>·_arm (braço)</span><span><i style="background:var(--red)"></i>·_grasp (squeeze)</span></div>
 <div class="chartbox"><div class="chartttl"><span>eval × step — a DESCOLAGEM</span></div><svg id="c_eval" viewBox="0 0 600 200"></svg></div>
 <div class="call"><b>A descolagem (o gráfico acima):</b> a <span class="hla">val_loss sobe</span>, mas a <span class="hlg">val_action_mse do braço cai/fica baixa (~0,039)</span>. Ou seja: a proxy assusta, mas a <b>ação gerada do braço melhora</b>. Olhe a <b>val_action_mse</b> (e o split abaixo), nunca só a val_loss.</div>
</div>

<div class="grid2">
 <div class="card"><h3><span class="k">eval/val_action_mse_arm</span><span class="val" style="color:var(--green)">{S['eval/val_action_mse_arm'][-1][1]:.3f}</span></h3>
  <p>Só as <b>7 juntas do braço</b>. <span class="hlg">Baixo (~0,039)</span> → o braço <b>generaliza bem</b> (alcança o copo em episódios novos).</p></div>
 <div class="card amber"><h3><span class="k">eval/val_action_mse_grasp</span><span class="val" style="color:var(--red)">{S['eval/val_action_mse_grasp'][-1][1]:.3f}</span></h3>
  <p>Só o <b>squeeze (dim 7, a mão)</b>. <span class="hlr">Alto (~0,238)</span> → a <b>pega é o gargalo</b> (≈6× o erro do braço). Bate com todo o diagnóstico desta frente.</p></div>
</div>

<div class="card">
 <h3><span class="k">dim_eval/val_action_mse_dim_XX</span> — por dimensão</h3>
 <p>A val_action_mse aberta por dimensão da ação. O <b>braço (0-6) fica entre 0,014 e 0,079</b>; o <span class="hlr">squeeze (dim 7) destoa em 0,238</span> — visivelmente o ponto fraco. (No <code>val_loss</code> por-dim o squeeze também lidera: 0,42.)</p>
 <div class="chartbox"><div class="chartttl"><span>val_action_mse por dimensão (final)</span><span>0-6 braço · 7 squeeze</span></div><svg id="c_bars" viewBox="0 0 600 180"></svg></div>
</div>

<div class="call"><b>Como ler tudo junto:</b> 1) <b>train/loss</b> baixa = aprende. 2) <b>val_loss</b> subindo NÃO é alarme — é proxy frouxa. 3) <b>val_action_mse</b> (e o split <b>arm</b> vs <b>grasp</b>) é o veredito real: o <span class="hlg">braço aprende</span>, a <span class="hlr">pega é o gargalo</span>. 4) o <b>best</b> é o menor val_action_mse (step 8000).</div>

<div class="foot">
 Fonte: wandb run <b>prometheus-lcad/prometheus_g1/6kr7d8nz</b> (puxado via API). Estado: <b>crashed</b> @ step {M['last_step']} (queda de rede/energia no lab — retoma do checkpoint <code>last</code> quando a Atena voltar).<br>
 Organização dos blocos (train/ · eval/ · dim_train/ · dim_eval/ · split arm/grasp) configurada em <code>lerobot-ext/train/run_train.py</code>.
</div>

</div>
<script>
const DATA = {{
 loss: {js(S['train/loss'])}, lr: {js(S['train/lr'])}, gn: {js(S['train/grad_norm'])},
 vloss: {js(S['eval/val_loss'])}, vmse: {js(S['eval/val_action_mse'])},
 varm: {js(S['eval/val_action_mse_arm'])}, vgrasp: {js(S['eval/val_action_mse_grasp'])},
 dims: {js([DIMS[f'vmse_{i:02d}'] for i in range(8)])}
}};
const W=600, PADL=44, PADR=10, PADT=10, PADB=22;
function lineChart(id, series, color, h, opts){{
 opts=opts||{{}}; const svg=document.getElementById(id); if(!svg)return;
 const H=h, all=[].concat(...series.map(s=>s.data));
 let ymin=opts.ymin!=null?opts.ymin:Math.min(...all.map(p=>p[1]));
 let ymax=opts.ymax!=null?opts.ymax:Math.max(...all.map(p=>p[1]));
 if(ymax===ymin)ymax=ymin+1;
 const xs=all.map(p=>p[0]), xmin=Math.min(...xs), xmax=Math.max(...xs);
 const X=s=>PADL+(s-xmin)/(xmax-xmin||1)*(W-PADL-PADR);
 const Y=v=>PADT+(1-(v-ymin)/(ymax-ymin))*(H-PADT-PADB);
 let g='';
 // grid y
 for(let i=0;i<=3;i++){{const v=ymin+(ymax-ymin)*i/3, y=Y(v);
  g+=`<line x1="${{PADL}}" y1="${{y.toFixed(1)}}" x2="${{W-PADR}}" y2="${{y.toFixed(1)}}" stroke="#1b2532" stroke-width="1"/>`;
  g+=`<text x="${{PADL-6}}" y="${{(y+3).toFixed(1)}}" fill="#5c6c80" font-size="9" font-family="IBM Plex Mono" text-anchor="end">${{v.toFixed(opts.dec||2)}}</text>`;}}
 // x labels
 [xmin,Math.round((xmin+xmax)/2),xmax].forEach(s=>{{g+=`<text x="${{X(s).toFixed(1)}}" y="${{H-6}}" fill="#5c6c80" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">${{(s/1000).toFixed(0)}}k</text>`;}});
 series.forEach(s=>{{
  const pts=s.data.map(p=>`${{X(p[0]).toFixed(1)}},${{Y(p[1]).toFixed(1)}}`).join(' ');
  g+=`<polyline points="${{pts}}" fill="none" stroke="${{s.color}}" stroke-width="2" stroke-linejoin="round"/>`;
  const last=s.data[s.data.length-1]; g+=`<circle cx="${{X(last[0]).toFixed(1)}}" cy="${{Y(last[1]).toFixed(1)}}" r="3" fill="${{s.color}}"/>`;
 }});
 svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`); svg.innerHTML=g;
}}
lineChart('c_loss',[{{data:DATA.loss,color:'#22d3ee'}}],'#22d3ee',150,{{dec:2,ymin:0}});
lineChart('c_lr',[{{data:DATA.lr,color:'#8ea0b6'}}],'#8ea0b6',110,{{dec:5,ymin:0}});
lineChart('c_gn',[{{data:DATA.gn,color:'#8ea0b6'}}],'#8ea0b6',110,{{dec:2,ymin:0}});
lineChart('c_eval',[
 {{data:DATA.vloss,color:'#ff9e3d'}},{{data:DATA.vmse,color:'#22d3ee'}},
 {{data:DATA.varm,color:'#34d399'}},{{data:DATA.vgrasp,color:'#f2607a'}}],'',200,{{dec:2,ymin:0}});
// barras por-dim
(function(){{const svg=document.getElementById('c_bars');const d=DATA.dims,H=180;
 const ymax=Math.max(...d)*1.1, n=d.length, bw=(W-PADL-PADR)/n*0.62, gap=(W-PADL-PADR)/n;
 const Y=v=>PADT+(1-v/ymax)*(H-PADT-PADB); let g='';
 for(let i=0;i<=3;i++){{const v=ymax*i/3,y=Y(v);
  g+=`<line x1="${{PADL}}" y1="${{y.toFixed(1)}}" x2="${{W-PADR}}" y2="${{y.toFixed(1)}}" stroke="#1b2532"/>`;
  g+=`<text x="${{PADL-6}}" y="${{(y+3).toFixed(1)}}" fill="#5c6c80" font-size="9" font-family="IBM Plex Mono" text-anchor="end">${{v.toFixed(2)}}</text>`;}}
 const lbl=['d0','d1','d2','d3','d4','d5','d6','sqz'];
 d.forEach((v,i)=>{{const x=PADL+gap*i+(gap-bw)/2, y=Y(v), hh=H-PADB-y, c=i===7?'#f2607a':'#34d399';
  g+=`<rect x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{bw.toFixed(1)}}" height="${{hh.toFixed(1)}}" rx="2" fill="${{c}}"/>`;
  g+=`<text x="${{(x+bw/2).toFixed(1)}}" y="${{(y-4).toFixed(1)}}" fill="#8ea0b6" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">${{v.toFixed(2)}}</text>`;
  g+=`<text x="${{(x+bw/2).toFixed(1)}}" y="${{H-7}}" fill="#5c6c80" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">${{lbl[i]}}</text>`;}});
 svg.innerHTML=g;}})();
document.getElementById('t_loss_last').textContent='último: '+DATA.loss[DATA.loss.length-1][1].toFixed(4);
</script></body></html>"""

OUT.write_text(HTML)
print(f"[ok] {OUT} ({len(HTML)//1024} KB)")
