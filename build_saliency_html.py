#!/usr/bin/env python3
# Monta o HTML visual (RGB + saliencia + barras de uso da visao) a partir dos dois
# manifests do probe_saliency.py (baseline e armstate7), PNGs embutidos em base64.
# Honestidade: ONDE (mapas, normalizados ao proprio pico) e separado de QUANTO
# (magnitude = sq_img_sens, no painel). Mostra os 2 metodos de oclusao (cinza+troca).
import base64, json, sys
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/sal_pull")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path("docs/diagnostico_treino_grasp/RESULTADO_VISUAL_SALIENCIA.html")

def load(name):
    m = json.loads((BASE / name / "manifest.json").read_text())
    m["_dir"] = BASE / name
    return m

def b64(d, fname):
    return base64.b64encode((d / fname).read_bytes()).decode()

bl = load("sal_baseline")
ar = load("sal_armstate7")

def fr(m, tag):
    return next(f for f in m["frames"] if f["tag"] == tag)

def src(m, fname):
    return f'data:image/png;base64,{b64(m["_dir"], fname)}'

def bar(val, vmax=1.0, color="#2f9e44"):
    pct = max(0, min(100, 100 * val / vmax))
    return (f'<div class="barwrap"><div class="bar" style="width:{pct:.0f}%;background:{color}"></div>'
            f'<span class="barlab">{val:.2f}</span></div>')

blc, arc = fr(bl, "closed"), fr(ar, "closed")
ratio = ar["sq_img_sens"] / max(bl["sq_img_sens"], 1e-6)

def usepanel():
    VS = 0.8
    rows = [("👁️ Visão (imagem)", bl["sq_img_sens"], ar["sq_img_sens"], "#1c7ed6"),
            ("💪 Propriocep. braço", bl["arm_img_sens"], ar["arm_img_sens"], "#e8590c")]
    h = ""
    for lab, b, a, col in rows:
        h += (f'<tr><td class="ul">{lab}</td>'
              f'<td>{bar(b, VS, "#adb5bd")}</td><td>{bar(a, VS, col)}</td></tr>')
    h += ('<tr><td class="ul">🟦 Depth</td>'
          '<td colspan="2" class="na">N/A — o modelo é <b>RGB-only</b> (1 câmera, sem stream de depth). '
          'Não vê profundidade. Depth de verdade seria outro experimento.</td></tr>')
    return h

HTML = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>O que o modelo VÊ — antigo × novo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Patrick+Hand&display=swap" rel="stylesheet">
<style>
 :root{{--ink:#2b2b2b;--paper:#fdf6ec;--blue:#d4ecff;--blue-b:#1c7ed6;--green:#d6f5dd;--green-b:#2f9e44;
   --red:#ffd0d0;--red-b:#e03131;--orange:#ffe0bf;--orange-b:#e8590c;--gray:#e9ecef;}}
 *{{box-sizing:border-box;}}
 body{{margin:0;background:var(--paper);background-image:radial-gradient(#0000000a 1px,transparent 1px);
   background-size:22px 22px;color:var(--ink);font-family:'Patrick Hand',cursive;font-size:18px;
   line-height:1.4;padding:26px 14px 70px;}}
 .poster{{max-width:1120px;margin:0 auto;}}
 h1{{font-family:'Caveat',cursive;font-size:58px;text-align:center;margin:.1em 0 0;}}
 h1 .a{{color:var(--green-b);}} h1 .b{{color:var(--blue-b);}} h1 .c{{color:var(--orange-b);}}
 .sub{{font-family:'Caveat',cursive;font-size:27px;text-align:center;color:#555;margin:2px 0 4px;}}
 .credits{{text-align:center;color:#888;font-size:15px;margin-bottom:14px;}}
 .box{{background:#fff;border:3px solid var(--ink);border-radius:230px 14px 220px 16px/16px 215px 14px 230px;
   padding:14px 18px;margin:12px 0;box-shadow:3px 4px 0 #00000018;}}
 .box.alt{{border-radius:14px 230px 16px 220px/210px 16px 225px 14px;}}
 .section{{margin-top:26px;}}
 .head{{display:flex;align-items:center;gap:12px;margin-bottom:2px;}}
 .num{{font-family:'Caveat',cursive;font-weight:700;font-size:30px;width:50px;height:50px;flex:0 0 50px;
   display:flex;align-items:center;justify-content:center;border:3px solid var(--ink);border-radius:50%;
   background:#fff;box-shadow:2px 3px 0 #00000018;}}
 .title{{font-family:'Caveat',cursive;font-weight:700;font-size:31px;}}
 .title span{{padding:0 2px;border-bottom:7px solid var(--blue-b);}}
 .title.tg span{{border-color:var(--green-b);}} .title.to span{{border-color:var(--orange-b);}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
 .col h3{{font-family:'Caveat',cursive;font-size:24px;margin:2px 0 6px;text-align:center;}}
 .col.old h3{{color:#868e96;}} .col.new h3{{color:var(--green-b);}}
 img.cam{{width:100%;border:2px solid var(--ink);border-radius:8px;display:block;}}
 .imlab{{font-size:14px;color:#666;text-align:center;margin:2px 0 6px;}}
 .barwrap{{position:relative;background:var(--gray);border:2px solid var(--ink);border-radius:7px;
   height:26px;margin:4px 0;overflow:hidden;}}
 .bar{{height:100%;}} .barlab{{position:absolute;right:8px;top:2px;font-weight:bold;font-size:15px;}}
 .blab{{font-size:14px;color:#555;}}
 table.use{{width:100%;border-collapse:collapse;}}
 table.use td{{padding:5px 7px;vertical-align:middle;}} table.use td.ul{{width:210px;font-weight:bold;}}
 table.use th{{font-family:'Caveat',cursive;font-size:22px;padding:4px;}}
 .na{{color:#888;font-size:15px;}}
 .hlg{{background:linear-gradient(transparent 55%,#b2f2bb 55%);padding:0 2px;font-weight:bold;}}
 .hlr{{background:linear-gradient(transparent 55%,#ffc9c9 55%);padding:0 2px;font-weight:bold;}}
 ul{{margin:6px 0;padding-left:22px;}} li{{margin:3px 0;}}
 .cap{{font-size:15px;color:#444;margin-top:6px;}}
 .big{{font-size:21px;}} .center{{text-align:center;}}
 .legend{{display:flex;gap:8px;align-items:center;justify-content:center;font-size:14px;color:#555;margin-top:4px;}}
 .swatch{{height:12px;width:120px;border:1px solid var(--ink);border-radius:3px;
   background:linear-gradient(90deg,#1c46d6,#19c3c3,#36d63a,#f5d000,#e03131);}}
</style></head><body><div class="poster">

 <p class="sub" style="margin-bottom:0">🤖 G1 Dex3 · π0.5 · "pick up the white cup"</p>
 <h1><span class="a">O que o modelo</span> <span class="b">VÊ</span> <span class="c">— antigo × novo</span></h1>
 <p class="sub">câmera RGB · mapa de saliência · força do aperto · quanto usa cada sinal</p>
 <p class="credits">oclusão (seed fixa do flow-matching) no <code>best</code> · antigo <code>8hajpdab</code> state[14] × novo <code>armstate7</code> ~5k state[7] · 2026-06-14</p>

 <div class="box alt center" style="background:var(--blue)">
   <b>Como ler:</b> o modelo recebe <b>1 câmera RGB</b> (a <code>head_camera</code> abaixo) — <b>sem depth</b>.
   A pergunta: <span class="hlg">ele decide fechar a mão OLHANDO o copo, ou ignora a imagem?</span>
   Mexemos só na imagem e medimos o aperto (squeeze).
 </div>

 <!-- 1. A CAMERA -->
 <div class="section"><div class="head"><div class="num">1</div>
   <div class="title to"><span>A câmera que ele vê</span> 📷</div></div>
   <div class="box"><div class="grid2">
     <div class="col"><h3>momento de FECHAR</h3>
       <img class="cam" src="{src(ar, blc['rgb'])}">
       <div class="imlab">mão envolvendo o copo · idx {blc['idx']}</div></div>
     <div class="col"><h3>momento ABERTO</h3>
       <img class="cam" src="{src(ar, fr(ar,'open')['rgb'])}">
       <div class="imlab">mão aberta, copo fora do quadro · idx {fr(ar,'open')['idx']}</div></div>
   </div></div>
 </div>

 <!-- 2. TESTE DE TROCA (headline) -->
 <div class="section"><div class="head"><div class="num">2</div>
   <div class="title tg"><span>Ele usa a visão? (teste de troca)</span> 🎯</div></div>
   <div class="box"><p class="center big">No momento de <b>FECHAR</b>: prevê o aperto com a imagem <b>real</b>,
   depois <b>zerada</b> (preta) e <b>trocada</b> (pela imagem de mão aberta). Se o aperto <b>cai</b> ao trocar → ele estava OLHANDO.</p>
   <div class="grid2">
     <div class="col old"><h3>Antigo (8hajpdab)</h3>
       <div class="blab">real</div>{bar(bl['closed_real'])}<div class="blab">zerada</div>{bar(bl['closed_zero'])}<div class="blab">trocada</div>{bar(bl['closed_swap'])}
       <p class="cap">Trocar a imagem <span class="hlr">quase não mexe</span> ({bl['closed_real']:.2f}→{bl['closed_swap']:.2f}) → <b>ignora a visão</b>, fecha pela propriocepção.</p></div>
     <div class="col new"><h3>Novo (armstate7)</h3>
       <div class="blab">real</div>{bar(ar['closed_real'])}<div class="blab">zerada</div>{bar(ar['closed_zero'])}<div class="blab">trocada</div>{bar(ar['closed_swap'],1.0,'#e03131')}
       <p class="cap">Trocar a imagem <span class="hlg">DESPENCA</span> o aperto ({ar['closed_real']:.2f}→{ar['closed_swap']:.2f}) → a mão <b>segue o que vê</b>.</p></div>
   </div></div>
 </div>

 <!-- 3. MAPA DE SALIENCIA -->
 <div class="section"><div class="head"><div class="num">3</div>
   <div class="title"><span>Mapa de saliência — onde ele olha</span> 🔥</div></div>
   <div class="box"><p class="center">Deslizamos um quadrado pela imagem e medimos quanto o aperto muda. <b>Quente = ali a imagem decide o fechar.</b>
   <span class="blab">(Mapas normalizados ao próprio pico → mostram <b>ONDE</b>, não <b>QUANTO</b>; o quanto está no painel 4.)</span></p>
   <div class="legend"><span>frio</span><span class="swatch"></span><span>quente</span></div>
   <div class="grid2" style="margin-top:8px">
     <div class="col old"><h3>Antigo — oclusão cinza</h3>
       <img class="cam" src="{src(bl, blc['sal'])}">
       <p class="cap"><b>Difuso/espalhado</b> — os focos caem na <b>mesa vazia</b> e no <b>braço</b>, NÃO no copo. Assinatura de quem <b>não olha</b>.</p></div>
     <div class="col new"><h3>Novo — oclusão cinza</h3>
       <img class="cam" src="{src(ar, arc['sal'])}">
       <p class="cap">Concentra na <b>borda do copo / contato dos dedos</b> 👆 — olha <b>onde agarra</b>.</p></div>
   </div>
   <div class="grid2" style="margin-top:10px">
     <div class="col old"><h3>Antigo — oclusão por troca</h3>
       <img class="cam" src="{src(bl, blc['salswap'])}">
       <p class="cap">Idem: focos espalhados pela cena → sem foco no copo.</p></div>
     <div class="col new"><h3>Novo — oclusão por troca</h3>
       <img class="cam" src="{src(ar, arc['salswap'])}">
       <p class="cap">Foco <b>no corpo do copo</b> 🥤 — os <b>2 métodos concordam</b>.</p></div>
   </div>
   <p class="cap center" style="margin-top:8px">⚠️ O efeito local é pequeno (nenhum bloco sozinho vira a decisão): o uso da imagem é <b>holístico</b> (a cena toda — só
   trocar a imagem INTEIRA vira, ver §2). Estes mapas mostram <b>onde a sensibilidade se concentra</b> — e no novo isso é o <b>copo</b>.</p>
   </div>
 </div>

 <!-- 4. QUANTO USA CADA SINAL -->
 <div class="section"><div class="head"><div class="num">4</div>
   <div class="title to"><span>Quanto usa cada sinal</span> 📊</div></div>
   <div class="box"><table class="use">
     <tr><th></th><th>Antigo</th><th>Novo</th></tr>
     {usepanel()}
   </table>
   <p class="center big" style="margin-top:10px">👁️ Uso da visão pro aperto: <span class="hlg">{bl['sq_img_sens']:.2f} → {ar['sq_img_sens']:.2f}</span>
   (≈ <b>{ratio:.0f}×</b>). O braço usa a imagem ~igual; o <b>ganho é todo na decisão de fechar</b>.</p>
   </div>
 </div>

 <!-- VEREDITO + RESSALVAS -->
 <div class="section center">
   <div class="box" style="background:#fff3bf;border-width:4px">
     <p class="title tg" style="margin:0"><span>Resumo</span></p>
     <p class="big" style="margin:8px 0">O modelo <b>antigo</b> fechava a mão <b>de olhos fechados</b> (propriocepção).
     O <b>novo</b> (sem os dedos no input) <span class="hlg">passou a olhar o COPO</span> pra decidir fechar.</p>
   </div>
   <div class="box alt" style="text-align:left;background:var(--orange)">
     <b>⚠️ Honestidade técnica:</b>
     <ul>
       <li><b>Depth:</b> modelo <b>RGB-only</b>, 1 câmera, <b>sem depth</b>. "Quanto usa depth" = <b>0 (não é input)</b>.</li>
       <li><b>Mapa:</b> oclusão com ruído do flow-matching <b>fixo</b> (sem isso o pico vira ruído do amostrador). É a versão honesta de "atenção" pra esse modelo. O efeito é holístico; o mapa mostra a concentração relativa.</li>
       <li><b>Parcial:</b> novo lido no <b>~5k de 20k</b> steps. Falta o treino fechar e o <b>juiz final: o robô</b>.</li>
     </ul>
   </div>
   <p class="credits">probe <code>probe_saliency.py</code> · antigo best step ~8000 · novo best step 4000 (~5k) · grid {ar['grid']}×{ar['grid']} (cinza) / {arc.get('swap_grid','?')}×{arc.get('swap_grid','?')} (troca) · seed fixa</p>
 </div>

</div></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"[ok] {OUT}  ({len(HTML)//1024} KB)")
print(f"  baseline sq_img={bl['sq_img_sens']:.3f}  armstate7 sq_img={ar['sq_img_sens']:.3f}  ratio={ratio:.0f}x")
print(f"  swap closed: bl={bl['closed_real']:.2f}->{bl['closed_swap']:.2f}  ar={ar['closed_real']:.2f}->{ar['closed_swap']:.2f}")
