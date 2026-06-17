#!/usr/bin/env python3
# Monta o HTML self-contained do PROBE DO BRACO/REACH (probe_arm_reach.py).
# Le /tmp/probe_pull/arm_r4.json (run4a best/ema) e arm_a7.json (armstate7 8k).
# Uso: python build_probe_braco_html.py [dir_jsons] [out.html]
import sys, json
from pathlib import Path
from statistics import mean

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_pull")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "docs/diagnostico_treino_grasp/PROBE_BRACO_REACH.html")
WHEN = "2026-06-16"

r4 = json.loads((SRC / "arm_r4.json").read_text())
a7 = json.loads((SRC / "arm_a7.json").read_text())


def bar(val, mx, color, txt):
    pct = max(2, min(100, abs(val) / mx * 100))
    return (f'<div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'<span class="bv">{txt}</span></div>')


# ---- scatter SVG: cup_disp x armE_swap (run4a) ----
def scatter_svg(d, color="#38bdf8"):
    P = [(p["cup_disp"], p["armE_swap"]) for p in d["pairs"] if p.get("cup_disp") is not None]
    W, H, pad = 520, 240, 38
    xs = [p[0] for p in P]; ys = [p[1] for p in P]
    xmin, xmax = 0, max(xs) * 1.05
    ymin, ymax = 0, max(ys) * 1.05

    def X(x): return pad + (x - xmin) / (xmax - xmin) * (W - pad - 12)
    def Y(y): return H - pad - (y - ymin) / (ymax - ymin) * (H - pad - 14)
    # reta de tendencia (min sq)
    n = len(P); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in P)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    x0, x1 = xmin, xmax
    dots = "".join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{color}" opacity="0.5"/>'
                   for x, y in P)
    med = sorted(xs)[len(xs) // 2]
    return f'''<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:560px">
      <rect x="0" y="0" width="{W}" height="{H}" fill="#070b12" rx="10"/>
      <line x1="{pad}" y1="{H-pad}" x2="{W-12}" y2="{H-pad}" stroke="#26324d"/>
      <line x1="{pad}" y1="14" x2="{pad}" y2="{H-pad}" stroke="#26324d"/>
      <line x1="{X(med):.1f}" y1="14" x2="{X(med):.1f}" y2="{H-pad}" stroke="#3a2a55" stroke-dasharray="4 4"/>
      {dots}
      <line x1="{X(x0):.1f}" y1="{Y(a+b*x0):.1f}" x2="{X(x1):.1f}" y2="{Y(a+b*x1):.1f}" stroke="#f472b6" stroke-width="2.5"/>
      <text x="{pad}" y="{H-12}" fill="#5e7194" font-size="11">desloc. do copo na imagem &rarr;</text>
      <text x="{pad+4}" y="24" fill="#5e7194" font-size="11">&uarr; quanto o braço muda (alvo)</text>
      <text x="{X(med)+6:.1f}" y="28" fill="#a78bfa" font-size="10">perto | longe</text>
    </svg>'''


MX_SENS = 0.7
HTML = f"""<!doctype html><html lang=pt-BR><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Probe do BRAÇO / reach — o braço usa a imagem?</title>
<style>
:root{{--bg:#0a0e16;--card:#121a2b;--card2:#0f1626;--ink:#e9eff8;--ink2:#9db0cc;--ink3:#5e7194;
 --line:#1d2942;--blue:#38bdf8;--green:#34d399;--amber:#fbbf24;--red:#f87171;--violet:#a78bfa;--pink:#f472b6}}
*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(1100px 560px at 72% -10%,#15203a 0%,#0a0e16 60%);color:var(--ink);
 font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;padding:38px 18px 90px}}
.wrap{{max-width:980px;margin:0 auto}}
h1{{font-size:28px;line-height:1.18;margin:0 0 6px;letter-spacing:-.01em}}
h1 .hl{{background:linear-gradient(90deg,var(--violet),var(--blue));-webkit-background-clip:text;background-clip:text;color:transparent}}
.sub{{color:var(--ink2);font-size:15px;max-width:740px;margin:0}}
.meta{{color:var(--ink3);font-size:12.5px;margin:14px 0 28px}}
.sec h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin:34px 0 12px;font-weight:700}}
.card{{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin:14px 0;box-shadow:0 10px 30px -18px #000}}
.lead{{font-size:16px;margin:0 0 4px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:9px 8px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}}
th{{color:var(--ink3);font-size:12px;letter-spacing:.04em;text-transform:uppercase;font-weight:600}}
td.k{{font-weight:600;width:300px}}
.bar{{position:relative;height:28px;background:#070b12;border-radius:6px;overflow:hidden;min-width:90px}}
.fill{{position:absolute;left:0;top:0;bottom:0;border-radius:6px;opacity:.85}}
.bv{{position:absolute;left:8px;top:50%;transform:translateY(-50%);font-size:13px;font-variant-numeric:tabular-nums;text-shadow:0 1px 2px #000}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
.verd{{border-radius:12px;padding:15px 18px;font-size:14px;color:var(--ink2)}}
.vg{{background:#0e2419;border-left:3px solid var(--green)}}
.va{{background:#241d0c;border-left:3px solid var(--amber)}}
.vr{{background:#260f12;border-left:3px solid var(--red)}}
.vb{{background:#0c1c30;border-left:3px solid var(--blue)}}
b.g{{color:var(--green)}}b.r{{color:var(--red)}}b.a{{color:var(--amber)}}b.bl{{color:var(--blue)}}b.v{{color:var(--violet)}}
code{{background:#070b12;border:1px solid var(--line);padding:1px 6px;border-radius:5px;color:var(--ink2);font-size:12.5px}}
.note{{color:var(--ink3);font-size:12.5px;margin-top:8px}}
.pill{{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:700}}
.ok{{background:#0f2e22;color:var(--green)}}.warn{{background:#332814;color:var(--amber)}}
.foot{{margin-top:38px;color:var(--ink3);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}
</style></head><body><div class=wrap>

<h1>Probe do <span class=hl>BRAÇO / reach</span> — o braço usa a imagem?</h1>
<p class=sub>A metade que faltava: o <code>PROBE_RUN4a</code> mediu só a <b>MÃO</b> (squeeze). Aqui medimos se o
<b>reach do braço</b> depende de <b>onde o copo está</b> na imagem — ou se vai por inércia/prior.</p>
<div class=meta>{WHEN} · probe <code>probe_arm_reach.py</code> · 40 frames de reach (1/episódio, squeeze aberto = copo
em posições variadas) · 200 pares de troca · OPEN-LOOP (frames memorizados) · run4a best/ema (deploy) × armstate7 8k</div>

<div class=sec><h2>1 · O braço responde à imagem? (sensibilidade)</h2></div>
<div class=card>
<p class=lead>Quanto a ação do braço (7 dims, normalizado) muda quando a imagem é <b>zerada</b> ou <b>trocada</b>.
Medido em dois pontos do chunk: o <b>1º passo</b> (o que executa já) e o <b>último passo</b> (o ALVO do reach).</p>
<table style="margin-top:8px">
<tr><th>Métrica</th><th>run4a (deploy)</th><th>armstate7 8k</th></tr>
<tr><td class=k>1º passo — commit <span class=note style="display:block">ação imediata, dominada por continuidade</span></td>
  <td>{bar(r4['arm_img_sens_commit'],MX_SENS,'#5e7194',f"{r4['arm_img_sens_commit']:.3f}")}</td>
  <td>{bar(a7['arm_img_sens_commit'],MX_SENS,'#5e7194',f"{a7['arm_img_sens_commit']:.3f}")}</td></tr>
<tr><td class=k>ÚLTIMO passo — alvo do reach <span class=note style="display:block">onde o braço quer chegar</span></td>
  <td>{bar(r4['arm_img_sens_target'],MX_SENS,'#34d399',f"{r4['arm_img_sens_target']:.3f}")}</td>
  <td>{bar(a7['arm_img_sens_target'],MX_SENS,'#a78bfa',f"{a7['arm_img_sens_target']:.3f}")}</td></tr>
</table>
<div class="verd vg" style="margin-top:14px"><b class=g>O braço NÃO ignora a imagem.</b> O 1º passo é baixo (~0,12)
porque a ação imediata segue a trajetória em curso; mas o <b>alvo do reach é bem sensível</b> à imagem
(0,45–0,65). E o <b>run4a SUBIU</b> isso: <b class=g>0,446 &rarr; 0,645</b> (+45%). <span class=note>Corrige a leitura
inicial de "o braço é 5× menos aterrado / ignora a imagem" — aquilo era artefato de olhar só o 1º passo.</span></div>
</div>

<div class=sec><h2>2 · O braço TRACKA a posição do copo?</h2></div>
<div class=card>
<p class=lead>Teste-chave: ao trocar a imagem por uma com o copo <b>mais longe</b>, o braço muda <b>mais</b>?
Se sim, o reach segue a posição do copo. Detectamos o centroide do copo (branco) em cada frame.</p>
<div class="grid2" style="margin-top:8px">
 <div>
  <table>
  <tr><th></th><th>run4a</th><th>armstate7</th></tr>
  <tr><td class=k>Correlação (desloc. copo × muda braço)</td><td><b class=bl>+{r4['track_corr_target']:.2f}</b></td><td>+{a7['track_corr_target']:.2f}</td></tr>
  <tr><td class=k>Braço muda — copo trocado PERTO</td><td>{r4['near']['armE']:.3f}</td><td>{a7['near']['armE']:.3f}</td></tr>
  <tr><td class=k>Braço muda — copo trocado LONGE</td><td>{r4['far']['armE']:.3f}</td><td>{a7['far']['armE']:.3f}</td></tr>
  <tr><td class=k>Razão longe / perto</td><td><span class="pill warn">{r4['far']['armE']/r4['near']['armE']:.2f}×</span></td><td><span class="pill warn">{a7['far']['armE']/a7['near']['armE']:.2f}×</span></td></tr>
  </table>
 </div>
 <div style="display:flex;flex-direction:column;justify-content:center">
   {scatter_svg(r4)}
   <div class=note style="text-align:center">run4a · cada ponto = um par de troca · reta rosa = tendência (fraca-positiva)</div>
 </div>
</div>
<div class="verd va" style="margin-top:14px"><b class=a>Tracka — mas FRACO.</b> A correlação é <b>positiva</b>
(+0,20 a +0,26) e copo mais longe &rarr; braço muda mais (~<b>1,3×</b> nos dois). Ele <b>sabe</b> onde o copo está e
mira na direção. <b class=r>Mas a correlação baixa (0,2)</b> diz que boa parte da resposta à troca de imagem <b>não</b>
é explicada pela posição do copo &mdash; tracking <b>real porém ruidoso</b>, não um servo limpo.</div>
</div>

<div class=sec><h2>3 · O que isso significa pro "desce em cima"</h2></div>
<div class=card>
<div class="verd vb"><b class=bl>A alavanca é loop fechado, não grounding do zero.</b> Se o braço <i>ignorasse</i> o
copo, o conserto seria aterrar do zero. Como ele <b>tracka fraco</b>, o "desce em cima" bate com
<b>distribution-shift de loop fechado</b>:
<div style="margin-top:8px;padding-left:14px;border-left:2px solid #1d2942;color:var(--ink2)">
em <b>open-loop</b> (este probe) o braço lê a imagem e mira ~no copo &rarr; no <b>robô (closed-loop)</b> os erros por
passo <b>compõem</b>, o state deriva pra OOD, e o tracking <b>fraco demais (corr 0,2)</b> não corrige a trajetória
&rarr; aterrissa fora, <b>em cima</b>.</div></div>
<p style="margin:14px 0 0;color:var(--ink2);font-size:14px"><b>Próximos passos:</b> (1) <b>eval de loop FECHADO</b>
(sim reagindo às próprias ações) pra <i>medir</i> a deriva; (2) <b>DAgger</b> (treinar nos próprios rollouts) ataca o
compounding-drift; (3) <b>reforçar o tracking</b> (corr 0,2&rarr;mais alto) com <b>copo deslocado sintético</b> ou
grounding loss amarrando o alvo do reach à posição do copo.</p>
</div>

<div class=foot>
<b>Ressalvas de método:</b> probe <b>OPEN-LOOP</b> (frames de treino memorizados, não rollout). O "swap" troca a
<b>imagem inteira</b> (copo + fundo + braço no quadro), não só o copo &mdash; por isso a <b>correlação com a posição
do copo</b> é o sinal mais limpo de que a posição importa, e ela é <b>fracamente positiva</b>. O teste <b>definitivo</b>
seria <b>copo deslocado sintético</b> (mover só o copo, fundo fixo), que ficou <b>pendente</b>. Centroide do copo
por limiar de branco (claro + baixa saturação), mediana dos pixels. Métricas em espaço normalizado.<br>
JSONs: <code>/tmp/arm_r4.json</code>, <code>/tmp/arm_a7.json</code> · probe <code>probe_arm_reach.py</code> ·
companheiro do <code>PROBE_RUN4a.html</code> (mão).
</div>

</div></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"[ok] {OUT}  ({len(HTML)//1024} KB)")
