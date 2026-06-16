#!/usr/bin/env python3
# Gera um relatório EDITORIAL (estilo blog da Physical Intelligence) da investigação
# do grasp do copo: 3 runs no robô, diagnóstico open-loop, raiz e conserto.
# Auto-contido (imagens em base64). Uso: python3 build_editorial_html.py
import base64, pathlib

OUT = "docs/probs/relatorio_grasp_editorial.html"

def b64(p, mime="image/jpeg"):
    fp = pathlib.Path(p)
    if not fp.exists():
        print("AVISO: faltou", p); return ""
    return f"data:{mime};base64," + base64.b64encode(fp.read_bytes()).decode()

IMGS = {
    "@@FAIL@@":  b64("/tmp/emaframes/000270.jpg"),
    "@@DEMO1@@": b64("/tmp/dsframe_first.jpg"),
    "@@DEMO2@@": b64("/tmp/dsframe_mid.jpg"),
    "@@DEMO3@@": b64("/tmp/dsframe_last.jpg"),
    "@@CUP@@":   b64("/tmp/rgbframes/000000.jpg"),
    "@@ANALYSIS@@": b64("docs/probs/deploy_ema_run20260615_160710_analysis.png", "image/png"),
}

HTML = r"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Por que a política fecha a mão no ar</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--ink:#16181d;--ink2:#3d434d;--mut:#8a909a;--line:#e7e9ee;--bg:#ffffff;--soft:#f6f7f9;--accent:#2f6df6;--red:#d6455d;--green:#1f9d63;--amber:#c8851a;
--serif:'Newsreader',Georgia,serif;--sans:'Inter',system-ui,sans-serif;--mono:'IBM Plex Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-font-smoothing:antialiased}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.65}
.col{max-width:720px;margin:0 auto;padding:0 22px}
header.hero{padding:78px 0 26px}
.kicker{font-family:var(--mono);font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:18px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(34px,5.4vw,52px);line-height:1.07;letter-spacing:-.01em}
.dek{font-family:var(--serif);font-size:clamp(19px,2.6vw,23px);line-height:1.45;color:var(--ink2);margin-top:18px;font-weight:400}
.byline{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:26px;padding-top:18px;border-top:1px solid var(--line);font-size:13.5px;color:var(--mut);font-family:var(--mono)}
main{padding:14px 0 60px}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(25px,3.6vw,31px);line-height:1.15;margin:54px 0 6px;letter-spacing:-.01em}
h2 .n{color:var(--mut);font-family:var(--mono);font-size:.5em;vertical-align:middle;margin-right:.5em;font-weight:500}
p{font-size:18.5px;color:var(--ink2);margin:16px 0}
p b,p strong{color:var(--ink);font-weight:600}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid #c9d8fb}
figure{margin:38px 0}
figure.bleed{margin-left:calc((720px - 100vw)/2 + 22px);margin-right:calc((720px - 100vw)/2 + 22px);max-width:980px;margin-top:42px;margin-bottom:42px}
@media(max-width:1024px){figure.bleed{margin-left:0;margin-right:0;max-width:100%}}
figure img{width:100%;display:block;border-radius:12px;border:1px solid var(--line)}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.trio{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.trio img,.duo img{border-radius:10px}
figcaption{font-size:14px;color:var(--mut);margin-top:12px;line-height:1.5}
figcaption b{color:var(--ink2)}
.callout{background:var(--soft);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:18px 20px;margin:30px 0}
.callout.bad{border-left-color:var(--red)} .callout.good{border-left-color:var(--green)}
.callout h4{font-family:var(--sans);font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin-bottom:8px}
.callout p{margin:0;font-size:17px;color:var(--ink)}
.pull{font-family:var(--serif);font-size:clamp(23px,3.4vw,30px);line-height:1.28;color:var(--ink);margin:44px 0;padding-left:22px;border-left:3px solid var(--accent)}
table{width:100%;border-collapse:collapse;margin:26px 0;font-size:15px}
caption{caption-side:top;text-align:left;font-size:13px;color:var(--mut);font-family:var(--mono);letter-spacing:.04em;margin-bottom:10px;text-transform:uppercase}
th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;color:var(--ink2);font-weight:500}
thead th{font-size:12.5px;color:var(--mut);font-weight:600;border-bottom:1.5px solid var(--ink2)}
td{font-family:var(--mono);color:var(--ink)}
tr td.hl{color:var(--red);font-weight:600}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px;background:var(--soft);border:1px solid var(--line);color:var(--ink2)}
hr{border:0;border-top:1px solid var(--line);margin:50px 0}
.foot{font-size:13px;color:var(--mut);font-family:var(--mono);padding:24px 0 70px;border-top:1px solid var(--line);margin-top:40px;line-height:1.7}
.lead::first-letter{font-family:var(--serif);float:left;font-size:62px;line-height:.82;padding:6px 10px 0 0;color:var(--ink)}
</style></head><body>

<header class="hero"><div class="col">
  <div class="kicker">Diagnóstico · G1 Dex3 · π0.5</div>
  <h1>A política que fecha a mão no ar</h1>
  <p class="dek">Rodamos três variantes de uma política π0.5 no Unitree G1 para pegar um copo. As três falham — e, depois de descartar duas pistas falsas, a causa acaba sendo uma só, medida no robô.</p>
  <div class="byline"><span>Pick-up-the-white-cup · right8/armstate7</span><span>15 jun 2026</span><span>deploy + dataset + probes causais</span></div>
</div></header>

<main><div class="col">

<p class="lead">O objetivo era simples: pôr no robô o "melhor" de três runs de treino (chamadas <b>as-is</b>, <b>valfix</b> e <b>EMA</b>) e ver a mão pegar um copo branco em cima da mesa. Para poder auditar depois, gravamos <b>tudo</b> de cada execução — RGB, profundidade, mapa de atenção da VLA, os <i>chunks</i> previstos e as ações executadas — num formato que toca offline como um vídeo.</p>

<p>O que vimos foi consistente, e estranho: o braço vai até perto da mesa, <b>fecha a mão</b>… e o copo continua parado, intocado. A mão fecha <b>no ar</b>.</p>

<figure>
  <img src="@@FAIL@@" alt="Mão fechada no ar, copo na mesa">
  <figcaption><b>O sintoma, num frame.</b> Run 3 (EMA), frame 270: o squeeze está em ~1.0 (mão fechada) há mais de dois minutos — mas a mão está fechada <b>no ar, à direita</b>, e o copo segue em pé na mesa, à esquerda. Fechar a mão ≠ pegar o copo.</figcaption>
</figure>

<p>O copo não era o problema de percepção. Ele aparece <b>nítido e bem enquadrado</b> na câmera da cabeça em todos os frames:</p>

<figure>
  <img src="@@CUP@@" alt="Copo nítido na câmera">
  <figcaption>Câmera da cabeça no deploy: caneca branca, alça, bom contraste. O modelo <b>vê</b> um copo claro — e ainda assim erra.</figcaption>
</figure>

<h2><span class="n">01</span>A primeira pista falsa: o "lag"</h2>
<p>A suspeita inicial era mecânica: um clamp de velocidade muito agressivo (<span class="tag">--max-delta 0.01</span>) fazia o braço ficar <i>atrás</i> do que o modelo mandava, descendo curto e batendo no copo. Tinha lógica — e some quando voltamos o clamp ao padrão. Medimos o gap entre a ação <b>prevista</b> e a <b>executada</b>: ~0.02–0.05 rad. Pequeno. O robô segue o plano do modelo de perto. <b>Não era o lag</b> — o resíduo continuava.</p>

<h2><span class="n">02</span>A prova causal: o braço ignora a imagem</h2>
<p>Em vez de inferir pela atenção (que, descobrimos, é um artefato de agregação e <b>não serve de prova</b>), fizemos um teste de causa e efeito: <b>apagar/trocar a imagem</b> e medir se a ação muda. Se não muda, o modelo não está usando a visão.</p>

<div class="callout bad"><h4>Resultado da probe (as 3 runs)</h4>
<p>Apagar ou trocar a imagem <b>quase não muda a ação do braço</b> — sensibilidade ~0.07 nas três. O braço é <b>open-loop</b>: ele se move pela propriocepção, não pelo que vê.</p></div>

<table><caption>Grounding — sensibilidade à imagem (probe causal)</caption>
<thead><tr><th>métrica</th><th>as-is</th><th>valfix</th><th>EMA</th></tr></thead>
<tbody>
<tr><td>braço → imagem</td><td class="hl">0.069</td><td class="hl">0.078</td><td class="hl">0.072</td></tr>
<tr><td>squeeze → imagem</td><td>0.376</td><td>0.515</td><td>0.549</td></tr>
<tr><td>erro de alcance (on-dist.)</td><td>0.043</td><td>0.043</td><td>0.045</td></tr>
</tbody></table>

<p>O <b>squeeze</b> (a decisão de fechar) usa um pouco a imagem; o <b>braço</b> (o reach que leva o gripper até o copo) não usa quase nada. E é o braço que falha — ele vai para um ponto aprendido, não para onde o copo está, e fecha no vazio.</p>

<h2><span class="n">03</span>As três, lado a lado</h2>
<p>A correção de medição (valfix) e a média de pesos (EMA) eram as duas variáveis do A/B. Nenhuma toca o núcleo:</p>

<table><caption>Comportamento no robô</caption>
<thead><tr><th>métrica</th><th>Run 1 as-is</th><th>Run 2 valfix</th><th>Run 3 EMA</th></tr></thead>
<tbody>
<tr><td>grasp (média)</td><td>0.80</td><td>0.69</td><td>0.79</td></tr>
<tr><td>% mão fechada</td><td>79%</td><td>66%</td><td>77%</td></tr>
<tr><td>flip-flops / decisão</td><td>0.052</td><td>0.176</td><td>0.117</td></tr>
<tr><td>corr(squeeze, ombro)</td><td>−0.46</td><td>−0.26</td><td>0.00</td></tr>
<tr><td>ombro std (reach)</td><td>0.09</td><td>0.11</td><td>0.14</td></tr>
</tbody></table>

<figure class="bleed">
  <img src="@@ANALYSIS@@" alt="Painéis de análise da run EMA">
  <figcaption><b>Run 3 (EMA), o run inteiro.</b> Grasp quase sempre fechado e instável; ombro quase parado (todo o movimento no cotovelo); atenção difusa; e o chunk previsto ≈ executado (sem lag). Os mesmos padrões aparecem nas três.</figcaption>
</figure>

<p>valfix e EMA <b>nudgaram só o grasp</b> — usa um pouco mais a imagem (squeeze 0.38→0.55), menos a pose (corr −0.46→0.00) — mas deixaram a pega mais instável e <b>não mexeram no braço open-loop</b>. As três falham a pega.</p>

<h2><span class="n">04</span>A segunda pista falsa: o dado</h2>
<p>Antes de culpar o treino, auditamos o dataset. E ele está <b>bom</b>. O vídeo de cada demonstração casa com a ação: mão aberta perto do copo → fecha <b>no copo</b> → levanta. Pega limpa, alinhada, bem-sucedida.</p>

<figure>
  <div class="trio"><img src="@@DEMO1@@" alt="demo início"><img src="@@DEMO2@@" alt="demo pega"><img src="@@DEMO3@@" alt="demo levanta"></div>
  <figcaption><b>Uma demonstração de treino (episódio 0).</b> Aberta perto do copo → <b>fecha no copo</b> → <b>levanta</b>. Imagem e ação batem (squeeze 0 → 1 → 1). O dataset não é o problema.</figcaption>
</figure>

<p>Duas hipóteses minhas caíram aqui, derrubadas por medição — e vale registrar:</p>
<div class="callout"><h4>O que foi descartado</h4>
<p><b>"A mesa muda de cor (bege no treino, azul no deploy)."</b> Errado: medindo os pixels, as duas mesas são quase cinza (RGB ~[121,122,116] vs ~[139,144,143]); trocar R↔B não aproxima nada. Era a luz/JPEG me enganando — não é bug de cor.<br><br>
<b>"O reach das demos é estreito demais."</b> Errado: em 82 episódios, a pose inicial varia (std 0.16), a pose de pega <b>varia</b> (ombro 0.24, cotovelo 0.38 → o copo aparece em lugares diferentes) e os reaches são grandes (cotovelo ~60°). O dado <b>pediria</b> visão.</p></div>

<h2><span class="n">05</span>A raiz: atalho proprioceptivo</h2>
<p>Se o dado é bom e variado, e ainda assim o braço ignora a imagem, sobra uma explicação — clássica em imitação: <b>causal confusion</b>. As trajetórias do braço são suaves, então o <b>estado</b> (a propriocepção) prediz a próxima ação <i>quase perfeitamente</i>. O modelo aprende o caminho mais fácil: <b>seguir a trajetória pelo estado e ignorar a imagem</b>.</p>

<p class="pull">On-distribution funciona — ele segue a trajetória pelo estado. No robô, sem âncora visual, a trajetória deriva para um ponto aprendido e a mão fecha no vazio.</p>

<h2><span class="n">06</span>O conserto</h2>
<p>O alvo não é o dado nem a cor nem a medição — é forçar o modelo a <b>usar a imagem para o reach</b>. A próxima run ataca isso direto:</p>
<div class="callout good"><h4>Run 4</h4>
<p><b>state-dropout</b> (esconder a propriocepção em parte do treino → quebrar o atalho) <b>+ grounding loss</b> (premiar localizar o copo na imagem). A diversidade de reach já existe no dado; não é preciso coletar mais.</p></div>

<hr>
<div class="foot">
Metodologia: 3 runs gravadas no G1 com RGB+depth+atenção+chunks+ações (replays interativos por run em <span class="tag">docs/probs/</span>); probe causal de oclusão (zera/troca a imagem) no checkpoint exato de cada run; auditoria do dataset (alinhamento imagem↔ação, grasp-timing, variância de reach em 82 episódios). wandb: prometheus-lcad/prometheus_g1 — runs 6kr7d8nz · y32omum0 · 6ivtoov9.
</div>

</div></main></body></html>"""

for k, v in IMGS.items():
    HTML = HTML.replace(k, v)
pathlib.Path("docs/probs").mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT).write_text(HTML)
print(f"[ok] {OUT}  ({len(HTML)//1024} KB)")
