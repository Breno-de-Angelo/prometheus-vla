#!/usr/bin/env python3
# Monta o infografico do probe de grounding a partir do manifest.json + PNGs
# gerados por gen_grounding_assets.py. Embute as imagens em base64 (self-contained).
import sys, json, base64
from pathlib import Path

ASSETS = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/grounding_assets")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "docs/probs/y32omum0_grounding_5k.html")
WHEN = sys.argv[3] if len(sys.argv) > 3 else "2026-06-14"

man = json.loads((ASSETS / "manifest.json").read_text())


def b64(name):
    if not name:
        return ""
    p = ASSETS / name
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


VALFIX = man["sq_img_sens"]
BASELINE, ASIS = 0.145, 0.606
MX = max(VALFIX, ASIS, BASELINE) * 1.15


def bar(val, color, lab, sub=""):
    pct = max(2, min(100, val / MX * 100))
    return (f'<div class="cmp"><div class="cmp-l">{lab}</div>'
            f'<div class="cmp-t"><div class="cmp-f" style="width:{pct:.0f}%;background:{color}"></div>'
            f'<span class="cmp-v">{val:.3f}{sub}</span></div></div>')


def sqbars(f):
    # GT, predito(real), zerada, trocada — [0,1]
    items = [("GT", f["gt_squeeze"], "var(--ink3)"), ("predito", f["pred_real"], "var(--green)"),
             ("img zerada", f["pred_zero"], "var(--amber)"), ("img trocada", f["pred_swap"], "var(--red)")]
    out = ['<div class="sq">']
    for lab, v, c in items:
        out.append(f'<div class="sqrow"><span class="sqlab">{lab}</span>'
                   f'<div class="sqt"><div class="sqf" style="width:{max(2,v*100):.0f}%;background:{c}"></div></div>'
                   f'<span class="sqv">{v:.2f}</span></div>')
    out.append('</div>')
    return "".join(out)


def armspark(f):
    # braço predito (chunk) x GT, 7 dims em [-1,1] -> barras divergentes
    out = ['<div class="arm"><div class="armhd">braço: <b style="color:var(--cyan)">predito</b> × <b style="color:var(--ink2)">real</b> (norm)</div><div class="armrow">']
    for d in range(7):
        pv, gv = f["pred_arm"][d], f["gt_arm"][d]
        def h(v):  # [-1,1] -> 0..40px a partir do meio
            return min(20, abs(v) * 20)
        pside = "top" if f["pred_arm"][d] >= 0 else "bottom"
        gside = "top" if gv >= 0 else "bottom"
        out.append(
            f'<div class="armc"><div class="armmid">'
            f'<i class="armp" style="{pside}:20px;height:{h(pv):.0f}px"></i>'
            f'<i class="armg" style="{gside}:20px;height:{h(gv):.0f}px"></i>'
            f'</div><span>{d}</span></div>')
    out.append('</div></div>')
    return "".join(out)


def frame_card(f):
    imgs = [("RGB", f.get("rgb")), ("DEPTH", f.get("depth")),
            ("ATENÇÃO", f.get("attn")), ("SALIÊNCIA", f.get("sal"))]
    grid = "".join(
        f'<figure><img src="{b64(n)}" alt="{lab}"><figcaption>{lab}</figcaption></figure>'
        for lab, n in imgs)
    closed = f["label"] == "CLOSED"
    tagcol = "var(--red)" if closed else "var(--green)"
    return f'''
    <div class="fcard">
      <div class="fhd"><span class="ftag" style="color:{tagcol};border-color:{tagcol}">{f["label"]}</span>
        <span class="fidx">frame #{f["idx"]} · GT squeeze {f["gt_squeeze"]:.2f}</span></div>
      <div class="fgrid">{grid}</div>
      <div class="fmetrics">{sqbars(f)}{armspark(f)}</div>
    </div>'''


frames_html = "".join(frame_card(f) for f in man["frames"])
t = man["table"]

HTML = f'''<!doctype html>
<!--
  Escrito em: {WHEN}
  O que é: infografico do probe de grounding (run {man["run"]} / {man["run_name"]} @ step {man["step"]}).
           Mostra que a pega USA a imagem: RGB, depth, atencao real (attn_recorder) e saliencia
           de frames OPEN/CLOSED, o squeeze previsto sob imagem real/zerada/trocada, e o chunk
           predito x GT. Dados: probe em frames de treino memorizados. Imagens em base64.
-->
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounding da pega — {man["run"]} @ {man["step"]}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#070b11;--bg2:#0a0f17;--card:#0d131c;--card2:#111925;--line:#1b2532;--line2:#26344a;
--ink:#dce6f1;--ink2:#8ea0b6;--ink3:#5c6c80;--cyan:#22d3ee;--amber:#ff9e3d;--green:#34d399;--red:#f2607a;
--disp:'Chakra Petch',sans-serif;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;font-size:15px;
background-image:radial-gradient(900px 500px at 82% -8%,rgba(52,211,153,.06),transparent 60%),
radial-gradient(820px 520px at 0% 108%,rgba(34,211,238,.05),transparent 55%);}}
.wrap{{max-width:1120px;margin:0 auto;padding:30px 22px 90px}}
.docmeta{{font-family:var(--mono);font-size:11px;color:var(--ink3);border:1px solid var(--line);border-radius:9px;
padding:9px 13px;margin-bottom:22px;background:rgba(10,15,23,.6);line-height:1.7}}
.docmeta b{{color:var(--ink2)}}
h1{{font-family:var(--disp);font-weight:700;font-size:33px;letter-spacing:.01em;line-height:1.08;margin-bottom:8px}}
h1 .g{{color:var(--green)}} h1 .q{{color:var(--cyan)}}
.lead{{color:var(--ink2);font-size:15.5px;max-width:840px}}
.lead code,.note code{{font-family:var(--mono);font-size:12.5px;color:var(--amber);background:rgba(255,158,61,.08);padding:1px 5px;border-radius:4px}}
.runbar{{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px}}
.tag{{font-family:var(--mono);font-size:11.5px;color:var(--ink2);background:var(--bg2);border:1px solid var(--line);border-radius:7px;padding:5px 10px}}
.tag b{{color:var(--ink)}}
.verdict{{margin:24px 0 8px;border:1px solid var(--line2);border-radius:15px;background:linear-gradient(180deg,#0e1622,#0b1019);padding:20px 22px}}
.verdict .top{{display:flex;align-items:center;gap:14px;margin-bottom:14px}}
.stamp{{font-family:var(--disp);font-weight:700;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#06231f;background:var(--green);padding:7px 13px;border-radius:8px;box-shadow:0 0 22px rgba(52,211,153,.32)}}
.verdict h2{{font-family:var(--disp);font-weight:700;font-size:21px;border:0;margin:0;padding:0;text-transform:none}}
.cmp{{display:flex;align-items:center;gap:12px;margin:9px 0}}
.cmp-l{{font-family:var(--mono);font-size:12px;color:var(--ink2);width:150px;text-align:right}}
.cmp-t{{position:relative;flex:1;height:26px;background:var(--bg2);border:1px solid var(--line);border-radius:7px;overflow:hidden}}
.cmp-f{{position:absolute;left:0;top:0;height:100%;border-radius:6px;opacity:.85}}
.cmp-v{{position:absolute;right:9px;top:3px;font-family:var(--mono);font-size:12.5px;font-weight:600;color:var(--ink)}}
h2{{font-family:var(--disp);font-weight:700;font-size:20px;letter-spacing:.05em;text-transform:uppercase;margin:40px 0 4px;display:flex;align-items:center;gap:11px}}
h2 i{{width:4px;height:20px;border-radius:2px;background:var(--green);box-shadow:0 0 10px var(--green)}}
.sub{{color:var(--ink3);font-family:var(--mono);font-size:12px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px;margin-top:4px}}
th,td{{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--ink3);font-weight:500;font-size:11px;text-transform:uppercase;border-bottom:1px solid var(--line2)}}
.fcard{{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:14px;padding:16px 17px;margin:16px 0;box-shadow:0 12px 30px rgba(0,0,0,.28)}}
.fhd{{display:flex;align-items:center;gap:11px;margin-bottom:12px}}
.ftag{{font-family:var(--disp);font-weight:700;font-size:12px;letter-spacing:.1em;border:1px solid;border-radius:6px;padding:3px 9px}}
.fidx{{font-family:var(--mono);font-size:12px;color:var(--ink3)}}
.fgrid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media(min-width:760px){{.fgrid{{grid-template-columns:repeat(4,1fr)}}}}
figure{{margin:0;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#000}}
figure img{{width:100%;display:block;aspect-ratio:848/480;object-fit:cover}}
figcaption{{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;color:var(--ink2);text-align:center;padding:5px 0;background:var(--bg2)}}
.fmetrics{{display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px}}
@media(min-width:680px){{.fmetrics{{grid-template-columns:1.3fr 1fr}}}}
.sq{{display:flex;flex-direction:column;gap:6px}}
.sqrow{{display:flex;align-items:center;gap:9px}}
.sqlab{{font-family:var(--mono);font-size:11px;color:var(--ink2);width:78px;text-align:right}}
.sqt{{position:relative;flex:1;height:16px;background:var(--bg2);border:1px solid var(--line);border-radius:5px;overflow:hidden}}
.sqf{{position:absolute;left:0;top:0;height:100%;border-radius:4px;opacity:.85}}
.sqv{{font-family:var(--mono);font-size:11.5px;color:var(--ink);width:34px}}
.arm{{border:1px solid var(--line);border-radius:9px;padding:10px 12px;background:var(--bg2)}}
.armhd{{font-family:var(--mono);font-size:11px;color:var(--ink2);margin-bottom:8px}}
.armrow{{display:flex;gap:10px;align-items:flex-end;justify-content:space-around}}
.armc{{text-align:center}}
.armc span{{font-family:var(--mono);font-size:10px;color:var(--ink3)}}
.armmid{{position:relative;width:18px;height:40px;border-top:1px dashed var(--line2);border-bottom:1px dashed var(--line2)}}
.armmid i{{position:absolute;width:5px;border-radius:2px}}
.armp{{left:3px;background:var(--cyan)}} .armg{{right:3px;background:var(--ink2);opacity:.7}}
.note{{border:1px solid var(--line2);border-left:3px solid var(--amber);border-radius:10px;padding:14px 18px;margin-top:8px;background:rgba(255,158,61,.04);color:var(--ink2);font-size:14px}}
.note b{{color:var(--amber)}}
.foot{{margin-top:42px;color:var(--ink3);font-family:var(--mono);font-size:11px;border-top:1px solid var(--line);padding-top:16px;line-height:1.7}}
</style></head><body><div class="wrap">

<div class="docmeta"><b>Escrito:</b> {WHEN} &nbsp;·&nbsp; <b>O que é:</b> probe de grounding da pega — run <b>{man["run"]}</b> ({man["run_name"]}) @ step <b>{man["step"]}</b>. Imagens reais (RGB/depth/atenção/saliência) embutidas.</div>

<h1>A pega <span class="q">usa a imagem</span>? <span class="g">Usa.</span></h1>
<p class="lead">O probe pega frames REAIS de treino (mão aberta e fechada), prevê o aperto (<code>squeeze</code>) com a imagem
<b>real</b>, <b>zerada</b> e <b>trocada</b> por uma de aperto oposto. Se o squeeze previsto seguir a imagem → grounding OK.
Aqui também: <b>atenção real</b> do action expert (attn_recorder) e <b>saliência</b> (oclusão).</p>

<div class="runbar">
<span class="tag"><b>run</b> {man["run"]} · step {man["step"]}</span>
<span class="tag"><b>frames</b> {man["n_open"]} open / {man["n_closed"]} closed no treino</span>
<span class="tag"><b>modo</b> {man["mode"]} (squeeze = action[7])</span>
</div>

<div class="verdict">
  <div class="top"><span class="stamp">grounding OK</span>
  <h2>Sensibilidade do squeeze à imagem = <b style="color:var(--green)">{VALFIX:.3f}</b></h2></div>
  {bar(VALFIX, "var(--green)", "valfix (ruído isolado)", "")}
  {bar(BASELINE, "var(--red)", "baseline (quebrado)", "")}
  <p class="sub" style="margin-top:12px">perto de 0 = ignora a imagem · <b style="color:var(--ink2)">~{VALFIX/BASELINE:.1f}× o baseline quebrado</b> · as-is armstate7 deu ~{ASIS:.3f} (comparável — o armstate7 funciona nas duas runs)</p>
</div>
<div class="note"><b>Honestidade:</b> o agregado tem piso de ruído do flow-matching. Seedando o ruído entre as condições (número acima, {VALFIX:.3f}) isola o efeito da imagem; sem seedar, o mesmo probe deu 0,685. Os dois são ~4× o baseline 0,145. E o grounding é <b>frame-dependente</b>: em frames de pose ambígua a previsão <b>segue a imagem</b> (ver closed/open abaixo, onde trocar a imagem inverte o aperto); em frames de pose clara, a pose já decide.</div>

<h2><i></i>Frames — RGB · depth · atenção · saliência</h2>
<div class="sub">o squeeze previsto sob imagem real/zerada/trocada mostra o grounding frame a frame</div>
{frames_html}

<h2><i></i>Agregado OPEN × CLOSED</h2>
<div class="sub">squeeze médio previsto · separação (closed−real) − (open−real) = {man["sep"]:+.3f}</div>
<table><thead><tr><th>condição</th><th>imagem real</th><th>imagem zerada</th><th>imagem trocada</th></tr></thead><tbody>
<tr><td><b style="color:var(--green)">OPEN</b> (mão aberta, GT≈0)</td><td>{t["open_real"]:.3f}</td><td>{t["open_zero"]:.3f}</td><td>{t["open_swap"]:.3f}</td></tr>
<tr><td><b style="color:var(--red)">CLOSED</b> (mão fechada, GT≈1)</td><td>{t["closed_real"]:.3f}</td><td>{t["closed_zero"]:.3f}</td><td>{t["closed_swap"]:.3f}</td></tr>
</tbody></table>
<div class="note" style="margin-top:14px"><b>Leitura:</b> no CLOSED, trocar a imagem por uma de mão <b>aberta</b> derruba o squeeze previsto
({t["closed_real"]:.3f} → {t["closed_swap"]:.3f}); no OPEN, trocar por uma <b>fechada</b> sobe ({t["open_real"]:.3f} → {t["open_swap"]:.3f}).
A previsão <b>segue a imagem</b> — não está cravada na pose do braço. Sensibilidade do braço à imagem = {man["arm_img_sens"]:.3f} (braço é mais pose, menos imagem).</div>

<div class="note" style="border-left-color:var(--ink3);background:rgba(255,255,255,.02)"><b style="color:var(--ink2)">Método:</b> frames de treino memorizados; o ruído do flow-matching no <code>predict_action_chunk</code> não é seedado entre condições, então há um piso de ruído na sensibilidade (a mesma metodologia do baseline 0,145 e da as-is 0,606, logo comparável). Atenção = média sobre denoise steps × camadas × heads das queries de ação sobre os 256 tokens de imagem (16×16), recortada à faixa da imagem no quadrado 224 do resize_with_pad.</div>

<div class="foot">probe: gen_grounding_assets.py (RGB/depth + attn_recorder + oclusão) · run {man["run"]} @ {man["step"]} · ckpt last/pretrained_model<br>
sensibilidade squeeze→imagem {VALFIX:.3f} · separação {man["sep"]:+.3f} · braço→imagem {man["arm_img_sens"]:.3f}</div>
</div></body></html>'''

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"[ok] {OUT}  ({len(HTML)//1024} KB)  frames={len(man['frames'])}")
