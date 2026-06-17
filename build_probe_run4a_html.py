#!/usr/bin/env python3
# Monta o infografico self-contained da PROBE ESTATICA do run4a (state-dropout),
# comparado com armstate7 (pre-run4a) e o baseline quebrado (8hajpdab, doc).
# Le os JSONs gerados por probe_grasp_grounding.py / probe_proprio.py.
# Uso: python build_probe_run4a_html.py <dir_jsons> <out.html>
import sys, json
from pathlib import Path
from statistics import mean

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_pull")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "docs/diagnostico_treino_grasp/PROBE_RUN4a.html")
WHEN = sys.argv[3] if len(sys.argv) > 3 else "2026-06-16"


def load(name):
    p = SRC / name
    return json.loads(p.read_text()) if p.exists() else None


def gmetrics(g):
    """metricas do grounding json"""
    if not g:
        return None
    rows = g["rows"]
    OPEN = [r for r in rows if r["label"] == "OPEN"]
    CLOSED = [r for r in rows if r["label"] == "CLOSED"]
    return {
        "sep": g["sep"],
        "sq_img": g["sq_img_sens"],
        # tampar (zerar img) num frame ABERTO -> quanto sobe rumo a fechado (RUIM se alto)
        "cover_close": mean(r["pred_sq_zero"] for r in OPEN),
        # trocar img num frame FECHADO -> quanto cai o fechamento (BOM se alto = segue a imagem)
        "swap_drop": mean(r["pred_sq_real"] for r in CLOSED) - mean(r["pred_sq_swap"] for r in CLOSED),
    }


def psep(p, cond):
    rows = p["rows"]
    O = mean(r[cond] for r in rows if r["label"] == "OPEN")
    C = mean(r[cond] for r in rows if r["label"] == "CLOSED")
    return C - O


def pmetrics(p):
    if not p:
        return None
    return {c: psep(p, c) for c in ["baseline", "state_zero", "state_swap", "arm_zero", "fingers_zero"]}


# --- dados ---
a7g = gmetrics(load("cmp_a7_gnd.json"))
a7p = pmetrics(load("cmp_a7_prp.json"))
r4g = gmetrics(load("cmp_r4_gnd.json"))           # run4a NON-ema (comparavel ao a7 non-ema)
r4p = pmetrics(load("cmp_r4_prp.json"))
r4ge = gmetrics(load("probe_run4a_grounding.json"))  # run4a EMA (deployado)
r4pe = pmetrics(load("probe_run4a_proprio.json"))

# baseline 8hajpdab quebrado — numeros dos docs (PROBE_GROUNDING/PROPRIO_RESULTADO)
BASE = {"sep": 0.84, "sq_img": 0.145, "cover_close": None, "swap_drop": None,
        "state_swap": -0.88, "fingers_zero": 0.11, "state_zero": None}

COLS = [
    ("baseline 8hajpdab", "#f87171", "modelo quebrado (doc)"),
    ("armstate7 8k", "#fbbf24", "pre-run4a (non-ema)"),
    ("run4a 15k non-ema", "#38bdf8", "state-dropout"),
    ("run4a 15k EMA", "#34d399", "DEPLOYADO no robo"),
]


def fmt(v, sign=True):
    if v is None:
        return "<span class=na>n/d</span>"
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def bar(v, mx, color, txt):
    if v is None:
        return '<div class="bar"><div class="na">n/d</div></div>'
    pct = max(2, min(100, abs(v) / mx * 100))
    return (f'<div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'<span class="bv">{txt}</span></div>')


def row_metric(title, key, vals, mx, good, note):
    cells = ""
    for (lab, color, _), v in zip(COLS, vals):
        cells += f'<td>{bar(v, mx, color, fmt(v))}</td>'
    return (f'<tr><th>{title}<div class="gd">{good}</div></th>{cells}</tr>'
            f'<tr class="nt"><td colspan="5">{note}</td></tr>')


# valores por metrica, na ordem das COLS
SEP = [BASE["sep"], a7g["sep"], r4g["sep"], r4ge["sep"]]
SQIMG = [BASE["sq_img"], a7g["sq_img"], r4g["sq_img"], r4ge["sq_img"]]
COVER = [BASE["cover_close"], a7g["cover_close"], r4g["cover_close"], r4ge["cover_close"]]
SWAPD = [BASE["swap_drop"], a7g["swap_drop"], r4g["swap_drop"], r4ge["swap_drop"]]
SWST = [BASE["state_swap"], a7p["state_swap"], r4p["state_swap"], r4pe["state_swap"]]
ZRST = [BASE["state_zero"], a7p["state_zero"], r4p["state_zero"], r4pe["state_zero"]]

head_cells = "".join(
    f'<th class=hc><span style="color:{c}">{l}</span><div class="sub">{s}</div></th>'
    for l, c, s in COLS)

HTML = f"""<!doctype html><html lang=pt-BR><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Probe estatica — run4a (state-dropout)</title>
<style>
:root{{--bg:#0b0f17;--card:#121826;--ink:#e6edf6;--ink2:#9fb0c7;--ink3:#5b6b85;--line:#1e2740}}
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0f17;color:#e6edf6;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:32px 18px 80px}}
.wrap{{max-width:980px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 4px}}
.meta{{color:#5b6b85;font-size:13px;margin-bottom:24px}}
.card{{background:#121826;border:1px solid #1e2740;border-radius:14px;padding:22px 24px;margin:18px 0}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px 6px;text-align:left;vertical-align:middle}}
th{{font-weight:600}}
.hc{{font-size:13px;text-align:center;border-bottom:1px solid #1e2740;padding-bottom:10px}}
.hc .sub{{color:#5b6b85;font-weight:400;font-size:11px;margin-top:2px}}
tr>th:first-child{{width:230px;color:#e6edf6;font-size:14px}}
.gd{{color:#5b6b85;font-size:11px;font-weight:400;margin-top:2px}}
.bar{{position:relative;height:30px;background:#0b0f17;border-radius:6px;overflow:hidden;min-width:90px}}
.fill{{position:absolute;left:0;top:0;bottom:0;border-radius:6px;opacity:.85}}
.bv{{position:absolute;left:8px;top:50%;transform:translateY(-50%);font-size:13px;font-variant-numeric:tabular-nums;text-shadow:0 1px 2px #000}}
.na{{color:#5b6b85;font-size:12px;padding-left:8px}}
.nt td{{color:#9fb0c7;font-size:12.5px;padding:2px 6px 14px;border-bottom:1px solid #161e30}}
.lead{{color:#9fb0c7;font-size:15px;margin:6px 0 0}}
.verd{{background:#0e1830;border-left:3px solid #34d399;padding:14px 18px;border-radius:8px;margin-top:10px}}
.bad{{border-left-color:#f87171}}
.amb{{border-left-color:#fbbf24}}
b.g{{color:#34d399}} b.r{{color:#f87171}} b.a{{color:#fbbf24}}
code{{background:#0b0f17;padding:1px 5px;border-radius:4px;color:#9fb0c7;font-size:13px}}
</style></head><body><div class=wrap>

<h1>Probe estatica — o run4a (state-dropout) ficou aterrado na visao?</h1>
<div class=meta>{WHEN} · best/ema step 15000 · 8dim · frames de treino memorizados (offline, sem robo) ·
mesma bateria das outras runs (probe_grasp_grounding + probe_proprio)</div>

<p class=lead>Pergunta: a decisao de <b>fechar a mao</b> depende de <b>ver o copo</b>, ou o modelo fecha
por inercia/propriocepcao? Medimos mexendo so na imagem (tampar/trocar) e so no estado (zerar/trocar),
e comparamos o baseline quebrado, o armstate7 (pre-run4a) e o run4a.</p>

<div class=card>
<table>
<tr><th></th>{head_cells}</tr>
{row_metric("Crava a decisao? (separacao)", "sep", SEP, max(x for x in SEP if x), "perto de 1 = decide forte",
  "Todos cravam aberto&rarr;0 / fechado&rarr;~0.8. So cravar nao basta — a pergunta e <i>por que</i> crava.")}
{row_metric("Usa a imagem? (sensib.)", "sq_img", SQIMG, max(x for x in SQIMG if x), "perto de 0 = IGNORA a imagem",
  "Baseline <b class=r>0.145</b> (quase cego) &rarr; armstate7 <b class=a>0.63</b> (salto grande, ~4&times;) &rarr; run4a <b class=g>0.72&ndash;0.80</b>. O state-dropout <b>subiu mais</b> a dependencia da imagem (+13&ndash;26%).")}
{row_metric("Trocar a imagem (segue?)", "swap_drop", SWAPD, max(x for x in SWAPD if x), "alto = segue a imagem",
  "Num frame fechado, trocar por uma imagem de &lsquo;copo longe&rsquo; <b>derruba</b> o fechamento &mdash; o modelo segue o que ve.")}
{row_metric("TAMPAR a imagem &rarr; fecha?", "cover_close", COVER, 1.0, "ALTO = fecha cego (RUIM)",
  "&#128308; Frame ABERTO com a imagem zerada: o squeeze <b>deriva pra FECHADO</b> (fallback sem-visao = fechar). Persiste e e <b>levemente pior</b> no run4a (0.09&rarr;0.17&ndash;0.25) &mdash; e o &lsquo;age cego / acha que segura&rsquo; visto no robo.")}
{row_metric("TROCAR o state (dita a pega?)", "state_swap", SWST, 1.0, "negativo/zero = state ainda pesa",
  "Baseline <b class=r>&minus;0.88</b> (propriocepcao INVERTIA = ditava tudo) &rarr; run4a <b class=a>&minus;0.06</b>: state errado <b>neutraliza</b> a pega, mas nao <b>inverte</b> mais. Melhorou, mas ainda modula.")}
{row_metric("ZERAR todo o state (imagem decide?)", "state_zero", ZRST, 1.0, "alto = imagem decide sozinha",
  "&#11088; O ganho do state-dropout: sem nenhuma propriocepcao, armstate7 cai p/ <b class=a>0.50</b>, mas o run4a mantem <b class=g>0.68&ndash;0.75</b> &mdash; a <b>imagem sozinha</b> ja decide a pega. E exatamente o que o state-dropout treina.")}
</table>
</div>

<div class=card>
<h3 style="margin:0 0 10px">Veredito</h3>
<div class="verd"><b class=g>O run4a corrigiu a confusao causal grave.</b> A pega deixou de ser ditada pela
propriocepcao (state_swap saiu de <b class=r>&minus;0.88</b> p/ <b class=a>&minus;0.06</b>) e a imagem passou a
carregar a decisao (sensib. <b class=g>~0.72</b>, segue a troca de imagem, decide mesmo com state zerado).</div>
<div class="verd amb" style="margin-top:10px"><b class=a>O state-dropout (run4a) deu ganho real, mensuravel.</b>
Pelo mesmo probe: sensib. a imagem <b>0.63&rarr;0.72&ndash;0.80</b>, e &mdash; o ganho-chave &mdash; com o state
removido a imagem ja decide sozinha (<b>state_zero 0.50&rarr;0.75</b>). O salto grande veio do armstate7 (remover
o state dos dedos); o run4a <b>empilhou em cima</b> reduzindo a dependencia do braco. Os dois fixes somam.</div>
<div class="verd bad" style="margin-top:10px"><b class=r>E sobra o vies que quebra no robo:</b> sem imagem, o
default e <b>FECHAR</b>. A probe <b>preve</b> os 4 sintomas do robo (age cego, acha que segura, demora re-reach,
desce em cima) &mdash; nenhum exige re-curar dados; pedem sinal de <b>posse</b> (tatil / pi05-D) e <b>cobertura</b>
de posicao do copo.</div>
</div>

</div></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
print(f"[ok] {OUT}  ({len(HTML)//1024} KB)")
