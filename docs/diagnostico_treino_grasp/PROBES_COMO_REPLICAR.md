2026-06-13 (criado) · última edição 2026-06-14 05:26

> **O que é (breve):** como rodar/replicar os probes de diagnóstico (ablação de imagem, propriocepção, state-check, saliência).
> **Correções:** em 14/06 adicionei o probe #4 (`probe_saliency.py` + `build_saliency_html.py`). Cabeçalho adicionado retroativamente em 2026-06-14 12:21.

---

# Probes de diagnóstico — como foram feitos e como replicar

> Três scripts offline (sem robô) que rodamos no `best` dos checkpoints pra diagnosticar o grasp.
> Todos rodam na **Atena**, env **ms3**, **GPU2**. Ficam na **raiz do repo** (importam `REPO_ROOT/lerobot-ext`).
> Avaliam o `predict_action_chunk` em **frames REAIS de treino** (memorizados) — não é rollout.

## Ambiente (comum aos três)
```bash
ssh hercules@10.9.8.252
cd ~/Prometheus/Luiz/prometheus-vla
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ms3
export HF_HOME=/data/huggingface-models HUGGING_FACE_HUB_TOKEN=<hf> CUDA_VISIBLE_DEVICES=2
```
**Gotcha:** o `torchcodec` do ms3 quebra com `seek_mode` ao decodar vídeo → os scripts usam
`LeRobotDataset(..., video_backend="pyav")`. Sem isso, erro no decode do mp4.

---

## 1. `probe_grasp_grounding.py` — ABLAÇÃO DE IMAGEM (o grasp olha o copo?)

**Ideia:** em frames de treino onde o squeeze GT é **aberto (0)** e **fechado (1)**, gerar a ação e ver
(a) se o squeeze previsto sai **cravado** (open→0 / closed→1) e (b) se ele **segue a imagem**. Para cada
frame, roda `predict_action_chunk` em **3 condições de imagem** (mexendo SÓ na `head_camera`):

| condição | o que faz |
|---|---|
| **real** | imagem do próprio frame |
| **zerada** | `head_camera = zeros` (tela preta) |
| **trocada** | imagem de um frame do bucket OPOSTO (a um frame CLOSED dá a imagem de um OPEN, e vice-versa) |

**Leitura:**
- **Separação** = `mean(pred_squeeze | CLOSED) − mean(pred_squeeze | OPEN)`. Perto de 1 = crava; perto de 0 = morno.
- **Sensibilidade à imagem** = `mean|real−zerada| + mean|real−trocada|`. Perto de 0 = **ignora a imagem**.
- Se trocar por uma imagem "aberta/copo longe" **não** faz o squeeze cair → o fechamento **não** é gatilhado pela visão.

**Detalhe técnico que importa:** o `predict_action_chunk` devolve a ação **NORMALIZADA**. Pro squeeze do
8-dim (q01=0/q99=1) o raw = `(norm+1)/2` (exato). Pro 14-dim os dedos são `squeeze×RIGHT_TARGET` →
recupera o squeeze por-dim (sinal depende de `RIGHT_TARGET[d]`). Frames OPEN/CLOSED são lidos da coluna
`action` do `hf_dataset` **sem decodar vídeo** (rápido); só os ~24 selecionados decodam a imagem.

**Rodar:**
```bash
# 8-dim
python probe_grasp_grounding.py --mode 8dim \
  --ckpt train_output/cup_pi05_right8_1squeeze_lf/checkpoints/best/pretrained_model \
  --repo-id lewislf/G1_Dex3_pick_white_cup_right8_1squeeze \
  --root datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze --out /tmp/probe_8dim.json
# 14-dim: --mode 14dim, --ckpt .../cup_pi05_right14_rgb238_lf/checkpoints/best/pretrained_model,
#         --repo-id lewislf/G1_Dex3_right14_dataset, --root lerobot-ext/datasets/G1_Dex3_right14_dataset/v3_238ep
```
**Resultado obtido:** 8-dim separação +0,84 / sensib. imagem 0,145; 14-dim +0,995 / 0,010. → squeeze
cravado no treino, mas **cego à imagem**. (`PROBE_GROUNDING_RESULTADO.md`.)

---

## 2. `probe_proprio.py` — ABLAÇÃO DA PROPRIOCEPÇÃO (o que dita o fechamento?)

**Ideia:** imagem SEMPRE real; mexe só no `observation.state`. Mede a separação do squeeze sob:
`baseline` · `state_zero` (zera tudo) · `state_swap` (state de um frame oposto) · `arm_zero` (zera só
dims 0-6) · `fingers_zero` (zera só dims 7-13). Se a separação **colapsa/inverte** numa condição, esse
input é o que carrega a decisão.

**Rodar:** igual ao #1, com `probe_proprio.py` (mesmos `--ckpt/--repo-id/--root/--mode`).

**Resultado obtido (8-dim):** `state_swap` **inverte** (+0,95→−0,88) → a propriocepção DITA o fechamento;
`fingers_zero` colapsa (+0,95→+0,11) e `arm_zero` mantém (+0,74) → **dominado pelos dedos medidos** (autocorrelação). (`PROBE_PROPRIO_RESULTADO.md`.)

---

## 3. `probe_state_check.py` — o modelo USA o state? (teste de seed fixa)

**Ideia:** com `torch.manual_seed` FIXO (tira o ruído do flow-matching), compara `predict_action_chunk`
com state real vs **zerado** vs **+5**. Se `seed-repro Δ=0` e `state-ZERO Δ>0`, o state **é usado**.

**Por que foi preciso:** a leitura de código sugeria que o pi05 ignora o state (`sample_actions` sem
arg de state). O teste provou que **usa** (Δ 0,2–2,0). Motivo: o state entra como **texto** no prompt,
no pré-processador (ver `g1-pi05-state-como-texto`). Lição: **medir, não confiar só na leitura de código.**

**Rodar:** `python probe_state_check.py` (paths hardcoded pro 8-dim best).

---

## 4. `probe_saliency.py` — SALIÊNCIA VISUAL (onde/quanto olha) + `build_saliency_html.py`

**Ideia:** gerar o pôster visual antigo×novo. Pra 1 frame CLOSED e 1 OPEN, mede 3 coisas, **com a seed do
flow-matching FIXA** (`torch.manual_seed` em cada predição — senão o "pico" da oclusão vira ruído do amostrador):
1. **Quanto usa a imagem** (igual ao probe #1): `sq_img_sens = mean|real−zerada| + mean|real−trocada|`.
2. **Mapa de saliência por OCLUSÃO**: desliza um quadrado pela imagem (grid GxG) e mede |Δsqueeze|. Duas
   variantes — **cinza** (`--grid`, remove conteúdo) e **por troca** (`--swap-grid`, injeta o frame oposto bloco a bloco).
3. Salva `rgb_*.png`, `sal_*.png` (oclusão cinza), `salswap_*.png` (troca) + `manifest.json`.

**Resultado obtido (~5k, seed fixa):** antigo `8hajpdab` sq_img=**0,014** (mapa difuso, foca mesa/braço);
novo `armstate7` sq_img=**0,693** (~50×; mapa concentra **no copo**). Troca da imagem INTEIRA: antigo
1,01→1,01 (ignora), novo 0,99→**0,34** (segue). **Uso holístico:** nenhum bloco isolado vira a decisão
(pico ~0,005) — só a imagem inteira (Δ0,65); os mapas mostram a concentração *relativa* (onde), não a magnitude (quanto).

**Rodar (na Atena, GPU livre ≠ a do treino; python do env por path):**
```bash
PY=/home/hercules/miniconda3/envs/ms3/bin/python
CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/huggingface-models HF_HUB_OFFLINE=1 $PY probe_saliency.py \
  --ckpt train_output/cup_pi05_right8_armstate7_lf/checkpoints/best/pretrained_model \
  --repo-id lewislf/G1_Dex3_pick_white_cup_right8_1squeeze_armstate7 \
  --root datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze_armstate7 \
  --mode 8dim --grid 12 --swap-grid 8 --label armstate7 --outdir /tmp/sal_armstate7
# baseline: --ckpt .../cup_pi05_right8_1squeeze_lf/... --repo-id .../right8_1squeeze --root .../right8_1squeeze --label baseline_8hajpdab
```
Depois, no laptop: `scp -r` os dois `/tmp/sal_*` → `/tmp/sal_pull/` e `python3 build_saliency_html.py /tmp/sal_pull
docs/diagnostico_treino_grasp/RESULTADO_VISUAL_SALIENCIA.html` (embute os PNGs em base64). Render headless:
`google-chrome --headless=new --window-size=1180,5200 --screenshot=/tmp/r.png file://.../RESULTADO_VISUAL_SALIENCIA.html`.

**Gotcha:** os overlays são normalizados ao próprio pico (mostram ONDE) → NUNCA leia magnitude deles; a
magnitude (QUANTO) é o `sq_img_sens`. Mostrar overlay sem essa ressalva é enganoso.

---

## Scripts (versionados no repo)
- `probe_grasp_grounding.py`, `probe_proprio.py`, `probe_state_check.py` — raiz do repo (laptop + Atena).
- `probe_saliency.py` + `build_saliency_html.py` — raiz do repo; geram o pôster visual de saliência.
- `lerobot-ext/tools/slice_drop_finger_state.py` — gera o dataset `state[7]` do experimento #0.
