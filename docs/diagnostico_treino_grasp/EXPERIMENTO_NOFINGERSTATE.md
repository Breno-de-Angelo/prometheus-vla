2026-06-14 (criado) · última edição 2026-06-14 04:45

> **O que é (breve):** registro do experimento #0 — tirar os dedos medidos do `observation.state` (state[14]→[7]), hipótese, como foi feito e o resultado parcial (~5k).
> **Correções:** em 14/06 acrescentei o resultado ~5k (INDO BEM) e o comando literal do probe final. Cabeçalho adicionado retroativamente em 2026-06-14 12:21.

---

# Experimento #0 — tirar os dedos medidos do `observation.state`

> Primeiro experimento da frente "aterrar a visão". RGB-only. Disparado 2026-06-14.
> Run wandb: **`cup_pi05_right8_armstate7_lf`** (`prometheus-lcad/prometheus_g1`, id `6kr7d8nz`), GPU2 Atena, ~20h.

## Hipótese
O diagnóstico mediu que a mão **fecha pela propriocepção, dominada pela autocorrelação dos dedos
medidos** (probe `drop-proprio`: zerar SÓ os dedos do `state` colapsa a decisão +0,95→+0,11; trocar o
`state` inverte a decisão), **ignorando a imagem** (sensib. do squeeze à imagem 0,01–0,15). Ou seja: o
modelo prevê o squeeze ecoando o estado atual dos dedos, sem olhar o copo.

**Tese do experimento:** se a gente **remover os dedos medidos do input** (`state[14]→[7]`, só o braço),
o modelo perde esse atalho e é forçado a achar o sinal de "quando fechar" em outro lugar — idealmente a
imagem. É o teste mais barato e direto à causa-raiz, **sem mexer no modelo nem precisar de rótulo**.

## Como o state entra (por que mexer no dataset basta)
No π0.5 o `observation.state` **não** é um vetor projetado — ele é **discretizado em 256 bins e injetado
como TEXTO no prompt** (`State: 12 200 ...;`), virando tokens de linguagem no **prefixo** do PaliGemma
(`processor_pi05.py:58-89` → `embed_prefix`/`embed_language_tokens`). Logo, **tirar os dedos do state =
tirar esses tokens de texto** — basta o dataset ter `state[7]`. Detalhe: `g1-pi05-state-como-texto` (memória).

## O que foi feito
1. **Slice do dataset** (`lerobot-ext/tools/slice_drop_finger_state.py`): do `right8_1squeeze`
   (state=14, action=8) → **`..._right8_1squeeze_armstate7`** com `state[7]` (7 juntas do braço; remove
   os 7 dedos medidos), **action[8] INTACTA** (o squeeze continua na ação), stats/episodes recomputados.
2. **Config** `lerobot-ext/config/train/train_cup_pi05_right8_armstate7.yaml`: cópia exata do baseline
   right8, mudando só `observation.state.shape [14]→[7]`, o dataset (root → armstate7) e o nome do job;
   `keep_only_best_and_last=true` (segurança de disco). RGB-only, action[8], 20k steps, lr — tudo idêntico.
3. **Smoke test** (30 steps): eval com `state[7]` rodou sem erro de shape → pipeline OK.
4. **Launch** `launch_armstate7_lf.sh` (GPU2, wandb luiz-coutinho).

## Critério de sucesso (sem robô)
Depois do treino, re-rodar **o mesmo probe** (`probe_grasp_grounding.py --mode 8dim`) no `best` novo e
comparar com o baseline `8hajpdab`:
- ✅ **sensibilidade do squeeze à imagem SOBE** (baseline 0,145 → algo maior); **→ ✅ deu 0,606 no ~5k**
- ✅ **`val_action_mse` do braço NÃO regride** vs baseline; **→ ✅ 0,041 no ~5k**
- (E rodar `probe_proprio.py` pra ver se a decisão migrou pro braço — ver caveat.)

### Comando literal do probe (rodar na Atena ao bater 20k — GPU1, treino fica na GPU2)
```bash
cd ~/Prometheus/Luiz/prometheus-vla
CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/huggingface-models HF_HUB_OFFLINE=1 \
  conda run -n ms3 python probe_grasp_grounding.py \
    --ckpt  train_output/cup_pi05_right8_armstate7_lf/checkpoints/best/pretrained_model \
    --repo-id lewislf/G1_Dex3_pick_white_cup_right8_1squeeze_armstate7 \
    --root    datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze_armstate7 \
    --mode 8dim --out /tmp/probe_armstate7_final.json
```
Comparar `sensibilidade SQUEEZE a imagem` vs **0,606 (~5k)** e vs baseline **0,145**, e `val_action_mse_arm`
final vs 0,041. Se subiu/manteve → confirmado; aí atualizar `RESULTADO_NOFINGERSTATE.html` com o número final.

## Caveat honesto
Tirar só os dedos remove o atalho **mais forte**, mas o **braço continua no state-texto** e correlaciona
com a fase do grasp (no `drop-proprio`, zerar só o braço ainda manteve +0,74). Então o fechamento pode
**migrar pro state do braço** em vez de ir pra imagem. É um teste isolado e válido, com essa ressalva — se
não bastar, o próximo passo é a grounding loss (`PLANO_GROUNDING.md`).

## Pendência pro deploy real (se este experimento vingar)
**Coerência treino↔deploy:** o `inference_realtime_pi05d_right14.py` precisa enviar `state[7]` (cortar os
dedos do state também no deploy), senão os bins desalinham e o state vira ruído. (Regra: cortou no treino,
corta no deploy — ver `g1-pi05-state-como-texto` e `g1-lerobot-dataset-compat-controle`.)

## Arquivos
- Slice: `lerobot-ext/tools/slice_drop_finger_state.py`
- Config: `lerobot-ext/config/train/train_cup_pi05_right8_armstate7.yaml`
- Launch (Atena): `launch_armstate7_lf.sh`
- Probes p/ avaliar: `probe_grasp_grounding.py`, `probe_proprio.py` (ver `PROBES_COMO_REPLICAR.md`)

## ✅ RESULTADO PARCIAL (~5000 steps, 14/06) — INDO BEM

Probe rodado no `best` (~5k) na GPU1 (treino seguindo na GPU2), dataset `..._armstate7`:

| | baseline `8hajpdab` | **armstate7 (~5k)** |
|---|---|---|
| **Sensib. do squeeze à imagem** | 0,145 | **0,606** (≈ **4×**) |
| Trocar a imagem | não mexe o squeeze | **FLIPA**: CLOSED 0,978→0,327 ; OPEN −0,008→0,229 |
| Separação squeeze (crava?) | +0,84 | **+0,986** |
| `val_action_mse` braço | ~0,04 | **0,041** (não regrediu) |
| `val_action_mse` grasp | ~0,20 | 0,238 (≈, leve ↑) |

**Veredito:** ✅ os **dois critérios bateram** — a sensibilidade do squeeze à imagem **subiu ~4×** (0,145→0,606)
e o braço **não regrediu** (0,041). Mais forte ainda: **trocar a imagem agora INVERTE a decisão de fechar**
(no baseline a troca era ignorada). A pega está **se aterrando na visão** — a hipótese do experimento se
confirmou. É leitura **parcial (~5k/20k)**; o `val_mse` do grasp subiu um tiquinho (esperado: usar a imagem
é mais difícil que o atalho dos dedos). **Decisão: deixar treinar até 20k** e re-rodar o probe no best final.

Evolução do eval (arm/grasp) até o ~5k:
```
 step  val_action_mse  arm     grasp
 3500     0.0658       0.0478  0.1918
 4000     0.0614       0.0417  0.1987
 4500     0.0641       0.0388  0.2412
 5000     0.0658       0.0412  0.2383
```
HTML pro Luiz: **`RESULTADO_NOFINGERSTATE.html`** (estilo dos outros). Probe salvo em `/tmp/probe_armstate7_best.json` (Atena).

## Estado atual (overnight autônomo, 14/06)
- **Run vivo:** wandb `6kr7d8nz` · PID **3640623** na Atena · GPU2 · ~3 s/step (5k ≈ 4h, 20k ≈ 17h).
- **Logging patchado** (run_train.py): `train/`+`eval/` principais; per-dim em `dim_train/`/`dim_eval/`;
  `eval/val_action_mse_arm` + `_grasp` (split). Confirmado funcionando no train-side.
- **Watcher armado** (background, `~/tmp_prometheus/watch_armstate7.sh`): dispara no **step 5000** (ou se o run morrer).
- **Plano quando disparar (autônomo):** (1) rodar `probe_grasp_grounding.py --mode 8dim` + `probe_proprio.py`
  no `best` (na **GPU1**, sem atrapalhar o treino na GPU2), dataset `..._armstate7`; (2) comparar sensib.
  do squeeze à imagem vs **baseline 0,145** e `val_action_mse_arm` vs baseline; (3) **se subiu** = indo bem,
  deixar treinar; **se ~0** = ainda cego à imagem (migrou pro braço) → investigar em **pasta nova**,
  decidir/implementar próximo experimento, relançar; (4) gerar **HTML** (estilo `INFOGRAFICO.html`) com
  problema + tentativa + resultado, pro Luiz validar de manhã.
