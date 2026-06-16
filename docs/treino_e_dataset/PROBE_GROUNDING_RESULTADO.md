# Probe de grounding do grasp — resultado (8-dim best, frames de TREINO)

> Teste rodado na Atena (GPU2, env ms3) com `probe_grasp_grounding.py` no checkpoint
> `cup_pi05_right8_1squeeze_lf/checkpoints/best` (step 8000). Frames REAIS de treino (eps 0–213):
> 12 com squeeze GT aberto (0) + 12 fechado (1). Para cada um, `predict_action_chunk` em 3 condições
> de imagem: real / zerada / trocada (imagem de um frame do bucket OPOSTO). squeeze = action[7],
> q01=0/q99=1 → raw = (norm+1)/2 (exato). Data: 2026-06-13.

## Resultado

| condição | GT squeeze | pred real | imagem zerada | imagem trocada |
|---|---|---|---|---|
| OPEN | 0.000 | **0.005** ±0.019 | 0.011 | 0.024 |
| CLOSED | 0.998 | **0.847** ±0.288 | 0.909 | 0.975 |

- **Separação** (closed_real − open_real) = **+0.842** (perto de 1.0 = crava; perto de 0 = morno).
- **Sensibilidade do squeeze à imagem** (|real−zerada| + |real−trocada|) = **0.145** (perto de 0 = ignora).
- **Sensibilidade do braço à imagem** (norm) = **0.109**.

## Leitura

1. **O objetivo NÃO está quebrado.** Em dado memorizado o squeeze sai CRAVADO (~0 pra abrir, ~0.85
   pra fechar; separação 0.84) — **não regride pra média**. Confirma a correção: não há defeito de
   flow-matching, e a premissa "mesmo overfitando deveria cravar a pega" se sustenta. (O CLOSED tem
   dispersão ±0.29 = o resíduo de ~5–6% de frames que vimos nas métricas.)

2. **✅ MEDIDO: a IMAGEM não é o gatilho do fechamento.** Zerar a imagem (0.847→0.909) ou trocá-la pela
   de um frame ABERTO (0.847→0.975) **não** empurra o squeeze pra abrir — ele continua fechando.
   Sensibilidade do squeeze à imagem = 0.145 (baixa). Isso prova diretamente que a visão não dispara o grasp.
   **⏳ INFERIDO (por eliminação, NÃO medido ainda): o gatilho é a propriocepção (pose).** Com a imagem
   neutralizada, o único outro input que distingue um frame fechado é o `observation.state`. Dois caveats
   honestos: (a) falta rodar o teste complementar `drop-proprio` (zerar o state) pra confirmar; (b) o
   `state` inclui os **7 dedos medidos**, então parte do "prever fechado" é **autocorrelação** (continuar
   o que a mão já faz), não necessariamente a pose do braço.

## Causa raiz

A pega **não está aterrada na visão** (medido: a imagem não gatilha o fechamento). No dataset funciona;
no robô, o braço tem erro de **6–16°/junta** → a garra para alguns cm fora do copo, e a mão **fecha assim
mesmo** (gatilho não é a visão — medido; ser a pose = inferido). Bate com o observado: 8-dim fechou no ar; 14-dim ciclou.

**Não é** bug de treino, **não é** dado corrompido, **não é** "falta dado". É **grounding visual fraco
do grasp** — exatamente o que a *grounding loss* ataca, agora empiricamente justificada. Como o VLM
está congelado (features visuais boas), o conserto é **forçar o action expert a usar a imagem pra
decidir o grasp**, não VRA.

## Caveats / próximos
- Perturbações de frame único (sensibilidade local), não rollout.
- Falta corroborar no **14-dim** (precisa ajustar o probe pros dedos 8–13 / projeção do squeeze).
- Script: `probe_grasp_grounding.py` (no repo da Atena e em `~/tmp_prometheus/`).
