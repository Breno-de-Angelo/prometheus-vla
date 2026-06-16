# 8-DIM — `cup_pi05_right8_1squeeze_lf` (wandb `8hajpdab`)

> SÓ dados do modelo de **8 dimensões** (7 braço + 1 squeeze). Não misturar com o 14-dim (`14DIM.md`).

## Identidade
- **action[8]** = 7 juntas do braço + **dim 7 = `squeeze`** (escalar ∈[0,1]; dedos no deploy = squeeze×RIGHT_TARGET).
- Dataset: `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze` (mesmo conteúdo bruto do 14-dim, só a action mudou).
- Run: **crashed ~step 15900** (não chegou a 20000). **best por val_action_mse = step 8000** (0,0456).

## Métricas (nos marcos) — fonte: `docs/treino_e_dataset/METRICAS_COMPLETAS.md`
| | 1000 | best 8000 | 5000 | 10000 | 15500 (fim) |
|---|---|---|---|---|---|
| train/loss | 0.063 | 0.014 | 0.028 | 0.017 | 0.006 |
| val_loss (flow, proxy) | 0.102 | 0.228 | 0.160 | 0.291 | **0.363** ↑ |
| val_action_mse (ação gerada) | 0.073 | **0.046** | 0.054 | 0.050 | 0.054 |
| squeeze dim_07 (action_mse) | 0.228 | 0.200 | 0.220 | 0.233 | 0.244 |

- Braço (ação gerada) cai e fica baixo (~0,01–0,05). Squeeze é o pior (~0,20–0,24) — mas em espaço
  normalizado ({−1,+1}, piso da média = 0,95) isso é **~4× melhor que a média**, não hesitação.

## Probe de grounding (frames de TREINO, GPU2/ms3 · `probe_grasp_grounding.py --mode 8dim`)
| condição | GT squeeze | pred real | imagem zerada | imagem trocada |
|---|---|---|---|---|
| OPEN | 0.000 | **0.005** ±0.019 | 0.011 | 0.024 |
| CLOSED | 0.998 | **0.847** ±0.288 | 0.909 | 0.975 |

- **Separação = +0,842** → squeeze **CRAVADO** no treino (não regride pra média). Objetivo OK.
- **Sensib. squeeze à imagem = 0,145** (baixa); braço = 0,109. Trocar pela imagem de um frame ABERTO
  não empurra pra abrir (0,847→0,975). → ✅ **MEDIDO: a imagem NÃO gatilha o fechamento.**
- Dispersão do CLOSED (±0,29) = o resíduo de ~5–6% de frames com timing flipado.

> ✅ **MEDIDO (`drop-proprio`):** a ablação de imagem prova que a **imagem não dispara** o grasp; a
> ablação de `state` prova que **a propriocepção dita o fechamento** — trocar o `state` INVERTE a
> decisão (sep +0,95→−0,88). E é dominado pela **autocorrelação dos dedos medidos** (zerar só os dedos
> colapsa: +0,95→+0,11; zerar só o braço quase não muda: +0,74). Pose do braço = secundária.
> Detalhe em `PROBE_PROPRIO_RESULTADO.md`.

## Deploy no robô (best 8000) — `docs/investigacao_deploy_robo/`
- **Mão fechou 1× e segurou (sem ciclar)**, mas o **braço errou o alvo** — fechou no ar, no canto,
  longe do copo. Coerente: o reach tem erro de 6–16°/junta (compõe na cadeia, ~cm no end-effector), e
  a mão fecha pela pose mesmo fora do copo.

## Conclusão (8-dim)
Não há bug nem dado errado. O braço aprende mas erra fino o reach. **Medido:** a imagem não gatilha o
fechamento, e a propriocepção dita (dominada pela **autocorrelação dos dedos medidos** > pose do braço).
Falha no robô = reach fora + grasp cego à visão.
