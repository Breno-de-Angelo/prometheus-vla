# 14-DIM — `cup_pi05_right14_rgb238_lf` (wandb `35jrrbk0`)

> SÓ dados do modelo de **14 dimensões** (7 braço + 7 dedos). Não misturar com o 8-dim (`8DIM.md`).

## Identidade
- **action[14]** = 7 juntas do braço + **dims 7–13 = dedos**. Mas os dedos = `squeeze × RIGHT_TARGET`
  (1-DOF efetivo); **dim 7 (thumb_0) é morta** (T=0, ~constante). RIGHT_TARGET=[0,−0.92,−1.74,1.57,1.74,1.57,1.74].
- Dataset: `lewislf/G1_Dex3_right14_dataset` (v3_238ep). Run: **finished 20000**. **best = step 5500** (val_action_mse 0,0759).

## Métricas (nos marcos) — fonte: `docs/treino_e_dataset/METRICAS_COMPLETAS.md`
| | 1000 | best 5500 | 5000 | 10000 | 15000 | 20000 |
|---|---|---|---|---|---|---|
| train/loss | 0.036 | 0.018 | 0.020 | 0.027 | 0.008 | 0.009 |
| val_loss (flow, proxy) | 0.099 | 0.124 | 0.131 | 0.215 | 0.285 | **0.317** ↑ |
| val_action_mse (ação gerada) | 0.108 | **0.076** | 0.125 | 0.092 | 0.185 | 0.190 |
| dedos 08–13 (action_mse) | ~0.18 | **~0.146** | ~0.25 | ~0.184 | ~0.394 | **~0.405** ↑ |
| thumb_0 dim_07 | ~0.0007 | ~0.0003 | — | ~0.0002 | ~0.0001 | ~0.00005 (morta) |

- Braço (ação gerada) ~0,01–0,07. **Dedos:** mínimo no best (~0,146) e depois **degradam pra ~0,40**
  (overfit real da mão — aqui loss E mse sobem juntos). Os 6 dedos andam em bloco (mesma variável).

## Probe de grounding (frames de TREINO, GPU2/ms3 · `probe_grasp_grounding.py --mode 14dim`)
| condição | GT squeeze | pred real | imagem zerada | imagem trocada |
|---|---|---|---|---|
| OPEN | 0.000 | **0.006** ±0.005 | 0.007 | 0.005 |
| CLOSED | 0.998 | **1.001** ±0.006 | 1.000 | 1.002 |

- **Separação = +0,995** → squeeze **CRAVADÍSSIMO** (mais crisp que o 8-dim). Objetivo OK, sem hesitação.
- **Sensib. squeeze à imagem = 0,010 (≈ZERO)**; braço = 0,075. Zerar ou trocar a imagem **não muda nada**.
  → ✅ **MEDIDO: a imagem não tem efeito nenhum no fechamento** (ainda mais cego que o 8-dim). Que o
  gatilho é a propriocepção (✅ medido no `drop-proprio`: trocar o `state` inverte +0,99→−0,99), dominada
  pela **autocorrelação dos dedos medidos** (zerar dedos: +0,99→+0,41; zerar braço: quase não muda, +0,91).

## Deploy no robô (best 5500) — `docs/investigacao_deploy_robo/`
- **Mão CICLOU abre/fecha 7×** (early-close f69), pressão travada no baseline (sem grip). Coerente com
  o probe: como o squeeze é cravado por pose e ignora a visão, ao mover o braço por poses diferentes
  (deploy ≠ treino) ele **slama abre/fecha seguindo a pose memorizada**, sem o copo gatilhar nada.

## Conclusão (14-dim)
Igual ao 8-dim, em versão mais extrema: squeeze cravadíssimo no treino (objetivo OK), e a **imagem não
tem efeito no fechamento** (sensib. 0,010 — medido). ✅ **Medido (`drop-proprio`):** o gatilho é a
propriocepção, dominada pela **autocorrelação dos dedos medidos** (> pose do braço). O ciclar no robô é
o sintoma direto de uma pega cega à visão. Além disso, os dedos sofrem overfit real após o best
(0,146→0,405), reforçando parar no best.
