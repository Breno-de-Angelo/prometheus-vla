# Ablação da propriocepção (`drop-proprio`) — resultado

> Complemento do probe de grounding. Rodado na Atena (GPU2/ms3) com `probe_proprio.py` nos checkpoints
> best, em frames REAIS de treino (12 abertos + 12 fechados). **Imagem sempre REAL**; mexe só no
> `observation.state`. Mede a separação do squeeze previsto (CLOSED − OPEN) por condição: se ela
> **colapsa (→0)** ou **inverte**, esse input é o que carrega a decisão de abrir/fechar. Data: 2026-06-13.

## 8-dim (`8hajpdab`)
| condição | OPEN | CLOSED | separação |
|---|---|---|---|
| baseline (state real) | 0.015 | 0.968 | **+0.953** |
| state ZERADO | −0.030 | −0.044 | **−0.014** (colapsou) |
| state TROCADO (bucket oposto) | 0.896 | 0.016 | **−0.880** (INVERTEU) |
| só braço zerado (dims 0–6) | 0.002 | 0.746 | +0.744 |
| só dedos zerados (dims 7–13) | −0.017 | 0.095 | **+0.111** (quase colapsou) |

## 14-dim (`35jrrbk0`)
| condição | OPEN | CLOSED | separação |
|---|---|---|---|
| baseline (state real) | 0.003 | 0.997 | **+0.994** |
| state ZERADO | 0.003 | 0.011 | **+0.008** (colapsou) |
| state TROCADO | 0.999 | 0.004 | **−0.995** (INVERTEU total) |
| só braço zerado | 0.002 | 0.916 | +0.913 |
| só dedos zerados | 0.003 | 0.409 | **+0.406** (caiu pela metade) |

## Leitura

1. ✅ **MEDIDO: a propriocepção DITA o fechamento.** O `state TROCADO` **inverte** a decisão (separação
   +0.95→−0.88 no 8-dim; +0.99→−0.99 no 14-dim): dar a um frame fechado o `state` de um aberto faz ele
   prever ABRIR, com a imagem real do frame fechado. É causalidade direta. (`state ZERADO` também colapsa.)
   Junto com o probe de imagem (que mostrou a imagem **não** mudar nada), fecha: **grasp = proprioceptivo, cego à visão.**

2. 🔧 **Dominado pela AUTOCORRELAÇÃO dos dedos medidos, não pela pose do braço** (resolve o caveat):
   - zerar **só os dedos** colapsa a separação (8-dim +0.95→+0.11; 14-dim +0.99→+0.41);
   - zerar **só o braço** quase não muda (8-dim +0.74; 14-dim +0.91).
   → o modelo **ecoa o estado atual da mão** ("continua o que os dedos já fazem"); a pose do braço é secundária.

3. **Correção honesta:** a formulação anterior "fecha pela pose do braço" estava imprecisa. O preciso é:
   **fecha porque a propriocepção (sobretudo os dedos medidos) manda — não a visão.**

## Implicação pra solução
Reforça a grounding loss: o expert aprendeu a prever o grasp **ecoando a propriocepção**, ignorando a
imagem. Além de forçar dependência da imagem, vale discutir se **alimentar os dedos medidos como input**
cria justamente esse atalho de autocorrelação (decisão de design pro Luiz).

Script: `probe_proprio.py` (repo da Atena e `~/tmp_prometheus/`). JSONs: `/tmp/proprio_8dim.json`, `/tmp/proprio_14dim.json`.
