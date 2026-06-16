2026-06-13 (criado) · última edição 2026-06-14 05:26

> **O que é (breve):** índice/mapa autoritativo do diagnóstico do treino do grasp (veredito + status das hipóteses + links pros docs da pasta).
> **Correções:** em 14/06 acrescentei os links dos pôsteres de resultado (RESULTADO_NOFINGERSTATE / RESULTADO_VISUAL_SALIENCIA) e o estado ~5k do experimento #0. Cabeçalho adicionado retroativamente em 2026-06-14 12:21.

---

# Diagnóstico do treino / grasp — ÍNDICE (a frente que estamos atacando agora)

> **Pergunta que origina tudo:** "Tenho 238 demos pegando o copo, deveria aprender — por que o robô
> não pega? Tem algo errado no treino, ou o dado está errado?"
> Esta pasta é o mapa autoritativo. Onde houver conflito com docs antigos, **vale o que está aqui.**
> Data: 2026-06-13.

## VEREDITO ATUAL (correto, com evidência medida)

**Não há bug de treino, nem dado corrompido, nem é falta de dado.** O que está errado é:
**a pega não está aterrada na visão.**
- ✅ **MEDIDO** (ablação de imagem no probe): a **imagem NÃO é o gatilho do fechamento** — zerar/trocar a câmera quase não muda o squeeze.
- ✅ **MEDIDO** (ablação `drop-proprio`): o gatilho é a **propriocepção** — mais preciso, a **autocorrelação dos dedos medidos** (trocar o `state` INVERTE a decisão; zerar só os dedos colapsa; zerar só o braço quase não muda). A pose do braço é **secundária**. *(Refina o que antes eu chamava de "pose do braço".)*

Cadeia de evidência:
1. O **braço aprende** (val_action_mse cai 0,077→0,025; generaliza). A `val_loss` que "só sobe" é a
   **proxy de flow-matching** (frouxa), não a ação gerada — as duas DESCOLAM.
2. O **squeeze sai CRAVADO no treino** (probe: separação +0,84 no 8-dim, +0,995 no 14-dim) → o
   objetivo NÃO está quebrado, o modelo decide aberto/fechado.
3. **MEDIDO:** o squeeze **ignora a imagem** (probe ablação: sensib. 0,145 no 8-dim, **0,010** no 14-dim) →
   a visão não gatilha o fechamento. **MEDIDO** (`drop-proprio`): o gatilho é a propriocepção — autocorrelação dos dedos medidos > pose do braço.
4. No robô: o braço erra fino o reach (6–16°/junta → ~cm no end-effector) e a mão fecha pela pose
   mesmo fora do copo → 8-dim fecha no ar; 14-dim cicla.

**Conserto indicado:** forçar o action expert a **usar a imagem pra decidir o grasp** (grounding loss),
agora justificada empiricamente. Como o VLM está congelado (features visuais boas), **não** é VRA.
Secundário: precisão do reach (menos variação de pose / mais dado na região da pega).

## Status das hipóteses

| Hipótese | Status |
|---|---|
| Braço aprende e generaliza | ✅ CONFIRMADO |
| `val_loss` sobe = proxy de flow-matching, não a ação | ✅ CONFIRMADO (descola de val_action_mse) |
| Squeeze cravado no treino (objetivo OK) | ✅ CONFIRMADO (probe) |
| **Imagem NÃO gatilha o grasp** | ✅ **MEDIDO (probe ablação) — causa raiz** |
| Gatilho = propriocepção (autocorrelação dos dedos medidos > pose) | ✅ MEDIDO (ablação `drop-proprio`) |
| Erro fino do braço compõe → erra reach | ✅ CONFIRMADO (6–16°/junta) |
| Dados variam (não é "pega única repetida") | ✅ MEDIDO (pose ~10°, duração CV 32%, copo ~9cm) |
| Rótulo do squeeze limpo (não é dado errado) | ✅ CONFIRMADO (94% transição única) |
| "Squeeze regride pra média / hesita" | ❌ ERRADO (erro de unidade) → ver `CORRECOES.md` |
| "val só sobe = modelo quebrado / braço esquece" | ❌ ERRADO → ver `CORRECOES.md` |
| Drift do VLM → VRA (tese do prof) | ❌ REFUTADO (VLM congelado) → `CORRECOES.md` |
| Falta chunking → ACT (tese do prof) | ❌ REFUTADO (já usa RTC) → `CORRECOES.md` |
| Imagem real usa a visão? (sensib. arm 0,075–0,11) | ⏳ EM ABERTO (toda a política é fracamente visual) |

## Como esta pasta está organizada (separação que você pediu)

- **`00_INDEX.md`** (este) — mapa + veredito + status.
- **`CORRECOES.md`** — ❌ tudo que foi achado ERRADO e a correção (pra não carregar pra frente).
- **`8DIM.md`** — SÓ o modelo de 8 dim (`8hajpdab`): métricas + probe + deploy.
- **`14DIM.md`** — SÓ o modelo de 14 dim (`35jrrbk0`): métricas + probe + deploy.
- **`PLANO_GROUNDING.md` / `.html`** — a SOLUÇÃO planejada (grounding loss, RGB-only).
- **`EXPERIMENTO_NOFINGERSTATE.md`** — 🔬 experimento #0 (run `cup_pi05_right8_armstate7_lf`, wandb `6kr7d8nz`): tirar os dedos do `state`. **✅ leitura ~5k INDO BEM** — sensib. à imagem 0,145→**0,606** (~4×), braço não regrediu; treinando até 20k. Pôster: **`RESULTADO_NOFINGERSTATE.html`**.
- **`RESULTADO_VISUAL_SALIENCIA.html`** — 🔥 pôster VISUAL antigo×novo: câmera RGB, mapa de saliência (oclusão seed-fixa), barras do aperto (teste de troca) e painel "quanto usa cada sinal". Achado: o novo **concentra a saliência no COPO** e usa a imagem ~**50×** mais (0,014→0,693); uso é **holístico** (só trocar a imagem inteira vira a decisão). Gerado por `probe_saliency.py` + `build_saliency_html.py`.
- **`PROBES_COMO_REPLICAR.md`** — como rodar os 3 probes (ablação de imagem, propriocepção, state-check) pra replicar.

### Comparação rápida 8-dim × 14-dim (probe de grounding, frames de TREINO)
| | 8-dim (`8hajpdab`) | 14-dim (`35jrrbk0`) |
|---|---|---|
| Separação squeeze (crisp?) | +0,842 (cravado) | **+0,995 (cravadíssimo)** |
| Sensib. squeeze à imagem | 0,145 | **0,010 (≈zero)** |
| Sensib. braço à imagem | 0,109 | 0,075 |
| Deploy no robô | fecha 1× e segura, **reach errou** | **mão ciclou 7×** |
| best (val_action_mse) | step 8000 | step 5500 |

## Docs de referência (detalhe bruto, fora desta pasta)
- `docs/treino_e_dataset/METRICAS_COMPLETAS.md` — todas as métricas por-dim nos marcos.
- `docs/treino_e_dataset/DIAGNOSTICO_VAL_SOBE.md` — diagnóstico do "val sobe" (contém parte da
  hipótese depois corrigida — ver `CORRECOES.md`).
- `docs/treino_e_dataset/PROBE_GROUNDING_RESULTADO.md` — 1º probe (8-dim) standalone.
- `docs/treino_e_dataset/RELATORIO_TREINO_DATASET.md` — como o treino foi feito + estrutura do dataset.
- `docs/prof-review/` — review do professor + nossas respostas (grounding loss).
- Script do probe: `probe_grasp_grounding.py` (repo da Atena e `~/tmp_prometheus/`).
