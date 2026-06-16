# Run 4 — conserto do grasp (π0.5 / G1): diagnóstico, proposta e decisões

> **15/06/2026** · Proposta para revisão (1ª versão — nada a corrigir).
> **O que é:** resumo do diagnóstico (deploy no robô + probes causais + auditoria do dataset) da política de pegar o copo, a proposta de conserto (**run 4**) e as **decisões abertas** que gostaria de revisar.

---

## 1. O problema

Política **π0.5** (PaliGemma + action expert Gemma-300m, flow-matching), espaço de ação **right8/armstate7** (7 juntas do braço direito + 1 *squeeze*; o `state` de entrada são só as 7 juntas do braço), task *"pick up the white cup"*, no **Unitree G1 (mão Dex3)**.

No robô, a política **não pega o copo**: o braço vai a uma posição aprendida e a **mão fecha no ar**, longe do copo — que fica parado na mesa. As três variantes de treino que rodamos falham igual. Medimos a causa.

---

## 2. Diagnóstico (com evidência)

**(a) Probe causal de grounding.** Apago/troco a imagem de entrada e meço o quanto a ação prevista muda (sensibilidade à imagem; baixo = a ação ignora a imagem). Rodado no checkpoint exato de cada run, n = 20 frames por fase:

| métrica | as-is | valfix | EMA |
|---|---|---|---|
| **braço → imagem** | **0.069** | **0.078** | **0.072** |
| squeeze → imagem | 0.376 | 0.515 | 0.549 |
| erro de alcance (on-distribution) | 0.043 | 0.043 | 0.045 |

→ **O braço é open-loop nas três** (~0.07): zerar/trocar a imagem quase não altera a ação do braço. O *squeeze* usa um pouco a imagem; **o reach não**.

**(b) A/B das 3 runs no robô** (gravadas frame-a-frame: RGB, profundidade, mapa de atenção, *chunks* previstos, ações executadas):

| métrica | as-is | valfix | EMA |
|---|---|---|---|
| grasp (média do squeeze) | 0.80 | 0.69 | 0.79 |
| % do tempo com a mão fechada | 79% | 66% | 77% |
| flip-flops abre/fecha por decisão | 0.052 | 0.176 | 0.117 |
| corr(squeeze, ombro) | −0.46 | −0.26 | 0.00 |

→ Nem a correção de medição da validação (valfix) nem a EMA tocam o **braço open-loop**; só ajustam o grasp marginalmente (um pouco mais de imagem, menos pose). **As três falham a pega.**

**(c) O dataset NÃO é a causa** (auditado):
- **imagem ↔ ação alinhada** — o vídeo das demonstrações casa com a ação (abre perto do copo → fecha **no copo** → levanta; é uma pega bem-sucedida);
- **grasp-timing limpo** — fecha ~no meio do episódio (depois do reach), ~1 alternância por episódio;
- **reach variado** — em 82 episódios a pose de pega varia (ombro std 0.24, cotovelo 0.38 rad → o copo aparece em posições diferentes), o que **exigiria visão** para acertar.
- *(Duas hipóteses iniciais — "mudança de cor da mesa entre treino e deploy" e "reach estreito demais nas demos" — foram medidas e **refutadas**.)*

**(d) Raiz: causal confusion / atalho proprioceptivo.** No π0.5 o `observation.state` é discretizado (256 bins) e **injetado como texto no prompt** (`"Task: …, State: <bins>; Action: "`). Como as trajetórias são suaves, o `state` prediz a próxima ação quase perfeitamente — então o modelo aprende a **seguir a trajetória pelo `state` e ignorar a imagem**. On-distribution funciona (erro de alcance 0.043); no robô, sem âncora visual, a trajetória deriva e a mão fecha no vazio.

---

## 3. Proposta — Run 4 (conserto)

Três candidatos, do mais barato ao mais caro, cada um com o ponto exato de plugue no código:

1. **state-dropout** *(principal)* — com probabilidade *p* (só no treino), montar o prompt **sem** o trecho `State: …`. Remove a muleta proprioceptiva → força o modelo a usar a imagem. Mudança mínima e localizada (1 step do processor, `Pi05PrepareStateTokenizerProcessorStep`).
2. **grounding loss** *(fase 2, se preciso)* — loss auxiliar de localização do copo, somada à MSE de flow-matching. Precisa de **rótulo do copo** (rodar um detector nos frames → centroide/bbox) + uma cabeça auxiliar.
3. **Knowledge Insulation** *(principiado, caro)* — insular o VLM da loss de ação (preserva o grounding visão-linguagem). É o conserto "certo" descrito na receita da PI, **mas não está implementado no port PyTorch** (estaria no openpi/JAX) — exigiria portar.

**Plano proposto:** Run 4a = **só state-dropout**, medir com a probe causal (a sensibilidade braço→imagem subiu de 0.07?) + deploy no robô; Run 4b = adicionar a **grounding loss** se a 4a não bastar.

---

## 4. Decisões a revisar

1. **p** do state-dropout — 0.3 / 0.5 / 1.0?
2. **Dropar o `state` só no treino** (deploy mantém o `state`) **ou também no deploy** (política puramente visual)?
3. Partir do **base pi05** ou **resumir** de um checkpoint existente?
4. **Prioridade:** rodar a 4a sozinha primeiro (mais barato), ou já 4a + grounding loss juntas? Vale o custo de **portar a Knowledge Insulation**?
5. **Pergunta aberta:** alguma técnica adicional contra causal confusion que você recomendaria aqui (ex.: augmentations visuais agressivas, *modality dropout*, perda de reconstrução auxiliar, etc.)? E concorda que **state-dropout** é o primeiro passo certo, dado que o dataset está limpo e variado?

---

## Anexos disponíveis
- 3 replays interativos (RGB + profundidade + atenção + chunks + ações, por run) + 1 relatório editorial do diagnóstico.
- wandb (`prometheus-lcad/prometheus_g1`): run 1 as-is `6kr7d8nz` · run 2 valfix `y32omum0` · run 3 EMA `6ivtoov9`.
