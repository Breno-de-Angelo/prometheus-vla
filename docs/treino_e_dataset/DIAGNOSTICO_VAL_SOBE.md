# Diagnóstico — "por que a val só sobe se eu tenho dado?"

> Investigação numérica (wandb per-dim + aprendibilidade do dado + sanidade de norm/métrica) da
> queixa: "238 demos, deveria aprender, mas a val_loss só sobe — está errado." Data: 2026-06-13.
> **Não foi assumido "falta dado".** Todos os números abaixo são reais (extraídos do wandb e dos parquet).

## Veredito em uma frase

A VLA **aprende o braço e generaliza** (val do braço cai 0,077→0,025); a "val que só sobe" é **~91%
seis dims de dedo que são a MESMA variável binária** (`squeeze × RIGHT_TARGET`) medida com a métrica
errada (L2 contínuo num alvo bimodal). **Não é falta de dado nem bug de normalização.** O que está
genuinamente em aberto — e decide se a mão um dia fecha na hora certa — é **se o expert está olhando
pra imagem**, e isso **ainda NÃO foi medido**.

## Pesos das causas (com evidência)

| Causa | Peso | Evidência dura |
|---|---|---|
| **Artefato de métrica** (mão 1-DOF redundante domina o MSE) | **~55%** | Dedos = 77,7% → 83,0% → **91,2%** do val_action_mse total (s500→s5500→s20000). As 6 dims efetivas (8-13) sobem em *lockstep* (todas ~0,405 @20k) — SVD: 2º valor singular = 1,8% do 1º → **rank-1, é uma variável só**. Dim 7 (thumb_0) está **morta** (~0,0001). |
| **Dado/alvo intrinsecamente não-aprendível na mão** | **~30%** | Squeeze é bimodal (58% aberto / 40% fechado / ~2% no meio); timing do fechamento varia **118±43 frames (~1,4 s)**. Decisivo: o squeeze-**state** logo antes de fechar = 0,077, **idêntico** a um frame aberto (0,083) → a **propriocepção NÃO carrega a pista do "quando fechar"**. kNN conditional-mean explica só **0,23** da variância do squeeze na val (RMSE 0,419) → propagado pelo target reproduz exatamente o "dedos val~0,40". É **piso irredutível**, não treino ruim. |
| **Overfit do braço + split sequencial** | **~12%** | Braço overfita no flow-loss (gap train-LOO vs val 20–100×), com domain-shift moderado (RElbow inicial desloca 0,51 std; range da val ~10-20% menor). MAS o `val_action_mse` do braço fica **baixo o treino todo (~0,025–0,033)** → cosmético. |
| **Bug mecânico de normalização** | **~0%** | Train e val usam o MESMO `stats.json` global (52952 frames); nenhuma dim com range < 1e-3 (guard não dispara); VISUAL=IDENTITY (sem bug BGR/RGB). **Descartado.** |
| **Conditioning quebrado (expert ignora a imagem)** | **INCERTO** | Não medido. É *inferido* que só a imagem poderia resolver o timing (a propriocepção não resolve). **É a única hipótese aberta de verdade — e a que mais importa pro sucesso real da mão.** |

## O braço está aprendendo? — SIM, decisivamente

- val_action_mse médio-por-dim do braço: 0,0771 (s500) → 0,0258 (s5500) → **0,0254 (s8000, mínimo)** → ~0,0334 (s20000). **Nunca explode.**
- Per-dim @20k (d0..d6): 0,067 / 0,019 / 0,011 / 0,059 / 0,021 / 0,034 / 0,022 — todas baixas.
- Aprendibilidade independente do modelo: kNN conditional-mean atinge **0,67–0,90 de variância explicada** no braço na val. O alcance *é* aprendível e o modelo *está* generalizando nele.

## O Luiz está certo que "está errado"?

- **SIM, no que importa:** a política real não fecha a mão de forma confiável — problema legítimo, e os números explicam **por quê** (alvo 1-DOF binário, timing não-observável pela propriocepção). E você está certo: **não é falta de dado** — adicionar demos NÃO resolve.
- **NÃO, no que você temia:** a *curva agregada subindo* **não** é modelo quebrado nem treino enviesado — é majoritariamente artefato de medir 6 cópias da mesma variável binária com L2. O **`best`@5500 que você deployou É o ponto útil** (a métrica, apesar de ruim, capturou o lugar certo).
- Ressalva: o base já está em val~0,10 e qualquer fine-tune piora a val de imediato (run do zero `1sozoy32`: val 0,105 ≥ base 0,10 já no s1000) — **esperado** pra fine-tune pi0 em tarefa estreita com alvo binário, não prova de quebra. (A run do zero crashou no s7000 sem logar per-dim → separação braço/dedos nela = INCERTO.)

## Próximos passos (baratos, offline, ANTES de re-treinar)

1. **Re-plotar val SÓ do braço (0-6) vs SÓ dedos (7-13)** — as colunas `eval/val_action_mse_dim_00..13` já estão no wandb da 35jrrbk0. Custo ~zero. É o slide que encerra a discussão.
2. **Medir a variância da métrica:** re-rodar o eval 5–10× no mesmo checkpoint (o `predict_action_chunk` usa `sample_noise()` **sem seed fixo**, só 4 batches). Quantifica quanto da oscilação é ruído de amostragem. Fix: fixar seed + subir `max_action_mse_batches` 4→16-32.
3. **★ TESTE DE GROUNDING VISUAL (o que decide o futuro da mão):** rodar o eval do checkpoint com a imagem **zerada/embaralhada** e ver se o val_action_mse dos **dedos** muda. Se NÃO mudar → o expert ignora a imagem → e como a propriocepção comprovadamente não tem o timing, a mão **nunca** vai fechar na hora certa. Barato, e é a hipótese aberta mais importante.
4. **Split aleatório por episódio** (não sequencial) ao recomputar a métrica — separa overfit de domain-shift (~12% do quadro).

## Fix mecânico?

**Não há bug de norm.** Os fixes reais são de **métrica** e de **modelagem do alvo**:
- **Métrica:** reportar val do braço e do squeeze (1 escalar) **separados**; não somar 6 cópias do squeeze (com targets grandes tipo 1,74) no agregado.
- **Modelagem do alvo (a discutir, não decidir):** o squeeze deveria ser previsto como **evento/binário condicionado na IMAGEM** (classificação/threshold), não regressão L2 contínua — porque a propriocepção não tem o "quando" e o L2 num alvo bimodal é inadequado por construção.
