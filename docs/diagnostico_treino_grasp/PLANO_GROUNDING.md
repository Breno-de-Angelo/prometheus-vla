# Plano — aterrar a visão (grounding loss), RGB-only

> Plano vigente pra próxima rodada. Foco e limpo (companion em texto do `PLANO_GROUNDING.html`).
> A exploração ampla do thread da review do prof (rotas 3D/depth/extrínseca/cross-attention) está em
> `docs/prof-review/ANALISE_GROUNDING_LOSS.md` — material de referência, **não** é o plano.

## Por quê (o diagnóstico, medido)
A pega não está aterrada na visão: a mão fecha por **propriocepção** — dominada pela **autocorrelação
dos dedos medidos** — e **ignora a imagem** (zerar/trocar a imagem não muda o squeeze; trocar a
propriocepção INVERTE a decisão). O braço aprende; o gargalo é o grasp não-visual. *(Detalhe:
`00_INDEX.md`, `PROBE_GROUNDING_RESULTADO.md`, `PROBE_PROPRIO_RESULTADO.md`.)*

## A ideia
`L_total = L_flow + λ · L_ground`.
- Uma **head MLP** pendurada nas features do **action expert** (em `suffix_out`, antes do
  `action_out_proj` — `modeling_pi05.py:890-896`) que prevê **onde o copo está**.
- Pra acertar isso, os tokens de ação têm que **ler o RGB** → força o grounding visual do grasp.
- **Não é VRA**: o VLM está congelado (features boas); o que falta é o expert **usar** o RGB.

## Modalidade: RGB-only (igual o pi05 base)
pi05 base é **RGB por construção** (PaliGemma/SigLIP: RGB + state + linguagem). As runs usam
`depth_fusion=false`. A grounding loss **não muda a modalidade** — só obriga o modelo a usar o RGB que
já tem. O `depth_fusion`/PointNet é um bolt-on custom **não usado**.

## Rota escolhida: A — alvo 2D na imagem
- **Alvo = posição do copo NA IMAGEM** (pixel/centro/heatmap), gerado **offline** por um detector/
  segmentação 2D (zero-shot tipo OWLv2/Grounding-DINO/SAM2, ou cor HSV no copo branco), **1× sobre os
  238 eps**, cacheado num sidecar (`cup_xy` + `valid`).
- **Dispensa:** extrínseca câmera→base, 3D monocular, FK, p_ee, e depth como input. (Tudo isso era a
  rota 3D — ver abaixo.) → caminho mais barato pro 1º experimento.
- O label vem da **VISÃO**, nunca da trajetória da garra (seria circular e não forçaria grounding).
- **Onde prender a head importa:** tem que ser na representação que o **expert consome** (`suffix_out`),
  não nos patches SigLIP crus — senão não força o expert a usar a imagem.
- Alternativas (se a 2D não bastar): **B** = alvo 3D monocular (precisa extrínseca); **C** = ligar
  depth-fusion + 3D (mudança de arquitetura). Ficam pra depois.

## Plano de ponderação do λ (pra não dominar o treino)
1. **Igualar a escala primeiro** — normalizar o alvo (variância unitária); senão nenhum λ fixo é estável.
2. **Começar pequeno + ramp (warmup)** — λ: ~0 → sobe gradual (0→~2k steps). Imitação se estabelece
   primeiro, grounding refina depois.
3. **Balancear por gradiente** — medir ‖∇L_flow‖ vs ‖∇L_ground‖; λ pra o grounding ser **10–30%** do
   gradiente, nunca dominar.
4. **Sweep curto + critério certo** — λ ∈ {≈0,05 · 0,1 · 0,25}; escolher **não** pela loss de treino,
   e sim por: **(a)** sensibilidade à imagem ↑ (o probe) **e (b)** `eval/val_action_mse_arm` **não** regride.
5. **Guardrail anti-dominação** — logar separado `loss_flow` vs `loss_ground`. Se o braço piorar ou a
   flow-loss parar de convergir → **λ alto demais, reduz**.

## Como saber que funcionou (sem robô)
- Re-rodar o probe de ablação (`probe_grasp_grounding.py`): a **sensibilidade do squeeze à imagem deve
  SUBIR** (hoje 0,01–0,15 → algo real).
- **`val_action_mse_arm` ≤ baseline** (o reach não pode regredir).
- Juiz final: no robô, a mão fecha **quando vê o copo**.

## Decisões abertas (do Luiz, não invento)
- λ **inicial** + **schedule** (ramp vs constante).
- **Rota do alvo** (A 2D recomendada; B/C se precisar).
- **Tirar os dedos medidos do input?** — eles criam o atalho de autocorrelação medido (a mão fecha por
  eco da propriocepção). Removê-los pode ser tão ou mais decisivo que a loss.
- **Detector 2D** pro alvo: zero-shot (OWLv2/Grounding-DINO/SAM2) vs cor (HSV branco).
- **Supervisão por-frame vs só na fase de reach** (no grasp o copo some atrás da mão → label degenerado).

## O que precisa ser implementado
1. **Script offline** — detecta o copo (2D) em cada frame → sidecar (`cup_xy` + `valid`).
2. **Head MLP** no expert (`suffix_out`) + termo de loss **mascarado** onde `valid=False`.
3. **Logging** separado `loss_flow` / `loss_ground` (pro guardrail do passo 5).

## Referências
- Diagnóstico: `00_INDEX.md`, `8DIM.md`, `14DIM.md`, `PROBE_GROUNDING_RESULTADO.md`, `PROBE_PROPRIO_RESULTADO.md`.
- Infográfico: `PLANO_GROUNDING.html`.
- Exploração ampla (review do prof, rotas 3D): `docs/prof-review/ANALISE_GROUNDING_LOSS.md` (referência, não o plano).
