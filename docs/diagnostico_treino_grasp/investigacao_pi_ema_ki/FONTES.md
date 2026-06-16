# Fontes e citações — investigação da receita PI (EMA × Knowledge Insulation)

**Escrito:** 2026-06-14 23:50 (-03)
**O que é:** as citações verbatim + URLs que sustentam o relatório `INVESTIGACAO_PI_EMA_KI.html`. Pesquisa feita em fontes oficiais (pi.website, openpi GitHub, docs LeRobot, papers arxiv) em 2026-06-14.
**Nota:** arquivo novo — nada sobrescrito. Citações conferidas contra HTML do arxiv (não o PDF — uma extração de PDF do KI devolveu números de página suspeitos e foi descartada).

---

## 1. EMA no openpi (código oficial JAX da PI) — VERDICT: usa EMA

Repo: https://github.com/Physical-Intelligence/openpi (branch `main`, conferido 2026-06-14)

- **Default ligado** — `src/openpi/training/config.py`:
  ```python
  ema_decay: float | None = 0.99    # default do TrainConfig
  ```
- **Update Polyak por step** — `scripts/train.py`:
  ```python
  ema_params=jax.tree.map(
      lambda old, new: state.ema_decay*old + (1-state.ema_decay)*new,
      state.ema_params, new_params)
  ```
  (decay **flat**, sem warmup ramp — nós adicionamos warmup.)
- **`pi05_libero` (exemplo de fine-tune do pi05_base)** — `config.py`: `ema_decay=0.999`, `batch_size=256`, `num_train_steps=30_000`, `CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_lr=5e-5)`.
- **`pi05_droid_finetune`** (template p/ dataset custom): `num_train_steps=20_000`, `batch_size=32`, `ema_decay=0.99`. ← igual à nossa config exceto lr.
- **LoRA desliga EMA** — `config.py`:
  ```python
  # Turn off EMA for LoRA finetuning.
  ema_decay=None,
  ```
- **PyTorch port NÃO suporta** — `README.md` lista como "not currently supported": `LoRA training` e `EMA (exponential moving average) weights during training`.
- Memória: LoRA fine-tune `>22.5 GB` (RTX 4090); **full fine-tune `>70 GB`** (A100-80GB/H100).

## 2. Freeze do VLM — a PI co-treina (full FT) ou usa LoRA; nunca o nosso híbrido

- `src/openpi/models/pi0_config.py` → `get_freeze_filter()`: freezing controlado por filtro de path; **full FT = `freeze_filter` vazio → tudo treina, inclusive PaliGemma**. LoRA = congela full-rank, treina adapters.
- **Não existe `train_expert_only` no openpi** — é um flag do LeRobot.
- **Docs LeRobot/pi05** (https://huggingface.co/docs/lerobot/en/pi05), comando de treino recomendado:
  ```bash
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false   # co-treina o VLM
  --steps=3000 --batch_size=32
  ```
  Tabela de parâmetros: `train_expert_only` default `false` = "Do not freeze the VLM, train all parameters". E a dica: *"Setting `train_expert_only=true` freezes the VLM and trains only the action expert and projections, allowing finetuning with **reduced memory usage**."* → economia de memória, não recomendação de qualidade.

## 3. Knowledge Insulation (KI) — o conserto principiado do grounding

Paper: https://arxiv.org/abs/2505.23705 (Driess, Springenberg, Ichter, Pertsch, Levine et al.) · HTML: https://arxiv.org/html/2505.23705v1 · PDF hospedado pela PI: https://www.pi.website/download/pi05_KI.pdf · página: https://www.pi.website/research/knowledge_insulation

- Problema (abstract): *"naively including such [continuous flow/diffusion action] experts significantly harms both training speed and knowledge transfer."*
- Mecanismo:
  - *"fine-tune the VLM backbone with discretized actions while simultaneously adapting an action expert to produce continuous actions ... without propagating its gradients back into the VLM backbone."*
  - *"we propose to stop the gradient flow from the action expert to the pre-trained weights ... where sg denotes the stop-gradient operator."*
  - *"using next-token prediction [FAST tokens] makes the model learn much faster and more stably."*
  - *"co-train the model on non-action datasets such as general vision-language data."*
  - *"stopping the gradient flow ... is an effective way of improving language following."*
- Explicador NeurIPS 2025 (paráfrase): os gradientes do expert recém-iniciado, se entram no backbone, *"flood the VLM, destroying the delicate feature detectors it learned during pre-training"* → *"a robot that moves smoothly but can't tell an apple from a rock."* (https://neurips2025.pages.dev/explainers/knowledge_insulating_vision_language/)
- **⚠️ DOIS modos de falha distintos (correção — congelar NÃO é o "destruidor"):**
  - **(a) Congelar = INSUFICIENTE** (não destrutivo) — §4: *"VLM pretraining does not have sufficient representations for robotics — freezing doesn't work … their representations, when frozen, are insufficient for training highly performant policies."* Preserva conhecimento; falha por falta de adaptação à tarefa (Fig 4a: ~0% no cenário from-scratch). **É o NOSSO regime.**
  - **(b) Co-treino naive = DESTRUIDOR do grounding** — §4: *"naive training with such a randomly initialized action expert harms the models' ability to follow language commands (presumably due to gradient interference)."* É o **RISCO de destravar mal**.
  - Caveat: o expert é **random** no cenário do paper (*"initialized from scratch"*); o NOSSO vem pré-treinado do `pi05_base` → interferência mais branda. E o `pi05_base` já foi treinado COM KI → nosso VLM congelado já tem grounding de robótica (não é o "0%" do paper).
- **Distinção crucial:** KI **não congela** o VLM — treina ele na loss de ação discreta COM stop-grad no expert (resolve (a) e (b) juntos). O nosso `train_expert_only=true` congela o VLM inteiro E remove a loss discreta.

## 4. Como o π0.5 é treinado (co-treino + hierárquico)

Paper: https://arxiv.org/abs/2504.16054 · HTML: https://arxiv.org/html/2504.16054v1 · blog: https://www.pi.website/blog/pi05

- *"π0.5 ... uses co-training on heterogeneous tasks to enable broad generalization"* — mix: mobile manipulation, multi-environment, cross-embodiment, high-level subtask prediction, multimodal web data (captioning/VQA/detecção).
- Hierárquico: *"the model first predicts the semantic subtask ... and then predicts the low-level robot action chunk based on this subtask."* (a ação de pega é condicionada numa subtarefa de linguagem que o MESMO VLM tira da imagem.)
- Discreto→contínuo: pré-treino com FAST tokens; pós-treino adiciona o action expert de flow-matching.

## 5. Dados / overfit

- π0 open-sourcing (https://www.pi.website/blog/openpi): *"between 1 and 20 hours of data was sufficient to fine-tune to a variety of tasks"* → nossas 238 demos de 1 tarefa estão no/abaixo do piso; overfit é o modo de falha esperado.
- Norm stats (openpi `docs/norm_stats.md`): reusar stats do pretrain só ajuda se *"your robot was part of our pre-training mixture"* (ALOHA/DROID). O G1/Dex3 não está → quantis frescos (o que fazemos) é o certo.

## 6. EMA — ausência nos papers

- Nenhuma menção a EMA/exponential moving average nos papers π0 (2410.24164), π0.5 (2504.16054) ou KI (2505.23705). O controle de variância/qualidade da PI nos papers é por **dados** e **arquitetura (KI)**, não EMA. (A EMA aparece só no código `openpi` como default de otimização.)

---

## Confiança e lacunas (honesto)

- **Alta:** EMA é default do openpi (0.99 / 0.999) — citação file:line, corroborada por 2 buscas independentes. Mecanismo do KI (stop-grad + loss FAST + não-congela VLM) — múltiplas fontes concordando.
- **Média:** que o KI mova a NOSSA métrica de grasp. O paper mede "language following", não timing de fechamento de dedo. Direção certa, magnitude a medir.
- **Média-alta:** "congelar o VLM é a causa do nosso grounding fraco" — consistente com o nosso diagnóstico + teoria do KI, mas o armstate7 já ataca o atalho de propriocepção dos dedos; não é a história inteira.
- π0.6/π0.7 confirmados **não-abertos**; não mudam nossas opções (seguimos no pi05_base).
