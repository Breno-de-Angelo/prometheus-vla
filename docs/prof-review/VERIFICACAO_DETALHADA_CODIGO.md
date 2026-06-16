# Verificação detalhada contra o código — lastro do `RESPOSTA_REVIEW.md`

> Documento de trabalho (não enviado ao professor — é o *show-your-work*). Cada afirmação da review
> foi confrontada com o código/config/dados reais por investigação dedicada. Veredito:
> **CONFIRMA / REFUTA / PARCIAL / INCERTO**, com `arquivo:linha` ou dado concreto.
>
> Data: 2026-06-13. Arquivos-fonte: `lerobot-ext/policies/pi0_depth/modeling_pi05.py`,
> `inference_realtime_pi05d_right14.py`, `lerobot-ext/config/train/train_cup_pi05_right14_rgb_238.yaml`,
> `train_cup_pi05_right8_1squeeze.yaml`, `lerobot/src/lerobot/policies/rtc/modeling_rtc.py`,
> `lerobot/src/lerobot/datasets/utils.py`.

---

## Área 1 — Controle, latência e action chunking

| Veredito | Afirmação da review | O que o código mostra (evidência) |
|---|---|---|
| **PARCIAL** | "Roda a ~3 fps (333 ms), observações obsoletas → covariate shift" | A taxa de **inferência** é ~3 fps (1 chunk a cada ~333 ms), mas a de **controle é 30 fps** (`inference_realtime_pi05d_right14.py:417` `step_period = 1.0/args.fps`, fps=30) e a **mão a 100 Hz** (hand-streamer). O "stale" é **dentro do chunk executado**, não entre observações. |
| **REFUTA** | "Inferência autoregressiva passo-a-passo, sem action chunking" | `predict_action_chunk` gera **chunk de 50 ações** nos dois loops: síncrono (`:631`/`:634`) e RTC (`:563-567`). **Não existe caminho step-wise no código.** |
| **REFUTA** | "Solução: implementar ACT + Temporal Ensembling" | Já existe via **RTC (Real-Time Chunking)**: `_run_rtc_loop()` (`:509-620`) prediz com `inference_delay` + `prev_chunk_left_over` e faz **inpainting** na emenda (`modeling_rtc.py:116-145`, merge em `actionqueue.py:128-154`). Ativo nas runs: `--rtc-execution-horizon 25 --rtc-max-guidance 1.0`. |
| **PARCIAL** | "policy_view.mp4 gravado a ~3 fps (1 frame/chunk)" | Vídeo segue o frame rate da câmera ZMQ amostrada por observação (≈30 fps), **não** 3 fps fixos. A ilusão de 3 fps vem da taxa de *replanejamento*. Convém logar timestamp por frame. |
| **REFUTA** | "Degraus do loop síncrono causam o jitter/ciclagem" | O jitter é propriedade **do chunk predito**, não do loop: o **8-dim com o mesmo código (síncrono+RTC) NÃO cicla** (fecha 1× e segura). Se fosse o loop, ambos ciclariam. |
| **CONFIRMA** | "RTC usa inpainting pra suavizar emendas" | Implementado e ativo (`modeling_rtc.py:116-145` `denoise_step()`; `actionqueue.py:128-154` `merge()`). |
| **REFUTA** | "Envio de ações ao robô a 3 fps" | Envio a **30 fps** (`:417`, loop a cada 33 ms `:669-671`) + mão a 100 Hz. Inferência roda em thread separada; a fila (`:585-607`) consome a 30 fps sem pausa. |

**Conclusão Área 1:** o diagnóstico causal do jitter está errado. Chunking + RTC já existem; o jitter do
14-dim é *aprendizado errado dentro do chunk* (overfit + grounding pobre), não artefato de loop.

---

## Área 2 — Congelamento do VLM e *representational drift*

| Veredito | Afirmação da review | O que o código mostra (evidência) |
|---|---|---|
| **PARCIAL** | "Fine-tuning erode o VLM (drift); rede esquece o copo, aprende atalho" | **Sintoma real** (grounding colapsou), mas **causa falsa**: com `train_expert_only=true`, `_set_requires_grad()` congela toda a PaliGemma (`modeling_pi05.py:467-470`: `self.paligemma.eval()` + `requires_grad=False`). VLM congelada **não pode driftar**. |
| **CONFIRMA** | "`train_expert_only=true` congela a PaliGemma; só o expert treina" | `modeling_pi05.py:467-470`; configs `:88`/`:91`. Treinam só action expert (Gemma-300M) + projeções. |
| **REFUTA** | "`freeze_vision_encoder=false` → vision tower treina" | `train_expert_only` é **mais restritivo e vence**: congela TODA a PaliGemma, vision tower incluso, independente de `freeze_vision_encoder`. O encoder de visão **fica congelado**. |
| **CONFIRMA** | "~693 M treináveis são o expert + projeções (não o VLM)" | Consistente: PaliGemma (2B) congelada; treinam `gemma_expert` (~300M) + `action_in/out_proj` + `time_mlp`. *(O número exato ainda merece reconciliação num run real.)* |
| **INCERTO→MOOT** | "Solução: VRA (alinhar features contra SigLIP professor congelado)" | A premissa de VRA (encoder treinando e degradando) **não vale aqui** (encoder congelado). VRA seria relevante só num cenário `train_expert_only=false`, **não testado**. |

**Conclusão Área 2:** a falha de grounding é real, mas a causa não é drift do VLM (impossível). É o
**action expert decorando atalhos** (frame-number/fundo) sem acoplamento visual-motor + multimodalidade
nos dados. VRA é moot no setup atual.

---

## Área 3 — Espaço de ação (joint vs task, absoluto vs delta)

| Veredito | Afirmação da review | O que o código mostra (evidência) |
|---|---|---|
| **CONFIRMA** | "Usa joint-space com ações absolutas (não delta)" | `meta/info.json` `action.names` terminam em `.q` (posição); inferência trata como `target_q` absoluto. |
| **CONFIRMA** | "É joint-space (7 braço + 7 dedos), não task-space" | Nenhuma dim é xyz/rpy de EE; `action` shape [14]/[8] = juntas diretas. |
| **PARCIAL** | "Joint-space cria 'abismo semântico' (rede aprende cinemática)" | Correto que prediz posições sem cinemática explícita — **mas pi05 base, OpenPI e RT-2 também usam joint-space** com sucesso; com chunking aprende trajetórias. |
| **INCERTO** | "Solução: task-space (Cartesiano EE) + delta (RT-2/OpenVLA)" | Válido em princípio, mas **força causal baixa**: não há evidência de que seja a causa-raiz vs. dados insuficientes / overfit. Refactoring grande (IK/controle cartesiano). |
| **CONFIRMA** | "Joint-space absoluto é frágil a descalibração" | Verdade em geral — mas **sem evidência** de que cup_pi05 falhou por descalibração (há soft-start de abertura da mão antes de cada run). |

**Conclusão Área 3:** factualmente correto (joint-space absoluto), mas a recomendação task-space+delta é
ortogonal e cara, não um quick-fix com força causal demonstrada.

---

## Área 4 — Split e augmentation

| Veredito | Afirmação da review | O que o código mostra (evidência) |
|---|---|---|
| **CONFIRMA (geral)** | "Cuidado com split aleatório por frame (data leakage)" | Verdadeiro como princípio. |
| **NÃO SE APLICA** | "(implícito: vocês podem estar vazando frames)" | Split é **por episódio (trajectory-level) e sequencial**: treino eps `[0..213]`, val `[214..237]`; filtro por `episode_index` (`lerobot/.../datasets/utils.py:130-132`). Zero vazamento de frame. |
| **PARCIAL** | "Color jitter anula o rótulo; usar VISTA" | `ColorJitter` só no **treino** (não no val); `RandomResizedCrop` (scale [0.75,1.0]) já dá variação **espacial** tipo VISTA-lite. **VISTA não está implementado** (`grep VISTA` = 0). |

**Conclusão Área 4:** o split já é o que a review recomenda como correto. Augmentation é pragmática, não
principiada (VISTA), mas funcional. *(Ironia: split sequencial introduz domain-shift leve, medido como
não-dominante; overfitting é o efeito dominante.)*

---

## Área 5 — Solidez e aplicabilidade da literatura

| Veredito | Recomendação | Avaliação |
|---|---|---|
| **MOOT** | VRA / *Don't Blind Your VLA* | Técnica real, mas **inaplicável** (VLM congelado → sem drift a corrigir). |
| **JÁ FEITO** | ACT (chunking) + Temporal Ensembling | **Já implementado** via RTC (chunk 50 + inpainting). |
| **FORÇA BAIXA** | Task-space + delta (RT-2/OpenVLA) | Real, mas reformulação grande; joint-space prova funcionar; sem evidência de ser a raiz. |
| **VÁLIDO, DEPOIS** | Co-training 50/50 (Open-X / Bridge V2) | Real e estabelecido. **Não usado** (treinos só com 238 eps). Custo alto. Aplicável como passo posterior. |
| **EXAGERO P/ AGORA** | RL-Co (sim-real, PPO) | Real, mas pesado/especulativo com 238 demos e sim de baixa fidelidade. |
| **CORRETO** | Congelar o VLM com poucos dados | Decisão correta dado o regime de dados. |
| **CORRETO** | Mão 1-DOF limita preensão fina | Confirmado pelos dados (right8 fecha/segura sem ciclar; right14 cicla). |
| **PARCIAL** | Domain shift do split sequencial | Medido **leve/não-significativo**; overfitting domina. |
| **CONFIRMA** | Overfitting → `best` no val ≠ generalização | `train_loss→0,008`, `val` sobe; A/B best vs last só ~11%; ambos `best` falharam no robô. |

**Conclusão Área 5:** das recomendações, **3 são já-feitas/inaplicáveis** (VRA, ACT, VLM-frozen-é-correto),
**2 são válidas-mas-posteriores** (co-training, RL), **1 é reformulação ortogonal** (task-space). O
diagnóstico clínico (overfit + grounding + 1-DOF) é sólido; as alavancas reais são **dados + encoding da
mão + regularização**, não VRA nem chunking.
