# Análise — loss de grounding do grasp (depth/cinemática vs cross-attention)

> ⚠️ **Isto é a EXPLORAÇÃO do thread da review do professor** (rotas 3D/depth/extrínseca/cross-attention,
> + as 2 teses dele). Muita coisa aqui NÃO é a direção escolhida. **O plano vigente** (RGB-only + alvo 2D
> na imagem + ponderação do λ) está limpo em **`../diagnostico_treino_grasp/PLANO_GROUNDING.md`**. Mantido
> como material de referência.


> Documento de trabalho (lastro da resposta `03_resposta_grounding_loss.md`). Resposta aos 2 pontos
> novos do professor + design da loss de grounding, **aterrado no código real** (arquivo:linha).
> Data: 2026-06-13.

## Fatos-base verificados no código (fundamentam tudo)

1. **VLM 100% congelada:** `_set_requires_grad()` põe `requires_grad=False` em todo o `paligemma`
   (vision tower incluso) quando `train_expert_only=true` (`modeling_pi05.py:461-472`). Os 3 YAMLs
   (right14 rgb/depth, right8) usam `train_expert_only:true`.
2. **Loss = flow-matching MSE puro, sem termo auxiliar:** `return F.mse_loss(u_t, v_t, reduction="none")`
   (`modeling_pi05.py:898`). Nada na loss obriga a política a depender da imagem.
3. **Atenção = self-attention CONJUNTA** sobre prefixo `[patches SigLIP | lang | (1 token depth) |
   (1 token pressão)]` + suffix `[50 tokens de ação]`; os Q,K,V da PaliGemma e do expert são
   concatenados e passam pelo mesmo `eager_attention_forward` (`:251-273`). Os tokens de ação
   atendem ao prefixo (`att_mask=0`, `:833-834`).
4. **Os pesos de atenção EXISTEM mas são DESCARTADOS:** `att_output, _ = ...eager_attention_forward(...)`
   (`:273`) — o `_` joga fora os `attention_probs`. Expor exige refatorar `compute_layer_complete`
   nas 18+18 camadas (caminho quente).
5. **O expert recebe a observação via atenção ao prefixo:** não há `state_proj` (era do π0), mas o
   `observation.state` ESTÁ no prefixo — discretizado em texto no prompt (ver ⚠️ correção abaixo). A info
   de observação (imagem + state-texto + linguagem) chega toda via atenção ao prefixo.
6. **Depth entra como UM token global:** o PointNet faz `torch.max(x, 2)` (`depth_encoder.py:78`),
   o que **destrói a estrutura espacial** — vira 1 feature global, não por-patch.
7. **No right8 os 7 dedos são determinísticos:** `squeeze × RIGHT_TARGET`; só o escalar squeeze é
   predito (`inference_realtime_pi05d_right14.py:224-227`).

## Inventário de sinais label-free

| Sinal | Disponível? | Onde / como |
|---|---|---|
| **Depth** | ✅ SIM | `head_camera_depth` PNG16 uint16 mm, 480×848, alinhado ao RGB (D435 hardware). `depth_to_pointcloud()` (`depth_encoder.py:118`) já converte em nuvem 3D com crop de workspace + FPS. |
| **FK (pose da garra)** | ✅ SIM | Pinocchio, frame `R_ee` filho do `right_wrist_yaw_joint`, offset 50 mm em X (`robot_arm_ik.py:87-92`). Derivável de `observation.state[14]`. |
| **Pressão tátil** | ✅ SIM (bruto) | 108 slots (33 taxels reais). Contato derivável por baseline (~10 primeiros frames) + threshold τ (a calibrar). |
| **Pose do copo (real)** | ❌ NÃO | Só no sim (`action_logger.py` `cup_position`). No dado real exigiria mini-detector (SAM2/YOLO em ~20-30 frames) ou cluster geométrico no depth. |
| **Cross-attention por patch** | ⚠️ só com cirurgia | Pesos calculados mas descartados (`:273`); expor = refatorar o core da atenção × 36 camadas. |

## a) Ponto 1 do prof — "congelar o VLM dá tokens genéricos → expert memoriza"

**Driver SECUNDÁRIO, com contradição interna.** Se as features congeladas são boas o bastante pra
tornar VRA moot (sabem "o que é um copo"), não podem ser "genéricas demais" a ponto de impedir o
grounding — a info está no prefixo; o problema é o expert **não usá-la**. Prova: a única via de obs
é a atenção ao prefixo, e a loss (MSE puro) pode ser minimizada **decorando** a trajetória com 238
demos. Cura = **forçar dependência da imagem (grounding loss) + mais dados**, não necessariamente
descongelar. O prof não está 100% errado quanto ao **domínio** (SigLIP é web, não o nosso lab) —
por isso o **meio-termo** (LoRA / últimas N camadas do vision tower) é sensato, mas **depois** da
grounding loss e só se medirmos que precisa (aí o VRA volta a ser relevante, pois a representação
passa a mudar).

## b) Ponto 2 do prof — "14-dim incerto → oscila no chunk → RTC interpola 2 predições ruidosas"

**Inverte causa e efeito.** O jitter é **intra-chunk** (a predição de 50 passos já oscila — os 7
dedos), e o RTC só atua na **fronteira** entre chunks. O RTC pode amplificar na borda, não é a
fonte. **Teste decisivo (barato, código já expõe):** (1) logar o **chunk BRUTO** (pré-RTC) e ver a
trajetória dos dedos — se já treme no cru, é a política; (2) sweep `--rtc` on/off pra isolar a
borda. Fazer **antes** de qualquer mudança de arquitetura.

## c) Ranking das opções de loss de grounding (decisão final é do Luiz)

0. **(diagnóstico, antes de tudo)** logar chunk bruto + sweep RTC — confirma se 14-dim é o problema.
1. **RECOMENDADO — head de PROFUNDIDADE / DISTÂNCIA CINEMÁTICA.** Label-free (depth + FK). Ataca a
   raiz (força o caminho que LÊ a imagem a carregar posição). Custo baixo-médio; invasividade média
   (precisa **expor as features pós-atenção** no forward — hoje só `suffix_out` é usado). Sem risco
   de drift. **Melhor custo/benefício.**
2. **Timing de fechamento via tátil.** Barato, label-free, mas **ortogonal** à mira (resolve "quando
   apertar", não "onde"). Usar como TARGET, nunca entrada (causalidade contato→fechar). Complemento.
3. **Descongelamento PARCIAL (LoRA / últimas camadas).** Só se (1) não bastar — meio-termo do ponto 1.
4. **Cross-attention nos pesos internos.** Mais invasivo (core da atenção × 36 camadas) e efeito
   menos garantido (atenção alta ≠ feature usada); exige os mesmos labels de patch que (1).
5. **Affordance/keypoint.** Labels caros, ganho marginal pro gargalo atual. Baixa prioridade.

## d) A decidir ANTES de implementar (com o Luiz)

1. **ONDE prender a head** (decisão crítica): tem que ser na representação que o **expert consome**
   (features pós-self-attn dos tokens de ação, ou um token-resumo que atende à imagem). Se ler os
   patches SigLIP **crus** (congelados), só re-aprende o que já está lá **sem forçar grounding**.
2. Rodar o diagnóstico barato (b) primeiro.
3. Como obter o "label do copo" no dado real (cluster no depth vs mini-detector).
4. Validar τ do baseline de pressão (se formos pra opção 2); checar se as demos chegam a tocar o copo.
5. Confirmar intrínsecos reais do D435 a 848×480 (`dump_realsense_intrinsics.py`).

---

## e) Extração de p_copo do dado real (resposta à pergunta do prof)

Pergunta do prof: como extrair o centroide 3D do copo (`p_copo`) do **dado real** de forma **escalável**,
já que ele só existe nativamente no sim (`action_logger.py` `cup_position`).

### Onde a head de grounding pluga (confirmado)

`suffix_out` em `modeling_pi05.py:890-896` (hidden states dos tokens de ação, antes do
`action_out_proj` = `nn.Linear(width, max_action_dim)`, `:643`). A head é um MLP paralelo a partir
de `suffix_out` — exatamente o que o prof propôs.

### ⚠️ CORREÇÃO: o `observation.state` ENTRA no modelo — como TEXTO no prompt

> Eu tinha escrito aqui que "o state NÃO entra no modelo" — **ERRADO** (verificado depois: empírico +
> paper π0.5; ver memória `g1-pi05-state-como-texto`). O `state_proj` realmente não existe (era do π0),
> mas no **π0.5 o state é discretizado em 256 bins e injetado como TEXTO no prompt/prefixo**
> (`processor_pi05.py:58-89` → `embed_language_tokens`). Então o modelo **usa** o state.

Implicação pra head de grounding: o `p_ee` (derivável do state) **está** disponível ao modelo, mas só na
forma de tokens de texto discretizado — **não** como vetor limpo que uma MLP lê. Então ainda pode valer
**pré-computar o `p_ee` via FK e passá-lo como tensor** no batch junto com o label — mas o motivo NÃO é
"o state não entra".

### ⚠️ Contexto: o modelo é RGB-only (pi05 base) → o depth é só pro ALVO

As runs (`35jrrbk0`, `8hajpdab`) usam pi05 **RGB-only** (`depth_fusion=false`): o modelo vê só
`head_camera` RGB + state + linguagem. **pi05 base é RGB por construção** (PaliGemma/SigLIP); o
`depth_fusion`/PointNet é um bolt-on custom do repo, **não usado** nessas runs. Logo:
- O **depth (e a FK) entram só OFFLINE pra COMPUTAR o alvo** — **não** viram input do modelo.
- A grounding loss força as **features do RGB** a codificar a posição do copo (= forçar o expert a usar o RGB).

**3 rotas pro alvo (decisão de design):**
- **(A) RGB-only + alvo 2D na imagem** (pixel/heatmap do copo, via detector/segmentação 2D): mais
  simples, **dispensa extrínseca e 3D**, casa com o modelo RGB. **Recomendada pro 1º experimento.**
- **(B) RGB-only + alvo 3D** (distância garra→copo): exige monocular-3D (difícil) + extrínseca.
- **(C) ligar depth-fusion + alvo 3D**: dá o 3D direto, mas é mudança de arquitetura (PointNet).

O "Método primário" abaixo (geometria no depth → 3D) descreve a rota **B/C**; a **rota A** troca isso por
um alvo 2D e **elimina o elo bloqueante da extrínseca**.

### Método primário — geometria no depth (label-free, escalável)

Pipeline (boa parte já existe em `lerobot-ext/train/depth_encoder.py`, a versão usada no treino):
1. depth (PNG16 mm) → nuvem 3D via `depth_to_pointcloud()` (`:118-189`, projeção pinhole).
2. crop de workspace (`:159-163`; `depth_workspace: z[0.2,1.5] x[-0.8,0.8] y[-0.8,0.8]` em
   `train_cup_pi05_right14_depth_238.yaml:152-155`).
3. **(a implementar)** remoção do plano da mesa (RANSAC `segment_plane`) + cluster (DBSCAN) → centroide
   (mediana) do copo. Não existe no repo (open3d só usado p/ viz passiva).
4. transformar pro frame da base via **extrínseco T_base_cam**.

### Elo bloqueante DA ROTA 3D: T_base_cam (extrínseco câmera→base) não existe no real

> Só bloqueia as rotas **B/C** (alvo 3D na base). A **rota A** (alvo 2D na imagem) **dispensa** isso.

No sim a pose vem do MuJoCo (`pickup_experiments/depth_pointcloud.py:140-154`, `d.cam_xpos/cam_xmat`);
no **robô real não foi calibrada** e não está no dataset/config. Sem ela, `p_copo` (frame câmera) e
`p_ee` (frame base, FK) não são comparáveis. Caminhos: (a) calibração fiducial (ArUco na pelvis, ~10 min)
→ JSON da coleta; (b) supervisionar `t` no frame da câmera (dispensa extrínseco, perde a interpretação
garra→copo na base). Intrínsecos hoje são nominais (`fx=fy=425,cx=424,cy=240`, "PROVISÓRIO");
`dump_realsense_intrinsics.py` lê os reais.

### Fallback p/ o copo branco (depth holey)

Detector zero-shot RGB ("white cup": OWLv2/Grounding-DINO, opc. SAM2) → máscara → **mediana** do depth
na máscara → backproject. **Nenhum detector existe no repo** (busca SAM/YOLO/owl/grounding-dino = 0);
é trabalho novo, mas roda 1× offline. RGB (`head_camera`, 848×480) é alinhado ao depth → a máscara indexa
direto o depth. Marcar a fonte de cada label (`source: geom|owlv2`).

### Circularidade (por que FK-no-contato NÃO pode ser o label)

Se `p_copo` viesse da FK no contato, `t = p_copo − p_ee` seria função da trajetória que o expert decora
→ ele prediz sem ler a imagem → anula o grounding. O label **tem** que ser vision-derived. A FK-no-contato
entra só como **cross-check** automático (offline gate): demos são pegas bem-sucedidas, então no contato a
garra está ~no copo; se `p_copo_visão` divergir de `p_ee_contato` em >5-8 cm, marca o label como inválido.
(INCERTO: não há flag de sucesso no schema real; o proxy `hand_cup_contacts` só existe no sim.)

### FK p_ee — detalhe

`forward_kinematics(q)` (`robot_arm_ik.py:315-323`) devolve `T_right` no frame da base; `q` é do modelo
reduzido 14-DOF (braço esq 7 + braço dir 7, **dedos lockados**). `observation.state[14]` = braço dir(7,
idx orig 7-13) + mão dir(7). Logo: pega os **7 primeiros** (braço), monta `[zeros(7), arm(7)]`, lê
`T_right`. **A verificar 1×:** ordem das 7 juntas do braço (state ↔ modelo reduzido).

### Escalabilidade

Script offline 1× sobre 238+24 eps → sidecar por-frame (`cup_center_3d_base [3]` + `grounding_valid [bool]`).
No treino só carrega o `[3]` + máscara; loss = `MSE(head(suffix_out), t)` mascarada onde inválido. Zero
custo de detector no loop.

### A decidir com o Luiz/prof antes de codar
1. **Rota do alvo (RGB-only):** (A) 2D na imagem — **recomendada, dispensa extrínseca** · (B) 3D monocular · (C) ligar depth-fusion + 3D.
2. Supervisão por-frame vs. só na fase de reach (no grasp o copo some atrás da mão).
3. Rodar `dump_realsense_intrinsics.py` no robô (trocar os nominais).
4. Verificar ordem das 7 juntas do braço antes do FK.
5. Confirmar empiricamente que as 238 demos são sucessos (valida o cross-check).
6. Peso/escala da loss de grounding (não inventar — discutir).
