# Relatório — Correções da Auditoria do Pipeline π0.5 (branch Luiz-pi05d)

Relatório incremental: cada fase é documentada ao ser concluída (o que foi feito,
como, resultado). Data de início: 2026-06-10, madrugada.

---

## FASE 1 — Diagnóstico: o token de depth está sendo usado? ✅ CONCLUÍDA

**Suspeita da auditoria:** o token injetado poderia estar marcado como padding e
ser ignorado pela atenção.

**O que foi feito:**
1. Análise estática de `embed_prefix` (`lerobot/src/lerobot/policies/pi05/modeling_pi05.py:634-675`)
   e do patch do injector (`lerobot-ext/train/pi05_depth_injector.py:86-105`):
   - tokens de **imagem**: `pad_mask = img_mask` (True quando a câmera existe), `att_mask = 0`;
   - tokens de **linguagem**: `pad_mask = máscara do tokenizer` (True), `att_mask = 0`;
   - token de **depth** (injector): `extra_pad = torch.ones(...)` → pad_mask = **True**
     (linha 131) e `extra_att = torch.zeros(...)` → att_mask = **0** (linha 100).
   - Em `make_att_2d_masks` (`modeling_pi05.py:101-130`), att=0 mantém o token no
     mesmo bloco bidirecional; pad=True o mantém visível. Ou seja: o token de depth
     recebe tratamento IDÊNTICO ao dos tokens de imagem.
2. Teste empírico `lerobot-ext/tests/test_depth_token_effect.py`, rodado na **Atena
   (GPU 0)** com o **checkpoint real** do run depth
   (`train_output/cup_pi05_right14_depth_lf/checkpoints/005000/pretrained_model`):
   mesma observação + mesmo ruído de flow matching, depth real vs depth zerado.
   Notas de implementação: seed antes de cada chamada (o subsample de 1024 pontos
   usa `torch.randperm` e mudaria o token entre chamadas); `config.json` do
   checkpoint saneado em cópia temp (skew de versão: campo `tokenizer_name` não
   existe mais no PI05Config); no modo checkpoint os intrínsecos usados são os
   MESMOS do treino (default fx=600/cx=320) de propósito, para medir o modelo como
   ele foi treinado.

**Resultado:**
```
[máscaras do prefixo] total=305 tokens (img=256, lang=48, depth=1)
  imagem : pad=[True] att=[False]
  língua : pad=[True] att=[False]
  depth  : pad=[True] att=[False]
[determinismo] mesmo input 2x: max abs diff = 0.000e+00 (esperado 0)
[efeito depth] real vs zerado : max abs diff = 3.172e-02, MSE = 8.710e-05
```

**Veredito: suspeita NÃO confirmada.** As máscaras estão corretas e o token de
depth influencia o chunk de ação do modelo treinado (diff 3.17e-2 ≫ 1e-6, com
determinismo exato). Nenhuma correção foi necessária no injector. Efeito modesto
(~3% no espaço normalizado), consistente com 5k steps e geometria de nuvem
distorcida pelos intrínsecos errados (corrigidos na FASE 3).

**Arquivos tocados:** `lerobot-ext/tests/test_depth_token_effect.py` (novo).

---

## FASE 2 — Diagnóstico: NaN/Inf na normalização (dim 7) ✅ CONCLUÍDA

**Suspeita da auditoria:** q01 = q99 na dim 7 (right_hand_thumb_0) → divisão por
zero → NaN/Inf.

**O que foi feito:**
1. Leitura do código da quantile norm
   (`lerobot/src/lerobot/processor/normalize_processor.py:362-377`): existe guard
   `denom == 0 → eps=1e-8`, então NaN/Inf direto não acontece.
2. Check empírico replicando a fórmula exata (com o guard) sobre **todos os 3027
   frames** do right14 (`/tmp/check_norm_finite.py` + `/tmp/dim7_precise.py`).

**Resultado do diagnóstico — NaN/Inf NÃO confirmado, mas problema real encontrado:**
```
state  dim7: q01=-4.586695e-02 q99=-4.585977e-02 → range=7.19e-06 (54 valores únicos = ruído LSB do encoder)
             norm resultante: [-2.82, +3.01]   ← ruído amplificado ×2.8e5 à escala dos sinais reais
action dim7: raw EXATAMENTE 0.0 sempre, mas q01=4.0e-16, q99=4.0e-14 (poeira numérica do quantile)
             norm resultante: -1.020408 constante (≠ -1)
isfinite: True em todas as dims (o guard eps segura)
```
Detalhe crucial: o range da dim 7 do state NÃO é exatamente zero (7.2e-6), então
nem o guard `==0` do LeRobot nem um fix por igualdade disparam — o denominador
minúsculo amplifica ruído do mesmo jeito. O critério correto é **threshold de
range mínimo**.

3. Fix aplicado (o menos invasivo, sem mudar shapes):
   - `lerobot-ext/tools/slice_right_arm_only.py`: `_guard_zero_range_quantiles()` —
     ao gerar stats do dataset fatiado, qualquer dim de state/action com
     `q99 - q01 < QUANTILE_MIN_RANGE` (1e-3 rad) recebe `q99 = q01 + 1.0`, com
     warning. TODO right13 documentado no docstring.
   - `lerobot-ext/train/run_train.py`: `_guard_zero_range_quantile_stats()` —
     safety net em memória no load do dataset (antes do normalizer ser construído),
     mesmo critério, cobre datasets já fatiados e futuros (ex.: v3_grasp 32-dim).

**Re-check com o fix:**
```
observation.state: finito=True  dim7 norm=[-1.000013, -0.999971]  (≈ -1 constante ✓)
action:            finito=True  dim7 norm=[-1.000000, -1.000000]  (exato ✓)
```
(max|norm| global de ~4.3 em outras dims é comportamento esperado de quantile
norm para outliers além de q01/q99.)

**Arquivos tocados:** `lerobot-ext/tools/slice_right_arm_only.py`,
`lerobot-ext/train/run_train.py`.

---

## FASE 3 — Intrínsecos reais + depth_scale seguro ✅ CONCLUÍDA

**O que foi feito:**

1. **Leitura dos intrínsecos reais do sensor:** o `realsense_server_depth16.py`
   vive apenas no robô (não há cópia neste repo; robô offline durante a
   implementação). Criado `lerobot-ext/tools/dump_realsense_intrinsics.py` —
   roda no robô, abre o stream de depth na resolução pedida, lê via
   `profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()`,
   salva JSON {fx, fy, cx, cy, width, height, model, coeffs, serial} e imprime o
   bloco YAML pronto pra colar na config. O docstring traz o snippet exato pra
   embutir no server na próxima coleta.

2. **Intrínsecos via YAML:** novo campo `depth_intrinsics: dict[str, float] | None`
   no `CustomTrainPipelineConfig` (`lerobot-ext/train/run_train.py`), encaminhado
   para `inject_pi05_depth(...)` e `inject_pi05_d(...)` (o modo `full` nem recebia
   esses parâmetros antes). Parse draccus do dict validado por smoke test.
   `train_cup_pi05_right14_depth.yaml` ganhou o bloco com os nominais D435 848x480
   (fx=fy=425, cx=424, cy=240) e comentário PROVISÓRIO apontando o script acima.

3. **Default 640x480 REMOVIDO:** `validate_intrinsics()` em
   `lerobot-ext/train/depth_encoder.py` — `camera_intrinsics=None` ou incompleto
   levanta ValueError explicando onde configurar (YAML / CLI / script do robô).
   Aplicado nos dois injectors.

4. **depth_scale obrigatório:** default `2.0` removido do
   `CustomTrainPipelineConfig` (agora `float | None = None`); `run_train.py`
   levanta ValueError se `depth_fusion: true` sem `depth_scale`; os dois injectors
   também exigem o parâmetro (o `pi05_d_injector` tinha 0.001 hardcoded inline —
   agora parametrizado). Inferência (`inference_realtime_pi05d_right14.py`) ganhou
   `--depth-fx/fy/cx/cy` e `--depth-scale`, repassados ao injector.

5. **Sanity-assert do primeiro batch:** `depth_cloud_sanity_check()` em
   `depth_encoder.py`, executado UMA VEZ por injector (flag `_depth_sanity_done`)
   — cobre o 1º batch do treino E a inicialização da inferência. Loga
   min/p5/mediana/p95/max da primeira nuvem em metros; mediana dos pontos válidos
   (z > 0.05) fora de [0.2, 3.0] m → RuntimeError com a mediana na mensagem.

**Resultado dos testes (`/tmp/test_fase3.py`):**
```
✓ ValueError p/ None            (mensagem aponta YAML + dump_realsense_intrinsics.py)
✓ ValueError p/ {'fx': 425.0}   (faltam ['cx','cy','fy'])
✓ sanity passou com depth_scale=0.001 (mediana ~0.6m)
✓ RuntimeError p/ scale=1.0 → "mediana = 599.716 m, fora de [0.2, 3.0]"
✓ RuntimeError p/ scale=2.0 → "mediana = 1199.431 m, fora de [0.2, 3.0]"
✓ inject_pi05_depth exige camera_intrinsics e depth_scale
✓ draccus decodifica depth_intrinsics: dict[str,float] | None
```

**Arquivos tocados:** `lerobot-ext/train/depth_encoder.py`,
`lerobot-ext/train/pi05_depth_injector.py`, `lerobot-ext/train/pi05_d_injector.py`,
`lerobot-ext/train/run_train.py`, `lerobot-ext/config/train/train_cup_pi05_right14_depth.yaml`,
`inference_realtime_pi05d_right14.py`, `lerobot-ext/tools/dump_realsense_intrinsics.py`
(novo), `lerobot-ext/tests/test_depth_token_effect.py` (intrínsecos da era do treino
explícitos no modo checkpoint).

---

## FASE 4 — Workspace crop + Farthest Point Sampling ✅ CONCLUÍDA

**O que foi feito** (`lerobot-ext/train/depth_encoder.py`):

1. **Crop de workspace configurável:** `depth_to_pointcloud()` ganhou o parâmetro
   `workspace` ({"z": [min,max], "x": ..., "y": ...} em metros, no frame da
   câmera; eixos ausentes não são cropados). Plumbing completo: campo
   `depth_workspace: dict[str, list[float]] | None` no `CustomTrainPipelineConfig`
   → injectors → `depth_to_pointcloud`. Valores no YAML
   (`train_cup_pi05_right14_depth.yaml`): z ∈ [0.2, 1.5], x ∈ [-0.8, 0.8],
   y ∈ [-0.8, 0.8] — nada hardcoded.

2. **FPS no lugar do randperm:** `farthest_point_sampling()` puro em torch,
   batched (ops vetorizadas por batch, loop só em K). Pipeline: filtro z>0.05 +
   crop → pré-subsample uniforme a 16384 candidatos (teto de custo, mesmo padrão
   do 3D Diffusion Policy) → FPS para 1024. Padding com zeros + warning com a
   contagem quando sobram < 1024 pontos. pytorch3d não está no env ms3 → FPS puro
   conforme previsto pela auditoria (O(M·K) com M≤16384 é barato).

3. **Script de inspeção:** `lerobot-ext/tools/inspect_depth_cloud.py` — N frames
   aleatórios → pipeline completo da config → scatter 3D + vista de topo + vista
   frontal em PNG.

**Resultados:**
```
unit tests (/tmp/test_fase45.py):
  crop ON : z range [0.60, 0.60]m — fundo sintético a 5m excluído ✓
  crop OFF: fundo presente (max 5.0m) ✓
  FPS: menor dist entre pontos = 0.028m (cobertura espacial, sem duplicatas) ✓
  crop agressivo: 4/1024 não-nulos + warning + zero-pad ✓
  determinismo com seed ✓ | custo CPU: 0.15s/amostra (GPU: bem menor)

inspeção em 5 frames reais do right14 (intrínsecos nominais 848x480 + workspace):
  1024/1024 pontos em todos; z mediana 0.535–0.577 m (mesa dominando ✓)
  PNGs: lerobot-ext/docs/img_auditoria/cloud_frame*.png
```

**Arquivos tocados:** `lerobot-ext/train/depth_encoder.py`,
`lerobot-ext/train/pi05_depth_injector.py`, `lerobot-ext/train/pi05_d_injector.py`,
`lerobot-ext/train/run_train.py`, `train_cup_pi05_right14_depth.yaml`,
`lerobot-ext/tools/inspect_depth_cloud.py` (novo).

---

## FASE 5 — Zero-init da projeção final do PointNet ✅ CONCLUÍDA

**O que foi feito:** em `PointNetEncoder.__init__` (`depth_encoder.py`),
`nn.init.zeros_` em weight E bias APENAS da `fc2` (512→hidden_size). Camadas
anteriores (conv1/2/3, fc1) mantêm init padrão. Comentário no código explica o
padrão (estilo ControlNet: token entra como no-op, o modelo aprende a usá-lo
gradualmente, sem ruído de init aleatória no prefixo do modelo pré-treinado).

**Resultado dos testes:**
```
PointNet zero-init: |saída| max = 0.0e+00 para input aleatório ✓
load_state_dict (caminho do load_injected_from) sobrescreve o zero-init ✓
```

**Arquivos tocados:** `lerobot-ext/train/depth_encoder.py`.

---

## FASE 6 — Higiene de docs ✅ CONCLUÍDA

1. `README_treino_pi05_right14.md`: corrigida a seção "O que é treinado/congelado"
   — SigLIP está **congelado** por `train_expert_only: true`
   (`modeling_pi05.py:417-420` congela `paligemma.parameters()` inteiro, incluindo
   a vision tower); `freeze_vision_encoder: false` é inerte nesta config.
2. `README_depth16.md`: shape de validação corrigido para `[480, 848, 1]`.
3. `README_treino_pi05_right14.md`: nota sobre execução open-loop do chunk de 50
   (~1.67 s @ 30 Hz) e recomendação de replanejar após ~25 ações
   (`--actions-per-chunk 25`) — apenas documentado, sem mudar a inferência.

---

## NÃO FEITO (conforme escopo da auditoria)

- Injeção do token no action expert (planejada pós-validação do pipeline).
- Depth dropout (entra junto com a fase de mais dados).
- Hiperparâmetros/augmentations/split/chunk size: intocados.
- Nenhum retreino automático disparado *por esta auditoria* (o treino noturno no
  dataset novo de 238 eps é tarefa separada, pedida pelo usuário).

## Trabalho noturno pós-auditoria (madrugada 2026-06-10)

1. **Commit/push:** auditoria commitada na `Luiz-pi05d` (`c360cb3`, 14 arquivos) e
   puxada na Atena.
2. **Dataset 238 eps:** `slice_right_arm_only.py` adaptado p/ action[32] →
   `lerobot-ext/datasets/G1_Dex3_right14_dataset/v3_238ep` (238 eps, 52952 frames,
   action/state[14]); rsync completo pra Atena (data+meta+vídeo real).
   Guard da FASE 2 atuou no slice (action dim 7). Bug extra encontrado e corrigido:
   o merge_datasets do LeRobot descarta as keys de imagem do stats.json → KeyError
   no make_dataset; entradas de imagem restauradas no v3_238ep e no v3_grasp.
3. **Treino RGB 238 eps NO AR na Atena (GPU 0):**
   config `train_cup_pi05_right14_rgb_238.yaml` (split 214/24, 20k steps, warmup
   2k, decay 18k, bs 32, save a cada 5k — checkpoint tem 9.1G), wandb
   `prometheus_g1/1sozoy32` (entity prometheus-lcad, chave do Luiz injetada).
   step 100: loss 0.304→0.173. ~3s/step → ETA ~16h.
   Pegadinhas do launch (documentadas na memória): SSH não-interativo não lê o
   .bashrc → `HF_HOME=/data/huggingface-models` + `HF_HUB_OFFLINE=1` obrigatórios
   (senão pega snapshot novo incompatível do pi05_base e 403 no paligemma gated).
4. **Treino depth 238 encadeado:** watcher `run_depth_after_rgb_lf.sh` na Atena
   dispara o `train_cup_pi05_right14_depth_238.yaml` (intrínsecos nominais 848x480,
   crop, FPS, zero-init, sanity) quando o RGB terminar — vigia o padrão de
   processo (não PID, que é frágil com workers do dataloader).
5. **Push do v3_grasp pro HF: BLOQUEADO em credencial.** Não há token HF do
   lewislf no laptop e o token da Atena é do Rafael-LCAD (não usei). Para subir:
   `hf auth login` (laptop, conta lewislf) e depois
   `conda run -n g1 python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset(repo_id='lewislf/G1_Dex3_pick_white_cup_v3_grasp', root='datasets/_merged/G1_Dex3_pick_white_cup_v3_grasp', video_backend='pyav').push_to_hub()"`

## Pendências para a próxima coleta de dados

1. **Intrínsecos reais do sensor**: rodar no robô
   `python lerobot-ext/tools/dump_realsense_intrinsics.py --width 848 --height 480`
   e substituir os nominais (fx=fy=425, cx=424, cy=240) no
   `train_cup_pi05_right14_depth.yaml`; opcionalmente embutir o snippet do
   docstring no `realsense_server_depth16.py` (robô) pra salvar o JSON junto de
   cada gravação.
2. **Dim 7 (right_hand_thumb_0)**: continua congelada e protegida pelo guard de
   stats; na fase de escala, decidir entre destravar o polegar na teleop ou migrar
   para **right13** (remover a dim — muda shapes, invalida comparações atuais).
3. **Re-treinar o run depth** com intrínsecos corretos + crop + FPS + zero-init
   para uma ablação RGB vs depth honesta (o checkpoint atual treinou com nuvem
   geometricamente distorcida).
4. Validar visualmente o workspace crop em frames da PRÓXIMA cena/posição de
   gravação (`inspect_depth_cloud.py`) — os limites x/y ∈ [-0.8, 0.8] foram
   calibrados na cena atual.
