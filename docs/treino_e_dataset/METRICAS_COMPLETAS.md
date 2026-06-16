# Auditoria de Métricas — VLA π0.5 no Unitree G1 Dex3 ("pick up the white cup")

**Projeto wandb:** `prometheus-lcad/prometheus_g1` · **Runs auditadas:** `35jrrbk0` (14-dim) e `8hajpdab` (8-dim)
**Data:** 2026-06-13

## 1. Método

- Todos os números vêm da **wandb API** (`run.history()` / `scan_history()`), alinhados pelo `global_step` **real** logado pelo treino (não pelo `_step` interno do wandb).
- **Marcos pedidos:** step **1000**, o **BEST** (por `val_action_mse`), e de 5k em 5k (**5000, 10000, 15000, 20000**).
- **Eval** logado a cada **500** steps; **train** a cada **100**.
- **BEST por `val_action_mse`:** 14-dim = **step 5500** (mínimo absoluto 0.075899); 8-dim = **step 8000** (mínimo absoluto 0.045644).
- A run **8-dim CRASHOU ~15900** (último train logado = 15500; último eval = 15500) — **não há step 20000**, e o marco "15000" usa o último eval real (15500).
- Nenhum valor inventado. Onde uma chave não existe, fica marcado **n/d** (nas duas runs todas as chaves esperadas existiam; n/d = 0).

## 2. Legenda e as DUAS famílias de métrica

### Dimensões da action

**14-dim (`35jrrbk0`)** — braço 0–6, mão 7–13:

| dim | junta | grupo |
|---|---|---|
| 00 | RShoulderPitch | braço |
| 01 | RShoulderRoll | braço |
| 02 | RShoulderYaw | braço |
| 03 | RElbow | braço |
| 04 | RWristRoll | braço |
| 05 | RWristPitch | braço |
| 06 | RWristYaw | braço |
| 07 | thumb_0 | mão (≈constante) |
| 08 | thumb_1 | mão |
| 09 | thumb_2 | mão |
| 10 | index_0 | mão |
| 11 | index_1 | mão |
| 12 | middle_0 | mão |
| 13 | middle_1 | mão |

Os dedos (08–13) são, na prática, **1-DOF efetivo** = `squeeze × RIGHT_TARGET`; movem-se em bloco. `thumb_0` (dim 07) é praticamente invariante no dataset.

**8-dim (`8hajpdab`)** — braço 0–6 idêntico ao acima; **dim 7 = `squeeze`**, escalar de fechamento da mão ∈ [0,1].

### As duas famílias de métrica de validação (confirmado em `run_train.py`)

Elas medem **coisas diferentes** e, como veremos, **DESCOLAM**:

- **`eval/val_loss`, `val_loss_std`, `val_loss_dim_XX` = LOSS DE FLOW-MATCHING no val.** Roda `policy.forward` com **ruído + timestep aleatórios**. É só um **proxy de treino, FROUXA** — sobe com overfit mesmo sem a ação piorar.
- **`eval/val_action_mse`, `val_action_mse_std`, `val_action_mse_dim_XX` = ERRO DA AÇÃO GERADA.** Roda `predict_action_chunk` (**denoising completo**) e compara com o ground-truth. **É o que importa pro robô.**

Critério de **BEST = `val_action_mse`** (a ação real), não a proxy de flow-matching.

---

## 3. Run `35jrrbk0` — 14-dim (`cup_pi05_right14_rgb238_lf`)

**Chaves logadas** (confirmadas via `run.history().columns`):
- **train/**: `loss`, `loss_std`, `lr`, `grad_norm`, `loss_dim_00..13`, `global_step`, `steps`, `samples`, `episodes`, `epochs`, `update_s`, `dataloading_s`.
- **eval/**: `val_loss`, `val_loss_std`, `val_loss_dim_00..13`, `val_action_mse`, `val_action_mse_std`, `val_action_mse_dim_00..13`, `global_step`.
- Eval a cada 500 steps (40 pontos: 500…20000); train a cada 100 (200 pontos). **BEST por `val_action_mse` = step 5500** (val_action_mse = 0.075899, mínimo absoluto da run).

### (a) Escalares por marco

| marco | train/loss | train/loss_std | lr | grad_norm | val_loss | val_loss_std | val_action_mse | val_action_mse_std |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.036217 | 0.018591 | 4.7551e-05 | 0.33624 | 0.099159 | 0.093559 | 0.107688 | 0.068431 |
| **5500 (BEST)** | 0.018212 | 0.010102 | 8.1128e-05 | 0.19388 | 0.124385 | 0.148697 | **0.075899** | 0.047554 |
| 5000 | 0.020360 | 0.010806 | 8.4222e-05 | 0.19006 | 0.131499 | 0.145751 | 0.125166 | 0.088956 |
| 10000 | 0.027294 | 0.006534 | 4.7569e-05 | 0.17842 | 0.215192 | 0.282910 | 0.092215 | 0.059655 |
| 15000 | 0.007992 | 0.002432 | 1.6225e-05 | 0.15215 | 0.285113 | 0.359973 | 0.185215 | 0.108958 |
| 20000 | 0.008788 | 0.002390 | 1.0000e-05 | 0.15695 | 0.317392 | 0.400821 | 0.190237 | 0.118025 |

### (b) TRAIN per-dim (`train/loss_dim_XX`)

| dim (junta) | 1000 | 5500 (best) | 5000 | 10000 | 15000 | 20000 |
|---|---|---|---|---|---|---|
| 00 RShoulderPitch | 0.068537 | 0.040470 | 0.020885 | 0.013638 | 0.010461 | 0.011241 |
| 01 RShoulderRoll | 0.036318 | 0.023541 | 0.015251 | 0.011538 | 0.012888 | 0.008942 |
| 02 RShoulderYaw | 0.035226 | 0.015492 | 0.012405 | 0.007271 | 0.006430 | 0.009770 |
| 03 RElbow | 0.045350 | 0.025725 | 0.033247 | 0.020784 | 0.007866 | 0.008832 |
| 04 RWristRoll | 0.056658 | 0.029434 | 0.025292 | 0.014323 | 0.006954 | 0.013682 |
| 05 RWristPitch | 0.054590 | 0.033070 | 0.038928 | 0.021333 | 0.006590 | 0.013169 |
| 06 RWristYaw | 0.023072 | 0.017875 | 0.019091 | 0.015065 | 0.007767 | 0.008852 |
| 07 thumb_0 | 0.001523 | 0.001757 | 0.001314 | 0.001097 | 0.000483 | 0.000317 |
| 08 thumb_1 | 0.040765 | 0.010871 | 0.020256 | 0.046763 | 0.008756 | 0.007943 |
| 09 thumb_2 | 0.029913 | 0.010793 | 0.018933 | 0.046084 | 0.008648 | 0.008553 |
| 10 index_0 | 0.033227 | 0.011203 | 0.019517 | 0.046420 | 0.008506 | 0.007799 |
| 11 index_1 | 0.028949 | 0.011137 | 0.019870 | 0.044389 | 0.008800 | 0.007782 |
| 12 middle_0 | 0.030045 | 0.012395 | 0.020357 | 0.047178 | 0.008936 | 0.007992 |
| 13 middle_1 | 0.022863 | 0.011210 | 0.019697 | 0.046240 | 0.008807 | 0.008153 |

### (c) VAL FLOW-MATCHING per-dim (`eval/val_loss_dim_XX`) — proxy frouxa, ruído+t aleatório

| dim (junta) | 1000 | 5500 (best) | 5000 | 10000 | 15000 | 20000 |
|---|---|---|---|---|---|---|
| 00 RShoulderPitch | 0.080166 | 0.103489 | 0.102703 | 0.144901 | 0.168419 | 0.180902 |
| 01 RShoulderRoll | 0.063872 | 0.066539 | 0.063904 | 0.091140 | 0.112054 | 0.115520 |
| 02 RShoulderYaw | 0.043048 | 0.050389 | 0.051543 | 0.071922 | 0.086260 | 0.088274 |
| 03 RElbow | 0.103229 | 0.116685 | 0.117950 | 0.186220 | 0.219960 | 0.239299 |
| 04 RWristRoll | 0.049105 | 0.060351 | 0.056861 | 0.096734 | 0.106212 | 0.112378 |
| 05 RWristPitch | 0.108370 | 0.131509 | 0.129117 | 0.207074 | 0.249559 | 0.266115 |
| 06 RWristYaw | 0.046648 | 0.049505 | 0.046650 | 0.066553 | 0.076581 | 0.078321 |
| 07 thumb_0 | 0.004058 | 0.002484 | 0.002581 | 0.002011 | 0.001211 | 0.001071 |
| 08 thumb_1 | 0.145196 | 0.189277 | 0.205562 | 0.355361 | 0.494615 | 0.561749 |
| 09 thumb_2 | 0.142968 | 0.190050 | 0.209541 | 0.355541 | 0.494444 | 0.558254 |
| 10 index_0 | 0.151482 | 0.193391 | 0.210345 | 0.356762 | 0.493729 | 0.558352 |
| 11 index_1 | 0.151852 | 0.197427 | 0.214867 | 0.362645 | 0.500107 | 0.565525 |
| 12 middle_0 | 0.151536 | 0.194845 | 0.211210 | 0.360073 | 0.496895 | 0.564111 |
| 13 middle_1 | 0.146696 | 0.195450 | 0.218150 | 0.355751 | 0.491539 | 0.553619 |

### (d) VAL AÇÃO-GERADA per-dim (`eval/val_action_mse_dim_XX`) — denoising completo vs GT, o que importa pro robô

| dim (junta) | 1000 | 5500 (best) | 5000 | 10000 | 15000 | 20000 |
|---|---|---|---|---|---|---|
| 00 RShoulderPitch | 0.105801 | 0.039473 | 0.055677 | 0.042273 | 0.063927 | 0.067295 |
| 01 RShoulderRoll | 0.062530 | 0.014301 | 0.023758 | 0.017709 | 0.018970 | 0.018574 |
| 02 RShoulderYaw | 0.030191 | 0.010727 | 0.015045 | 0.010788 | 0.012970 | 0.011333 |
| 03 RElbow | 0.095050 | 0.035584 | 0.057502 | 0.040949 | 0.056017 | 0.059006 |
| 04 RWristRoll | 0.017273 | 0.019790 | 0.020463 | 0.022121 | 0.020190 | 0.021089 |
| 05 RWristPitch | 0.061470 | 0.030424 | 0.041908 | 0.025070 | 0.032499 | 0.033965 |
| 06 RWristYaw | 0.036235 | 0.030464 | 0.031910 | 0.024130 | 0.021677 | 0.022538 |
| 07 thumb_0 | 0.000698 | 0.000286 | 0.000271 | 0.000202 | 0.000082 | 0.000053 |
| 08 thumb_1 | 0.179693 | 0.146444 | 0.253916 | 0.184155 | 0.394644 | 0.404916 |
| 09 thumb_2 | 0.183540 | 0.146773 | 0.252064 | 0.184855 | 0.394066 | 0.404600 |
| 10 index_0 | 0.182525 | 0.147904 | 0.250666 | 0.184024 | 0.395376 | 0.405648 |
| 11 index_1 | 0.183344 | 0.145551 | 0.250745 | 0.184881 | 0.394542 | 0.404990 |
| 12 middle_0 | 0.187506 | 0.146740 | 0.248327 | 0.184296 | 0.394121 | 0.404408 |
| 13 middle_1 | 0.181770 | 0.148123 | 0.250066 | 0.185558 | 0.393936 | 0.404910 |

### (e) Leitura por dimensão (1000 → best 5500 → fim 20000)

**Descolamento global val_loss × val_action_mse (a marca da run):** o `val_loss` (flow-matching) só PIORA monotonicamente — 0.099 → 0.124 (best) → 0.317 no fim (3,2×). Já o `val_action_mse` (ação gerada) tem fundo em 5500 (0.0759), volta a subir e estabiliza em ~0.19 no fim. A partir de ~10k o modelo treina demais: o flow-loss de val explode, mas a ação gerada não acompanha na mesma escala — o sinal de overfit aparece MUITO mais cedo e mais forte no `val_loss` do que no `val_action_mse`. O train cai limpo até o fim (0.036 → 0.0088), confirmando memorização.

**Braço (dims 00–06):**
- **00 RShoulderPitch / 03 RElbow / 05 RWristPitch** — os "carros-chefe" do movimento. No action_mse caem forte 1000→best (00: 0.106→0.039; 03: 0.095→0.036; 05: 0.061→0.030), depois oscilam e pioram um pouco até o fim (00 volta a 0.067, 03 a 0.059). São onde o overfit mais morde o braço.
- **01 RShoulderRoll / 02 RShoulderYaw** — melhoram e FICAM bons: caem 1000→best (01: 0.063→0.014; 02: 0.030→0.011) e seguram ~0.018/0.011 até o fim. Dimensões mais estáveis do braço.
- **04 RWristRoll** — praticamente TRAVADA: action_mse ~0.017–0.022 do começo ao fim, sem ganho real com treino (já era a melhor em 1000). No train cai (0.057→0.014), mas em val não anda: memorização sem generalização.
- **06 RWristYaw** — melhora lenta e monotônica em action_mse (0.036→0.030→0.022), das poucas dims que continua ganhando até o fim.

**Mão / dedos (dims 07–13) — o ponto fraco da run:**
- **07 thumb_0** — efetivamente CONSTANTE/morta, como esperado (≈invariante no dataset). Loss minúsculo em tudo (action_mse 0.0007 → 0.00005), só decora um valor fixo. Não carrega informação de grasp.
- **08–13 (thumb_1/2, index_0/1, middle_0/1 = o squeeze efetivo de 1-DOF)** — movem-se em bloco (valores quase idênticos entre si em toda coluna, confirmando squeeze×RIGHT_TARGET). É AQUI que o overfit é catastrófico: no `val_loss` sobem ~0.145 → 0.19 (best) → ~0.56 no fim (quase 4×, muito pior que o braço); no `val_action_mse` o melhor ponto é o best (~0.146), explodem para ~0.40 no fim (≈2,7×). Atenção: o action_mse dos dedos em 5000 (~0.25) é PIOR que em 5500 (~0.146) e que em 10000 (~0.184) — o best 5500 é um vale localizado de mão; fora dele a mão fica ~0.18–0.40. Em todos os marcos os dedos dominam o erro de ação (≈0.15–0.40 contra ≈0.01–0.07 do braço): a mão é o gargalo, e piora muito mais que o braço com o overtraining.

**Resumo:** o BEST (5500) é o ponto certo — mínimo de val_action_mse global e mínimo dos dedos. Tudo após ~10k é overfit, visível primeiro e mais forte no val_loss (flow) e nas dims 08–13 (mão). Ombro (01/02) e wrist-roll (04) são robustos/travados; pitch/elbow (00/03/05) sofrem moderadamente; a mão (08–13) é o que estraga a run no fim.

---

## 4. Run `8hajpdab` — 8-dim (`cup_pi05_right8_1squeeze_lf`)

**Estado da run:** **crashed** (último train logado = step 15900; último eval = step 15500). Best por `val_action_mse` confirmado em **step 8000** (mse = 0.045644). Marco "15000" = último eval real (15500). Train alinhado pelo `train/global_step`, eval pelo `eval/global_step` (todos os marcos bateram exato; o "fim" usa train 15500 / eval 15500). Existem `grad_norm`, `update_s`, `epochs`, `samples`; nenhuma chave val_* faltou (n/d = 0).

Dims (8-dim): 0=RShoulderPitch, 1=RShoulderRoll, 2=RShoulderYaw, 3=RElbow, 4=RWristRoll, 5=RWristPitch, 6=RWristYaw, **7=squeeze** (∈[0,1]).

### (a) Escalares por marco

| métrica | 1000 | best 8000 | 5000 | 10000 | 15500 (fim) |
|---|---|---|---|---|---|
| train/loss | 0.063126 | 0.014005 | 0.027775 | 0.016512 | 0.006321 |
| train/loss_std | 0.011420 | 0.003464 | 0.005034 | 0.003309 | 0.002686 |
| train/lr | 4.7551e-05 | 6.3197e-05 | 8.4222e-05 | 4.7569e-05 | 1.4382e-05 |
| train/grad_norm | 0.325244 | 0.147418 | 0.162766 | 0.136294 | 0.123774 |
| eval/val_loss | 0.101559 | 0.227762 | 0.159644 | 0.291419 | 0.363374 |
| eval/val_loss_std | 0.081343 | 0.227620 | 0.142843 | 0.294475 | 0.362503 |
| eval/val_action_mse | 0.072521 | **0.045644** | 0.054141 | 0.049954 | 0.053861 |
| eval/val_action_mse_std | 0.029170 | 0.018139 | 0.019685 | 0.024235 | 0.023336 |

*(Auxiliares: train/epochs 0.671 / 5.370 / 3.357 / 6.713 / 10.405; train/samples 32000 / 256000 / 160000 / 320000 / 496000; train/update_s ~1.80s/it constante. lr é warmup→decay, pico ~8.4e-5 em 5k.)*

### (b) TRAIN per-dim (`train/loss_dim_XX`)

| dim (junta) | 1000 | best 8000 | 5000 | 10000 | 15500 (fim) |
|---|---|---|---|---|---|
| 00 RShoulderPitch | 0.064085 | 0.010477 | 0.019792 | 0.019848 | 0.007341 |
| 01 RShoulderRoll | 0.030277 | 0.020693 | 0.023650 | 0.010907 | 0.006810 |
| 02 RShoulderYaw | 0.034072 | 0.006466 | 0.008489 | 0.009129 | 0.004464 |
| 03 RElbow | 0.046382 | 0.011608 | 0.025232 | 0.021995 | 0.006899 |
| 04 RWristRoll | 0.058894 | 0.014848 | 0.015282 | 0.007411 | 0.005015 |
| 05 RWristPitch | 0.062319 | 0.019517 | 0.023097 | 0.020519 | 0.006666 |
| 06 RWristYaw | 0.019948 | 0.007162 | 0.014380 | 0.007949 | 0.004804 |
| 07 squeeze | 0.189029 | 0.021265 | 0.092280 | 0.034334 | 0.008571 |

### (c) VAL FLOW-MATCHING per-dim (`eval/val_loss_dim_XX`)

| dim (junta) | 1000 | best 8000 | 5000 | 10000 | 15500 (fim) |
|---|---|---|---|---|---|
| 00 RShoulderPitch | 0.089492 | 0.189793 | 0.134356 | 0.224720 | 0.308099 |
| 01 RShoulderRoll | 0.060290 | 0.102178 | 0.080620 | 0.122475 | 0.153864 |
| 02 RShoulderYaw | 0.044273 | 0.073372 | 0.057337 | 0.088447 | 0.110652 |
| 03 RElbow | 0.124065 | 0.247232 | 0.170477 | 0.293625 | 0.384744 |
| 04 RWristRoll | 0.049580 | 0.092799 | 0.072323 | 0.116464 | 0.142876 |
| 05 RWristPitch | 0.122238 | 0.284262 | 0.188905 | 0.325447 | 0.408220 |
| 06 RWristYaw | 0.045427 | 0.072196 | 0.053908 | 0.089150 | 0.106743 |
| 07 squeeze | 0.277106 | 0.760268 | 0.519228 | 1.071030 | 1.291800 |

### (d) VAL AÇÃO-GERADA per-dim (`eval/val_action_mse_dim_XX`)

| dim (junta) | 1000 | best 8000 | 5000 | 10000 | 15500 (fim) |
|---|---|---|---|---|---|
| 00 RShoulderPitch | 0.089770 | 0.037586 | 0.051737 | 0.035962 | 0.050579 |
| 01 RShoulderRoll | 0.051224 | 0.015695 | 0.019438 | 0.014412 | 0.014269 |
| 02 RShoulderYaw | 0.027946 | 0.010207 | 0.009543 | 0.007414 | 0.008014 |
| 03 RElbow | 0.077745 | 0.034304 | 0.051560 | 0.033393 | 0.042206 |
| 04 RWristRoll | 0.015854 | 0.018298 | 0.022355 | 0.023479 | 0.020759 |
| 05 RWristPitch | 0.051687 | 0.023864 | 0.033142 | 0.026269 | 0.028369 |
| 06 RWristYaw | 0.037624 | 0.025483 | 0.025816 | 0.025254 | 0.023088 |
| 07 squeeze | 0.228321 | 0.199716 | 0.219539 | 0.233449 | 0.243601 |

### (e) Leitura por dimensão (1000 → best 8000 → fim 15500)

**Descolamento global val_loss × val_action_mse (o achado central):** o `val_loss` (flow-matching) SOBE monotonicamente o tempo todo (0.102 → 0.228 no best → 0.363 no fim, ~3,6×), enquanto o `val_action_mse` (ação realmente gerada) CAI até o best (0.0725 → 0.0456) e depois fica praticamente FLAT/leve piora (0.0500 em 10k, 0.0539 no fim). Ou seja: a loss de val frouxa "explode" mas a ação gerada continua boa — o checkpoint 8000 é o melhor real, e treinar além disso não melhora a ação (a mão piora um pouco). O train/loss segue caindo (0.063 → 0.0063), confirmando overfit ao critério de flow-matching.

**Braço (dims 0–6):**
- **00 RShoulderPitch** — train despenca (0.064→0.010→0.0073). Ação-gerada: melhora forte até best (0.090→0.038), oscila e termina pior (0.051). É a junta de braço com maior erro de ação. val_loss sobe (0.089→0.308): descolamento clássico.
- **01 RShoulderRoll** — train cai limpo (0.030→0.021→0.0068). Ação-gerada melhora e SEGUE melhorando além do best (0.051→0.0157→0.0143 no fim) — uma das poucas que se beneficia de treino extra. val_loss sobe (0.060→0.154).
- **02 RShoulderYaw** — train cai (0.034→0.0065→0.0045). Ação-gerada das melhores e estável (0.028→0.0102→0.0080). val_loss sobe moderado (0.044→0.111).
- **03 RElbow** — train cai (0.046→0.012→0.0069). Ação-gerada: 2ª pior do braço; melhora até best (0.078→0.034), oscila, termina 0.042. val_loss sobe muito (0.124→0.385) — junta com forte descolamento.
- **04 RWristRoll** — ANOMALIA: ação-gerada PIORA com o treino (0.0159 em 1000 → 0.0183 no best → 0.0208 no fim). É a única dim cuja ação estava melhor no início; treinar a degrada. train cai normal (0.059→0.0050) e val_loss sobe (0.050→0.143) — overfit puro aqui.
- **05 RWristPitch** — train cai (0.062→0.020→0.0067). Ação-gerada melhora até best (0.052→0.024) e fica ~estável (0.028). val_loss um dos que mais sobem (0.122→0.408).
- **06 RWristYaw** — train cai (0.020→0.0072→0.0048). Ação-gerada melhora suave e contínua (0.038→0.0255→0.0231). val_loss sobe moderado (0.045→0.107).

**Mão (dim 7 = squeeze):**
- **07 squeeze** — DOMINA todo o erro. No train cai muito (0.189→0.021→0.0086), mas é sempre a maior loss de treino. No val flow-matching é a que mais explode: 0.277 → 0.760 no best → **1.292 no fim** (~4,7×, uma ordem de grandeza acima de qualquer braço). Na ação-gerada é de longe a pior dim e essencialmente TRAVADA / leve piora: 0.228 (1000) → 0.200 (best) → 0.234 (10k) → **0.244 (fim)** — nunca desce abaixo de ~0.20 enquanto o braço inteiro vive em 0.01–0.05. O squeeze é o gargalo da run: o modelo praticamente não aprende a fechar a mão de forma consistente, e o val_action_mse global (0.046–0.054) é puxado quase inteiramente por essa dim. A leve degradação após o best (0.200→0.244) reforça que treinar além de 8000 só piora a mão.

**Resumo:** best 8000 é correto (val_action_mse mínimo global e da maioria das juntas). Pós-best: o braço fica estável/levemente melhor em algumas (01, 06), mas wrist-roll (04) e mão (07) pioram, e o val_loss sobe em tudo — sinal de overfit sem ganho de ação. O problema de produto é a **dim 7 (squeeze)**: uma ordem de grandeza pior que o braço em ação-gerada e que nunca destravou.

---

## 5. Leitura geral (o que aconteceu)

- **(a) O train converge nas duas runs.** `train/loss` cai limpo e fundo: 14-dim 0.036 → 0.0088; 8-dim 0.063 → 0.0063 (~0.006–0.009 nos dois). `grad_norm` cai junto (0.34→0.16 / 0.33→0.12). O modelo claramente memoriza o conjunto de treino.

- **(b) O `val_loss` (flow-matching) sobe em QUASE TUDO — inclusive no braço.** Nas duas runs a proxy frouxa só piora de forma monotônica (14-dim 0.099→0.317, ~3,2×; 8-dim 0.102→0.363, ~3,6×), e isso vale **per-dim até em juntas de braço** (ex.: 14-dim dim00 0.080→0.181; 8-dim dim03 0.124→0.385). **Lido isolado, o `val_loss` diz "overfit total".** Mas ele é flow-matching com ruído+timestep aleatório: é um sinal de proxy, não a ação.

- **(c) MAS o `val_action_mse` do braço fica baixo/estável — a AÇÃO gerada do braço NÃO degrada na mesma escala.** No best, o braço inteiro vive em **~0.01–0.07** nas duas runs, e mesmo no fim a maioria das juntas de braço continua nessa faixa (8-dim: 01 e 06 até **melhoram** depois do best; 14-dim: 01/02/04 seguram ~0.011–0.022). O `val_action_mse_std` global até **cai** do step 1000 ao best (14-dim 0.068→0.048; 8-dim 0.029→0.018). Ou seja: **a explosão do `val_loss` é majoritariamente a proxy, não o braço perdendo a ação.** Exceções honestas, sem esconder: dims 00/03/05 (14-dim) e 00/03 (8-dim) sofrem uma piora **moderada e real** na ação após o best (ombro-pitch e cotovelo voltam de ~0.035–0.039 para ~0.05–0.067), e a **04 RWristRoll** da 8-dim **piora de verdade** com treino (0.016→0.021) — mas tudo isso é de centésimos, não a ordem de grandeza da mão.

- **(d) A MÃO é o problema, nas duas runs.** No `val_action_mse` ela fica **travada alta e nunca aprende de fato**: 14-dim dedos (08–13) ~0.146 no best, explodindo para ~0.40 no fim; 8-dim squeeze (07) ~0.20 no best e ~0.24 no fim, **sempre ≥0.20** enquanto o braço vive em 0.01–0.07. A mão **domina o agregado** — por ser um DOF efetivo redundante (squeeze×TARGET / 1 escalar) medido com L2, ela puxa quase sozinha o `val_action_mse` global. No `val_loss` é onde o overfit é mais catastrófico (8-dim squeeze chega a 1.29; 14-dim dedos ~0.56).

- **(e) O BEST é o ponto útil das duas runs.** 14-dim = **5500** (val_action_mse global 0.0759 e mínimo dos dedos ~0.146); 8-dim = **8000** (0.0456 e mínimo do squeeze ~0.200). Nas duas, treinar além do best não recupera a ação — só piora a mão e infla a proxy.

- **(f) Conclusão honesta:** **a val que "sobe" é majoritariamente (i) a proxy de flow-matching e (ii) a métrica da mão 1-DOF medida com L2 — não o braço degradando.** O braço gerado está bom e estável (com pioras de centésimos em pitch/elbow/wrist-roll). Se o robô falha, **falha pela MÃO** (squeeze/dedos que nunca destravam no `val_action_mse`) **e por grounding** — não por o braço esquecer a trajetória. O critério de seleção por `val_action_mse` (e não por `val_loss`) está certo justamente porque captura esse descolamento.

## 6. Próximo passo

A causa provável da mão travada é o **expert não estar usando a imagem** (grounding fraco): se o squeeze é decidido por viés do prior de ação e não pelo que a câmera vê, ele nunca generaliza o "quando fechar". O teste é **de graça e offline**: rodar a **ablação de imagem no eval do checkpoint best** — recomputar o `val_action_mse` (a) com a imagem **zerada** e (b) com a imagem **embaralhada** entre amostras do batch, e comparar com o eval normal. Se o `val_action_mse` (especialmente das dims de mão) **quase não mudar** sob imagem zerada/embaralhada, está provado que o expert ignora a visão — e a mão depende de grounding, não de mais treino. Se mudar muito (degradar), o modelo está de fato olhando a imagem e o gargalo é outro (dados/representação do squeeze). Em qualquer caso, o resultado direciona o conserto sem gastar uma run de treino nova.
