# Investigação — deploy dos checkpoints `best` no ROBÔ REAL (G1, 10.9.8.73)

> Log vivo. Contexto: o checkpoint `last` (20000) do 14-dim estava overfit; testamos os `best`
> (melhores no val) no robô real pra ver se a pega melhora. VLA roda na **Atena GPU2** (`ms3`),
> conecta direto no robô (`--robot-ip 10.9.8.73`). Robô precisa do `run_g1_server.py` rodando.
> Cada run grava: `train/log/infer_right14_<ts>.log` (log) + `train/log/run_<ts>/` (chunks.jsonl,
> events.jsonl, policy_view.mp4, first_policy_frame.jpg, meta.json).

## Mapa das runs (LEMBRAR)

| # | modelo | checkpoint | wandb run | log da inferência | artefatos | resultado |
|---|---|---|---|---|---|---|
| 1 | **14-dim** `cup_pi05_right14_rgb238_lf` | `best` = step **5500** | **`35jrrbk0`** | `infer_right14_20260613_122710.log` | `run_20260613_122710` | ❌ **FALHOU** (ver abaixo) |
| 2 | **8-dim** `cup_pi05_right8_1squeeze_lf` | `best` (~step 8000, sem best_meta) | **`8hajpdab`** | `infer_right14_20260613_123837.log` | `run_20260613_123837` | ⏳ em análise |

(Caminho dos logs/artefatos na Atena: `~/Prometheus/Luiz/prometheus-vla/train/log/`)

---

## Run 1 — 14-dim best (5500) no robô · 2026-06-13 12:27→12:29 · `run_20260613_122710`

**Comando:** `--robot-ip 10.9.8.73 --checkpoint .../cup_pi05_right14_rgb238_lf/checkpoints/best/pretrained_model --fps 30 --hand-kp 0.8 --rtc --live --denoising-steps 5 --rtc-execution-horizon 25 --rtc-max-guidance 1.0 --home-arm-s 3 --open-hand-s 2 --rehome-idle-s 12`

**Resultado: NÃO pegou o copo.** ❌

- **Soft-start ok:** braço foi do standby (Elbow 0.76) pra HOME das demos (Elbow −0.107) em 3s. Mão abriu.
- **Mão CICLOU 7× abre/fecha** (eventos do log): FECHOU f69→ABRIU f120 (1.7s) · FECHOU f293→ABRIU f576 (9.5s) · f618→f634 (0.5s) · f659→f738 (2.6s) · f866→f1122 (8.6s) · f1220→f1574 (11.8s) · FECHOU f2315 (parou 7s depois). **Early-close já no f69.**
- **Pressão travada no baseline ~104,1–104,3k** o tempo todo → **nenhum grip real** (não apertou o copo).
- hand-gate disparou **26×** (suprimiu fechamento com braço fora de reach).
- **Vídeo (`policy_view.mp4`):** copo na mesa → mão do robô aberta ao lado → **uma pessoa teve que apresentar o copo na mão aberta** (meio do vídeo) e o robô não fechou em cima → no fim o copo está na mesa e o **braço recolhido**. Sem pega autônoma.
- **Conclusão:** o `best` (5500) reproduziu o MESMO problema do `last` — braço chega na região mas a mão cicla e não agarra. Trocar pro checkpoint melhor de validação **não consertou a pega**. Bate com o A/B em sim (best só ~11% melhor; ambos generalizam mal). Gargalo real = overfitting + mão 1-DOF (squeeze).

---

## Run 2 — 8-dim best no robô · 2026-06-13 12:38→ · `run_20260613_123837`

**Comando:** igual à Run 1, mas `--checkpoint .../cup_pi05_right8_1squeeze_lf/checkpoints/best/pretrained_model` (action [8] = 7 braço + 1 squeeze; dedos reconstruídos = squeeze × RIGHT_TARGET).

**Resultado: NÃO pegou o copo.** ❌ (falha DIFERENTE do 14-dim)

- Modo **RIGHT8 detectado certo** (8 dims → squeeze; dedos = squeeze × RIGHT_TARGET).
- **Mão fechou só 1× — FECHOU no frame 136 e NÃO reabriu** (segurou até o fim) · hand-gate só **1×**. Ou seja, **NÃO ciclou** (diferente do 14-dim, que ciclou 7×). Nesse aspecto o 8-dim é mais estável.
- **MAS o braço errou o alvo:** vídeo mostra a mão fechando **no ar, no canto inferior-esquerdo**, longe do copo (centro). Início: mão aberta no canto superior-direito. Fim: braço recolhido, humano segurando o copo. **Nunca chegou no copo.**
- pressão max ~105k (≈baseline; sem grip real num objeto).
- **Conclusão:** o 8-dim fecha cedo e segura (sem ciclar), mas o **reach não aponta pro copo** → fecha a mão no lugar errado. Sem pega.

---

## Veredito (2026-06-13): os DOIS `best` FALHARAM no robô real

- **14-dim best (5500):** braço chega na região, **mão cicla** abre/fecha (timing), nunca agarra.
- **8-dim best:** mão fecha 1× e segura (sem ciclar), mas **braço erra o alvo** (fecha no ar).
- **Nenhum pega o copo autonomamente.** Trocar pro checkpoint `best` (melhor no val) **não resolveu** o deploy. Confirma a hipótese: o gargalo não é o checkpoint — é **overfitting + falta de grounding visual do grasp** (timing E mira), agravado pela mão ser **1-DOF (squeeze)**. As alavancas reais são re-treino (mais dados/diversidade, split aleatório, menos overfit) e/ou repensar o encoding da mão — não a escolha de checkpoint.
