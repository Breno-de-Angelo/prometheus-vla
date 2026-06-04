# Resumo — Experimento "G1 pega o copo" (replay do dataset real no sim)

## Objetivo
Reproduzir no simulador (`run_sim_visible.py` + ZMQ) a teleoperação real gravada no dataset
HF `Mrwlker/pick_up_the_cup_2026-04-30` (mão DIREITA, ep18) e fazer o robô pegar o copo,
com o mundo simulado fiel ao real (mesa, copo, pose).

## O que construímos (tudo em `pickup_experiments/`)
- **`replay_dataset.py`** — streama a trajetória do dataset (28 juntas → motores do sim) via ZMQ.
  Flags: `--torso`, `--arm-kp`, `--close-frame`, `--author-grasp`, `--lift-rad`, `--loop`.
- **`depth_pointcloud.py`** — reconstrói a nuvem de pontos do depth da D435i no MuJoCo (validação).
- **`eval_grasp_headless.py`** + **`sweep_real.py`** — sweeps de configuração (headless e fiel).
- **`ca.md`** — 12 critérios de aceitação com status.
- Hooks no `base_sim.py` (reset/move do copo por teclado, `BAND_Z`, `ROOT_X`, nuvem de pontos)
  e `run_sim.py` (overrides por env). Logger ganhou os dedos da mão esquerda.

## O que funcionou / validamos (CA1–CA8 ✅)
- **Mapeamento** action(28)→sim: braço dir = ds7-13 → motores 22-28; mão dir = ds21-27 (index/middle
  TROCADOS). Fechar mão dir: `[0,-1.5,-1.5,1.5,1.5,1.5,1.5]`.
- **Frame correto do episódio** via `from_timestamp×30` do `meta/episodes` (NÃO offset cumulativo).
  ep18 = frame 6023. Cada episódio tem o copo em posição diferente.
- **Posição do copo** (ep18): X≈0.26–0.30, Y≈0–(-0.02) — corroborada por 2 métodos (raio do depth→
  plano da mesa E onde a mão fecha no replay).
- **Robô RETO e aterrado**: `BAND_Z=0.847` (pés no chão, pélvis ~0.79). Sem lean de tronco.
- **Mesa real** 1.80×0.75m, tampo fino topo ~0.78 no quadril, pernas sem furar o chão.
- **Copo assenta em pé**: contato rígido (`solref 0.01`, `solimp 0.95 0.99`) — soft demais penetrava.
- A mão **vai ao copo e encosta** (até 5 contatos) e **levanta ~5cm no pico**.

## Barreiras / o que deu errado (CA9–CA10 ❌)
1. **GARGALO RAIZ — pose do corpo não gravada**: o dataset só tem braços+mãos (28 dims), sem
   cintura/pélvis. Sem a pose real do tronco, a trajetória replicada **varre/raspa** o copo em
   vez de envolvê-lo → não segura. Causa de quase tudo.
2. **Sem compensação de gravidade no sim**: o braço **cai (sag)** abaixo do alvo. kp alto NÃO
   resolve (mão fica no alvo alto da FK, acima do copo). O sag ajudava a mão a descer ao copo.
3. **Avaliador headless DIVERGE do sim real**: a config "vencedora" do sweep headless (10 agentes,
   496 configs) deu **0 contatos no sim real**. Lição: só confiar no sim real (fiel).
4. **Sweep fiel de física do copo** (atrito/massa/tamanho, 12 configs): **0 seguraram**. Física do
   copo não resolve o escorregamento.
5. **Pega autorada** (congela no copo→fecha→levanta): a pose da mão do dataset é uma "passada de
   lado", não envolvente → fechar só **empurra** o copo (+2–3cm, escorrega).
6. **Webcam teleop (MediaPipe)**: abandonado — mediapipe quebrava o numpy da stack `record_v2`
   (resolvido isolando em venv), mas a abordagem não vingou.
7. Operacional: depth 8-bit (escala métrica perdida, copo branco = inválido no depth); câmera
   fixa no torso (sem pescoço); spawn do copo sensível a altura de queda/contato.

## Estado atual
Melhor resultado fiel: a mão alcança o copo, faz contato e o levanta **~5cm no pico**, mas
**escorrega** (líquido ~+3cm). Não há pega segura (CA10).

## ✅ RESOLVIDO — o robô PEGA E LEVANTA o copo (no NOSSO sim)
A pega finalmente funciona: copo levantado **+7cm e mantido** (3 contatos, não escorrega).
O que destravou:
1. **Robô reto + aterrado** (`BAND_Z=0.847`, pés no chão), mesa real no quadril (topo 0.78).
2. **Colisão do copo = CILINDRO limpo** (raio 0.038) — o casco convexo do mug furava o tampo;
   o mesh do mug vira só visual. + contato rígido (`solref 0.005`, `solimp 0.99 0.999`) na mesa
   e no copo → assenta certo (0.4mm), não afunda.
3. **`<option impratio="50" cone="elliptic" noslip_iterations="15"/>`** ← A CHAVE da pega.
   Sem isso o atrito "amolece" e o copo desliza pra fora mesmo a mão envolvendo (4 contatos).
   impratio alto deixa o atrito rígido vs normal — técnica padrão do MuJoCo p/ grasp (achada
   pesquisando "como pegam objeto no mujoco"). Mão: condim=4, friction 8, priority=2, kp~60.
4. **Pick autorado pela AÇÃO REVERSA** (ideia do usuário): captura a pose de grasp (tecla T no
   sim → `/tmp/grasp_state.json`), depois `pickup_experiments/pick_from_grasp.py` faz
   segura-grasp → fecha → levanta (ombro sobe). Repetível (reset copo com R).
Controles do sim (`base_sim.viewer_key_callback`): setas=move copo XY, N/M=Z, O/P=gira,
F/G=fecha/abre mão (gradual), T=captura, R/Espaço=reset robô+copo.

## DESCOBERTA — repo `luckyrobots/g1-manipulation-challenge` (clonado em /tmp)
É a MESMA tarefa (pick & place de um cilindro vermelho com o G1 no MuJoCo) e entrega o que
faltava: **políticas RL ONNX** — `right_reacher.onnx` (alvo em frame da pélvis → 7D ações do
braço dir, 36D obs; sobrepõe ao walker), `walker.onnx`, `rotator`, `croucher`, + função "grab".
Isso RESOLVE nosso gargalo: em vez de replay cego das juntas (que raspa o copo), damos o ALVO
(posição do copo) e a política alcança com controle de corpo inteiro. Cilindro 4cm Ø×7cm
(agarrável). Plano: rodar `run.py`, automatizar reacher→alvo no copo→grab→levanta→mesa azul.
Status: EM ANDAMENTO (agents paralelos + integração).

## INTEGRAÇÃO com o repo do desafio (políticas RL) — progresso e achados

Clonei `luckyrobots/g1-manipulation-challenge` em `/tmp/g1-manipulation-challenge` e escrevi um
controlador automático **`pickup_experiments/lucky_auto_pick.py`** (cópia do `/tmp/.../auto_pick.py`)
que reusa as políticas ONNX (`walker` + `right_reacher`) via as classes do `run.py`, dirigindo por
máquina de estados: stand → walk (anda até a mesa) → reach/above/descend (reacher mira no cilindro,
top-down) → grab → lift. Roda no venv isolado `/tmp/lucky_env` (mujoco+onnxruntime), headless com
`MUJOCO_GL=egl`. Agents paralelos documentaram o pipeline (obs reacher 36D, ação 7D, grab, cena).

**Progresso real:** o robô anda autonomamente até a mesa, ativa o reacher e leva a palma à região
do cilindro (fwd≈0.33, centralizada via offset de side +0.13, altura certa via offset up +0.06).

**Gargalos que travaram a pega limpa (a parte difícil do desafio):**
1. **Envelope do reacher é limitado e NÃO-LINEAR**: pra um alvo, a palma vai a uma posição com
   offset grande e dependente da config (ex.: alvo side 0.06 → palma side −0.06). Calibração
   open-loop vira whack-a-mole; servo de alto ganho **diverge** (alvo dispara pros cantos do clamp,
   robô cai). Servo precisa ser estável/bounded.
2. **Aproximação lateral EMPURRA o cilindro** (leve, 9g) antes de fechar — a mão direita varrendo
   até o centro chuta o cilindro pro lado (+0.26). Top-down (subir→centrar no alto→descer vertical)
   resolve o empurrão (cilindro fica parado), mas precisa do robô bem posicionado.
3. **Marcha do walker é INCONSISTENTE**: às vezes anda até pelvis_x≈−0.34 (perto), às vezes trava
   no início (−0.60, cilindro fora do alcance fwd 0.60). Controle de yaw ou strafe forte **derruba**
   o robô. Reposicionar com precisão (andar+strafe) é frágil.

**O que faltou pra fechar**: combinar (robô anda perto de forma confiável) + (top-down sem empurrar)
+ (servo estável bounded calibrando o offset não-linear do reacher). É afinável, mas exige mais
engenharia do que a afinação cega permitiu na sessão. Comando: `source /tmp/lucky_env/bin/activate;
cd /tmp/g1-manipulation-challenge; MUJOCO_GL=egl python auto_pick.py --secs 45` (ou `--view` com DISPLAY).

## Caminhos pra fechar (decisão pendente)
- **A) Pega sintética com IK** — projetar a pega do zero (mão abre em torno do copo → fecha →
  levanta). Pega de verdade, mas abandona a trajetória real (usa só a posição do copo).
- **B) Obter a pose real do corpo** (cintura/pélvis do ep18) com a fonte do dataset — única forma
  de pegar **com a trajetória real fiel**.
- **C) Aceitar a demonstração atual** (encosta + levanta parcial) como o máximo que o dado permite.
</content>
