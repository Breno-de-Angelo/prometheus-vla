# Inferência Offline / Replay — Log de Progresso (autônomo)

> Documento vivo. Se uma sessão recomeçar, **leia isto + a memória `g1-inferencia-offline-replay`** pra retomar.
> Objetivo: rodar a VLA (Atena GPU1) vendo um episódio real do dataset (replay aberto) + MuJoCo (neste notebook) visualizando as decisões. Construir, consertar, analisar, rodar. Documentar tudo.

## Arquitetura
- **Notebook (esta máquina, /home/luiz-aumo/I2CA/prometheus-vla):** `unitree-g1-mujoco/offline_sim_host.py` — serve imagem+state do dataset por ZMQ (5555/6001/6002) e recebe cmd da VLA (6000/6003). Camada `--mujoco` (a fazer): aplica os cmd a um MuJoCo e renderiza/grava vídeo.
- **Atena (GPU1):** `inference_realtime_pi05d_right14.py --robot-ip 127.0.0.1` (via túnel SSH reverso). Zero mudança no script (is_simulation=False já usa o caminho ZMQ-socket).
- **Rede:** túnel SSH reverso do notebook→Atena (-R nas 5 portas) pra a Atena alcançar o notebook.

## Estado dos passos
- [x] Core `offline_sim_host.py` (serve dataset) — testado (roundtrip ZMQ OK)
- [x] P0: checagem de ambiente (mujoco/zmq local, GPU notebook, rede Atena, env da Atena)
- [x] P1: pipeline end-to-end FUNCIONA — VLA na Atena (GPU1, HF_HOME=/data/huggingface-models) vê o episódio via túnel e decide; warm-up cortou cold-start (chunk0 285ms)
- [x] P2: camada MuJoCo (--mujoco) FUNCIONA — robô 3ª pessoa dirigido pelos cmd, grava vídeo (testado com mock = trajetória real do braço; robô alcança o copo)
- [x] P3: rodado+analisado. VLA produz motions no RANGE do dataset; hand-gate atuou 21x; raw fecha cedo (frame~1 vs 50%). Vídeo /tmp/offline_real_viz.mp4. Correlação baixa = artefato do loop+rate; single-pass slow-replay daria audit limpo.
- [x] P4: README (docs/README_INFERENCIA_OFFLINE.md) + ferramenta de análise (analyze_offline_decisions.py)

## Notas / decisões
- Replay ABERTO: imagem+state do dataset; MuJoCo só visualiza. (decidido pelo usuário)
- Fonte: episódio do dataset real (_merged right8 tem observation.state 14-dim, serve pro right14).
- GPU1 da Atena é reservada pro Luiz — pode usar.
- MuJoCo headless (MUJOCO_GL=egl) grava vídeo offscreen (sem janela) pra revisão.

## Estado detalhado (00:50 12/06)
- P1 PROVADO em conexão: a VLA na Atena conectou via túnel (head_camera+depth+lowstate+pressão) e recebeu imagem+state do dataset. FALHOU no load do modelo: GatedRepoError do `google/paligemma-3b-pt-224` (ssh não-interativo sem HF_TOKEN/HF_HOME). FIX: `HF_HUB_OFFLINE=1` (paligemma já cacheado).
- BLOQUEIO: VPN/rede pra Atena caiu (ping falha) — intermitente, igual antes.
- Servidor offline (episódio 0) segue no notebook (5 portas). Túnel caiu com a rede.

## Comando VLA pronto (quando a rede voltar) — com HF offline
`cd ~/Prometheus/Luiz/prometheus-vla && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 /home/hercules/miniconda3/envs/ms3/bin/python3 -u inference_realtime_pi05d_right14.py --robot-ip 127.0.0.1 --checkpoint train_output/cup_pi05_right14_rgb238_lf/checkpoints/last/pretrained_model --fps 30 --hand-kp 0.8 --rtc --live --denoising-steps 5 --rtc-execution-horizon 25 --rtc-max-guidance 1.0 --home-arm-s 0 --open-hand-s 0 --rehome-idle-s 0`
Túnel: `ssh -N -R 5555:127.0.0.1:5555 -R 6000:127.0.0.1:6000 -R 6001:127.0.0.1:6001 -R 6002:127.0.0.1:6002 -R 6003:127.0.0.1:6003 hercules@10.9.8.252`

## Próxima ação
[01:1x] VPN reconectada via `nmcli up lfsccardoso`. End-to-end REAL rodando: servidor --mujoco (vídeo /tmp/offline_real_viz.mp4) + túnel + VLA (HF_HUB_OFFLINE=1, GPU1). Aguardando 1º chunk.


## ⚠️ HF na Atena (achado importante)
A VLA precisa de `HF_HOME=/data/huggingface-models` (onde está o paligemma-3b-pt-224) — NÃO o cache padrão. `HF_HUB_OFFLINE=1` sozinho falha (paligemma não está em ~/.cache). Comando certo: `HF_HOME=/data/huggingface-models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 ...`

## Experimento de LOOP FECHADO em sim (12/06 ~09h) — RODADO, mas DOMINADO pelo gap visual
Objetivo: medir a degradação de loop fechado (erro composto) — a VLA vê a **head_camera do sim reagindo às próprias ações** (não o dataset), controla o robô do MuJoCo, e medimos se a mão alcança+levanta o copo. Arquivo: `unitree-g1-mujoco/closed_loop_sim_host.py` (renderiza head_camera com Renderer próprio → publica 5555; publica state do sim 6001/6002; dirige o sim via DDS rt/lowcmd com os cmd da VLA; loga dist mão-copo + lift do copo em `/tmp/closed_loop_metric.jsonl`; vídeo global_view).

**Rodou end-to-end** (VLA right14 `last` na Atena GPU1 via túnel; host no notebook env `g1`). Soft-start braço+mão ligados (fiel ao deploy). Métrica medida:
- VLA assume (~t199s): dist mão-copo 21cm → **aproxima até 16cm, lift +4.2cm** (encostou/empurrou o copo).
- Depois **afasta** (16→40cm) e em t225s o **copo CAI da mesa** (lift −71cm = altura da mesa; dist 106→133cm = copo no chão). **NÃO pegou** — lift_max real só +4.2cm (empurrão), e a mão subiu (z 0.92→1.16, afastando do copo em z=0.756).
- Log da VLA: o `hand-gate` suprimiu **TODO** fechamento ("braço fora de reach", Elbow≈−0.7, ShPitch≈−0.2) → a mão nunca fechou pra agarrar; o braço empurrou o copo aberto e derrubou.

**CONFOUND validado visualmente (decisivo):** a head_camera do sim é um render "clay" — `mean=239`, mesa **branca lisa**, mãos/copo **estourados pelo headlight**, fundo branco, FOV mostra as duas mãos. A head_camera **real** (treino) tem `mean=114`, mesa **bege texturizada**, copo branco com brilho/sombra, ambiente de lab. Totalmente **OOD**. O `global_view` do MESMO renderer tem texturas (chão xadrez, mesa, céu) → não é bug do renderer, é a head_camera em close sobre superfícies claras. Comparação: `/tmp/cl_run/cmp_real_vs_sim.png`. Vídeo: `/tmp/cl_run/closed_loop.mp4`. Diag: `/tmp/cl_run/diag_view.py`.

**Conclusão:** este loop fechado **NÃO isola** o erro composto — está dominado pelo gap visual (imagem clay OOD). A comparação cruzada é o achado de valor: **replay ABERTO (imagem REAL) → VLA decide bem** (braço corr +0.67, erro 0.1rad); **loop com imagem SINTÉTICA → degenera** (afasta, derruba, gate trava a mão). A única variável que mudou e quebrou foi a **FONTE DA IMAGEM**. Reforça: o dataset NÃO é o problema; o modelo é fortemente sensível à **distribuição visual** (idem o bug BGR). Pra um sim útil offline seria preciso render realista da head_camera (texturas reais + iluminação + domain randomization no treino) — alto custo, retorno incerto p/ um VLM fine-tunado em 1 ambiente. Diagnóstico offline confiável segue sendo o **replay aberto**; a degradação de loop fechado REAL só mede no robô.