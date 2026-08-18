# Inferência remota do FastWAM-D — servidor na athena, robô no seu PC

O FastWAM-D tem 6 bilhões de parâmetros e depende do VAE do Wan e do encoder de
texto UMT5. Isso não cabe no notebook que fica ao lado do robô, e nem deveria: o
que precisa estar perto do G1 é o loop de controle a 30 Hz, não a GPU.

```
   SEU PC (robô)                                 ATHENA (A100)
┌────────────────────────┐                  ┌──────────────────────────┐
│ init_..._client.py     │   obs (~3 MB)    │ init_..._server.py       │
│                        │ ───────────────► │                          │
│  robot.get_observation │                  │  preprocessor            │
│  cameras (ZMQ)         │                  │  FastWAM-D (6 B)         │
│  painel de debug       │ ◄─────────────── │  postprocessor           │
│  robot.send_action     │  chunk + debug   │                          │
└────────────────────────┘                  └──────────────────────────┘
      30 Hz, sem GPU                          ~1,5 s por inferência
```

Arquivos:

| arquivo | onde roda | o que faz |
|---|---|---|
| `init_lerobot_inference_fastwamd_server.py` | athena | carrega o modelo, responde chunks por ZMQ |
| `init_lerobot_inference_fastwamd_client.py` | seu PC | lê o robô, executa as ações, desenha o painel |
| `viz_debug_fastwamd.py` | seu PC | os quatro quadrantes do painel |
| `policies/fastwam_depth/debug_inferencia.py` | athena | captura atenção e profundidade de uma inferência |
| `avaliar_episodio_fastwamd.py` | athena | replay de um episódio gravado contra o servidor, com métricas por grupo de juntas |
| `athena/*.sh` | athena | os launchers (ver `athena/README.md`) |

## Como rodar

**Na athena** (o checkpoint é o `best`, que o treino publica sozinho):

```bash
cd ~/DEV/prometheus-vla/lerobot-ext
HF_HOME=/data/.cache/huggingface CUDA_VISIBLE_DEVICES=2 \
python init_lerobot_inference_fastwamd_server.py \
    --checkpoint=/data/train_output/fastwamdepth_white_cup_on_dripper/checkpoints/best/pretrained_model \
    --port=5600 --debug
```

**No seu PC**:

```bash
cd lerobot-ext
python init_lerobot_inference_fastwamd_client.py \
    --server=10.9.8.252 --cam-robot=192.168.123.164 \
    --chunk=32 --lead=24 --fps=30 --v-debug --debug
```

Para testar sem robô, acrescente `--sim` no cliente.

## O painel de depuração (`--v-debug`)

Uma janela, quatro quadrantes. É uma janela só porque o loop de controle roda a
30 Hz e cada `imshow` extra é tempo tirado do ciclo.

**1. Atenção do DiT.** No MoT, as consultas do expert de ação atendem a
`[K de vídeo | K de ação]` (`wan/modular.py::_forward_action_cached`). A fatia
de vídeo dessa atenção é, literalmente, que pedaço da cena decidiu a ação. O
mapa vem na grade de tokens (7x14 para o mosaico de 224x448 — cada célula cobre
32x32 px) e é ampliado por interpolação; a mancha suave é honesta quanto à
resolução real.

⚠️ **O painel mostra o desvio da linha de base, não a atenção crua.** O mapa cru
é dominado por *attention sink*: as duas primeiras colunas da grade recebem a
maior parte do peso em TODO quadro, independente da cena.

```
media por coluna, tres quadros muito diferentes do episodio 25:
  quadro  60:  0.48 0.27 0.05 0.04 0.03 0.03 ...
  quadro 150:  0.44 0.29 0.05 0.03 0.03 0.04 ...
  quadro 250:  0.48 0.27 0.05 0.04 0.03 0.02 ...
```

Isso é destino "nulo" do transformer, não informação — e o pico do mapa cru cai
na coluna 0 sempre, em qualquer quadro. Como o sumidouro é praticamente
estacionário e o conteúdo varia, o servidor mantém uma média móvel e envia
`(atual - base) / base`. A razão, e não a subtração pura, porque o sumidouro
flutua: +0,10 sobre uma base de 0,48 é 20%, enquanto +0,10 sobre 0,03 é 300%.

Efeito medido:

| quadro | pico do mapa cru | pico do relativo |
|---|---|---|
| 60 | (2,0) borda esquerda | (5,6) meio da cabeça |
| 150 | (0,0) borda esquerda | (1,11) **pulso** |
| 250 | (1,0) borda esquerda | (6,8) **pulso** |

Três quadros, três lugares diferentes — que é a assinatura de atenção guiada por
conteúdo. O cru dava o mesmo canto nas três. O título do quadrante diz qual dos
dois está na tela, e avisa enquanto a base ainda está aquecendo.

**2. Profundidade, crua e como o modelo recebeu.** Em cima o que a RealSense
mandou, em milímetros; embaixo o mosaico depois do log, do recorte de faixa e do
posicionamento por câmera. Se a de cima tem geometria e a de baixo está preta, a
profundidade não está chegando ao modelo — e nenhuma métrica de treino contaria
isso, porque a loss continua caindo com o modelo ignorando o canal.

**3. Nuvem de pontos.** Topo (XZ) e lateral (YZ), a mesma projeção que alimenta
o encoder de profundidade das outras políticas, refeita em numpy para o cliente
não precisar de torch. É onde se vê se a mesa está na altura certa e se a parede
está onde deveria.

**4. Temperatura dos motores.** Lida do `lowstate` do SDK, não do
`get_observation()` — aquele dict é o mesmo que a gravação de dataset consome, e
acrescentar chave nele mudaria o schema do que é gravado. Limiares em 45 °C
(amarelo) e 60 °C (vermelho), conservadores de propósito: o objetivo é ver a
subida antes do desarme.

## Protocolo

ZMQ REQ/REP com msgpack, porta 5600:

```
cliente  → { obs: {juntas..., head_camera, right_wrist_camera, head_camera_depth},
             obs_step, actions_per_chunk, want_debug, task? }
servidor → { chunk_np, obs_step, infer_ms, debug?: { attn, depth } }
erro     → { error }
```

`want_debug` só custa quando ligado: os ganchos de captura são instalados
naquela inferência e removidos no fim.

## O que difere do cliente/servidor antigos

| | ACT-D / PI05 | FastWAM-D |
|---|---|---|
| câmeras de cor | 1 (cabeça) | **2** (cabeça + pulso direito) |
| profundidade | encoder próprio (PointNet) | pelo latente do VAE |
| linguagem | só PI05 | **sempre** (o modelo é condicionado por texto) |
| juntas | 28 | **29** (entra o `kWaistYaw`) |
| chunk | 60 | **32** (`action_horizon` do modelo) |

Pedir `--chunk` maior que 32 não gera mais ação: o servidor recorta. Como o
`lead` precisa cobrir uma inferência inteira (~1,5 s ≈ 45 ciclos a 30 Hz) e o
chunk só tem 32, **o buffer vai zerar entre chunks** — na prática o robô vai
executar em passos, com pausa. As saídas honestas são: baixar o `--fps` do loop
de controle, ou aceitar a pausa. Não dá para inventar ação que o modelo não
emitiu.

## Cadência: quantos Hz o loop aguenta de verdade

O modelo emite **32 ações por inferência**. O tempo de uma inferência é o que
decide o `--fps` honesto:

```
                     payload    ida e volta    servidor
observação crua      2,19 MB        4905 ms      2528 ms
observação reduzida  0,40 MB        1493 ms       856 ms
```

A redução (ligada por padrão, `--sem-reducao` desliga) redimensiona as câmeras
de cor para 224x224 — **exatamente o que o servidor faria de qualquer jeito**, com
a mesma interpolação — e comprime a profundidade em PNG de 16 bits, que é sem
perda. O efeito nas ações previstas é 0,0017 rad em média (pior caso 0,0143),
contra os ~0,14 rad de erro do próprio modelo: irrelevante.

Com 1,5 s por inferência e 32 ações no chunk:

```
30 Hz → 32 ações duram 1,07 s  <  1,5 s  →  o buffer ZERA entre chunks
20 Hz → 32 ações duram 1,60 s  >  1,5 s  →  cobre, sem folga
15 Hz → 32 ações duram 2,13 s  >  1,5 s  →  cobre com margem
```

É por isso que a 30 Hz aparece `⏳ Aguardando o servidor...` e o robô anda em
solavancos: ele executa 1 segundo de movimento e congela esperando o próximo
chunk. **Use `--fps=15` ou `--fps=20`.** Não dá para inventar ação que o modelo
não emitiu — o horizonte é 32 e ponto.

## A trava de faixa (leia antes de ligar o robô)

O servidor limita toda ação à faixa que o dataset demonstrou (`action.min` /
`action.max` das estatísticas do próprio checkpoint), com 10% de folga por junta,
e **descarta** o chunk inteiro se vier NaN/inf — clamp não conserta lixo.

Ela existe por um motivo medido, não por precaução genérica:

| entrada | valores travados | faixa da saída |
|---|---|---|
| quadro real do dataset | **0** de 928 | −0,951 a 1,047 rad |
| imagem de ruído puro | **640** de 928 | −1,225 a 1,650 rad |

Sem a trava, o ruído produzia ação da ordem de **10⁵ rad**. Um expert de ação
por flow matching integra um campo de velocidade em 10 passos; fora da
distribuição de treino esse campo aponta para qualquer lugar e a integração
diverge. O modelo não tem como saber que aquilo é impossível — a trava tem.

Repare que ela é **inerte com entrada boa**: zero travadas no quadro real. Ou
seja, o contador `TRAVADAS n` que aparece no cabeçalho do painel não é ruído de
operação, é diagnóstico: **enquanto ele não for zero, o modelo está vendo algo
que não reconhece** e não faz sentido julgar o comportamento.

Fontes conhecidas de "fora da distribuição" neste projeto: render do MuJoCo
(o modelo só viu quadro real da RealSense), câmera preta por falha de stream,
`--depth-legado` (8 bits em vez de 16), e cena diferente da que foi gravada.

## Armadilhas de ambiente (custaram uma hora)

O servidor morria com **`Segmentation fault (core dumped)` sem imprimir uma
linha** — nem com `python -u`, nem com `PYTHONFAULTHANDLER=1`. São dois
problemas empilhados, os dois de biblioteca C++, e os dois só aparecem no
servidor porque é o único processo que carrega **zmq e torch juntos**.

**1. O libstdc++ do sistema é velho demais.** O `_multiarray_umath.so` do numpy
exige `GLIBCXX_3.4.29`, e o `/lib/x86_64-linux-gnu/libstdc++.so.6` da athena não
tem. Quem for carregado primeiro vale para o processo inteiro. Ativar o conda
**não resolve** — a `activate.d` não mexe em `LD_LIBRARY_PATH`. O launcher força
os dois:

```bash
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
```

O `LD_PRELOAD` é o que realmente resolve: o `libzmq` puxa o libstdc++ do sistema
antes de qualquer coisa, e sem preload as extensões nativas do
diffusers/transformers estouram depois, no meio do import do lerobot.

Teste de bancada, se você suspeitar disso de novo:

```bash
python -c "import zmq, torch, lerobot.policies.fastwam"   # sem o preload: segfault
```

**2. Ordem de import.** Por precaução o `zmq` é importado **antes** do torch nos
dois scripts, com a nota no topo de cada um. Sozinho isso não bastava (o preload
é que resolve), mas a ordem trocada dá um `ImportError` diferente e igualmente
confuso: `python -c "import zmq, torch"` funciona e `python -c "import torch, zmq"`
não.

**3. `python` não existe no PATH de um shell não interativo.** `screen -dmS ...`
e `ssh host comando` não passam pelo `.bashrc`, então o conda não está ativo. Os
launchers usam o caminho absoluto do interpretador do ambiente.

## Ressalvas conhecidas

- **Banda:** a observação vai crua, ~3 MB por inferência. Numa LAN a ~1 Hz é
  irrelevante; mandar comprimido com perda mudaria o que o modelo vê em relação
  ao treino.
- **Intrínsecos nominais.** A nuvem usa `fx=fy=617, cx=424, cy=240`. Os reais
  ainda não foram medidos: rodar `Scripts_Prometheus_int/print_camera_intrinsics.py`
  no robô e passar `--intrinsics=fx,fy,cx,cy`. A forma da nuvem está certa; a
  escala absoluta pode andar alguns por cento.
- **Temperatura das mãos** depende do Dex3 publicar o campo. Se vier zerada, o
  painel simplesmente não mostra a linha — o mesmo sensor que já aparece zerado
  na pressão (ver `PROFUNDIDADE_NATIVA.md`).
- **Nada disso rodou com o robô ainda.** O que foi testado está abaixo.

## Validado de ponta a ponta em 18/08/2026

Servidor na GPU 0 da athena, com a cópia congelada do checkpoint de **step 1000**
(5% do treino), enquanto o treino seguia rodando na GPU 2:

```
✅ FastWAM-D carregado — depth_mode=latent | horizonte=32 | câmeras=2
🔌 Escutando em tcp://0.0.0.0:5600
```

**Com quadro real do dataset** (episódio 0, quadro 120), comparado com a ação
que foi gravada naquele instante:

```
estado atual  : [ 0.007  0.068  0.145  1.416  0.234  0.066  0.285]
acao predita  : [-0.037  0.275  0.730 -0.088 -0.302 -0.240  0.000]
acao gravada  : [-0.052  0.224  0.650 -0.087 -0.203 -0.273  0.000]
erro medio nas 29 juntas: 0.0406 rad (2,3°)
inferencia: ~900 ms   depth chegando ao modelo: [0.000, 0.775]
```

**Do PC do robô até a athena** (o caminho de rede real, com observação
sintética): chunk `(32, 29)` de volta, `debug` com `attn` e `depth`, ~1,3 s de
ida e volta.

⚠️ **Com imagem de ruído puro a predição explode** (valores da ordem de 10⁵ rad).
Não é bug do caminho — é um modelo com 5% de treino recebendo entrada fora de
qualquer distribuição que ele viu. Vale como aviso de operação: **se a câmera
falhar e entregar quadro preto ou lixo, a ação sai absurda.** Rode com `--sim`
antes de qualquer coisa com o robô energizado.

## O que foi verificado

Sem GPU e sem robô nesta máquina, então são testes de unidade:

```
monta_obs: estado de 29 na ordem do dataset (kWaistYaw no índice 14)   OK
           rgb em [0,1], depth preservado em milímetros                OK
           faltando câmera ou depth → erro claro, não zeros            OK
           depth de 3 canais (servidor antigo) → erro                  OK
CapturaDebug: ignora o caminho de vídeo, captura o de ação             OK
           pico da atenção no token exato que foi amplificado          OK
           grade incompatível → devolve None em vez de mapa errado     OK
           ganchos removidos ao sair do contexto                       OK
painel:    quatro quadrantes renderizados, proporção preservada        OK
           casos degenerados (sem imagem/sem depth/sem telemetria)     OK
```
