# Gravação com depth 16-bit (lossless) — pi05-D

> 📖 **Guia completo de teleop + record (com este depth16):** veja [`README_TELEOP_RECORD.md`](README_TELEOP_RECORD.md).
> Este arquivo cobre só o lado da profundidade.

Variante do fluxo de gravação que publica e grava a profundidade como **PNG uint16 em milímetros** (lossless), no lugar do uint8 (0–2 m) do fluxo padrão. O 8-bit colapsava uma cena de mesa em ~8–12 dos 256 níveis, deixando a depth como peso morto na fusão; o 16-bit preserva a precisão métrica.

Depende da branch `Luiz-pi05d` — os patches de transporte e gravação estão no `init_lerobot_record_v2.py`.

## Topologia

- **Robô** (`unitree@192.168.68.71`): roda os servidores — câmera + `run_g1_server.py`.
- **Laptop**: roda o `init_lerobot_record_v2.py` (branch `Luiz-pi05d`), conectando no robô via DDS + câmera ZMQ na porta 5555.

## Rodar

Suba na ordem abaixo: terminais 1 e 2 (robô) primeiro; o terminal 3 (laptop) **só depois** que a câmera e o servidor do G1 estiverem rodando.

> Antes de subir a câmera, confirme que **nenhum outro `realsense_server` está rodando** no robô — a porta 5555 fica ocupada e a gravação pegaria a stream errada.

### Terminal 1 — Robô: câmera (depth 16-bit)

```bash
ssh unitree@192.168.68.71                                # se acessando do laptop
source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1
cd ~/lerobot/src/lerobot/cameras/zmq
python realsense_server_depth16.py --serial 327122071538
```

### Terminal 2 — Robô: servidor de baixo nível do G1

```bash
ssh unitree@192.168.68.71                                # se acessando do laptop
source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1
cd ~/lerobot/src/lerobot/robots/unitree_g1
python run_g1_server.py
```

### Terminal 3 — Laptop: gravação

> ⚠️ Ao iniciar, o robô faz o **SOFT START** (rampa de ~3 s) e move o braço/mãos para a pose comandada. Antes de rodar: área livre, robô apoiado e E-stop ao alcance.

```bash
cd ~/I2CA/prometheus-vla/lerobot-ext
git branch --show-current                                # confirme: Luiz-pi05d
conda activate g1
python init_lerobot_record_v2.py --config_path config/record/record_televuer.yaml
```

Os dois lados precisam casar: `realsense_server_depth16` (robô) ↔ `init_record` da `Luiz-pi05d` (laptop). Rodar com o realsense padrão, ou o record fora da branch, gera depth corrompida ou quebra a gravação por shape.

## O que muda no dataset

`observation.images.head_camera_depth` passa de vídeo h264 (8-bit, 3 canais) para **imagem PNG uint16 de 1 canal**, em milímetros. RGB (`head_camera`) e o tátil (`left/right_hand_pressure`) seguem iguais.

## Validar a gravação

Depois de gravar um episódio, a feature de depth tem que estar como `image` (não `video`):

```bash
python -c "import json,glob; f=sorted(glob.glob('datasets/G1_Dex3_depth_tactil_dataset/*/meta/info.json'))[-1]; print(json.load(open(f))['features']['observation.images.head_camera_depth'])"
# esperado: dtype 'image', shape [480, 848, 1]
```

Um PNG de depth aberto com PIL deve ter modo `I;16` (16-bit), e os valores são distâncias em mm.

## Treino (atena `hercules@10.9.8.252`)

O treino lê a depth em mm e converte pra metros na back-projection: `pi05_d_injector` chama `depth_to_pointcloud(..., depth_scale=0.001)`. Config: `config/train/train_cup_depth_tactil_pi05_full.yaml`. Os patches de treino (`pi05_d_injector` + config) precisam estar na atena antes de treinar.

## Notas

- **Clip em 32767 mm**: o `ToTensor` do LeRobot decodifica PNG `I;16` como `int16` (com sinal), então manter ≤ 32767 evita overflow. 32767 mm (~32 m) é muito além do workspace de manipulação.
- **Escala**: o z16 da D435i já vem em mm (depth scale 0.001 m/unidade); no treino, `z_metros = mm / 1000`.
- `realsense_server_depth16.py` é self-contained (encode PNG inline, não depende do `encode_depth_image`); o `realsense_server.py` original (uint8 JPEG) fica intocado no robô.
- Datasets gravados em 8-bit são incompatíveis com este fluxo — regrave em 16-bit.
