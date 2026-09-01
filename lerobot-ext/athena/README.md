# Scripts da athena

Os arquivos daqui vivem em `/data/train_output/` na athena
(`hercules@10.9.8.252`) e estão versionados nesta pasta porque **cada linha
deles é uma armadilha que já custou tempo**. Recriá-los do zero significaria
tropeçar de novo em todas.

| arquivo | o que faz |
|---|---|
| `launch_fastwamd.sh` | sobe o TREINO. Escolhe a GPU mais livre, aponta `HF_HOME` para o `/data`, fixa `OMP_NUM_THREADS=1`. |
| `launch_server_fastwamd.sh` | sobe o SERVIDOR de inferência sobre uma cópia congelada do checkpoint. |
| `launch_client_fastwamd.sh` | sobe o LOOP DE CONTROLE, com o robô real na LAN (`10.9.8.73`) e o painel web na 8088. |
| `launch_grounding.sh` | desenha onde o TEXTO olha na imagem. Não treina nem infere ação — só a cross-attention vídeo→texto. Roda com qualquer login (ver os dois caches abaixo). |
| `wandb_sidecar.py` | espelha para o wandb um treino **já rodando**, lendo o log. Para quando alguém esqueceu de ligar o wandb e o treino já está adiantado demais para reiniciar. |

## O que os launchers resolvem, e por quê

**Os dois caches do HuggingFace, separados** (no `launch_grounding.sh`). O
`/data/.cache/huggingface` é do `hercules` e guarda 25 GB de pesos do Wan que
ninguém quer baixar de novo — mas a biblioteca também ESCREVE lá (locks do
`datasets`, metadados). Apontar `HF_HOME` para ele com outro login quebra, e o
erro só aparece depois de dois minutos carregando o modelo:

```
PermissionError: [Errno 13] Permission denied:
'/data/.cache/huggingface/datasets/..._0.0.0_....lock'
```

O conserto é dividir: `HF_HUB_CACHE=/data/.cache/huggingface/hub` (os pesos, só
leitura, compartilhado) e `HF_HOME=$HOME/.cache/huggingface` (a escrita, no home
de quem rodou). Vale para qualquer script, não só este.

**O python vem de `$HOME`, não de um caminho fixo** (também só no
`launch_grounding.sh`). Os launchers de treino fixam
`/home/hercules/miniconda3/...` porque só o `hercules` treina; o grounding roda
com mais de um login na mesma máquina, cada um com o seu miniconda. Continua
sendo caminho absoluto e não `conda activate`, pelo motivo do parágrafo abaixo.

**`LD_PRELOAD` do libstdc++ do conda** (no servidor e no cliente: o cliente não
carrega modelo, mas importa torch pela cadeia do `init_lerobot_inference_async_v2`
e paga o mesmo pedágio). Sem isso o processo morre em `Segmentation fault (core dumped)`
**sem imprimir uma linha** — nem com `python -u`, nem com `PYTHONFAULTHANDLER=1`.
O libstdc++ do sistema não tem `GLIBCXX_3.4.29`, que o numpy exige; o libzmq
carrega o do sistema primeiro e ele passa a valer para o processo inteiro.
`conda activate` **não** resolve.

**Caminho absoluto do interpretador.** `screen -dmS` e `ssh host comando` não
passam pelo `.bashrc`: o conda não está ativo e `python` nem existe no PATH.

**`HF_HOME=/data/.cache/huggingface`.** O disco de sistema da athena vive perto
de 100% e os pesos do Wan passam de 25 GB.

**Escolha da GPU por memória livre.** As três A100 são compartilhadas com outras
pessoas. Fixar a GPU 0 foi o que causou um `OutOfMemoryError` com outro job
ocupando 57 dos 80 GB dela. Passe o número como primeiro argumento para forçar.

**Servidor lê uma CÓPIA do checkpoint**, não o `checkpoints/best` do treino:
aquele é reescrito por rename atômico a cada melhora, e o servidor leria o
diretório sendo trocado embaixo dele.

## Como usar

```bash
# treino (GPU automática, ou passe o número)
bash /data/train_output/launch_fastwamd.sh
bash /data/train_output/launch_fastwamd.sh 0 --policy.depth_mode=token

# servidor de inferência (GPU 0, checkpoint padrão)
bash /data/train_output/launch_server_fastwamd.sh 0 [caminho/do/pretrained_model]
```

### Inferência no robô real, com tudo na athena

Dois screens no mesmo host: o modelo de 6 B numa GPU, e o loop de controle
falando com o robô pela LAN. **O bridge do robô é pré-requisito e sobe à mão**,
no próprio robô:

```bash
# 1. NO ROBÔ (à mão), três processos — os mesmos que gravaram o dataset:
python Scripts_Prometheus_int/dex3_g1_server_v2.py --loco   # bridge, 6000-6005
python Scripts_Prometheus_int/full_realsenser_server.py     # cabeça RGB+depth, 5555
python Scripts_Prometheus_int/right_arm_realsense_server.py # pulso direito, 5556
```

⚠️ **O `--loco` não é opcional.** O dataset foi gravado com ele
(`config/record/step1_white_cup_on_dripper.yaml`, `use_loco: true`), e o cliente
descobre o modo por handshake na porta 6004 — sem `--loco` o robô roda em low
level, com o WBC desligado e o tronco mole, que é a causa conhecida de ele
tombar para a frente.

```bash
# 2. NA ATHENA
screen -dmS infer   bash /data/train_output/launch_server_fastwamd.sh 2
screen -dmS control bash /data/train_output/launch_client_fastwamd.sh
```

```bash
# 3. DE QUALQUER LUGAR DA LAN: o painel
http://10.9.8.252:8088/
```

Confira que o robô está de pé antes de subir o controle — o bridge escuta em
`0.0.0.0`, então da athena isto tem que responder:

```bash
for p in 5555 5556 6000 6001 6002 6003 6004; do
  timeout 2 bash -c "echo > /dev/tcp/10.9.8.73/$p" && echo "$p ok" || echo "$p FECHADA"
done
```

**Por que `--server=127.0.0.1` no cliente:** com os dois processos na athena, a
observação (~600 KB reduzida) não sai da máquina. O que atravessa a rede agora é
só o que o robô manda — que é o tráfego que existiria de qualquer jeito.

**Por que `CUDA_VISIBLE_DEVICES=""` no cliente:** ele não infere nada. Sem isso o
torch reserva contexto CUDA numa placa à toa e passa a disputar com o servidor
ao lado.

Para atualizar a athena depois de mexer aqui, copie de volta:

```bash
scp lerobot-ext/athena/*.sh lerobot-ext/athena/*.py hercules@10.9.8.252:/data/train_output/
```

Ver `../docs/INFERENCIA_FASTWAMD.md` para o desenho completo do servidor,
do cliente e do painel de depuração.
