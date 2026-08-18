# Scripts da athena

Os arquivos daqui vivem em `/data/train_output/` na athena
(`hercules@10.9.8.252`) e estão versionados nesta pasta porque **cada linha
deles é uma armadilha que já custou tempo**. Recriá-los do zero significaria
tropeçar de novo em todas.

| arquivo | o que faz |
|---|---|
| `launch_fastwamd.sh` | sobe o TREINO. Escolhe a GPU mais livre, aponta `HF_HOME` para o `/data`, fixa `OMP_NUM_THREADS=1`. |
| `launch_server_fastwamd.sh` | sobe o SERVIDOR de inferência sobre uma cópia congelada do checkpoint. |
| `wandb_sidecar.py` | espelha para o wandb um treino **já rodando**, lendo o log. Para quando alguém esqueceu de ligar o wandb e o treino já está adiantado demais para reiniciar. |

## O que os launchers resolvem, e por quê

**`LD_PRELOAD` do libstdc++ do conda** (só no servidor, que é quem carrega zmq e
torch juntos). Sem isso o processo morre em `Segmentation fault (core dumped)`
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

Para atualizar a athena depois de mexer aqui, copie de volta:

```bash
scp lerobot-ext/athena/*.sh lerobot-ext/athena/*.py hercules@10.9.8.252:/data/train_output/
```

Ver `../docs/INFERENCIA_FASTWAMD.md` para o desenho completo do servidor,
do cliente e do painel de depuração.
