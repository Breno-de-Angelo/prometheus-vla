# Documentação — `lerobot-ext`

Extensões do LeRobot para o Unitree G1 + mãos Dex3 do projeto Prometheus-VLA.

## Fluxo de trabalho

Na ordem em que você vai precisar:

| Doc | Sobre |
|---|---|
| [DATASETS_MULTITAREFA.md](DATASETS_MULTITAREFA.md) | Como gravar uma tarefa por dataset e juntar tudo num dataset multi-tarefa. Congelar o schema antes de gravar, string de `task`, balanceamento, merge, teste no MuJoCo. **Leia antes de gravar a primeira demo.** |
| [TREINO_PASSO_A_PASSO.md](TREINO_PASSO_A_PASSO.md) | Roteiro linear: treinar a tarefa que já temos, e depois acrescentar a segunda. Orçamento de tempo medido, o que acompanhar no log, e por que fine-tune sequencial não produz multi-tarefa. |
| [SCHEMA_G1_V2.md](SCHEMA_G1_V2.md) | Yaw do tronco (28→29 dims) e câmera de pulso. O que muda em cada arquivo, por que roll/pitch ficam travados, e a medição do ponto de suporte no MuJoCo. **Muda o schema — leia antes de gravar.** |
| [PROFUNDIDADE_NATIVA.md](PROFUNDIDADE_NATIVA.md) | Profundidade métrica de ponta a ponta: uint16 em mm no ZMQ, mapa de 1 canal no dataset, milímetros na nuvem de pontos. As duas pontas têm que casar — mudar uma só dá dado errado em silêncio. **Leia antes de mexer em câmera, servidor ou política com depth.** |
| [INFERENCIA_FASTWAMD.md](INFERENCIA_FASTWAMD.md) | Inferência remota do FastWAM-D: o modelo de 6 B na athena, o loop de controle no seu PC, e o painel de depuração de quatro quadrantes (atenção do DiT, profundidade crua × a do modelo, nuvem de pontos e temperatura dos motores). |
| [MIGRACAO_CODIGO_061.md](MIGRACAO_CODIGO_061.md) | O que quebrou no nosso código na subida para a 0.6.1: símbolos que mudaram de módulo, `eval_freq`/`vcodec` que sumiram dos configs, o índice do sampler, e os monkeypatches que viraram comportamento nativo. |
| [SIM_REMOTO.md](SIM_REMOTO.md) | MuJoCo no seu PC, modelo de 7B na atena. Portas, IPs, e por que testar condicionamento por linguagem no simulador e não no robô. |
| [INFERENCIA_COMANDO_TEXTO.md](INFERENCIA_COMANDO_TEXTO.md) | Rodar o robô mandando o comando em texto. `--task`, troca de comando em tempo de execução, e o teste que revela se o modelo está mesmo lendo o prompt. |

## Ferramentas

| Script | Para quê |
|---|---|
| [`../train/build_multitask_dataset.py`](../train/build_multitask_dataset.py) | Junta datasets de tarefa única num multi-tarefa. `--dry-run` valida o schema antes de copiar GB. |
| [`../train/rename_dataset_task.py`](../train/rename_dataset_task.py) | Corrige a string de `task` de um dataset já gravado, sem tocar em vídeos ou dados. |
| [`../Scripts_Prometheus_int/print_camera_intrinsics.py`](../Scripts_Prometheus_int/print_camera_intrinsics.py) | Lê os intrínsecos reais da RealSense, no formato do YAML. Rode no robô. |

## Políticas

| Doc | Sobre |
|---|---|
| [OPENVLA_DEPTH.md](OPENVLA_DEPTH.md) | **OpenVLA-Depth** — VLA multi-tarefa condicionado por texto, com profundidade e tato. Head OFT paralelo. |
| [../policies/pi0_depth/README.md](../policies/pi0_depth/README.md) | **PI05-Depth** — flow matching + depth + tato. Como funciona a validação e o mapa das 28 juntas. |
| [../policies/act_depth/README.md](../policies/act_depth/README.md) | **ACT-D** — ACT + PointNet/PointTransformer + tato. Sem linguagem. |
| [../policies/fastwam_depth/README.md](../policies/fastwam_depth/README.md) | **FastWAM-D** — world action model (Wan2.2-5B) com profundidade métrica no latente. Texto, prior espacial de pré-treino em vídeo, e ablação embutida (latent/token/off). Precisa de GPU grande. |
| [../train/train.md](../train/train.md) | Arquitetura do ACT-D e o pipeline de ingestão de dados. |

### Qual usar

```
Precisa que o robô obedeça a comandos em texto diferentes?
├── sim → o dataset tem mais de uma string de `task`?
│         ├── sim → openvladepth  (generalização melhor, 7B, precisa de GPU grande)
│         │         pi05depth     (3B, mais barato, já funciona hoje)
│         └── não → grave demos das outras tarefas primeiro; nenhum VLA
│                   aprende a atender a um comando que nunca variou
└── não → actdepth (mais leve e rápido para tarefa única)
```

> Atenção: `override_task` no YAML do `pi05depth` **ou** do `openvladepth` força
> um prompt fixo e desliga o multi-tarefa. Use só para depurar.

## Operação

Os documentos de teleoperação, inferência assíncrona e ambiente estão em
[`../../docs/`](../../docs/) (raiz do repositório):

- `ambiente.md` — setup do ambiente
- `Teleoperacao_JoystickG1Arms.md` — teleoperação
- `unitree_g1_async_inference.md`, `g1_async_inference_atena.md` — inferência assíncrona
- `Erro_DataSets.md` — problemas comuns com datasets

Gravação de datasets: `init_lerobot_record_v2.py` nesta branch (o tutorial
`README_RECORD.md` está na branch `main`).
