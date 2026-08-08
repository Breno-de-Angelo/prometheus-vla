# Documentação — `lerobot-ext`

Extensões do LeRobot para o Unitree G1 + mãos Dex3 do projeto Prometheus-VLA.

## Políticas

| Doc | Sobre |
|---|---|
| [OPENVLA_DEPTH.md](OPENVLA_DEPTH.md) | **OpenVLA-Depth** — VLA multi-tarefa condicionado por texto, com profundidade e tato. Head OFT paralelo. |
| [../policies/pi0_depth/README.md](../policies/pi0_depth/README.md) | **PI05-Depth** — flow matching + depth + tato. Como funciona a validação e o mapa das 28 juntas. |
| [../policies/act_depth/README.md](../policies/act_depth/README.md) | **ACT-D** — ACT + PointNet/PointTransformer + tato. Sem linguagem. |
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
