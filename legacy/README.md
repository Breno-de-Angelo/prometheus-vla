# legacy/

Arquivos **aposentados** (não deletados) durante a organização da branch `Luiz-pi05d`.
Cada arquivo aqui era duplicata, versão antiga (v1) ou cópia sem referências de algo que
continua em uso no repositório. Foram mantidos com prefixo `legacy_` para preservar o
histórico e permitir consulta/restauração.

Há pastas `legacy/` espelhadas em subdiretórios — sempre dentro do diretório onde o arquivo
morava:
- `legacy/` (raiz) — entry point obsoleto, scripts soltos duplicados, docs duplicadas
- `lerobot-ext/legacy/` — entry points v1 (record, inference, train, train_v2, play, viz, teleoparate)
- `lerobot-ext/Scripts_Prometheus_int/legacy/` — cópias de servidores idênticas às da raiz
- `lerobot-ext/teleop/utils/legacy/` e `.../teleop/unitree_g1/utils/legacy/` — cópias órfãs de IK/retargeting

## Critério
- Só foi para `legacy/` o que tinha **0 referências/imports** (verificado por `grep`/`git grep`)
  ou era **idêntico** a um arquivo canônico mantido.
- **Não** inclui features de depth nem código em uso: `realsense_server` com depth,
  `sensor_utils`, `policies/act_depth/*`, e os dois diretórios de treino (`train/` e
  `lerobot-ext/train/`) ficaram intactos.

## Restaurar um arquivo
```
git mv legacy/legacy_<nome> <destino_original>/<nome>     # traz de volta
git log --oneline --follow legacy/legacy_<nome>           # vê o histórico
```
