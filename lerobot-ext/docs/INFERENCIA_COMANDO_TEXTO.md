# Rodar o robô por comando de texto

> Como mandar o G1 executar uma tarefa dizendo o que fazer — `"pick up the white
> mug and place it to the right"` — em vez de trocar de checkpoint.

Vale para `pi05depth` e `openvladepth`. O `actdepth` não tem linguagem nenhuma e
ignora tudo desta página.

---

## 1. O mecanismo, em uma linha

O texto que você passa em `--task` vira o prompt do modelo:

```
--task="pick up the white mug and place it to the right"
        ↓
"In: What action should the robot take to pick up the white mug and place it to the right?\nOut:"
        ↓
tokenizer Llama-2 → tokens no prefixo → LLM → action queries → chunk [50, 28]
```

**Nada mais no modelo sabe qual tarefa está rodando.** Não há head por tarefa,
nem índice, nem seletor. Trocar o texto é a única forma de mudar o comportamento,
e é também a única coisa que pode dar errado.

---

## 2. Rodar

```bash
python init_lerobot_inference_v3.py \
    --checkpoint=train/output/openvla_depth_cup_2026-06-09/best_val_checkpoint/pretrained_model \
    --task="pick up the white mug and place it to the right" \
    --cam-robot=192.168.123.164 \
    --v
```

`--task` é **obrigatório** para políticas de linguagem. Sem ele o script para com
erro explícito, em vez de rodar com prompt vazio e produzir movimento aleatório.

Sempre use o `best_val_checkpoint/`, nunca o último checkpoint periódico — o
último pode ter sofrido overfitting.

### Trocar o comando com o robô rodando

```bash
python init_lerobot_inference_v3.py \
    --checkpoint=<PATH> \
    --task="pick up the white mug and place it to the right" \
    --interactive
```

Cada linha digitada vira o novo comando:

```
💬 Modo interativo: digite um comando e Enter para trocar a tarefa.

place the white mug back on the table
🗣️  Comando trocado para: "place the white mug back on the table"
```

Detalhe que importa: ao trocar o comando o script chama `policy.reset()`. Sem
isso o robô terminaria o chunk anterior — até 50 passos, ~1,7 s a 30 fps — antes
de obedecer. Com o reset, a mudança vale no próximo passo.

---

## 3. O texto tem que casar com o do treino

Esta é a causa mais comum de "o comando não funciona".

O modelo aprendeu a associar **as strings exatas que estavam no dataset**. Para
descobrir quais são:

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as M
m = M('local/x', root='meu_dataset/pick_up_the_cup_2026-06-09')
print(list(m.tasks.index))"
```

```
['pick up the white mug and place it to the right']
```

Use exatamente essa string. Variações próximas (`grab the white mug...`) podem
funcionar — o LLM generaliza — mas só se o dataset tiver variedade de redação
para a mesma tarefa (ver `DATASETS_MULTITAREFA.md`, seção 2.2). Com uma única
redação gravada, o modelo tende a só responder bem a ela.

> **Português não funciona bem.** O pré-treino robótico do OpenVLA é todo em
> inglês, e o template do prompt é `"What action should the robot take to
> {task}?"`. Comandos em português degradam bastante mesmo quando o texto está
> correto.

---

## 4. O teste que diz se o condicionamento existe

Um modelo treinado com **uma tarefa só** roda normalmente com `--task` e parece
funcionar — mas está ignorando o texto. O `val_loss` não denuncia isso.

O teste decisivo:

> Com o robô na **mesma posição inicial** e a **mesma cena**, rode dois comandos
> diferentes e compare as trajetórias.

```bash
# terminal 1 — comando A
python init_lerobot_inference_v3.py --checkpoint=<PATH> --sim \
    --task="pick up the white mug and place it to the right" --debug

# terminal 2 (depois, mesma cena) — comando B
python init_lerobot_inference_v3.py --checkpoint=<PATH> --sim \
    --task="pick up the kettle" --debug
```

| resultado | significa |
|---|---|
| trajetórias **diferentes** | o modelo está lendo o texto ✓ |
| trajetórias **iguais** | está ignorando o texto ✗ |

Se derem iguais, o problema quase sempre é o dataset, não a inferência:

1. o treino usou um dataset de tarefa única (o modelo aprendeu que o texto é
   constante e irrelevante)
2. `override_task` estava preenchido no YAML de treino, o que força um prompt
   fixo e anula o condicionamento
3. as tarefas do dataset unificado tinham a mesma string de `task`

Use o `--sim` para esse teste: o MuJoCo dá a mesma cena inicial nas duas
execuções, o que o robô real não dá.

---

## 5. Uncertainty gate

Opcional, para operação mais conservadora:

```bash
--uncertainty=0.1
```

No `openvladepth` a incerteza vem de MC-dropout no head: o LLM roda uma vez e só
o MLP final roda N vezes com dropout ativo. Acima do limiar, a ação é misturada
com a posição neutra proporcionalmente ao excesso.

Só ligue depois de calibrar a posição neutra — o motor de treino avisa se a
`default_positions` não está no mesmo espaço de ação do dataset
(`[NeutralPosition] N junta(s) com valor normalizado > 3σ`). Com o buffer
errado, o gate empurra o robô para uma pose que ele nunca viu.

---

## 6. Problemas comuns

| sintoma | causa provável |
|---|---|
| `a política 'openvladepth' é condicionada por linguagem e precisa de um comando` | faltou `--task` |
| `Tipo 'X' não mapeado. Adicione em _POLICY_CLASS_MAP` | política nova sem entrada em `load_policy` |
| `Nenhum 'task' em complementary_data` | `--task` chegou vazio no processador |
| robô faz sempre a mesma coisa | ver seção 4 |
| movimento errático desde o primeiro passo | checkpoint de tarefa diferente, ou `--task` com texto que não existe no dataset |
| robô demora ~2 s para reagir ao comando novo | faltou `policy.reset()` — só acontece se você chamou a política fora deste script |

---

## Referências no código

| o quê | onde |
|---|---|
| prompt montado a partir do `task` | `policies/openvla_depth/processor_openvla.py` |
| `--task` / `--interactive` | `init_lerobot_inference_v3.py::main` |
| mapa de políticas | `init_lerobot_inference_v3.py::load_policy` |
| observação bruta com `task` | `init_lerobot_inference_v3.py::make_raw_obs` |
| uncertainty gate | `policies/openvla_depth/modeling_openvla.py::_apply_uncertainty_gate` |
