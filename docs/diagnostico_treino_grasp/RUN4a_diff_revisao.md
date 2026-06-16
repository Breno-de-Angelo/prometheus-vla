# Run 4a — diff para revisão da lógica do step (state-dropout + state-noise)

> **15/06/2026 19:20** · 1ª versão (nada a corrigir).
> **O que é:** o diff da implementação da **run 4a** (conserto do grasp via regularização
> do `state`), para revisarmos **a lógica do step antes de disparar o treino**, como
> combinado. Branch `run4a-state-dropout` (off `Luiz-pi05d`). **O treino NÃO foi rodado.**

---

## 1. Resumo

Implementei os dois consertos aprovados — **state-dropout** (p=0.5) e **State Noise
Injection** (k=3) — **só no treino**, como flags independentes. Três arquivos:

| arquivo | o quê |
|---|---|
| `lerobot-ext/train/state_regularizer.py` *(novo, 138 l)* | o step (lógica de dropout/ruído) |
| `lerobot-ext/train/run_train_valfix.py` *(edit cirúrgico)* | flags na config + plugue train-only |
| `lerobot-ext/config/train/train_cup_pi05_right8_armstate7_run4a.yaml` *(novo)* | run 4a (= run 2 + as flags) |

---

## 2. Decisões de implementação (gostaria do seu aval)

**(a) Monkeypatch do `__call__`, não subclasse persistida.** O processor do π0.5 é
**salvo no checkpoint e reconstruído no deploy** a partir do *nome registrado* do step.
Se eu trocasse a classe por uma subclasse custom, o config salvo passaria a referenciá-la
→ a inferência teria de **importá-la/registrá-la** (mudança no deploy, risco). Em vez
disso, troco só o **método `__call__` dentro do processo de treino**. O config salvo
(nome + campos do dataclass) fica **idêntico ao original** → o deploy carrega o step
**original**, sem dropout/ruído, **zero mudança na inferência**. Mesma lógica que você
pediu, sem o efeito colateral de persistência. (Se preferir a subclasse explícita,
faço — mas aí preciso registrá-la e importá-la também no deploy.)

**(b) Train-only de verdade — três caminhos separados:**
- **treino** (`run_train_valfix.py:725`): forward envolto em `state_regularizer_active(...)` → dropout/ruído **ligados**;
- **validação** (`:824`): `preprocessor(val_batch)` **fora** do wrapper → **state completo** (a val mede o comportamento de deploy, não o regime de treino);
- **deploy**: processor original (item a) → **state completo**.

**(c) EMA desligada nesta run.** 4a = a regularização do `state` como **única variável**
vs a run 2 (valfix) → ablação limpa do conserto open-loop. EMA é ortogonal e fica pra
depois (ou já religo se você preferir 4a sobre a run 3). **Confirma EMA off?**

**(d) Dropout + ruído juntos na 4a** (como você aprovou). As flags são **independentes**
no código (dá pra rodar só dropout, só ruído, ou os dois) — se quiser ablação separada
(p sozinho vs k sozinho), é trivial. **Mantemos os dois juntos na 4a?**

---

## 3. A lógica do step (o que peço pra você revisar)

Reimplemento o `Pi05PrepareStateTokenizerProcessorStep.__call__` **idêntico ao original**
(mesmo pad → discretização em 256 bins → formato do prompt), com duas únicas mudanças:

```python
# [-1,1] (já normalizado) -> 256 bins [0,255]  (idêntico ao original)
discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

# (1) State Noise Injection: perturba SÓ os bins das dims reais ±k, clip [0,255].
#     padding (dims 7..31, constantes) fica intacto — não carrega sinal.
k = cfg["noise_bins"]
if k > 0:
    sd = cfg["state_dim"] or discretized_states.shape[1]          # = 7 (armstate7)
    noise = np.random.randint(-k, k + 1, size=(B, sd))
    discretized_states[:, :sd] = np.clip(discretized_states[:, :sd] + noise, 0, 255)

# (2) State Dropout: por amostra, com prob p, prompt SEM "State: ..." -> força a imagem.
p = cfg["dropout_prob"]
for i, task in enumerate(tasks):
    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
    if p > 0.0 and np.random.random() < p:
        full_prompt = f"Task: {cleaned_text};\nAction: "                          # sem state
    else:
        state_str = " ".join(map(str, discretized_states[i]))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "      # = original
```

Decisões finas embutidas, pra você checar:
- **dropout por amostra** (cada item do batch decide independente), não por batch;
- **ruído só nas 7 dims reais** (o padding é constante → poluí-lo não ajuda);
- **ruído e dropout compõem**: com prob `p` → sem state; senão → state **com ruído**;
- `k=0` e `p=0` → **no-op estrito** (nem aplica o patch; a classe nem é tocada).

Teste numérico isolado (numpy): bins ∈ [0,255] após o clip ✓; ruído ∈ [−k,k] ✓; padding
intacto ✓; dropout por amostra ✓; os dois formatos de prompt corretos ✓.

---

## 4. O plugue no treino (edit cirúrgico)

```python
# em make_*: lê quantas dims reais o state tem (7), pra mirar o ruído
state_reg = install_state_regularizer(
    dropout_prob=cfg.state_dropout_prob,   # 0.5
    noise_bins=cfg.state_noise_bins,       # 3
    state_dim=_state_dim,                  # 7
)                                          # -> None (no-op) se ambas forem 0

# no loop de treino (e SÓ aqui):
with state_regularizer_active(state_reg):
    batch = preprocessor(batch)
```

Config da run 4a = a da run 2/3 com **só** estas linhas a mais:
```yaml
state_dropout_prob: 0.5
state_noise_bins: 3
ema_enabled: false      # 4a sozinha
# from base pi05 (pretrained_path = models/pi05_base_0601), não resume
```
(Augs visuais que você sugeriu — ColorJitter / RandomResizedCrop / RandomErasing — **já
estão** na config herdada da run 2/3.)

---

## 5. Validação da hipótese (como vou medir se funcionou)

Mesma probe causal do diagnóstico, no checkpoint da 4a: a sensibilidade **braço→imagem**
saiu de ~0.07? + deploy no robô. Se a 4a não bastar → run 4b (grounding loss).

---

## 6. O que falta (aguardando seu aval)

Está tudo escrito e compilando, **mas o treino não foi disparado** — como combinado,
quero seu OK na lógica do step (seção 3) e nas decisões (a)–(d) da seção 2 antes de
rodar no cluster.
