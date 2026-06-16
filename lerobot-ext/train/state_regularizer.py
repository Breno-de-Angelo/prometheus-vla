"""Run 4a — regularização do `state` (texto) contra causal confusion / atalho proprioceptivo.

Diagnóstico (deploy no robô + probes causais + auditoria do dataset): no π0.5 o
`observation.state` é discretizado (256 bins) e injetado como TEXTO no prompt
(`"Task: …, State: <bins>;\nAction: "`). Como as trajetórias são suaves, o `state`
prediz a próxima ação quase perfeitamente — então o modelo aprende a SEGUIR a
trajetória pelo `state` e a IGNORAR a imagem (braço→imagem ~0.07 nas 3 runs do A/B).
On-distribution funciona; no robô a trajetória deriva e a mão fecha no vazio.

Conserto (aprovado na revisão), aplicado SÓ NO TREINO:
  * state-dropout  — com prob p, montar o prompt SEM o trecho `State: …` → remove a
                     muleta proprioceptiva e força o modelo a usar a imagem.
  * state-noise    — perturbar os bins discretizados do `state` em ±k (clip 0..255) →
                     destrói a confiabilidade do atalho sem removê-lo de vez.

POR QUE MONKEYPATCH (e não subclasse persistida): o processor é SALVO no checkpoint e
RECONSTRUÍDO no deploy a partir do nome/config do step registrado. Trocar a classe por
uma subclasse faria o config salvo referenciar um step custom → a inferência teria de
importá-lo/registrá-lo. Aqui só trocamos o MÉTODO `__call__` da classe DENTRO do processo
de treino; o config salvo (nome + campos do dataclass) fica idêntico ao original →
o deploy carrega o step ORIGINAL, sem dropout/ruído, sem nenhuma mudança na inferência.

Train-only de verdade: o patch só age quando `handle["enabled"]` está ligado, e isso só
acontece dentro de `state_regularizer_active(handle)` envolvendo o forward de TREINO.
O forward de validação e o processor salvo p/ deploy passam pelo caminho original (state
completo). Default OFF: com as duas flags em 0, `install_state_regularizer` é no-op
estrito (nunca toca a classe).
"""

from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_STATE

# Estado global do regularizador (um único step de state-tokenizer no pipeline do π0.5).
# `enabled` é ligado/desligado por forward via state_regularizer_active().
_STATE_REG = {"enabled": False, "dropout_prob": 0.0, "noise_bins": 0, "state_dim": None, "rng": None}
_ORIG_CALL = None  # __call__ original, preservado para o caminho no-op e p/ idempotência


def _regularized_call(self, transition):
    """Reimplementa Pi05PrepareStateTokenizerProcessorStep.__call__ com dropout/ruído.

    Idêntico ao original quando `enabled` está desligado. Mantém EXATAMENTE o mesmo
    pad/discretização/formato de prompt do original — a única diferença é (a) perturbar
    os bins reais e (b) às vezes omitir o trecho `State: …`.
    """
    cfg = _STATE_REG
    if not cfg["enabled"]:
        return _ORIG_CALL(self, transition)

    transition = transition.copy()

    state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
    if state is None:
        raise ValueError("State is required for PI05")
    tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
    if tasks is None:
        raise ValueError("No task found in complementary data")

    state = deepcopy(state)
    state = pad_vector(state, self.max_state_dim)

    # [-1, 1] (já normalizado pelo NormalizerProcessorStep anterior) → 256 bins [0, 255].
    state_np = state.cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    # --- State Noise Injection (treino-only): perturba os bins REAIS ±k, clip [0, 255] ---
    # Só as primeiras `state_dim` dims (as reais); as dims de padding (constantes) ficam
    # intactas — não carregam sinal, não vale poluir. k=0 → desligado.
    rng = cfg["rng"]
    k = cfg["noise_bins"]
    if k > 0:
        sd = cfg["state_dim"] or discretized_states.shape[1]
        noise = rng.integers(-k, k + 1, size=(discretized_states.shape[0], sd))
        discretized_states[:, :sd] = np.clip(discretized_states[:, :sd] + noise, 0, 255)

    # --- State Dropout (treino-only): com prob p, prompt SEM "State: …" -> força a imagem ---
    # Decisão POR AMOSTRA (cada item do batch independente).
    p = cfg["dropout_prob"]
    full_prompts = []
    for i, task in enumerate(tasks):
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        if p > 0.0 and rng.random() < p:
            full_prompt = f"Task: {cleaned_text};\nAction: "  # sem state
        else:
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        full_prompts.append(full_prompt)

    transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
    return transition


def install_state_regularizer(dropout_prob=0.0, noise_bins=0, state_dim=None, seed=0):
    """Liga o regularizador via monkeypatch (idempotente). Train-only.

    No-op estrito se `dropout_prob<=0` e `noise_bins<=0` (retorna None, não toca a classe).
    Caso contrário, troca `Pi05PrepareStateTokenizerProcessorStep.__call__` por
    `_regularized_call` e retorna o handle (dict). O patch só AGE quando `handle["enabled"]`
    está True — use `state_regularizer_active(handle)` ao redor do forward de treino.

    Args:
        dropout_prob: prob. p de omitir `State: …` do prompt (por amostra).
        noise_bins:   k; perturba os bins reais do state em [-k, +k] (clip 0..255).
        state_dim:    nº de dims reais do state (sem padding) — alvo do ruído. None = todas.
        seed:         semente do RNG DEDICADO (np.random.Generator) do regularizador.
    """
    global _ORIG_CALL
    if dropout_prob <= 0.0 and noise_bins <= 0:
        return None
    _STATE_REG["dropout_prob"] = float(dropout_prob)
    _STATE_REG["noise_bins"] = int(noise_bins)
    _STATE_REG["state_dim"] = int(state_dim) if state_dim else None
    # RNG dedicado seedado por cfg.seed: isola a sequência de dropout/ruído de outras chamadas
    # np.random no processo (reprodutibilidade — revisão do prof). NB: o estado do gerador NÃO é
    # persistido no checkpoint, então um resume não é bit-exato (aceitável p/ a hipótese causal;
    # p/ bit-exatidão num resume, salvar/restaurar rng.bit_generator.state).
    _STATE_REG["rng"] = np.random.default_rng(seed)
    if _ORIG_CALL is None:  # idempotente: só patcheia uma vez
        _ORIG_CALL = Pi05PrepareStateTokenizerProcessorStep.__call__
        Pi05PrepareStateTokenizerProcessorStep.__call__ = _regularized_call
    return _STATE_REG


@contextmanager
def state_regularizer_active(handle):
    """Liga o regularizador só dentro do bloco (forward de treino). No-op se handle=None.

    Garante desligar mesmo em exceção → o forward de validação e o save ficam no caminho
    original (state completo).
    """
    if handle is None:
        yield
        return
    handle["enabled"] = True
    try:
        yield
    finally:
        handle["enabled"] = False
