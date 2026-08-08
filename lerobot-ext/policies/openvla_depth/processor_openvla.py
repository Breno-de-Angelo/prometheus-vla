#!/usr/bin/env python
"""
OpenVLA-Depth — pipelines de pré e pós-processamento.

A peça que interessa aqui é o `OpenVLADepthPromptProcessorStep`: é ele que
transforma o campo `task` de cada frame do dataset no prompt que condiciona a
rede. Esse é o mecanismo inteiro do multi-tarefa — nenhuma outra parte do modelo
sabe qual tarefa está sendo executada.

    task no dataset:  "pick up the cup"
    prompt gerado:    "In: What action should the robot take to pick up the cup?\\nOut:"

Se `state_as_token=False`, o estado também é discretizado em 256 bins e escrito
no prompt (formato do pi05depth). Com o padrão `state_as_token=True` o estado
entra como token contínuo e o prompt fica só com o texto da tarefa — sobra muito
mais orçamento de contexto e o prompt fica legível para depuração.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_openvla import OPENVLA_PROMPT_TEMPLATE, OPENVLADEPTHConfig


@ProcessorStepRegistry.register(name="openvladepth_prompt_processor_step")
@dataclass
class OpenVLADepthPromptProcessorStep(ProcessorStep):
    """Monta o prompt em linguagem natural a partir do `task` do dataset."""

    prompt_template: str = OPENVLA_PROMPT_TEMPLATE
    task_key: str = "task"
    # Quando False, o estado é discretizado e embutido no texto (modo pi05depth).
    state_as_token: bool = True
    max_state_dim: int = 32
    # Texto fixo que substitui o `task` do dataset. Só para debug — ver o aviso
    # em OPENVLADEPTHConfig.__post_init__.
    override_task: str | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError(
                "Nenhum `task` em complementary_data. O OpenVLA-Depth é condicionado "
                "por linguagem — o dataset precisa ter a descrição da tarefa."
            )

        state_strings = None
        if not self.state_as_token:
            state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
            if state is None:
                raise ValueError("state_as_token=False exige observation.state no batch.")
            # O estado já chega em [-1, 1] (o NormalizerProcessorStep roda antes).
            state_np = state.detach().cpu().numpy()
            bins = np.linspace(-1, 1, 256 + 1)[:-1]
            discretized = np.digitize(state_np, bins=bins) - 1
            state_strings = [" ".join(map(str, row)) for row in discretized]

        prompts = []
        for i, task in enumerate(tasks):
            raw = self.override_task if self.override_task is not None else task
            cleaned = raw.strip().replace("_", " ").replace("\n", " ")
            prompt = self.prompt_template.format(task=cleaned.lower())
            if state_strings is not None:
                # O estado é contexto, então entra ANTES do marcador de resposta.
                # Colocá-lo depois de "Out:" faria o modelo tratá-lo como parte da
                # saída que ele deveria gerar.
                state_str = f" State: {state_strings[i]};"
                marker = "\nOut:"
                prompt = (
                    prompt.replace(marker, state_str + marker)
                    if marker in prompt
                    else prompt + state_str
                )
            prompts.append(prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Não altera definições de feature — só reescreve o texto da tarefa."""
        return features


def make_openvladepth_pre_post_processors(
    config: OPENVLADEPTHConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Pré-processamento:
      1. Renomeia observações (compatibilidade com checkpoints).
      2. Adiciona dimensão de batch.
      3. Normaliza estado/ação (quantis) — imagens ficam em IDENTITY.
      4. Monta o prompt a partir do `task`.
      5. Tokeniza com o tokenizer do Llama-2 do próprio checkpoint OpenVLA.
      6. Move tudo para o device.

    A ordem 3 → 4 importa: com `state_as_token=False` o passo 4 discretiza o
    estado assumindo que ele já está em [-1, 1].

    Pós-processamento: desnormaliza a ação e volta para a CPU.
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        OpenVLADepthPromptProcessorStep(
            prompt_template=config.prompt_template,
            state_as_token=config.state_as_token,
            max_state_dim=config.max_state_dim,
            override_task=config.override_task,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name or config.pretrained_backbone,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
