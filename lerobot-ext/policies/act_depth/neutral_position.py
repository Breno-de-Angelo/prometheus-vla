"""
neutral_position.py — Calculador de Posição Neutra para o ACT-D

Responsabilidade única: converter os `default_positions` do UnitreeG1Config
para o espaço normalizado que o ACT usa internamente.

Por que isso é necessário:
  - O robô opera em radianos físicos (ex: cotovelo = -0.5 rad)
  - O ACT treina e infere em espaço normalizado (mean=0, std=1)
  - O uncertainty gate precisa da posição neutra no MESMO espaço da saída
    da rede, senão o blend vai misturar valores incompatíveis.

Fluxo:
  default_positions (29 juntas, radianos físicos)
      ↓  filtra para as juntas controladas (upper_body = 14, full_body = 29)
      ↓  normaliza: (valor - mean) / std   usando dataset.meta.stats["action"]
      ↓  tensor no espaço da ação normalizado
"""

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from robot.config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)


def compute_neutral_position(
    robot_config: "UnitreeG1Config",
    action_stats: dict[str, torch.Tensor],
    action_dim: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Converte os `default_positions` do robô para o espaço normalizado da ação.

    Args:
        robot_config: instância de UnitreeG1Config com default_positions preenchido.
        action_stats: dict com chaves "mean" e "std", shape [action_dim].
                      Geralmente: dataset.meta.stats["action"]
        action_dim: dimensão da ação que a política gera (ex: 28).
        device: onde colocar o tensor resultante.

    Returns:
        Tensor [action_dim] no espaço normalizado. Pronto para usar no
        uncertainty gate do ACTPolicy.
    """
    default_pos = robot_config.default_positions  # lista com 29 valores (física, radianos)
    control_mode = getattr(robot_config, "control_mode", "upper_body")

    # ── 1. Determina quais juntas o modo usa ─────────────────────────────────
    # Importação local para não criar dependência circular no topo do módulo
    try:
        from robot.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
        if control_mode == "upper_body":
            joint_index = list(G1_29_JointArmIndex)
        else:
            joint_index = list(G1_29_JointIndex)
    except ImportError:
        logger.warning(
            "[NeutralPosition] Não consegui importar G1_29_JointArmIndex. "
            "Usando os primeiros %d valores de default_positions.", action_dim
        )
        joint_index = None

    # ── 2. Extrai os valores físicos das juntas controladas ──────────────────
    if joint_index is not None:
        raw_values = [float(default_pos[motor.value]) for motor in joint_index]
    else:
        # Fallback: pega os primeiros action_dim valores
        raw_values = [float(v) for v in default_pos[:action_dim]]

    # Garante que o tamanho bate com action_dim
    if len(raw_values) != action_dim:
        logger.warning(
            "[NeutralPosition] Número de juntas (%d) != action_dim (%d). "
            "Ajustando com zeros.", len(raw_values), action_dim
        )
        if len(raw_values) < action_dim:
            raw_values += [0.0] * (action_dim - len(raw_values))
        else:
            raw_values = raw_values[:action_dim]

    neutral_physical = torch.tensor(raw_values, dtype=torch.float32, device=device)

    logger.info(
        "[NeutralPosition] Posição física (radianos): %s",
        [f"{v:.3f}" for v in raw_values]
    )

    # ── 3. Normaliza para o espaço da ação ──────────────────────────────────
    mean = action_stats["mean"].to(device=device, dtype=torch.float32)
    std  = action_stats["std"].to(device=device, dtype=torch.float32)

    # Evita divisão por zero em std muito pequeno (junta que nunca se move)
    std_safe = std.clamp(min=1e-6)

    neutral_normalized = (neutral_physical - mean) / std_safe

    logger.info(
        "[NeutralPosition] Posição normalizada (espaço ACT): %s",
        [f"{v:.3f}" for v in neutral_normalized.tolist()]
    )
    logger.info(
        "[NeutralPosition] Faixa: min=%.3f, max=%.3f",
        neutral_normalized.min().item(), neutral_normalized.max().item(),
    )

    # Aviso se o neutro estiver muito fora do centro — pode indicar erro de config
    outliers = (neutral_normalized.abs() > 3.0).sum().item()
    if outliers > 0:
        logger.warning(
            "[NeutralPosition] %d junta(s) com valor normalizado > 3σ. "
            "Verifique se default_positions corresponde ao mesmo espaço de ação do dataset.",
            outliers,
        )

    return neutral_normalized


def compute_neutral_from_dataset(
    dataset,
    action_dim: int,
    n_episodes: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Alternativa: calcula a posição neutra como a média dos primeiros frames
    de cada episódio no dataset, já no espaço normalizado.

    Útil quando você não tem o UnitreeG1Config disponível no momento do treino,
    ou quer validar que o neutro do robô coincide com o início dos episódios.

    Args:
        dataset: LeRobotDataset já carregado.
        action_dim: dimensão da ação.
        n_episodes: quantos episódios usar (None = todos).
        device: dispositivo de saída.

    Returns:
        Tensor [action_dim] — média das ações no primeiro frame de cada episódio,
        já no espaço normalizado (as ações no dataset estão em espaço físico;
        normalizamos com as stats do próprio dataset).
    """
    episodes = dataset.episodes if n_episodes is None else dataset.episodes[:n_episodes]
    first_frames = []

    for ep_idx in episodes:
        from_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
        item = dataset[from_idx]
        # Pega a OBSERVAÇÃO de estado (posição física das juntas no início)
        if "observation.state" in item:
            first_frames.append(item["observation.state"][:action_dim].float())

    if not first_frames:
        logger.warning("[NeutralPosition] Nenhum frame inicial encontrado. Usando zeros.")
        return torch.zeros(action_dim, device=device)

    neutral_physical = torch.stack(first_frames).mean(dim=0)  # [action_dim]

    logger.info(
        "[NeutralPosition] Neutro calculado dos primeiros frames (%d episódios): %s",
        len(first_frames), [f"{v:.3f}" for v in neutral_physical.tolist()]
    )

    # Normaliza com as stats do dataset
    stats = dataset.meta.stats
    if "observation.state" in stats:
        mean = torch.tensor(stats["observation.state"]["mean"][:action_dim], dtype=torch.float32)
        std  = torch.tensor(stats["observation.state"]["std"][:action_dim], dtype=torch.float32)
        std_safe = std.clamp(min=1e-6)
        neutral_normalized = (neutral_physical - mean) / std_safe
    else:
        logger.warning("[NeutralPosition] Stats de observation.state não encontradas. Usando valores físicos crus.")
        neutral_normalized = neutral_physical

    return neutral_normalized.to(device)