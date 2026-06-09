#!/usr/bin/env python3
"""Loop de inferência em tempo real do pi05 (right14) no Unitree G1 Dex3.

Variante do `inference_realtime_pi05d.py` para os modelos do pipeline RIGHT14
(`cup_pi05_right14_rgb_lf` / `cup_pi05_right14_depth_lf`): a política controla
SÓ o braço + mão DIREITA (14 dims), não o corpo todo (28).

Diferenças em relação ao script de 28 dims:
  1. STATE de entrada montado só com as 14 juntas da direita, na ordem exata do
     dataset right14 (braço direito 7 + mão direita 7) — não as 28 do corpo.
  2. AÇÃO de saída roteada SÓ para esses 14 motores da direita. A esquerda nunca
     entra no dicionário de ação → o modelo "não manda nada pra ela".
  3. Lado ESQUERDO fica MOLE (kp=0) via G1_LEFT_ARM_LIMP=1, setado antes do
     connect() — igual ao `--left-arm-limp` do record. Mesmo que o driver
     preencha juntas ausentes, kp=0 garante zero torque (sem tranco).
  4. Depth/tátil só são alimentados se o checkpoint os declarar em input_features
     (auto-detecção). O modelo RGB não recebe nenhum dos dois.

Pré-condições no ROBÔ (não nesta máquina):
  - run_g1_server.py rodando      (ponte ZMQ/DDS)
  - realsense_server.py rodando   (câmera ZMQ :5555)

Uso (depois dos serviços do robô no ar):
    python inference_realtime_pi05d_right14.py \
        --checkpoint train_output/cup_pi05_right14_rgb_lf/checkpoints/002500/pretrained_model \
        --robot-ip 10.9.8.73 \
        --task "Pick up the cup" \
        --fps 30 \
        --actions-per-chunk 50 \
        --dry-run            # tire isto só quando as ações do dry-run parecerem sãs
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

import safetensors.torch as _st  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.cameras.zmq.camera_zmq import ZMQCamera  # noqa: E402
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config  # noqa: E402

logger = logging.getLogger("pi05d_right14")

# Ordem EXATA das 14 juntas controladas, idêntica ao dataset G1_Dex3_right14_dataset
# (meta/info.json: 'action' e 'observation.state', ambos shape [14]). State e ação
# usam a mesma lista, na mesma ordem.
RIGHT14_FEATURES: list[str] = [
    "kRightShoulderPitch.q",
    "kRightShoulderRoll.q",
    "kRightShoulderYaw.q",
    "kRightElbow.q",
    "kRightWristRoll.q",
    "kRightWristPitch.q",
    "kRightWristYaw.q",
    "right_hand_thumb_0_joint.q",
    "right_hand_thumb_1_joint.q",
    "right_hand_thumb_2_joint.q",
    "right_hand_index_0_joint.q",
    "right_hand_index_1_joint.q",
    "right_hand_middle_0_joint.q",
    "right_hand_middle_1_joint.q",
]


def _policy_inputs(policy) -> set[str]:
    """Nomes das features de entrada que o checkpoint espera (p/ auto-detectar depth/tátil)."""
    feats = getattr(policy.config, "input_features", None) or {}
    try:
        return set(feats.keys())
    except AttributeError:
        return set(feats)


def build_observation_batch(
    robot: UnitreeG1Dex3,
    task: str,
    device: torch.device,
    wants_depth: bool,
    wants_pressure: bool,
    depth_camera: ZMQCamera | None = None,
) -> dict:
    """Monta o batch que a política espera, com STATE de 14 dims (só direita).

    - observation.state: as 14 juntas de RIGHT14_FEATURES, nessa ordem.
    - observation.images.head_camera: RGB do robô (CHW float [0,1]).
    - observation.images.head_camera_depth: só se o checkpoint pedir.
    - observation.{left,right}_hand_pressure: só se o checkpoint pedir.
    """
    obs = robot.get_observation()

    # STATE: só as 14 juntas da direita, na ordem do treino. Faltando alguma -> 0.0.
    missing = [n for n in RIGHT14_FEATURES if n not in obs]
    if missing:
        logger.warning("juntas ausentes na observação (usando 0.0): %s", missing)
    state_vec = [float(obs.get(n, 0.0)) for n in RIGHT14_FEATURES]
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

    def to_tensor(img: np.ndarray | None, fallback_hw=(480, 848)) -> torch.Tensor:
        # O preprocessor da pi05 redimensiona pra 224; aqui só normaliza CHW [0,1].
        if img is None:
            img = np.zeros((*fallback_hw, 3), dtype=np.uint8)
        if img.ndim == 2:
            img = img[:, :, None].repeat(3, axis=2)
        return torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    rgb_img = None
    for _k in ("cam_rgb_high", "head_camera", "cam_rgb_low", "rgb"):
        _v = obs.get(_k)
        if _v is not None:
            rgb_img = _v
            break
    if rgb_img is None:
        avail = [k for k in obs.keys() if any(s in k.lower() for s in ("cam", "rgb", "image"))]
        raise KeyError(f"nenhuma chave de RGB na obs; tentei cam_rgb_high/head_camera/rgb. Chaves de câmera: {avail}")

    batch: dict = {
        "observation.state": state_tensor,
        "observation.images.head_camera": to_tensor(rgb_img),
        "task": task,
    }

    if wants_depth:
        depth_img = depth_camera.async_read() if depth_camera is not None else None
        batch["observation.images.head_camera_depth"] = to_tensor(depth_img)

    if wants_pressure:
        left_p = obs.get("left_hand_pressure", [0.0] * 108)
        right_p = obs.get("right_hand_pressure", [0.0] * 108)
        batch["observation.left_hand_pressure"] = torch.from_numpy(
            np.asarray(left_p, dtype=np.float32)).to(device).unsqueeze(0)
        batch["observation.right_hand_pressure"] = torch.from_numpy(
            np.asarray(right_p, dtype=np.float32)).to(device).unsqueeze(0)

    return batch


def action_tensor_to_robot_action(action_vec: torch.Tensor) -> dict:
    """Roteia as 14 saídas (1as 14 dims) para os 14 motores da direita, por nome.

    NÃO inclui nenhuma junta da esquerda no dicionário → o robô não recebe comando
    pra elas (elas ficam moles via G1_LEFT_ARM_LIMP). A política pode devolver 14 ou
    32 dims (padding interno); pegamos só as 14 primeiras, que são as reais.
    """
    action = action_vec.detach().cpu().numpy().astype(float).reshape(-1).tolist()
    return {name: action[i] for i, name in enumerate(RIGHT14_FEATURES) if i < len(action)}


class GracefulKiller:
    def __init__(self):
        self.kill = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, *_):
        logger.warning("encerramento pedido; termina o chunk atual e para...")
        self.kill = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="diretório pretrained_model do checkpoint right14")
    parser.add_argument("--robot-ip", default="10.9.8.73")
    parser.add_argument("--task", default="Pick up the white cup",
                        help="DEVE bater com o task do treino (dataset right14: 'Pick up the white cup').")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Cadência do envio de ações. Baixe (ex: 10) pra mover mais devagar.")
    parser.add_argument("--actions-per-chunk", type=int, default=50,
                        help="Quantas ações executar de cada chunk previsto (chunk de treino=50; <=50 seguro).")
    parser.add_argument("--max-delta", type=float, default=0.02,
                        help="MOVIMENTO LENTO/SEGURO do BRAÇO: passo máx (rad/ciclo) das 7 juntas do braço. "
                             "Parte da posição medida e rampa devagar até o alvo. 0 = desliga (vai direto).")
    parser.add_argument("--hand-max-delta", type=float, default=0.10,
                        help="Passo máx (rad/ciclo) das 7 juntas da MÃO. Maior que o do braço pra fechar o "
                             "grip a tempo (dedos são seguros). 0 = sem clamp na mão (fecha direto).")
    parser.add_argument("--denoising-steps", type=int, default=0,
                        help="Passos de denoising do flow-matching (latência). 0 = usa o default do "
                             "checkpoint (num_inference_steps=10). Menor = mais rápido, menos preciso.")
    parser.add_argument("--hand-kp", type=float, default=0.0,
                        help="Sobrescreve o kp da mão direita (default do robô = 0.8, fraco). "
                             "Suba (ex: 2, 4) se a mão não fechar o grip. 0 = usa o default.")
    parser.add_argument("--control-mode", default="upper_body")
    parser.add_argument("--no-left-limp", action="store_true",
                        help="NÃO força a esquerda mole (kp=0). Por padrão a esquerda fica mole.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Loga as 14 ações previstas (com nome da junta) mas NÃO envia pro robô.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # force=True: alguns imports do lerobot já configuram o root logger, o que
    # tornaria este basicConfig um no-op e suprimiria nossos logs INFO (chunks,
    # ações do dry-run). force=True reconfigura e garante que apareçam.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logging.getLogger("pi05d_right14").setLevel(getattr(logging, args.log_level.upper()))

    # Lado esquerdo mole ANTES do connect() — igual ao --left-arm-limp do record.
    if not args.no_left_limp:
        os.environ["G1_LEFT_ARM_LIMP"] = "1"
        logger.info("braço/mão ESQUERDA mole (kp=0); a política só controla a direita.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("carregando política pi05 right14...")
    policy = PI05Policy.from_pretrained(args.checkpoint, strict=False)
    policy.to(device).eval()

    # Latência: desliga gradient checkpointing (otimização de TREINO, inútil em eval)
    # e, se pedido, reduz os passos de denoising do flow-matching.
    try:
        policy.config.gradient_checkpointing = False
    except Exception:
        pass
    if args.denoising_steps > 0:
        policy.config.num_inference_steps = args.denoising_steps
        logger.info("denoising steps = %d (default do checkpoint era 10)", args.denoising_steps)

    inputs = _policy_inputs(policy)
    wants_depth = "observation.images.head_camera_depth" in inputs
    wants_pressure = any("pressure" in k for k in inputs)

    # Injeção PI05-D (PointNet depth + tátil) SÓ se o checkpoint usa esses tokens.
    # O modelo RGB puro NÃO tem esses tensores -> injetar ativaria tokens que ele
    # nunca treinou (o bug do "missing=15" no load_pi05_d, que injeta sempre).
    if wants_depth or wants_pressure:
        from train.pi05_d_injector import inject_pi05_d
        inject_pi05_d(policy, device=device)
        sd = _st.load_file(str(Path(args.checkpoint) / "model.safetensors"), device=str(device))
        policy.load_state_dict(sd, strict=False)
        logger.info("injeção PI05-D aplicada (depth/tátil)")
    else:
        logger.info("modelo RGB puro — SEM injeção depth/tátil")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint
    )
    logger.info("checkpoint espera depth=%s tátil=%s (inputs=%s)", wants_depth, wants_pressure, sorted(inputs))
    logger.info("política pronta")

    robot_cfg = UnitreeG1Dex3Config(robot_ip=args.robot_ip, control_mode=args.control_mode, is_simulation=False)
    if args.hand_kp > 0:
        robot_cfg.hand_kp = args.hand_kp  # sobe o torque da mão (default 0.8 é fraco)
        logger.info("hand_kp sobrescrito = %.2f (default era 0.8)", args.hand_kp)
    robot = UnitreeG1Dex3(robot_cfg)
    robot.connect()
    logger.info(f"robô conectado em {args.robot_ip}")

    depth_camera = None
    if wants_depth:
        depth_camera = ZMQCamera(ZMQCameraConfig(
            server_address=args.robot_ip, port=5555,
            camera_name="head_camera_depth", width=848, height=480,
        ))
        depth_camera.connect()
        logger.info("câmera de depth conectada em %s:5555/head_camera_depth", args.robot_ip)

    killer = GracefulKiller()
    step_period = 1.0 / args.fps
    chunk_counter = 0

    # MOVIMENTO LENTO: seed do clamp com a posição MEDIDA atual da direita, pra
    # o 1º comando não saltar da pose atual pro alvo do modelo. A cada ciclo, cada
    # junta anda no máximo args.max_delta rad em direção ao alvo -> rampa devagar.
    obs0 = robot.get_observation()
    last_cmd = {n: float(obs0.get(n, 0.0)) for n in RIGHT14_FEATURES}

    def _clamp_slow(target: dict) -> dict:
        # Clamp por GRUPO: braço (lento/seguro) vs mão (mais rápido, pra fechar o grip).
        out = {}
        for k, v in target.items():
            md = args.hand_max_delta if "hand" in k else args.max_delta
            prev = last_cmd.get(k, v)
            if md <= 0:
                out[k] = v  # sem clamp -> vai direto pro alvo
            else:
                step = max(-md, min(md, v - prev))
                out[k] = prev + step
        return out

    if args.dry_run:
        logger.info("=== DRY-RUN: nenhuma ação será enviada ao robô ===")
    logger.info("movimento: braço max_delta=%.3f | mão max_delta=%.3f rad/ciclo @ %.0f fps",
                args.max_delta, args.hand_max_delta, args.fps)

    try:
        while not killer.kill:
            t_obs_start = time.perf_counter()
            batch = build_observation_batch(
                robot, args.task, device, wants_depth, wants_pressure, depth_camera)
            batch = preprocessor(batch)
            with torch.no_grad():
                action_chunk = policy.predict_action_chunk(batch)  # (1, chunk, action_dim)
            t_inf = time.perf_counter() - t_obs_start
            logger.info("chunk %d: previsto em %.0fms (shape %s)",
                        chunk_counter, t_inf * 1000, tuple(action_chunk.shape))

            steps_to_run = min(args.actions_per_chunk, action_chunk.shape[1])
            for i in range(steps_to_run):
                if killer.kill:
                    break
                loop_start = time.perf_counter()
                action_norm = action_chunk[:, i, :]
                action_out = postprocessor(action_norm).squeeze(0)
                target = action_tensor_to_robot_action(action_out)
                act_dict = _clamp_slow(target)           # rampa devagar até o alvo
                last_cmd = dict(act_dict)                 # próximo passo parte daqui
                if i == 0:
                    # loga a 1ª ação de cada chunk — no dry-run E no live (pra você
                    # ver no terminal o que o robô recebe enquanto se move).
                    arm = {k: round(v, 3) for k, v in act_dict.items() if "hand" not in k}
                    hand = {k: round(v, 3) for k, v in act_dict.items() if "hand" in k}
                    tag = "dry-run" if args.dry_run else "LIVE"
                    logger.info("%s chunk %d 1ª ação (clamp):\n  braço: %s\n  mão:   %s",
                                tag, chunk_counter, arm, hand)
                if not args.dry_run:
                    robot.send_action(act_dict)
                sleep_for = step_period - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            chunk_counter += 1
    finally:
        logger.info("desconectando")
        if depth_camera is not None:
            try:
                depth_camera.disconnect()
            except Exception as e:
                logger.warning(f"depth disconnect: {e}")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning(f"robot disconnect: {e}")


if __name__ == "__main__":
    main()
