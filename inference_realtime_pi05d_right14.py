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

# Pose de mão ABERTA (= pose de início dos episódios do dataset: todos os dedos
# em 0 rad; fechar = index/middle positivos, thumb_1/2 negativos).
OPEN_HAND_POSE: dict[str, float] = {n: 0.0 for n in RIGHT14_FEATURES if "hand" in n}


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
) -> tuple[dict, dict]:
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

    return batch, obs


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
    parser.add_argument("--depth-fx", type=float, default=None,
                        help="intrínseco fx do stream de DEPTH (obrigatório se o checkpoint usa depth)")
    parser.add_argument("--depth-fy", type=float, default=None)
    parser.add_argument("--depth-cx", type=float, default=None)
    parser.add_argument("--depth-cy", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="multiplicador depth→metros (PNG16 em mm → 0.001); obrigatório se usa depth")
    parser.add_argument("--control-mode", default="upper_body")
    parser.add_argument("--no-left-limp", action="store_true",
                        help="NÃO força a esquerda mole (kp=0). Por padrão a esquerda fica mole.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Loga as 14 ações previstas (com nome da junta) mas NÃO envia pro robô.")
    parser.add_argument("--rtc", action="store_true",
                        help="Real-Time Chunking (PI/lerobot): prediz o próximo chunk numa thread "
                             "ENQUANTO executa o atual, com inpainting na emenda — elimina os "
                             "degraus do loop síncrono. (ignorado no --dry-run)")
    parser.add_argument("--rtc-execution-horizon", type=int, default=20,
                        help="passos do chunk anterior tratados como prefixo fixo no inpainting")
    parser.add_argument("--rtc-max-guidance", type=float, default=1.0)
    parser.add_argument("--rtc-refill", type=int, default=30,
                        help="pede chunk novo quando a fila tem <= N ações (> horizon + delay)")
    parser.add_argument("--live", action="store_true",
                        help="publica telemetria (:5557) + mapa de ATENÇÃO da VLA pro dashboard "
                             "OmniView (rode tools/live_omniview.py na MESMA máquina e abra :8013)")
    parser.add_argument("--open-hand-s", type=float, default=2.0,
                        help="soft start: rampa a mão direita até a pose ABERTA em N segundos antes do "
                             "loop (episódios de treino começam de mão aberta; partir de mão fechada "
                             "joga o modelo no regime 'já agarrando'). 0 = desliga.")
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

    # Log em ARQUIVO além do terminal (pedido 2026-06-10): cada run de inferência
    # ganha um arquivo próprio com timestamp, com tudo que aparece no terminal.
    log_dir = REPO_ROOT / "train" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"infer_right14_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("log desta run também em: %s", log_path)

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
        # Auditoria FASE 3: intrínsecos e depth_scale agora são obrigatórios
        # (sem default silencioso). Os mesmos valores usados no TREINO do checkpoint.
        intr = None
        if None not in (args.depth_fx, args.depth_fy, args.depth_cx, args.depth_cy):
            intr = {"fx": args.depth_fx, "fy": args.depth_fy,
                    "cx": args.depth_cx, "cy": args.depth_cy}
        inject_pi05_d(policy, device=device, camera_intrinsics=intr,
                      depth_scale=args.depth_scale)
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

    # --- OmniView live: telemetria + mapa de atenção da VLA (--live) ---
    live_pub, attn_rec = None, None
    if args.live:
        import base64

        from tools.live_bridge import TelemetryPublisher
        from train.attn_recorder import AttnRecorder, overlay_heatmap

        live_pub = TelemetryPublisher()
        attn_rec = AttnRecorder().install()
        logger.info("OmniView live: telemetria em :5557 + atenção da VLA "
                    "(rode tools/live_omniview.py e abra :8013)")

        def _publish_attn(rgb_np):
            heat = attn_rec.heatmap()
            if heat is None or rgb_np is None:
                return
            try:
                import cv2 as _cv2
                ov = overlay_heatmap(rgb_np, heat)
                ok, buf = _cv2.imencode(".jpg", ov, [int(_cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    live_pub.publish_extra({
                        "attn_jpg": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
                    })
            except Exception as e:
                logger.debug("attn overlay falhou: %s", e)

    # MOVIMENTO LENTO: seed do clamp com a posição MEDIDA atual da direita, pra
    # o 1º comando não saltar da pose atual pro alvo do modelo. A cada ciclo, cada
    # junta anda no máximo args.max_delta rad em direção ao alvo -> rampa devagar.
    obs0 = robot.get_observation()
    last_cmd = {n: float(obs0.get(n, 0.0)) for n in RIGHT14_FEATURES}

    # SOFT START DA MÃO: rampa as 7 juntas da mão direita da pose medida até a
    # pose ABERTA antes do loop da política. Motivo: os episódios de treino
    # SEMPRE começam de mão aberta; se a run parte com a mão fechada (resto de
    # run anterior), o modelo vê um state que só existe na fase de grasp e fica
    # preso de mão fechada. O braço NÃO é tocado (parte da pose em que estiver).
    if args.open_hand_s > 0 and not args.dry_run:
        hand_start = {k: last_cmd[k] for k in OPEN_HAND_POSE}
        n_steps = max(1, int(args.open_hand_s * args.fps))
        logger.info("soft start: abrindo a mão direita em %.1fs (%d passos)...",
                    args.open_hand_s, n_steps)
        for i in range(1, n_steps + 1):
            a = i / n_steps
            robot.send_action({k: (1.0 - a) * hand_start[k] + a * tgt
                               for k, tgt in OPEN_HAND_POSE.items()})
            time.sleep(1.0 / args.fps)
        last_cmd.update(OPEN_HAND_POSE)

        # Espera a mão CONVERGIR de verdade (o streamer de 100Hz faz clip de
        # velocidade e pode ainda estar a caminho quando a rampa de comando acaba;
        # sem isso a política assume com a mão meio aberta e fecha de novo).
        SETTLE_TOL = 0.15   # rad
        deadline = time.time() + 2.0
        residual = {}
        while time.time() < deadline:
            obs_h = robot.get_observation()
            residual = {k: float(obs_h.get(k, 0.0)) for k in OPEN_HAND_POSE}
            if all(abs(v) < SETTLE_TOL for v in residual.values()):
                break
            time.sleep(0.1)
        laggards = {k.split("right_hand_")[-1]: round(v, 3)
                    for k, v in residual.items() if abs(v) >= SETTLE_TOL}
        if laggards:
            logger.warning("soft start: mão NÃO convergiu pra 0 em %s — juntas fora "
                           "(medido, rad): %s — checar kp/atrito/obstrução", "2.0s extra", laggards)
        else:
            logger.info("soft start: mão aberta e CONFIRMADA pela medição (|q| < %.2f rad).",
                        SETTLE_TOL)

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

    # ─────────────────── RTC: predição assíncrona + inpainting ───────────────────
    def _run_rtc_loop():
        """Real-Time Chunking (Black et al. 2025; impl. lerobot/policies/rtc).

        Thread de INFERÊNCIA: quando a fila cai abaixo de --rtc-refill, prevê um
        chunk novo condicionado ao resto do anterior (prev_chunk_left_over +
        inference_delay) e faz merge na ActionQueue. Thread ATUADORA (esta):
        consome a fila a --fps sem nunca parar → sem platôs nem saltos de replan.
        """
        import math
        from collections import deque
        from threading import Lock, Thread

        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.action_queue import ActionQueue
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        rtc_cfg = RTCConfig(
            enabled=True,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        policy.config.rtc_config = rtc_cfg
        policy.init_rtc_processor()
        queue = ActionQueue(rtc_cfg)
        # janela móvel (não máximo histórico): o chunk 0 leva ~900ms de warmup do
        # CUDA e congelaria a estimativa de delay lá em cima pra sempre.
        lat_win = deque(maxlen=5)
        robot_lock = Lock()
        time_per_step = 1.0 / args.fps
        refill = max(args.rtc_refill, args.rtc_execution_horizon + 2)
        latest_obs: dict = {}
        logger.info("RTC ON: execution_horizon=%d guidance=%.1f refill<=%d fps=%.0f",
                    args.rtc_execution_horizon, args.rtc_max_guidance, refill, args.fps)

        def _infer_loop():
            n = 0
            while not killer.kill:
                if queue.qsize() > refill:
                    time.sleep(0.02)
                    continue
                try:
                    t0 = time.perf_counter()
                    idx_before = queue.get_action_index()
                    left_over = queue.get_left_over()
                    delay = math.ceil(max(lat_win, default=0.0) / time_per_step)
                    with robot_lock:
                        batch, raw_obs = build_observation_batch(
                            robot, args.task, device, wants_depth, wants_pressure, depth_camera)
                    latest_obs.update(raw_obs)
                    rgb_np = None
                    if attn_rec is not None:
                        try:
                            t_img = batch["observation.images.head_camera"]
                            t_img = t_img[0] if t_img.dim() == 4 else t_img
                            rgb_np = (t_img.permute(1, 2, 0).float().cpu().numpy() * 255.0
                                      ).clip(0, 255).astype(np.uint8)
                        except Exception:
                            pass
                    b = preprocessor(batch)
                    with torch.no_grad():
                        if attn_rec is not None:
                            with attn_rec:
                                chunk = policy.predict_action_chunk(
                                    b, inference_delay=delay, prev_chunk_left_over=left_over)
                            _publish_attn(rgb_np)
                        else:
                            chunk = policy.predict_action_chunk(
                                b, inference_delay=delay, prev_chunk_left_over=left_over)
                    original = chunk.squeeze(0).clone()          # normalizado: p/ leftover do RTC
                    processed = postprocessor(chunk).squeeze(0)  # físico: p/ executar
                    dt = time.perf_counter() - t0
                    lat_win.append(dt)
                    queue.merge(original, processed, math.ceil(dt / time_per_step), idx_before)
                    logger.info("RTC chunk %d: %.0fms delay=%d fila=%d", n, dt * 1000, delay, queue.qsize())
                    n += 1
                except Exception as e:
                    logger.error("RTC infer falhou: %s", e)
                    time.sleep(0.2)

        th = Thread(target=_infer_loop, daemon=True)
        th.start()

        frame = 0
        while not killer.kill:
            t0 = time.perf_counter()
            a = queue.get()
            if a is not None:
                target = action_tensor_to_robot_action(a.cpu())
                act_dict = _clamp_slow(target)
                last_cmd.update(act_dict)
                with robot_lock:
                    robot.send_action(act_dict)
                if live_pub is not None:
                    _ls = getattr(robot, "_left_hand_state", None)
                    _rs = getattr(robot, "_right_hand_state", None)
                    live_pub.publish(latest_obs, act_dict,
                                     getattr(_ls, "pressure", None), getattr(_rs, "pressure", None),
                                     frame=frame, t=time.time(), robot_phase="unlocked")
                if frame % 150 == 0:
                    arm = {k: round(v, 3) for k, v in act_dict.items() if "hand" not in k}
                    hand = {k: round(v, 3) for k, v in act_dict.items() if "hand" in k}
                    logger.info("RTC frame %d:\n  braço: %s\n  mão:   %s", frame, arm, hand)
                frame += 1
            sleep_for = time_per_step - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
        th.join(timeout=2.0)

    if args.dry_run:
        logger.info("=== DRY-RUN: nenhuma ação será enviada ao robô ===")
    logger.info("movimento: braço max_delta=%.3f | mão max_delta=%.3f rad/ciclo @ %.0f fps",
                args.max_delta, args.hand_max_delta, args.fps)

    try:
        if args.rtc and not args.dry_run:
            _run_rtc_loop()
            killer.kill = True  # encerrado: não cai no loop síncrono abaixo
        while not killer.kill:
            t_obs_start = time.perf_counter()
            batch, raw_obs = build_observation_batch(
                robot, args.task, device, wants_depth, wants_pressure, depth_camera)

            # frame RGB cru (pré-processor) p/ desenhar o overlay de atenção
            rgb_np = None
            if attn_rec is not None:
                try:
                    t_img = batch["observation.images.head_camera"]
                    t_img = t_img[0] if t_img.dim() == 4 else t_img
                    rgb_np = (t_img.permute(1, 2, 0).float().cpu().numpy() * 255.0
                              ).clip(0, 255).astype(np.uint8)
                except Exception:
                    pass

            batch = preprocessor(batch)
            with torch.no_grad():
                if attn_rec is not None:
                    with attn_rec:
                        action_chunk = policy.predict_action_chunk(batch)  # (1, chunk, action_dim)
                    _publish_attn(rgb_np)
                else:
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
                if live_pub is not None:
                    # rampas/trajetória/tátil do dashboard (custo ~nulo: slot único)
                    _ls = getattr(robot, "_left_hand_state", None)
                    _rs = getattr(robot, "_right_hand_state", None)
                    live_pub.publish(
                        raw_obs, act_dict,
                        getattr(_ls, "pressure", None), getattr(_rs, "pressure", None),
                        frame=chunk_counter * args.actions_per_chunk + i,
                        t=time.time(), robot_phase="unlocked",
                    )
                sleep_for = step_period - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            chunk_counter += 1
    finally:
        logger.info("desconectando")
        if live_pub is not None:
            try:
                live_pub.close()
            except Exception:
                pass
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
