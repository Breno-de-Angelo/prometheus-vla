#!/usr/bin/env python
"""Inferência pi05 RGB-only com Real-Time Chunking (RTC) no Unitree G1 + Dex3.

Loop de 2 threads (padrão lerobot/examples/rtc/eval_with_real_robot.py):
  - get_actions:    gera chunks com predict_action_chunk(inference_delay, prev_chunk_left_over)
  - actor_control:  executa 1 ação/tick via ActionQueue e manda pro robô (DDS real ou ZMQ sim)

RTC é NATIVO no pi05 (config.rtc_config); NÃO usa o async_inference (o policy_server não
repassa o prev_chunk_left_over, então RTC ficaria inerte lá). É só pra modelo RGB-only
(sem depth/tátil) — modelos pi05-D precisam do injector e NÃO devem usar este script.

Uso:
  CUDA_VISIBLE_DEVICES=1 conda run -n ms3 python init_lerobot_inference_lf.py \\
      --checkpoint=/.../pretrained_model --task="Pick up the cup" \\
      --fps=30 --rtc-execution-horizon=10 [--sim] [--robot-ip=<ip>] [--debug]

  RTC OFF (baseline, executa chunk inteiro sem guidance): --rtc-enabled=false
"""
import argparse
import logging
import math
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import torch

# permite importar robot.unitree_g1 (igual init_lerobot_inference_v2.py)
_CUR = Path(__file__).resolve().parent
if str(_CUR) not in sys.path:
    sys.path.insert(0, str(_CUR))

from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.rtc.action_queue import ActionQueue  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.latency_tracker import LatencyTracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("infer_lf")

# Convenção 'cup' 28-dim (MESMA ordem do dataset/treino) — copiada de init_lerobot_inference_v2.py.
# Mão DIR é assimétrica (thumb->index->middle); a esquerda é thumb->middle->index.
JOINT_NAMES = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q", "kLeftElbow.q",
    "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q", "kRightElbow.q",
    "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
    "left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q", "left_hand_thumb_2_joint.q",
    "left_hand_middle_0_joint.q", "left_hand_middle_1_joint.q",
    "left_hand_index_0_joint.q", "left_hand_index_1_joint.q",
    "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q", "right_hand_thumb_2_joint.q",
    "right_hand_index_0_joint.q", "right_hand_index_1_joint.q",
    "right_hand_middle_0_joint.q", "right_hand_middle_1_joint.q",
]


class RobotIO:
    """Lock em volta do robô — get_observation/send_action são chamados por threads distintas."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action_dict):
        with self.lock:
            self.robot.send_action(action_dict)


def build_batch(obs, device, img_key):
    """Batch RGB-only do pi05: observation.state [1,28] + a imagem sob a key que a policy espera (img_key) em [0,1]."""
    state = torch.tensor([float(obs.get(n, 0.0)) for n in JOINT_NAMES], dtype=torch.float32)
    img = obs["head_camera"]                                       # HWC, uint8 (camera publicada pelo sim/robo)
    img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0  # CHW [0,1]
    return {
        "observation.state": state.unsqueeze(0).to(device),
        img_key: img_t.unsqueeze(0).to(device),                    # ego_view (GR00T) ou head_camera, conforme o config
    }


def get_actions(policy, preprocessor, postprocessor, robot, queue, fps, task,
                threshold, rtc_enabled, device, stop, img_key):
    tracker = LatencyTracker()
    tpc = 1.0 / fps
    if not rtc_enabled:
        threshold = 0  # sem RTC: re-infere só quando a fila zera
    while not stop.is_set():
        if queue.qsize() <= threshold:
            t0 = time.perf_counter()
            idx_before = queue.get_action_index()
            prev = queue.get_left_over()                       # cauda não-executada (ações ORIGINAIS) ou None
            inference_delay = math.ceil(tracker.max() / tpc)

            obs = robot.get_observation()
            if not obs:
                time.sleep(0.005)
                continue
            batch = build_batch(obs, device, img_key)
            batch["task"] = [task]                             # task é LISTA, não string
            batch = preprocessor(batch)

            # SEM torch.inference_mode/no_grad aqui: o RTC usa autograd.grad pro guidance.
            actions = policy.predict_action_chunk(
                batch, inference_delay=inference_delay, prev_chunk_left_over=prev
            )
            original = actions.squeeze(0).clone()              # ANTES do postprocess (espaço interno p/ RTC)
            processed = postprocessor(actions).squeeze(0)      # desnormalizado p/ o robô

            std = float(original.float().std())                # monitor de colapso QUANTILES
            if std < 1e-3:
                log.warning("[VAR] std do chunk ~0 (%.2e) — possível colapso de normalização", std)

            new_delay = math.ceil((time.perf_counter() - t0) / tpc)
            tracker.add(time.perf_counter() - t0)
            queue.merge(original, processed, new_delay, idx_before)
        else:
            time.sleep(0.05)


def actor_control(robot, queue, fps, stop, debug):
    interval = 1.0 / fps
    n = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        action = queue.get()
        if action is not None:
            a = action.cpu().numpy()
            action_dict = {JOINT_NAMES[i]: float(a[i]) for i in range(len(JOINT_NAMES))}
            robot.send_action(action_dict)                     # send_action faz low-pass (alpha=0.1) + clip por junta
            n += 1
            if debug:
                print("\r🤖 " + " ".join(f"{v:.2f}" for v in a), end="", flush=True)
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, interval - dt - 0.001))
    log.info("[actor] %d ações executadas", n)


def force_cup_home(robot, n=5, dt=0.05):
    """Reposiciona o copo no home da mesa. Só escreve qpos/qvel; o mj_step do sim propaga."""
    de = robot.sim_env.simulator.sim_env
    cidx, vidx = de.cup_qpos_adr, de.cup_dof_adr
    for _ in range(n):
        de.mj_data.qpos[cidx:cidx + 3] = [0.329, -0.081, 0.836]
        de.mj_data.qpos[cidx + 3:cidx + 7] = [1.0, 0.0, 0.0, 0.0]
        de.mj_data.qvel[vidx:vidx + 6] = np.zeros(6)
        time.sleep(dt)


def record_camera_zmq(out_path, fps, stop_event, port=5555, cam="head_camera"):
    """Grava MP4 da câmera publicada por ZMQ na porta 5555."""
    import base64
    import json
    import cv2
    import zmq
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 500)
    writer = None
    try:
        while not stop_event.is_set():
            try:
                data = json.loads(sub.recv_string())
            except zmq.Again:
                continue
            b64 = data.get("images", {}).get(cam)
            if not b64:
                continue
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()
        sub.close()
        ctx.term()


def main():
    ap = argparse.ArgumentParser(description="Inferência pi05 RGB-only + RTC no G1+Dex3")
    ap.add_argument("--checkpoint", required=True, help="dir pretrained_model do pi05 (RGB-only)")
    ap.add_argument("--task", default="Pick up the cup")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="IP da máquina onde o sim/robô roda (no modo distribuído, o IP do host do sim)")
    ap.add_argument("--sim", action="store_true", help="modo simulação (MuJoCo/Isaac via ZMQ)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rtc-enabled", type=lambda s: str(s).lower() != "false", default=True)
    ap.add_argument("--rtc-execution-horizon", type=int, default=10)
    ap.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)  # ótimo p/ 10 denoise steps
    ap.add_argument("--rtc-schedule", default="EXP", choices=["EXP", "LINEAR", "ONES", "ZEROS"])
    ap.add_argument("--queue-threshold", type=int, default=30,
                    help="re-infere quando a fila <= isso (deve ser > inference_delay + execution_horizon)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) policy pi05 + RTC (segue o padrão de examples/rtc/eval_with_real_robot.py)
    log.info("Carregando pi05 de %s", args.checkpoint)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if getattr(config, "type", "pi05") in ("pi05", "pi0"):
        config.compile_model = False
    policy = get_policy_class(config.type).from_pretrained(args.checkpoint, config=config)
    policy.config.rtc_config = RTCConfig(
        enabled=args.rtc_enabled,
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[args.rtc_schedule],
    )
    policy.init_rtc_processor()
    assert policy.name in ("pi05", "pi0", "smolvla"), f"RTC não suporta {policy.name}"
    policy = policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.checkpoint,
        dataset_stats=None,                                    # usa os stats embutidos no checkpoint (congelados do pretrain)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    log.info("RTC %s | horizon=%d guidance=%.1f sched=%s fps=%.0f",
             "ON" if args.rtc_enabled else "OFF", args.rtc_execution_horizon,
             args.rtc_max_guidance_weight, args.rtc_schedule, args.fps)

    # 2) robô G1+Dex3 (mesma classe do init_lerobot_inference_v2.py; sim e real pelo flag)
    robot = UnitreeG1Dex3(UnitreeG1Dex3Config(
        robot_ip=args.robot_ip, control_mode="upper_body", is_simulation=args.sim,
    ))
    robot.connect()
    io = RobotIO(robot)
    log.info("Robô conectado (sim=%s)", args.sim)

    if args.sim:
        force_cup_home(robot)

    # key da imagem que a policy espera (ex.: ego_view ou head_camera)
    img_key = next((k for k in config.input_features if "image" in k.lower()),
                   "observation.images.head_camera")
    log.info("Image key da policy: %s", img_key)

    # 3) RTC: ActionQueue + 2 threads
    queue = ActionQueue(policy.config.rtc_config)
    stop = Event()
    ta = Thread(target=get_actions, name="GetActions", daemon=True, args=(
        policy, preprocessor, postprocessor, io, queue, args.fps, args.task,
        args.queue_threshold, args.rtc_enabled, device, stop, img_key))
    tb = Thread(target=actor_control, name="Actor", daemon=True, args=(
        io, queue, args.fps, stop, args.debug))
    ta.start()
    tb.start()

    rec_stop, trec, out_mp4 = None, None, None
    if args.sim:
        rec_stop = Event()
        out_mp4 = f"rollout_grasp_{time.strftime('%H%M%S')}.mp4"
        trec = Thread(target=record_camera_zmq, name="Recorder", daemon=True,
                      args=(out_mp4, args.fps, rec_stop))
        trec.start()
        log.info("🎥 gravando head_camera (ZMQ) -> %s", out_mp4)

    log.info("🚀 inferência ativa (Ctrl-C p/ parar)")
    try:
        while True:
            time.sleep(2.0)
            log.info("[fila] %d", queue.qsize())
    except KeyboardInterrupt:
        log.info("parando...")
    finally:
        stop.set()
        if rec_stop is not None:                   # fecha o writer antes do disconnect
            rec_stop.set()
            trec.join(timeout=4.0)
            log.info("🎥 vídeo salvo -> %s", out_mp4)
        ta.join(timeout=2.0)
        tb.join(timeout=2.0)
        robot.disconnect()
        log.info("robô desconectado.")


if __name__ == "__main__":
    main()
