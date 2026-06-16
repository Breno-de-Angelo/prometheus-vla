#!/usr/bin/env python
"""Host de LOOP FECHADO em sim: o modelo vê a CÂMERA + STATE do MuJoCo reagindo às
próprias ações (não o dataset), controla o robô, e medimos se a mão alcança e levanta
o copo. Mede a degradação de loop fechado (erro composto) — ressalva: a câmera do sim é
sintética (gap visual sim-vs-real soma ao efeito).

Renderiza a head_camera com Renderer PRÓPRIO (o publish-subprocess do env falha headless)
e publica no schema do run_g1_server. ZMQ: 5555 PUB imagem | 6001/6002 PUB state (lido do
MuJoCo) | 6000/6003 PULL cmd (aplicado ao sim via DDS rt/lowcmd). A VLA não distingue.

Uso (notebook):  python closed_loop_sim_host.py --video /tmp/closed_loop.mp4
"""
import argparse, base64, json, time, os, sys
from pathlib import Path
import numpy as np
import cv2
import zmq

LOWCMD_PORT, LOWSTATE_PORT, HANDSTATE_PORT, HANDCMD_PORT, CAM_PORT = 6000, 6001, 6002, 6003, 5555
NUM_MOTORS, NUM_HAND = 35, 7
RIGHT_ARM_MOTOR_IDS = tuple(range(22, 29))
ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGER_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                 "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
                 "right_hand_middle_1_joint"]
kLowState, kRightHand, kLeftHand = "rt/lowstate", "rt/dex3/right/state", "rt/dex3/left/state"
_DUMMY_DEPTH = None


def _motor(q=0.0):
    return {"q": float(q), "dq": 0.0, "tau_est": 0.0, "temperature": 25.0}


def lowstate_msg(arm7):
    motors = [_motor() for _ in range(NUM_MOTORS)]
    for k, m in enumerate(RIGHT_ARM_MOTOR_IDS):
        motors[m] = _motor(arm7[k])
    return {"topic": kLowState, "data": {"motor_state": motors,
            "imu_state": {"quaternion": [1.0, 0, 0, 0], "gyroscope": [0]*3, "accelerometer": [0]*3, "rpy": [0]*3, "temperature": 25.0},
            "wireless_remote": base64.b64encode(bytes(40)).decode("ascii"), "mode_machine": 0}}


def handstate_msg(fingers7, side):
    qs = fingers7 if side == "right" else np.zeros(7)
    motors = [_motor(qs[i]) for i in range(NUM_HAND)]
    press = [{"pressure": [0.0]*12, "temperature": [25.0]*12} for _ in range(NUM_HAND)]
    return {"topic": kRightHand if side == "right" else kLeftHand,
            "data": {"side": side, "motor_state": motors, "press_sensor_state": press}}


def image_msg(rgb):
    global _DUMMY_DEPTH
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buf).decode("ascii") if ok else ""
    if _DUMMY_DEPTH is None:
        _, db = cv2.imencode(".png", np.zeros(rgb.shape[:2], np.uint16))
        _DUMMY_DEPTH = base64.b64encode(db).decode("ascii")
    t = time.time()
    return json.dumps({"images": {"head_camera": b64, "head_camera_depth": _DUMMY_DEPTH},
                       "timestamps": {"head_camera": t, "head_camera_depth": t}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--video", default="/tmp/closed_loop.mp4")
    ap.add_argument("--log", default="/tmp/closed_loop_metric.jsonl")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("G1_DDS_IFACE", "lo")
    here = Path(__file__).resolve().parent
    for p in (str(here), str(here.parent / "lerobot-ext/robot/unitree_g1")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import mujoco, yaml
    from env import make_env
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, HandCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.core.channel import ChannelPublisher

    # env headless SEM render do env (eu renderizo) -> sem conflito de contexto GL
    cfgp = here / "config.yaml"; orig = cfgp.read_text(); cfg = yaml.safe_load(orig)
    cfg["ENABLE_ONSCREEN"] = False; cfg["ENABLE_OFFSCREEN"] = False
    cfgp.write_text(yaml.safe_dump(cfg))
    try:
        env = make_env(cameras=[], publish_images=False)
    finally:
        cfgp.write_text(orig)
    de = env.sim_env; mjm, mjd = de.mj_model, de.mj_data

    def jadr(nm):
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, nm)
        return mjm.jnt_qposadr[jid] if jid >= 0 else None
    arm_adr = [jadr(n) for n in ARM_JOINTS]
    fin_adr = [jadr(n) for n in FINGER_JOINTS]
    hand_b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link")
    cup_b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "objeto_customizado")
    cup_z0 = float(mjd.xpos[cup_b][2]) if cup_b >= 0 else 0.0

    renderer = mujoco.Renderer(mjm, 480, 640)
    lowpub = ChannelPublisher("rt/lowcmd", hg_LowCmd); lowpub.Init()
    rhpub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_); rhpub.Init()
    cur_low = [None]; cur_rh = [None]
    n_sub = max(1, int(round((1.0/args.fps)/env.sim_dt)))

    def to_low(d):
        c = unitree_hg_msg_dds__LowCmd_(); data = d.get("data", d)
        c.mode_pr = data.get("mode_pr", 0); c.mode_machine = data.get("mode_machine", 0)
        for i, m in enumerate(data.get("motor_cmd", [])[:35]):
            mc = c.motor_cmd[i]; mc.mode = m.get("mode", 0); mc.q = m.get("q", 0.0); mc.dq = m.get("dq", 0.0)
            mc.kp = m.get("kp", 0.0); mc.kd = m.get("kd", 0.0); mc.tau = m.get("tau", 0.0)
        return c

    def to_hand(d):
        c = unitree_hg_msg_dds__HandCmd_(); data = d.get("data", d)
        for i, m in enumerate(data.get("motor_cmd", [])[:7]):
            mc = c.motor_cmd[i]; mc.mode = m.get("mode", 0); mc.q = m.get("q", 0.0); mc.dq = m.get("dq", 0.0)
            mc.kp = m.get("kp", 0.0); mc.kd = m.get("kd", 0.0); mc.tau = m.get("tau", 0.0)
        return c

    ctx = zmq.Context()
    campub = ctx.socket(zmq.PUB); campub.bind(f"tcp://{args.bind}:{CAM_PORT}")
    lowst = ctx.socket(zmq.PUB); lowst.bind(f"tcp://{args.bind}:{LOWSTATE_PORT}")
    handst = ctx.socket(zmq.PUB); handst.bind(f"tcp://{args.bind}:{HANDSTATE_PORT}")
    lowcmd = ctx.socket(zmq.PULL); lowcmd.bind(f"tcp://{args.bind}:{LOWCMD_PORT}")
    handcmd = ctx.socket(zmq.PULL); handcmd.bind(f"tcp://{args.bind}:{HANDCMD_PORT}")
    print(f"[closed-loop] env+ZMQ ok. cup_z0={cup_z0:.3f}, dist inicial mão-copo medida no 1º frame. "
          f"Esperando a VLA (--robot-ip 127.0.0.1)...", flush=True)

    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 480))
    logf = open(args.log, "w")
    period = 1.0/args.fps; frame = 0; n_cmd = 0; min_dist = 9.9; max_lift = 0.0
    t_start = time.time()
    try:
        while True:
            t0 = time.perf_counter()
            while True:
                try:
                    cur_low[0] = to_low(json.loads(lowcmd.recv_string(zmq.NOBLOCK))); n_cmd += 1
                except zmq.Again:
                    break
            while True:
                try:
                    cur_rh[0] = to_hand(json.loads(handcmd.recv_string(zmq.NOBLOCK)))
                except zmq.Again:
                    break
            for _ in range(n_sub):
                if cur_low[0] is not None:
                    lowpub.Write(cur_low[0])
                if cur_rh[0] is not None:
                    rhpub.Write(cur_rh[0])
                de.sim_step()
            # publica state do sim
            arm7 = [float(mjd.qpos[a]) if a is not None else 0.0 for a in arm_adr]
            fin7 = [float(mjd.qpos[a]) if a is not None else 0.0 for a in fin_adr]
            lowst.send_string(json.dumps(lowstate_msg(arm7)))
            handst.send_string(json.dumps(handstate_msg(fin7, "right")))
            handst.send_string(json.dumps(handstate_msg(fin7, "left")))
            # renderiza head_camera -> publica 5555 (a visão do modelo)
            renderer.update_scene(mjd, camera="head_camera")
            campub.send_string(image_msg(renderer.render()))
            # renderiza global_view -> vídeo 3ª pessoa
            renderer.update_scene(mjd, camera="global_view")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            # métrica
            if hand_b >= 0 and cup_b >= 0:
                dist = float(np.linalg.norm(mjd.xpos[hand_b] - mjd.xpos[cup_b]))
                lift = float(mjd.xpos[cup_b][2]) - cup_z0
                min_dist = min(min_dist, dist); max_lift = max(max_lift, lift)
                if frame % 5 == 0:
                    logf.write(json.dumps({"t": round(time.time()-t_start, 2), "frame": frame,
                                           "dist_cm": round(dist*100, 1), "lift_cm": round(lift*100, 1), "cmds": n_cmd}) + "\n")
                    logf.flush()
            if frame % 30 == 0:
                print(f"\r[closed-loop] frame {frame} cmds={n_cmd} dist_min={min_dist*100:.1f}cm lift_max={max_lift*100:.1f}cm  ", end="", flush=True)
            frame += 1
            sl = period - (time.perf_counter()-t0)
            if sl > 0:
                time.sleep(sl)
    except KeyboardInterrupt:
        print(f"\n[closed-loop] FIM. dist_min={min_dist*100:.1f}cm lift_max={max_lift*100:.1f}cm cmds={n_cmd}", flush=True)
    finally:
        logf.close()
        if writer:
            writer.release()
        try:
            renderer.close()
        except Exception:
            pass
        for s in (campub, lowst, handst, lowcmd, handcmd):
            s.close()
        ctx.term()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
