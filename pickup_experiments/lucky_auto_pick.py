#!/usr/bin/env python3
"""
Controlador AUTOMATICO de pick pro desafio luckyrobots/g1-manipulation-challenge.
Reusa as classes do run.py (walker + right_reacher ONNX) e dirige por uma maquina de
estados, sem teclado: estabiliza -> anda pra frente ate o cilindro entrar no alcance ->
reacher mira no cilindro -> fecha a mao -> levanta.

Rodar:
  source /tmp/lucky_env/bin/activate
  cd /tmp/g1-manipulation-challenge
  MUJOCO_GL=egl python auto_pick.py            # headless (so metricas)
  DISPLAY=:1 python auto_pick.py --view         # com viewer
"""
import sys, json, time, argparse
sys.path.insert(0, "/tmp/g1-manipulation-challenge")
import numpy as np
import mujoco
import run as R

ap = argparse.ArgumentParser()
ap.add_argument("--view", action="store_true", help="abre o viewer (precisa DISPLAY)")
ap.add_argument("--secs", type=float, default=30.0)
args = ap.parse_args()

cfg = json.load(open(R.SCRIPT_DIR / "model_config.json"))
jn = cfg["joint_names"]
model = mujoco.MjModel.from_xml_path(str(R.SCRIPT_DIR / "scene.xml"))
model.opt.timestep = 0.005
R.set_armature(model, jn)
data = mujoco.MjData(model)
# pose inicial (= run.py)
data.qpos[0] = -0.6; data.qpos[2] = 0.76; data.qpos[3:7] = [1, 0, 0, 0]
for n, v in cfg["default_joint_pos"].items():
    if n in jn: data.qpos[7 + jn.index(n)] = v
mujoco.mj_forward(model, data)

walker = R.ONNXPolicy(str(R.SCRIPT_DIR / "walker.onnx"))
croucher = R.ONNXPolicy(str(R.SCRIPT_DIR / "croucher.onnx"))
rotator = R.ONNXPolicy(str(R.SCRIPT_DIR / "rotator.onnx"))
reacher = R.ONNXPolicy(str(R.SCRIPT_DIR / "right_reacher.onnx"))
ctrl = R.G1Controller(model, data, walker, croucher, rotator, cfg, right_reacher=reacher)
for d_, p_ in [(99, walker), (101, croucher), (99, rotator), (36, reacher)]:
    p_(np.zeros((1, d_), np.float32))

bjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
BQ = model.jnt_qposadr[bjid]
PALM = ctrl.right_palm_site_id


def qinv(q, v):
    w, xyz = q[0], q[1:4]
    t = 2 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def cyl_pelvis():
    return qinv(data.qpos[3:7], data.qpos[BQ:BQ + 3] - data.qpos[:3])


def palm_pelvis():
    return qinv(data.qpos[3:7], data.site_xpos[PALM] - data.qpos[:3])


CLAMP_LO = np.array([-0.3, -0.6, -0.4]); CLAMP_HI = np.array([0.6, 0.3, 0.6])
phase = "stand"; pstep = 0; cstep = 0; decim = 4
target = ctrl.default_joint_pos.copy()
cup_z0 = float(data.qpos[BQ + 2])
log_t = 0


def control():
    """maquina de estados: ajusta ctrl.* conforme a fase. roda a 50Hz."""
    global phase, pstep
    pstep += 1
    cyl = cyl_pelvis(); palm = palm_pelvis()
    cup_h = float(data.qpos[BQ + 2])
    OFF = np.array([0.08, 0.13, 0.06])      # feedforward: palma -> centro/frente/altura do cilindro
    TGT_FWD = 0.30
    ctrl.lin_vel_x = ctrl.lin_vel_y = ctrl.ang_vel_z = 0.0   # default: parado (so walk anda)
    if phase == "stand":
        if pstep > 100:
            phase, _ = "walk", reset_p()
    elif phase == "walk":
        # anda RETO; strafe GENTIL so pra manter o cilindro centrado (~side 0)
        e_fwd = cyl[0] - TGT_FWD
        ctrl.lin_vel_x = 0.5 if cyl[0] > 0.34 else float(np.clip(e_fwd * 3.0, 0.0, 0.5))
        ctrl.lin_vel_y = float(np.clip(cyl[1] * 0.8, -0.12, 0.12))
        if abs(e_fwd) < 0.04:
            phase, _ = "settle", reset_p()
        if pstep > 900:
            phase, _ = "settle", reset_p()
    elif phase == "settle":
        if pstep == 1:
            ctrl.input_mode = "reach"; ctrl.reach_active = True
        # alvo ACIMA do cilindro (palma livre, sem empurrar)
        ctrl.reach_target[:] = np.clip(cyl + OFF + np.array([0, 0, 0.22]), CLAMP_LO, CLAMP_HI)
        if pstep > 200:
            phase, _ = "above", reset_p()
    elif phase == "above":
        # centraliza a palma ACIMA do cilindro (alto, livre)
        ctrl.reach_target[:] = np.clip(cyl + OFF + np.array([0, 0, 0.22]), CLAMP_LO, CLAMP_HI)
        if pstep > 150:
            phase, _ = "descend", reset_p()
    elif phase == "descend":
        # DESCE vertical sobre o cilindro (sem varrer lateralmente)
        frac = min(1.0, pstep / 150.0)
        up = 0.22 - frac * (0.22 - 0.06)
        ctrl.reach_target[:] = np.clip(cyl + OFF + np.array([0, 0, up]), CLAMP_LO, CLAMP_HI)
        if pstep > 160:
            phase, _ = "grab", reset_p()
    elif phase == "grab":
        ctrl.grip_closed = True
        ctrl.reach_target[:] = np.clip(cyl + OFF, CLAMP_LO, CLAMP_HI)
        if pstep > 120:
            phase, _ = "lift", reset_p()
    elif phase == "lift":
        ctrl.grip_closed = True
        ctrl.reach_target[2] = min(0.22, ctrl.reach_target[2] + 0.003)  # lift pequeno/estavel
        if pstep > 200:
            phase, _ = "hold", reset_p()
    elif phase == "hold":
        ctrl.grip_closed = True
    return cyl, palm, cup_h


def reset_p():
    global pstep
    pstep = 0
    print(f"  --> fase: {phase}")
    return None


def loop_body():
    global target, cstep, log_t
    if cstep % decim == 0:
        cyl, palm, cup_h = control()
        target = ctrl.step()
        if cstep % 50 == 0:
            rt = ctrl.reach_target
            print(f"[{phase:7s}] cyl_pel=[{cyl[0]:.2f},{cyl[1]:.2f},{cyl[2]:.2f}] "
                  f"palm_pel=[{palm[0]:.2f},{palm[1]:.2f},{palm[2]:.2f}] "
                  f"reach_tgt=[{rt[0]:.2f},{rt[1]:.2f},{rt[2]:.2f}] "
                  f"palm->cyl={np.linalg.norm(palm-cyl)*100:.0f}cm "
                  f"reacher_act|n|={np.linalg.norm(ctrl.last_arm_action):.2f} act={np.round(ctrl.last_arm_action,2)} "
                  f"grip={ctrl.grip_closed} cupz={cup_h:.3f}")
    ctrl.apply_pd_control(target)
    mujoco.mj_step(model, data)
    cstep += 1


print(f"[auto_pick] cilindro inicial z={cup_z0:.3f}")
nsteps = int(args.secs / 0.005)
if args.view:
    from mujoco import viewer
    with viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running() and cstep < nsteps:
            loop_body()
            if cstep % 4 == 0:
                v.sync()
            # pacing real-time
            w = time.time() - t0
            if cstep * 0.005 > w:
                time.sleep(cstep * 0.005 - w)
else:
    for _ in range(nsteps):
        loop_body()

cup_zf = float(data.qpos[BQ + 2])
print(f"\n[auto_pick] RESULTADO: cilindro z {cup_z0:.3f} -> {cup_zf:.3f} "
      f"(elevacao {(cup_zf-cup_z0)*100:+.1f}cm) | fase final={phase} | grip={ctrl.grip_closed}")
print(f"PEGOU E LEVANTOU? {'SIM ✓' if (cup_zf-cup_z0) > 0.05 else 'ainda nao'}")
