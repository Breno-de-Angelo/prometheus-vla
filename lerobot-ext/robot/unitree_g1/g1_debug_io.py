"""Debug I/O logger do robô G1 (opt-in via env G1_DEBUG_IO=<arquivo.jsonl>).

Grava, a cada frame, TUDO que é ENVIADO (send_action) e RECEBIDO (get_observation):
  - SEND: braços (q/kp), mão esq e dir (q/kp/kd/tau/mode por motor)
  - RECV: estado das juntas do braço, q das mãos, e PRESSÃO (tátil) esq/dir

Uso: o init seta G1_DEBUG_IO e embrulha send_action/get_observation. Não quebra o controle
(tudo em try/except). Para inspecionar: analyze_debug_io.py <arquivo>.
"""
import json
import os
import time

_F = {"fh": None, "t0": None, "path": None}


def _enabled():
    return bool(os.environ.get("G1_DEBUG_IO"))


def _fh():
    if not _enabled():
        return None
    p = os.environ["G1_DEBUG_IO"]
    if _F["fh"] is None or _F["path"] != p:
        try:
            _F["fh"] = open(p, "a")
            _F["path"] = p
            _F["t0"] = time.time()
        except Exception:
            return None
    return _F["fh"]


def _write(rec):
    fh = _fh()
    if fh is None:
        return
    try:
        rec["t"] = round(time.time() - (_F["t0"] or time.time()), 4)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
    except Exception:
        pass


def log_send(robot):
    """Loga o que ACABOU de ser publicado (lê as msgs do próprio robô)."""
    if not _enabled():
        return
    try:
        from .g1_utils import Dex3_1_Left_JointIndex, Dex3_1_Right_JointIndex, G1_29_JointArmIndex
        rec = {"io": "send"}
        # braços (q/kp) do LowCmd
        msg = getattr(robot, "msg", None)
        if msg is not None:
            rec["arm"] = {int(m.value): [round(float(msg.motor_cmd[m.value].q), 4),
                                          round(float(msg.motor_cmd[m.value].kp), 2)]
                          for m in G1_29_JointArmIndex}
        # mãos (q/kp/kd/tau/mode por motor)
        lh = getattr(robot, "_left_hand_msg", None)
        rh = getattr(robot, "_right_hand_msg", None)
        def hand(msgh, idxs):
            return [[round(float(msgh.motor_cmd[j].q), 4),
                     round(float(msgh.motor_cmd[j].kp), 3),
                     round(float(msgh.motor_cmd[j].kd), 3),
                     round(float(msgh.motor_cmd[j].tau), 4),
                     int(msgh.motor_cmd[j].mode)] for j in idxs]
        if lh is not None:
            rec["left_hand"] = hand(lh, list(Dex3_1_Left_JointIndex))   # [q,kp,kd,tau,mode]*7
        if rh is not None:
            rec["right_hand"] = hand(rh, list(Dex3_1_Right_JointIndex))
        _write(rec)
    except Exception as e:
        _write({"io": "send_err", "err": repr(e)})


def log_recv(robot, obs=None):
    """Loga o que foi RECEBIDO do robô (estado + pressão)."""
    if not _enabled():
        return
    try:
        from .g1_utils import G1_29_JointArmIndex, Dex3_Num_Motors
        rec = {"io": "recv"}
        ls = getattr(robot, "_lowstate", None)
        if ls is not None:
            rec["arm_state_q"] = {int(m.value): round(float(ls.motor_state[m.value].q), 4)
                                  for m in G1_29_JointArmIndex}
        lhs = getattr(robot, "_left_hand_state", None)
        rhs = getattr(robot, "_right_hand_state", None)
        if lhs is not None:
            rec["left_hand_q"] = [round(float(lhs.motor_state[i].q), 4) for i in range(Dex3_Num_Motors)]
            p = getattr(lhs, "pressure", None)
            if p is not None:
                rec["left_pressure"] = [round(float(x), 1) for x in p]
        if rhs is not None:
            rec["right_hand_q"] = [round(float(rhs.motor_state[i].q), 4) for i in range(Dex3_Num_Motors)]
            p = getattr(rhs, "pressure", None)
            if p is not None:
                rec["right_pressure"] = [round(float(x), 1) for x in p]
        _write(rec)
    except Exception as e:
        _write({"io": "recv_err", "err": repr(e)})
