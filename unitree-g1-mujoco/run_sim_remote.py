# zmq_bridge.py — roda no PC, expõe o DDS do MuJoCo via ZMQ para a rede
import json, base64, threading, time
import zmq
from sim.simulator_factory import init_channel
import yaml
from pathlib import Path

# Carrega config igual o env.py faz
config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(open(config_path))
init_channel(config=config)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, HandState_, HandCmd_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)

BIND_IP = "0.0.0.0"   # aceita de qualquer IP da rede
LOWCMD_PORT   = 6000
LOWSTATE_PORT = 6001
HANDSTATE_PORT= 6002
HANDCMD_PORT  = 6003

ctx = zmq.Context()

# ZMQ sockets
lowstate_pub  = ctx.socket(zmq.PUB)
lowstate_pub.bind(f"tcp://{BIND_IP}:{LOWSTATE_PORT}")

handstate_pub = ctx.socket(zmq.PUB)
handstate_pub.bind(f"tcp://{BIND_IP}:{HANDSTATE_PORT}")

lowcmd_pull   = ctx.socket(zmq.PULL)
lowcmd_pull.bind(f"tcp://{BIND_IP}:{LOWCMD_PORT}")

handcmd_pull  = ctx.socket(zmq.PULL)
handcmd_pull.bind(f"tcp://{BIND_IP}:{HANDCMD_PORT}")

print(f"✅ ZMQ Bridge ativo — ouvindo em 0.0.0.0:{LOWCMD_PORT}/{HANDCMD_PORT}")
print(f"   Publicando lowstate em :{LOWSTATE_PORT}, handstate em :{HANDSTATE_PORT}")

# ── DDS → ZMQ: publica lowstate para o servidor ──────────────────
_latest_lowstate = None
_lowstate_lock = threading.Lock()

def lowstate_handler(msg):
    global _latest_lowstate
    motors = []
    for i in range(config["NUM_MOTORS"]):
        m = msg.motor_state[i]
        motors.append({"q": m.q, "dq": m.dq, "tau_est": m.tau_est, "temperature": 0.0})
    imu = msg.imu_state
    data = {
        "motor_state": motors,
        "imu_state": {
            "quaternion": list(imu.quaternion),
            "gyroscope":  list(imu.gyroscope),
            "accelerometer": list(imu.accelerometer),
            "rpy": list(imu.rpy),
            "temperature": 0.0,
        },
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode(),
        "mode_machine": int(msg.mode_machine),
    }
    with _lowstate_lock:
        _latest_lowstate = json.dumps({"data": data}).encode()

lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
lowstate_sub.Init(lowstate_handler, 10)

def publish_lowstate_loop():
    while True:
        with _lowstate_lock:
            payload = _latest_lowstate
        if payload:
            lowstate_pub.send(payload)
        time.sleep(0.004)  # ~250Hz

threading.Thread(target=publish_lowstate_loop, daemon=True).start()

# ── DDS → ZMQ: publica handstate ─────────────────────────────────
_latest_left  = None
_latest_right = None
_hand_lock = threading.Lock()

def left_hand_handler(msg):
    global _latest_left
    motors = [{"q": m.q, "dq": m.dq, "tau_est": m.tau_est, "temperature": 0.0}
              for m in msg.motor_state]
    data = {"side": "left", "motor_state": motors, "press_sensor_state": []}
    with _hand_lock:
        _latest_left = json.dumps({
            "topic": "rt/dex3/left/state", "data": data
        }).encode()

def right_hand_handler(msg):
    global _latest_right
    motors = [{"q": m.q, "dq": m.dq, "tau_est": m.tau_est, "temperature": 0.0}
              for m in msg.motor_state]
    data = {"side": "right", "motor_state": motors, "press_sensor_state": []}
    with _hand_lock:
        _latest_right = json.dumps({
            "topic": "rt/dex3/right/state", "data": data
        }).encode()

left_sub  = ChannelSubscriber("rt/dex3/left/state",  HandState_)
right_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
left_sub.Init(left_hand_handler, 10)
right_sub.Init(right_hand_handler, 10)

def publish_handstate_loop():
    while True:
        with _hand_lock:
            l, r = _latest_left, _latest_right
        if l: handstate_pub.send(l)
        if r: handstate_pub.send(r)
        time.sleep(0.005)

threading.Thread(target=publish_handstate_loop, daemon=True).start()

# ── ZMQ → DDS: recebe lowcmd do servidor e manda pro MuJoCo ──────
low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
low_cmd_pub.Init()
low_cmd_msg = unitree_hg_msg_dds__LowCmd_()

def lowcmd_forward_loop():
    while True:
        try:
            payload = lowcmd_pull.recv(zmq.NOBLOCK)
            d = json.loads(payload)["data"]
            low_cmd_msg.mode_machine = d.get("mode_machine", 0)
            for i, mc in enumerate(d["motor_cmd"]):
                low_cmd_msg.motor_cmd[i].mode  = mc["mode"]
                low_cmd_msg.motor_cmd[i].q     = mc["q"]
                low_cmd_msg.motor_cmd[i].dq    = mc["dq"]
                low_cmd_msg.motor_cmd[i].kp    = mc["kp"]
                low_cmd_msg.motor_cmd[i].kd    = mc["kd"]
                low_cmd_msg.motor_cmd[i].tau   = mc["tau"]
            low_cmd_pub.Write(low_cmd_msg)
        except zmq.Again:
            pass
        time.sleep(0.001)

threading.Thread(target=lowcmd_forward_loop, daemon=True).start()

# ── ZMQ → DDS: recebe handcmd ────────────────────────────────────
left_hand_pub  = ChannelPublisher("rt/dex3/left/cmd",  HandCmd_)
right_hand_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
left_hand_pub.Init()
right_hand_pub.Init()
left_hand_msg  = HandCmd_default()
right_hand_msg = HandCmd_default()

def handcmd_forward_loop():
    while True:
        try:
            payload = handcmd_pull.recv(zmq.NOBLOCK)
            d = json.loads(payload)
            topic = d.get("topic", "")
            cmds  = d["data"]["motor_cmd"]
            msg   = left_hand_msg if "left" in topic else right_hand_msg
            pub   = left_hand_pub  if "left" in topic else right_hand_pub
            for i, mc in enumerate(cmds):
                msg.motor_cmd[i].q  = mc["q"]
                msg.motor_cmd[i].kp = mc["kp"]
                msg.motor_cmd[i].kd = mc["kd"]
            pub.Write(msg)
        except zmq.Again:
            pass
        time.sleep(0.001)

threading.Thread(target=handcmd_forward_loop, daemon=True).start()

print("🔁 Bridge rodando. Ctrl+C para parar.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Bridge encerrada.")