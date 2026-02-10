# Running Async Inference on Unitree G1 Dex3

This guide explains how to set up and run asynchronous inference for the Unitree G1 Dex3 robot using LeRobot.

## Prerequisites

- **Robot**: Unitree G1 Dex3 with the custom server script (`run_g1_server.py`) running.
- **Client**: A remote machine with the `lerobot` library installed and network access to the robot.
- **Network**: Ensure the robot and client can communicate (ping check).

---

## 1. Start Servers on Robot

SSH into the robot. You need to run **two separate processes**: one for control and one for vision.

### Terminal 1: Control Server
This script bridges the robot's internal DDS communication to ZMQ for body and hand control.

```bash
# On the robot
python lerobot/src/lerobot/robots/unitree_g1/run_g1_server.py
```

**Verify Output:**
```
bridge running (body + hands: lowstate/handstate -> zmq, lowcmd/handcmd -> dds)
```

### Terminal 2: Camera Server
This script streams camera images over ZMQ. You may need to identify your camera device index (e.g., `/dev/video0` is index 0).

```bash
# On the robot
python lerobot/src/lerobot/cameras/zmq/image_server.py \
    --device 0 \
    --fps 30 \
    --width 640 \
    --height 480 \
    --port 5555
```

**Note:** If your robot has multiple cameras or the main camera is on a different index (e.g., `/dev/video4`), change `--device 4`.

---

## 2. Start Policy Server (Remote Machine)

On your remote machine (where the GPU is), start the policy server. This process hosts the policy model and handles inference requests.

```bash
# On remote machine (Terminal 1)
python -m lerobot.async_inference.policy_server \
    --host 0.0.0.0 \
    --port 8080 \
    --fps 30 \
    --disable_obs_filtering True  # Optional: Run inference on every frame
```

>**Note:** If you encounter `CUDA unknown error`, try restarting your machine or run with `--policy_device=cpu` as a temporary workaround.

---

## 3. Start Robot Client (Remote Machine)

On the same remote machine, start the robot client. This process connects to the robot (via ZMQ) and the policy server (via gRPC).

```bash
# On remote machine (Terminal 2)
python -m lerobot.async_inference.robot_client \
    --server_address 127.0.0.1:8080 \
    --robot.type unitree_g1_dex3 \
    --robot.is_simulation false \
    --robot.control_mode upper_body \
    --task "Pick up the kettle" \
    --policy_type act \
    --pretrained_name_or_path models/last/pretrained_model \
    --policy_device cuda \
    --actions_per_chunk 100 \
    --chunk_size_threshold 0.5
```

---

## Parameter Tuning & Concepts

For a deep dive into these parameters and the asynchronous inference architecture, please consult the **SmolVLA paper** (https://arxiv.org/pdf/2506.01844).

| Parameter | Application | Description |
| :--- | :--- | :--- |
| **`fps`** | Server & Client | Target frequency for the control loop and inference. The server rate-limits inference to this value (e.g., 30 Hz). |
| **`actions_per_chunk`** | Client | Number of actions to fetch in a single inference step. Higher values allow longer open-loop execution but increase latency/jitter. **Must not exceed model's training chunk size.** |
| **`chunk_size_threshold`** | Client | Determines when to request the next chunk. <br>• **0.0**: Wait until queue is empty (synchronous-like).<br>• **1.0**: Request immediately (always prefetching).<br>• **0.5**: Request when queue is half empty (balanced). |
| **`disable_obs_filtering`** | Server | **False (Default)**: Skips inference if the observation is too similar to the previous one (saves compute).<br>**True**: Runs inference on every valid observation received. |

---

## Troubleshooting

- **Hands not moving?**
  - Check if `run_g1_server.py` on the robot is printing "bridge running".
  - Ensure the robot client sees the hands connecting via ZMQ in the logs.

- **Camera issues:**
  - If the client logs "Connecting to ZMQCamera..." but hangs or fails, ensure `image_server.py` is running on the robot at port 5555.
  - Verify camera device index on the robot using `v4l2-ctl --list-devices`.
