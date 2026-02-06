# Prometheus VLA

VR teleoperation, training, and deployment for the Unitree G1 humanoid robot with Dex3 hands.

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Breno-de-Angelo/prometheus-vla
cd prometheus-vla

# 1. Create Conda environment (Required for Pinocchio + CasADi bindings)
conda create -n g1 python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate g1

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Data Collection (VR Teleoperation)

Use Meta Quest 3 for demonstration collection:

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --teleop.type=televuer \
  --teleop.use_hand_tracking=true \
  --dataset.repo_id=your_user/g1_pick_kettle \
  --dataset.single_task="Pick up the kettle"
```

**VR Controls:**
- Hand tracking: Move hands to control robot arms
- Pinch gesture: Control fingers
- Voice: "Stop" to pause recording

---

## Training

### Using Config File (Recommended)

```bash
# Create config in train/config/my_task.yaml, then:
CUDA_VISIBLE_DEVICES=0 lerobot-train --config train/config/my_task.yaml
```

### Direct Command

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=your_user/g1_pick_kettle \
  --training.num_epochs=100 \
  --output_dir=outputs/train/g1_act
```

### Background Training

```bash
CUDA_VISIBLE_DEVICES=1 nohup lerobot-train \
  --config train/config/my_task.yaml \
  > train/log/my_task.log 2>&1 &

# Monitor
tail -f train/log/my_task.log
```

---

## Dataset Visualization

```bash
# Local
lerobot-dataset-viz \
  --repo-id=your_user/g1_pick_kettle \
  --episode-index=0 \
  --display-compressed-images=true

# Remote (stream to another machine)
lerobot-dataset-viz \
  --repo-id=your_user/g1_pick_kettle \
  --episode-index=0 \
  --mode=distant
# Then on your machine: rerun ws://server_ip:9087
```

---

## Robot Visualization

See [visualization/README.md](visualization/README.md) for detailed setup.

### 2D Dashboard
```bash
# Start robot servers first, then:
python visualization/visualize_g1.py
# Open http://localhost:5000
```

### 3D Viewer
```bash
python visualization/visualize_g1_3d.py
# Open http://localhost:8012
```

---

## Deployment

### Local Inference

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --policy.path=outputs/train/g1_act/checkpoints/last \
  --dataset.repo_id=your_user/g1_eval
```

---

## Network Setup

The G1 robot uses Ethernet (default: `192.168.123.x`):

```bash
# Set static IP on your machine
sudo ip addr add 192.168.123.100/24 dev eth0

# Test connection
ping 192.168.123.161  # Robot IP
```

---

## Project Structure

```
prometheus-vla/
├── lerobot/              # LeRobot submodule (fork)
│   └── src/lerobot/
│       ├── robots/unitree_g1/    # G1 + Dex3 robot code
│       ├── teleoperators/        # TeleVuer VR teleop
│       └── processor/            # VR → robot conversion
├── train/
│   ├── config/           # Training config files
│   └── log/              # Training logs
├── visualization/        # G1 visualization tools
└── assets/               # Robot meshes (STL files)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Robot not responding | Check network: `ping 192.168.123.161` |
| VR not connecting | Ensure Quest and PC on same network, check ports 8012-8013 |
| IK solver errors | Run: `python -c "from pinocchio import casadi as cpin"` |
| Missing unitree_sdk2py | Requires Python < 3.12 |
