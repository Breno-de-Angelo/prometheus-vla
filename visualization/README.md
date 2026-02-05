# G1 Visualization Tools

This package contains visualization scripts for the Unitree G1 robot.

## Scripts

- **visualize_g1.py** - 2D dashboard with camera feed, joint plots, and hand controls
- **visualize_g1_3d.py** - 3D URDF viewer using Vuer

## Robot Setup

Before running the visualizations, the robot must be streaming data.

### A. Start the ZMQ Bridge
This streams motor and IMU data.
**On the Robot:**
```bash
python lerobot/src/lerobot/robots/unitree_g1/run_g1_server.py
```

### B. Start the Camera Server
This streams the head camera feed.
**On the Robot:**
```bash
python lerobot/src/lerobot/cameras/zmq/image_server.py
```
*Note: If this is not running, the dashboard will show "Camera Disconnected" but will otherwise function normally.*

## Usage

```bash
# 2D Dashboard (http://localhost:5000)
uv run lerobot/src/lerobot/scripts/visualization/visualize_g1.py

# 3D Viewer (http://localhost:8012)
uv run lerobot/src/lerobot/scripts/visualization/visualize_g1_3d.py
```

See [G1_VISUALIZATION_GUIDE.md](./G1_VISUALIZATION_GUIDE.md) for detailed setup instructions.

## Details

### 2D Dashboard (`visualize_g1.py`)

A Flask-based web dashboard showing real-time plots of motor states, IMU data, and the camera feed.

**On the Local Machine:**
```bash
uv run lerobot/src/lerobot/scripts/visualize_g1.py
```

#### Accessing the Dashboard
Open your browser to: [http://localhost:5000](http://localhost:5000)

#### Features
- **Camera Feed:** Real-time stream from the robot's head camera (RGB).
- **Motor Plots:** Joint positions (q), velocities (dq), and torque (tau).
- **IMU Plots:** Accelerometer, Gyroscope, and Orientation (RPY).
- **Status Indicators:** Connection status and Update Rate (FPS).

#### Troubleshooting
- **"Camera Disconnected":** Ensure `image_server.py` is running on the robot (Port 5555).
- **"Robot not connected":** Ensure `run_g1_server.py` is running on the robot and the computer is on the same network (check `192.168.123.x`).


### 3D Visualization (`visualize_g1_3d.py`)

A 3D viewer using `vuer` to render the robot's URDF and real-time pose.

#### Running the Viewer
**On the Local Machine:**
```bash
uv run lerobot/src/lerobot/scripts/visualize_g1_3d.py
```

#### Accessing the Viewer
Open your browser to: [http://localhost:8012](http://localhost:8012)

#### Features
- **Real-time Pose:** The 3D model updates to match the physical robot.
- **Interactive Camera:** Orbit, zoom, and pan around the robot.
