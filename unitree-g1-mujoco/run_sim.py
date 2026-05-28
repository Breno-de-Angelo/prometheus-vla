#!/usr/bin/env python3
import sys
from pathlib import Path
import time
# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from sim.simulator_factory import SimulatorFactory, init_channel

def main(n_envs=1, use_async_envs: bool = False, 
             publish_images=True, camera_port=5555, cameras=None, **kwargs):
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with default values
    enable_offscreen = publish_images or config.get("ENABLE_OFFSCREEN", False)

    # Configure cameras if requested
    camera_configs = {}
    if enable_offscreen:
        # REMOVIDO: "d435i_rgb"
        camera_list = cameras or ["head_camera", "head_camera_depth"]
        for cam_name in camera_list:
            if cam_name == "head_camera":
                camera_configs[cam_name] = {"height": 480, "width": 640} # HD pro VR (e RGB da IA)
            else:
                camera_configs[cam_name] = {"height": 480, "width": 640} # SD pra IA (Depth/IR)
        print(f"📷 Cameras: {', '.join(camera_list)} → ZMQ port {camera_port}")
    
    print("="*60)
    
    # Initialize DDS channel
    init_channel(config=config)
    
    # Create simulator
    sim = SimulatorFactory.create_simulator(
        config=config,
        env_name="default",
        onscreen=config.get("ENABLE_ONSCREEN", True),
        offscreen=enable_offscreen,
        camera_configs=camera_configs,
    )
    
    # Start simulator
    print("\nSimulator running. Press Ctrl+C to exit.")
    if enable_offscreen and publish_images:
        print(f"Camera images publishing on tcp://localhost:{camera_port}")
    try:
        if publish_images:
            sim.start_image_publish_subprocess(
                start_method="spawn",
                camera_port=camera_port,
            )
            time.sleep(1)
        sim.start()
    except KeyboardInterrupt:
        print("+++++Simulator interrupted by user.")
    except Exception as e:
        print(f"++++error in simulator: {e} ++++")
    finally:
        print("++++closing simulator ++++")
        sim.close()

if __name__ == "__main__":
    main()

