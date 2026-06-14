from pathlib import Path
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from huggingface_hub import snapshot_download
import yaml
snapshot_download("lerobot/unitree-g1-mujoco")

# Ensure sim module is importable
sys.path.insert(0, str(Path(__file__).parent))

from sim.simulator_factory import SimulatorFactory, init_channel


def make_env(n_envs=1, use_async_envs=False, **kwargs):
    """Create a UnitreeG1 simulation environment.
    
    Args:
        n_envs: Number of environments (currently only 1 supported)
        use_async_envs: Whether to use async envs (not implemented)
        **kwargs: Additional arguments passed to simulator
            - publish_images: bool, whether to publish camera images via ZMQ
            - camera_port: int, ZMQ port for camera images
            - cameras: list of camera names
    """
    repo_dir = Path(__file__).parent
    
    # Load config
    config_path = repo_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Configure cameras if requested
    publish_images = kwargs.get("publish_images", True)
    camera_port = kwargs.get("camera_port", 5555)
    cameras = kwargs.get("cameras", ["head_camera"])
    
    enable_offscreen = publish_images or config.get("ENABLE_OFFSCREEN", False)
    camera_configs = {}
    if enable_offscreen:
        for cam_name in cameras:
            if cam_name == "head_camera":
                camera_configs[cam_name] = {"height": 480, "width": 848} # HD pro VR
            else:
                camera_configs[cam_name] = {"height": 480, "width": 848} # Leve pra IA
    
    # Initialize DDS channel
    init_channel(config=config)
    
    # Create simulator (runs in same process)
    simulator = SimulatorFactory.create_simulator(
        config=config,
        env_name="default",
        onscreen=config.get("ENABLE_ONSCREEN", True),
        offscreen=enable_offscreen,
        camera_configs=camera_configs,
    )
    
    class UnitreeG1Env(gym.Env):
        """Gymnasium environment wrapper for Unitree G1 MuJoCo simulation."""
        
        metadata = {"render_modes": ["human"]}

        def __init__(self, sim, cam_configs, pub_images, cam_port):
            super().__init__()
            self.simulator = sim
            self.sim_env = sim.sim_env
            self.step_count = 0
            self.camera_configs = cam_configs
            
            # Get timing from config
            self.sim_dt = config["SIMULATE_DT"]
            self.viewer_dt = config.get("VIEWER_DT", 0.02)
            self.image_dt = config.get("IMAGE_DT", 0.033333)
            
            # Start image publishing subprocess if requested
            if pub_images and len(cam_configs) > 0:
                sim.start_image_publish_subprocess(
                    start_method=config.get("MP_START_METHOD", "spawn"),
                    camera_port=cam_port
                )
                print(f"Camera images publishing on tcp://localhost:{cam_port}")
            
            # Define spaces
            num_joints = config.get("NUM_MOTORS", 29)
            self.action_space = spaces.Box(-np.pi, np.pi, shape=(num_joints,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(num_joints * 3 + 10,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            """Reset the simulation."""
            super().reset(seed=seed, options=options)
            self.simulator.reset()
            self.step_count = 0
            obs = self._get_obs()
            return obs, {}

        def step(self, action=None):
            """Execute one simulation step. Caller handles timing."""
            # Run physics step
            self.sim_env.sim_step()
            
            # Update viewer at viewer rate
            if self.step_count % int(self.viewer_dt / self.sim_dt) == 0:
                self.sim_env.update_viewer()
            
            # Update render caches at image rate (for ZMQ publishing)
            if self.step_count % int(self.image_dt / self.sim_dt) == 0:
                self.sim_env.update_render_caches()
            
            self.step_count += 1
            
            obs = self._get_obs()
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        def _get_obs(self):
            """Get current observation from simulation."""
            obs_dict = self.sim_env.prepare_obs()
            obs = np.concatenate([
                obs_dict.get("body_q", np.zeros(29)),
                obs_dict.get("body_dq", np.zeros(29)),
                obs_dict.get("body_tau_est", np.zeros(29)),
                obs_dict.get("floating_base_pose", np.zeros(7))[:4],
                obs_dict.get("floating_base_vel", np.zeros(6))[:3],
                obs_dict.get("floating_base_acc", np.zeros(3)),
            ]).astype(np.float32)
            return obs

        def close(self):
            """Close the simulation."""
            print("++++closing simulator ++++")
            self.simulator.close()

    return UnitreeG1Env(simulator, camera_configs, publish_images, camera_port)