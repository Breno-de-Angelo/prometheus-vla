import sys
from pathlib import Path
import configparser
import yaml

# Add src to path just in case
sys.path.append("/home/hercules/prometheus-vla/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def calculate_frames(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get('dataset', {})
    val_dataset_cfg = config.get('val_dataset', {})
    
    # Handle if they are None (though yaml.safe_load usually returns dict/None)
    if dataset_cfg is None: dataset_cfg = {}
    if val_dataset_cfg is None: val_dataset_cfg = {}

    repo_id = dataset_cfg.get('repo_id')
    train_episodes = dataset_cfg.get('episodes', [])
    val_episodes = val_dataset_cfg.get('episodes', [])
    
    # If repo_id not in dataset, check val? No, assume train is primary
    if not repo_id and val_dataset_cfg:
        repo_id = val_dataset_cfg.get('repo_id')

    print(f"Dataset Repo ID: {repo_id}")
    print(f"Train Episodes: {len(train_episodes)}")
    print(f"Val Episodes: {len(val_episodes)}")

    # Load dataset metadata (without downloading data if possible, but factory usually downloads)
    # Using LeRobotDataset directly to avoid full config parsing dependency issues if possible
    # But LeRobotDataset needs root. Let's try to use the standard factory if imports work.
    
    # Actually, let's just use LeRobotDataset. metadata is fetched from hub.
    dataset = LeRobotDataset(repo_id=repo_id)
    
    total_train_frames = 0
    total_val_frames = 0
    
    # lengths is a list of frame counts per episode
    # episodes index starts at 0
    
    print("Fetching dataset metadata...")
    # This might take a moment to download meta
    
    episode_data = dataset.meta.episodes
    # Accessing structured array or dict
    if isinstance(episode_data, dict):
        lengths = episode_data['length']
    else:
        # It might be a HF dataset object or similar.
        # In lerobot, meta.episodes is usually a arrow table or dict of numpy arrays
        lengths = dataset.meta.episodes['length']

    if train_episodes:
        for ep_idx in train_episodes:
            if 0 <= ep_idx < len(lengths):
                total_train_frames += lengths[ep_idx]
            else:
                print(f"Warning: Train episode index {ep_idx} out of bounds (max {len(lengths)-1})")
    else:
        print("No train episodes specified in dataset.episodes. Assuming all if val not separate?")
        # Logic is: if no episodes specified, it's all of them. But here we typically specify splits.
    
    if val_episodes:
        for ep_idx in val_episodes:
            if 0 <= ep_idx < len(lengths):
                total_val_frames += lengths[ep_idx]
            else:
                print(f"Warning: Val episode index {ep_idx} out of bounds (max {len(lengths)-1})")

    print("-" * 30)
    print(f"Total Train Frames: {total_train_frames}")
    print(f"Total Val Frames:   {total_val_frames}")
    print("-" * 30)
    
    batch_size = config.get('batch_size', 1)
    if batch_size > 0:
        steps_per_epoch = total_train_frames / batch_size
        print(f"Steps per epoch (batch_size={batch_size}): {steps_per_epoch:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_frames.py <path_to_config>")
        # Default for the user request
        default_path = "train/config/act_with_val.yaml"
        if Path(default_path).exists():
           calculate_frames(default_path)
        else:
           sys.exit(1)
    else:
        calculate_frames(sys.argv[1])
