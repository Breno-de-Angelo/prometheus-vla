#!/usr/bin/env python

"""
Script to convert a specific depth column in a LeRobot dataset from uint16 (mm) to float32 (meters).
Target column: observation.depths.depth_high
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.utils import write_info, write_stats

def convert_depth_column(repo_id: str, root: Path = None, column_name: str = "observation.depths.depth_high", scale: float = 1000.0, push_to_hub: bool = False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset {repo_id}...")
    
    if root is None:
        root = HF_LEROBOT_HOME / repo_id
        logger.info(f"Downloading/Syncing to {root}...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=root)
    else:
        root = Path(root)

    logger.info(f"Dataset root: {root}")

    # 1. Validate Column
    info_path = root / "meta/info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    if column_name not in info["features"]:
        logger.error(f"Column {column_name} not found in dataset features.")
        return

    current_dtype = info["features"][column_name]["dtype"]
    # Provide flexible check, sometimes just 'image' or specific types are listed, 
    # but user said it is array[uint16] which implies custom feature or specific shaping.
    # If it's stored as 'image', it might be path to png/tiff.
    # If it's stored in parquet, it's a list/array.
    
    logger.info(f"Current feature info for {column_name}: {info['features'][column_name]}")

    # 2. Iterate and Modify Parquet Files
    data_dir = root / "data"
    # match v3.0 (file-*.parquet) and v2.1 (episode_*.parquet)
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    
    logger.info(f"Found {len(parquet_files)} parquet files to process.")

    for file_path in parquet_files:
        logger.info(f"Processing {file_path}...")
        df = pd.read_parquet(file_path)
        
        if column_name in df.columns:
            # Check first value to see if it needs conversion
            # LeRobot datasets usually store arrays as lists in parquet
            if len(df) > 0:
                first_val = df[column_name].iloc[0]
                first_arr = np.array(first_val)
                if np.issubdtype(first_arr.dtype, np.floating):
                    logger.warning(f"File {file_path} column {column_name} seems to be float already. Skipping.")
                    continue

            # Apply conversion
            # x is likely a numpy array or list
            def convert_val(x):
                # Ensure x is a list if it's an object-array of arrays, otherwise np.array(x) might fail
                if isinstance(x, np.ndarray) and x.dtype == object:
                    x = x.tolist()
                
                arr = np.array(x, dtype=np.float32)
                res = arr / scale
                # Return list to ensure PyArrow compatibility (handles nested lists)
                # Raw numpy 2D arrays in Object columns cause ArrowInvalid
                return res.tolist()

            df[column_name] = df[column_name].apply(convert_val)
            
            # Save back
            df.to_parquet(file_path, index=False)
        else:
            logger.warning(f"Column {column_name} not found in {file_path}")

    # 3. Update info.json
    logger.info("Updating meta/info.json...")
    info["features"][column_name]["dtype"] = "float32"
    # Ensure shape is preserved
    write_info(info, root)

    # 4. Update stats.json
    stats_path = root / "meta/stats.json"
    if stats_path.exists():
        logger.info("Updating meta/stats.json...")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        if column_name in stats:
            col_stats = stats[column_name]
            # Keys to update: min, max, mean, std, q*
            keys_to_scale = ["min", "max", "mean", "std"]
            keys_to_scale += [k for k in col_stats.keys() if k.startswith("q")]
            
            for key in keys_to_scale:
                if key in col_stats:
                    # Depending on shape, it might be a list or single value
                    val = np.array(col_stats[key], dtype=np.float32)
                    col_stats[key] = (val / scale).tolist()
            
            write_stats(stats, root)
    
    logger.info("Conversion complete.")

    if push_to_hub:
        logger.info(f"Pushing to hub: {repo_id}")
        api = HfApi()
        api.upload_folder(
            repo_id=repo_id,
            folder_path=root,
            repo_type="dataset",
            commit_message="Convert depth maps to float32"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--root", type=str, default=None, help="Local root directory")
    parser.add_argument("--push-to-hub", action="store_true", help="Push changes back to Hub")
    args = parser.parse_args()

    convert_depth_column(args.repo_id, root=args.root, push_to_hub=args.push_to_hub)
