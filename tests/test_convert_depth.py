import unittest
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
from convert_depth import convert_depth_column


@patch("convert_depth.snapshot_download")
@patch("convert_depth.HfApi")
class TestConvertDepth(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)
        self.repo_id = "test_user/test_dataset"
        
        # Create dataset structure
        (self.root / "meta").mkdir(parents=True)
        (self.root / "data/chunk-000").mkdir(parents=True)
        
        # 1. Create info.json
        self.column_name = "observation.depths.depth_high"
        self.info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "features": {
                self.column_name: {
                    "dtype": "int64", # Use int64 to avoid pyarrow cast errors in test creation
                    # But convert script checks dtype field.
                    # If it was an image saved as png, conversion would be harder.
                    # As parquet column, it's just array values.
                    "shape": [480, 640],
                    "names": ["width", "height"] # Simplified
                },
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["dims"]
                }
            },
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "robot_type": "unitree_g1",
            "total_episodes": 1,
            "total_frames": 10,
            "total_tasks": 0,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 100
        }
        with open(self.root / "meta/info.json", "w") as f:
            json.dump(self.info, f)
            
        # 2. Create stats.json
        self.stats = {
            self.column_name: {
                "min": [1000],
                "max": [2000],
                "mean": [1500.0],
                "std": [500.0],
                "count": [10],
                "q01": [1000],
                "q99": [2000]
            }
        }
        with open(self.root / "meta/stats.json", "w") as f:
            json.dump(self.stats, f)
            
        # 3. Create parquet file
        # Create a single row with flattened array or just list of list
        # We simulate flattened array for simplicity or what pandas expects
        # "observation.depths.depth_high" -> array of shape (480*640,) typically if flattened
        # But let's just use a small array for testing, modifying shape in info
        
        # Update shape to small for test
        # Update shape to 1D for test to simplify parquet loading
        self.info["features"][self.column_name]["shape"] = [4]
        with open(self.root / "meta/info.json", "w") as f:
            json.dump(self.info, f)

        data = {
            "index": np.arange(10),
            # Use flattened lists
            self.column_name: [[1000, 2000, 1500, 1000] for _ in range(10)],
            "action": [[0.0]*6 for _ in range(10)]
        }
        df = pd.DataFrame(data)
        df.to_parquet(self.root / "data/chunk-000/file-000.parquet")
        
        # Need episodes metdata? LeRobotDataset loads episodes from meta/episodes/...
        # But our script iterates over data/chunk-* parquet files directly.
        # LeRobotDataset init might fail if episodes meta is missing.
        # Let's see if we can bypass LeRobotDataset init or if we need to mock it.
        # The script calls `LeRobotDataset(repo_id, root=root)`.
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_conversion(self, mock_hf_api, mock_snapshot_download):
        # Mock API
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance
        
        # We don't need episodes for this script if we bypass LeRobotDataset
        # But we create them just in case logic changes.
        
        (self.root / "meta/episodes/chunk-000").mkdir(parents=True)
        # Create dummy episodes parquet
        ep_df = pd.DataFrame({
            "episode_index": [0],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "dataset_from_index": [0],
            "dataset_to_index": [10],
             "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [0],
             "length": [10]
        })
        ep_df.to_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet")
        
        # Create dummy tasks
        # DEFAULT_TASKS_PATH is meta/tasks.parquet
        task_df = pd.DataFrame({"task_index": [], "task": []})
        task_df.to_parquet(self.root / "meta/tasks.parquet")


        # RUN CONVERSION
        convert_depth_column(self.repo_id, root=self.root, column_name=self.column_name)
        
        # Verify Info
        with open(self.root / "meta/info.json", "r") as f:
            new_info = json.load(f)
        self.assertEqual(new_info["features"][self.column_name]["dtype"], "float32")
        
        # Verify Stats
        with open(self.root / "meta/stats.json", "r") as f:
            new_stats = json.load(f)
        
        # 1000 -> 1.0, 2000 -> 2.0
        self.assertAlmostEqual(new_stats[self.column_name]["min"][0], 1.0)
        self.assertAlmostEqual(new_stats[self.column_name]["max"][0], 2.0)
        
        # Verify Parquet
        df = pd.read_parquet(self.root / "data/chunk-000/file-000.parquet")
        val = df[self.column_name].iloc[0]
        self.assertEqual(val.dtype, np.float32)
        print(f"Result value: {val}")
        np.testing.assert_array_almost_equal(val, np.array([1.0, 2.0, 1.5, 1.0], dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
