
import pandas as pd
import numpy as np

file_path = '/home/breno/.cache/huggingface/lerobot/Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-white-table/data/chunk-000/episode_000071.parquet'
try:
    df = pd.read_parquet(file_path)
    print(f"Loaded {file_path}")
    col = 'observation.depths.depth_high'
    if col not in df.columns:
        print(f"Column {col} not found")
        exit()
    
    val = df[col].iloc[0]
    print(f"Type of first element: {type(val)}")
    
    if isinstance(val, np.ndarray) and val.dtype == object:
        print(f"Shape: {val.shape}")
        print(f"Dtype: {val.dtype}")
        # Check lengths of elements
        lengths = [len(x) if hasattr(x, '__len__') else 'scalar' for x in val]
        unique_lengths = set(lengths)
        print(f"Unique element lengths: {unique_lengths}")
        print(f"First element type: {type(val[0])}")
    elif isinstance(val, list):
        print(f"Length: {len(val)}")
        print(f"Type of first item in list: {type(val[0])}")
        arr = np.array(val)
        print(f"Numpy conversion shape: {arr.shape}")
        print(f"Numpy conversion dtype: {arr.dtype}")
    
    # Try the failing operation
    try:
        scale = 1000.0
        def convert_val(x):
            arr = np.array(x, dtype=np.float32)
            return arr / scale
        
        print("Attempting conversion on first element...")
        converted = convert_val(val)
        print("Conversion successful for first element")
    except Exception as e:
        print(f"Conversion failed: {e}")

except Exception as e:
    print(f"Error: {e}")
