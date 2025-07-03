import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import yaml
from pathlib import Path

# ========== Load config ==========
try:
    with open("e:/intern/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load config.yaml: {e}")

behavioral_dirs = config["paths"]["behavioral_mat_dirs"]
output_csv = os.path.join("e:/intern/outputs", "event_timeline.csv")  # update if needed

# ========== Collect All Events ==========
event_log = []

def extract_events(mat):
    keys = mat.keys()
    targ_key = [k for k in keys if "TargOnT" in k][0]
    resp_key = [k for k in keys if "RespT" in k][0]
    targont = mat[targ_key].flatten()
    respt = mat[resp_key].flatten()
    return targont, respt

# ========== Main ==========
print("🔍 Extracting all TargOnT and RespT events...")

mat_files = []
for d in behavioral_dirs:
    mat_files.extend([
        os.path.join(d, f) for f in os.listdir(d)
        if f.endswith(".mat")
    ])

for mat_file in tqdm(mat_files, desc="Processing files"):
    try:
        mat = loadmat(mat_file, simplify_cells=True)
        file_id = Path(mat_file).stem
        targont, respt = extract_events(mat)

        for t in targont:
            event_log.append({
                "file_id": file_id,
                "event_type": "TargOnT",
                "timestamp": float(t)
            })

        for r in respt:
            event_log.append({
                "file_id": file_id,
                "event_type": "RespT",
                "timestamp": float(r)
            })

    except Exception as e:
        print(f"⚠️ Skipped {mat_file}: {e}")
        continue

# ========== Save ==========
df = pd.DataFrame(event_log)
df.to_csv(output_csv, index=False)
print(f"✅ Saved all TargOnT and RespT events to: {output_csv}")
