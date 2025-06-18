import os
import yaml
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

# ---------------------- Load Config ----------------------
with open("e:/intern/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("✅ YAML loaded:")
print("Top-level keys:", config.keys())
print("Paths keys:", config["paths"].keys())

# Load paths from config
input_dirs = config["paths"]["behavioral_mat_dirs"]
rt_save_path = config["paths"]["rt_csv"]
label_save_path = config["paths"]["label_csv"]

# ---------------------- Extract RTs ----------------------
all_rts = []

for folder in input_dirs:
    for file in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
        if not file.endswith(".mat"):
            continue

        filepath = os.path.join(folder, file)
        try:
            mat = loadmat(filepath)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
            continue

        # Extract response and stimulus times if available
        if 'RespT' in mat and 'TargOnT' in mat:
            resp_times = np.array(mat['RespT']).flatten()
            stim_times = np.array(mat['TargOnT']).flatten()
            print(f"📊 {file} → {len(resp_times)} RespT | {len(stim_times)} TargOnT")

            n_trials = min(len(resp_times), len(stim_times))
            for i in range(n_trials):
                rt = (resp_times[i] - stim_times[i]) * 1000  # convert to milliseconds
                all_rts.append({
                    "file_id": os.path.splitext(file)[0],
                    "trial": i,
                    "stim_time": stim_times[i],
                    "resp_time": resp_times[i],
                    "rt": rt
                })
        else:
            print(f"⚠️ No RT found in: {file}. Keys: {list(mat.keys())}")

# ---------------------- Save RT CSV ----------------------
rt_df = pd.DataFrame(all_rts)

if rt_df.empty:
    raise RuntimeError("❌ No reaction times extracted. Please check file contents.")

rt_df.to_csv(rt_save_path, index=False)
print(f"✅ Saved RTs to: {rt_save_path}")

# ---------------------- Label Fast/Slow ----------------------
median_rt = rt_df["rt"].median()
rt_df["label"] = rt_df["rt"].apply(lambda x: "fast" if x < median_rt else "slow")
rt_df.to_csv(label_save_path, index=False)
print(f"✅ Saved labeled RTs to: {label_save_path}")
