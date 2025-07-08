import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import yaml

# ------------------ Load Configuration ------------------
with open("e:/intern/config.yaml", "r") as f:
    config = yaml.safe_load(f)

source_dir = config["paths"]["features_dir"]
plv_graph_features_dir = os.path.join(source_dir, "plv_graph_features")

sfreq = 500  # Sampling rate (Hz)
window_size = 0.25  # seconds → 125 samples
step_size = 0.1     # seconds → 50 samples

# Ensure output directory exists
os.makedirs(plv_graph_features_dir, exist_ok=True)

# ------------------ Main Processing ------------------
all_features = []

for filename in tqdm(os.listdir(source_dir)):

    # ---- Strict File Filtering ----
    if not filename.endswith(".npy"):
        continue
    if "pre_stim" in filename or "channel" in filename:
        print(f"⏭️ Skipping pre-processed file: {filename}")
        continue
    if not filename.startswith("source_"):
        print(f"⏭️ Skipping unrelated file: {filename}")
        continue

    # ---- Load and Validate Data ----
    file_path = os.path.join(source_dir, filename)
    file_id = filename.replace("source_", "").replace(".npy", "")

    try:
        data = np.load(file_path)
        print(f"\n📊 {file_id}: loaded shape {data.shape}")
        print(f"   NaNs: {np.isnan(data).sum()}, Max abs value: {np.max(np.abs(data)):.6f}")

        # ---- Handle epoched or continuous ----
        if data.ndim == 3:  # (n_trials, n_parcels, n_times)
            print(f"⚠️ Epoched data detected: {data.shape}, concatenating trials")
            data = np.concatenate(data, axis=-1)
            print(f"   After concatenation: {data.shape}")
        elif data.ndim == 2:
            print(f"✅ Continuous data detected: {data.shape}")
        else:
            print(f"❌ Unexpected shape {data.shape}, skipping")
            continue

        n_parcels, n_times = data.shape
        w_size = int(window_size * sfreq)
        s_size = int(step_size * sfreq)
        n_windows = (n_times - w_size) // s_size + 1

        print(f"   Window size: {w_size} samples")
        print(f"   Step size: {s_size} samples")
        print(f"   Time points: {n_times}, Calculated windows: {n_windows}")

        if n_windows <= 0:
            print(f"⚠️ Skipping {file_id} — insufficient time points ({n_times})")
            continue

        file_features = []

        for w in range(n_windows):
            start = w * s_size
            end = start + w_size
            window_data = data[:, start:end]

            # Sanity check
            if np.isnan(window_data).any() or np.max(np.abs(window_data)) == 0:
                print(f"⚠️ Window {w} contains NaNs or zeros, skipping")
                continue

            # Compute PLV matrix
            plv_matrix = np.zeros((n_parcels, n_parcels))
            for i in range(n_parcels):
                for j in range(i + 1, n_parcels):
                    phase_i = np.angle(np.exp(1j * np.angle(window_data[i])))
                    phase_j = np.angle(np.exp(1j * np.angle(window_data[j])))
                    plv = np.abs(np.mean(np.exp(1j * (phase_i - phase_j))))
                    plv_matrix[i, j] = plv
                    plv_matrix[j, i] = plv

            # Validate matrix
            if np.isnan(plv_matrix).any() or np.max(plv_matrix) == 0:
                print(f"⚠️ Window {w}: PLV matrix invalid")
                continue

            # Build graph + centrality
            graph = nx.from_numpy_array(plv_matrix)
            centrality = nx.degree_centrality(graph)

            t_start = start / sfreq
            t_end = end / sfreq

            for node, cent_val in centrality.items():
                file_features.append({
                    "file_id": file_id,
                    "window_idx": w,
                    "t_start": t_start,
                    "t_end": t_end,
                    "node": node,
                    "centrality": cent_val
                })

        print(f"✔️ Extracted {len(file_features)} features from {file_id}")

        if len(file_features) == 0:
            print(f"⚠️ No features extracted from {file_id}, skipping CSV save.")
            continue

        # Save per-file CSV
        df = pd.DataFrame(file_features)
        csv_path = os.path.join(plv_graph_features_dir, f"{file_id}_graph.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path} ({len(df)} rows)")

        all_features.extend(file_features)

    except Exception as e:
        print(f"❌ Error processing {file_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ------------------ Save Combined Output ------------------
all_df = pd.DataFrame(all_features)
all_csv_path = os.path.join(plv_graph_features_dir, "plv_graph_features_ALL_FILES.csv")
all_df.to_csv(all_csv_path, index=False)

print("\n🎯 PROCESSING COMPLETE!")
print(f"✅ All features saved to: {all_csv_path}")
print(f"📊 Total features extracted: {len(all_features)}")
print(f"📁 Files processed: {len(all_df['file_id'].unique()) if not all_df.empty else 0}")
