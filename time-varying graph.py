import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import yaml
from scipy.signal import hilbert

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

def compute_plv_matrix_fast(data):
    """
    Optimized PLV computation using vectorized operations
    Input: data shape (n_parcels, n_timepoints)
    Output: PLV matrix shape (n_parcels, n_parcels)
    """
    n_parcels, n_times = data.shape
    
    # Get analytic signal (complex representation)
    analytic_signals = hilbert(data, axis=1)
    
    # Extract instantaneous phases
    phases = np.angle(analytic_signals)
    
    # Initialize PLV matrix
    plv_matrix = np.zeros((n_parcels, n_parcels))
    
    # Vectorized PLV computation
    for i in range(n_parcels):
        for j in range(i + 1, n_parcels):
            # Phase difference
            phase_diff = phases[i] - phases[j]
            # PLV is absolute value of mean complex exponential
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv  # Symmetric matrix
    
    return plv_matrix

# ------------------ Main Processing ------------------
all_features = []
processed_files = 0
start_time = time.time()

# Get list of source files to process
source_files = [f for f in os.listdir(source_dir) 
                if f.endswith(".npy") and f.startswith("source_") 
                and "pre_stim" not in f and "channel" not in f]

print(f"\n🚀 Found {len(source_files)} source files to process")

for filename in tqdm(source_files, desc="Processing files"):

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

        print(f"   Window size: {w_size} samples, Step: {s_size} samples")
        print(f"   Time points: {n_times}, Calculated windows: {n_windows}")

        if n_windows <= 0:
            print(f"⚠️ Skipping {file_id} — insufficient time points ({n_times})")
            continue

        file_features = []
        valid_windows = 0
        
        print(f"🔄 Processing {n_windows} windows...")
        window_start_time = time.time()

        for w in range(n_windows):
            start = w * s_size
            end = start + w_size
            window_data = data[:, start:end]
            
            # Show progress every 200 windows for better performance
            if w % 200 == 0 or n_windows <= 20:
                t_start = start / sfreq
                t_end = end / sfreq
                elapsed_windows = time.time() - window_start_time
                if w > 0:
                    windows_per_sec = w / elapsed_windows
                    eta_windows = (n_windows - w) / windows_per_sec if windows_per_sec > 0 else 0
                    print(f"   🌀 Window {w+1}/{n_windows}: t={t_start:.2f}–{t_end:.2f}s (ETA: {eta_windows/60:.1f}min)")
                else:
                    print(f"   🌀 Window {w+1}/{n_windows}: t={t_start:.2f}–{t_end:.2f}s")

            # Sanity check
            if np.isnan(window_data).any() or np.max(np.abs(window_data)) == 0:
                if w % 500 == 0:  # Only print occasionally to avoid spam
                    print(f"⚠️ Window {w} contains NaNs or zeros, skipping")
                continue

            # Compute PLV matrix using optimized function
            plv_matrix = compute_plv_matrix_fast(window_data)

            # Validate matrix
            if np.isnan(plv_matrix).any() or np.max(plv_matrix) == 0:
                if w % 500 == 0:  # Only print occasionally
                    print(f"⚠️ Window {w}: PLV matrix invalid")
                continue

            # Build graph + centrality
            graph = nx.from_numpy_array(plv_matrix)
            centrality = nx.degree_centrality(graph)
            
            valid_windows += 1

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
        print(f"   📊 Valid windows: {valid_windows}/{n_windows}")

        if len(file_features) == 0:
            print(f"⚠️ No features extracted from {file_id}, skipping CSV save.")
            continue

        # Save per-file CSV
        df = pd.DataFrame(file_features)
        csv_path = os.path.join(plv_graph_features_dir, f"{file_id}_graph.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path} ({len(df)} rows)")
        print(f"   📁 File size: {os.path.getsize(csv_path) / 1024:.1f} KB")

        all_features.extend(file_features)
        processed_files += 1
        
        # Show timing estimate
        elapsed = time.time() - start_time
        if processed_files > 0:
            avg_time = elapsed / processed_files
            remaining_files = len(source_files) - processed_files
            eta = avg_time * remaining_files
            print(f"   ⏱️ File ETA: {eta/60:.1f} min ({avg_time:.1f}s per file)")

    except Exception as e:
        print(f"❌ Error processing {file_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ------------------ Save Combined Output ------------------
if all_features:
    all_df = pd.DataFrame(all_features)
    all_csv_path = os.path.join(plv_graph_features_dir, "plv_graph_features_ALL_FILES.csv")
    all_df.to_csv(all_csv_path, index=False)
    
    total_time = time.time() - start_time
    print(f"\n🎯 PROCESSING COMPLETE! (took {total_time/60:.1f} minutes)")
    print(f"✅ All features saved to: {all_csv_path}")
    print(f"📊 Total features extracted: {len(all_features)}")
    print(f"📁 Files processed: {len(all_df['file_id'].unique())}/{len(source_files)}")
    print(f"📈 Final CSV size: {os.path.getsize(all_csv_path) / 1024 / 1024:.2f} MB")
else:
    print(f"\n⚠️ No features were extracted from any files! (took {(time.time() - start_time)/60:.1f} minutes)")
    print("   Check your data files and processing parameters.")
