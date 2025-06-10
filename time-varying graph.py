import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import hilbert
import yaml

# ---------------- Load config ----------------
with open("e:/intern/config.yaml", "r") as f:
    config = yaml.safe_load(f)

source_dir = config["paths"]["features_dir"]
output_dir = "e:/intern/plv_viz"
os.makedirs(output_dir, exist_ok=True)

sfreq = config.get("params", {}).get("sfreq", 1000)
window_size = config.get("params", {}).get("window_size", 0.2)
step_size = config.get("params", {}).get("step_size", 0.05)
threshold = config.get("params", {}).get("plv_threshold", 0.5)

file_list = [f for f in os.listdir(source_dir) if f.startswith("source_") and f.endswith(".npy")]
csv_path = os.path.join(output_dir, "plv_graph_features_ALL_FILES.csv")

# ---------------- Resuming Support ----------------
processed_files = set()
if os.path.exists(csv_path):
    try:
        existing_df = pd.read_csv(csv_path, usecols=["file_id"])
        processed_files = set(existing_df["file_id"].unique())
        print(f"⏩ Resuming... {len(processed_files)} files already processed.")
    except Exception as e:
        print(f"⚠️ Failed to load existing CSV: {e}")

# ---------------- Sliding Window PLV ----------------
def sliding_window_plv(source_ts, sfreq):
    n_parcels, n_times = source_ts.shape
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    plv_matrices = []

    analytic_signal = hilbert(source_ts, axis=1)
    phases = np.angle(analytic_signal)

    for start in range(0, n_times - window_samples + 1, step_samples):
        stop = start + window_samples
        phase_window = phases[:, start:stop]
        plv_matrix = np.abs(np.mean(np.exp(1j * (phase_window[:, None, :] - phase_window[None, :, :])), axis=2))
        plv_matrices.append(plv_matrix)

    return np.array(plv_matrices)

# ---------------- Graph Features ----------------
def extract_graph_features(plv_matrix, window_idx, file_id):
    adj = (plv_matrix > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    G = nx.from_numpy_array(adj)
    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        return None

    features = {
        "file_id": file_id,
        "window": window_idx,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "clustering_coeff": nx.average_clustering(G),
        "efficiency": nx.global_efficiency(G) if nx.is_connected(G) else 0.0,
        "avg_eigenvector_centrality": np.mean(list(nx.eigenvector_centrality_numpy(G, max_iter=500).values())) if G.number_of_nodes() > 1 else 0.0,
        "avg_degree_centrality": np.mean(list(nx.degree_centrality(G).values())),
        "avg_betweenness_centrality": np.mean(list(nx.betweenness_centrality(G).values())),
    }
    return features

# ---------------- Main Loop ----------------
all_features = []
for source_file in file_list:
    file_id = source_file.replace("source_", "").replace(".npy", "")
    if file_id in processed_files:
        print(f"⏭️ Skipping {file_id}, already processed.")
        continue

    print(f"\n🔄 Processing file: {file_id}")
    try:
        source_ts = np.load(os.path.join(source_dir, source_file))
        plv_matrices = sliding_window_plv(source_ts, sfreq)

        for window_idx, plv in enumerate(plv_matrices):
            feats = extract_graph_features(plv, window_idx, file_id)
            if feats:
                all_features.append(feats)

        # Save per file
        if all_features:
            df = pd.DataFrame(all_features)
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
            all_features.clear()

    except Exception as e:
        print(f"❌ Failed processing {file_id}: {e}")
        continue

print(f"\n✅ Finished processing {len(file_list)} files.")
