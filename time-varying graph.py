import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import hilbert
import yaml
import mne
from tqdm import tqdm

# ---------------- Load config ----------------
with open("e:/intern/config.yaml", "r") as f:
    config = yaml.safe_load(f)

source_dir = config["paths"]["features_dir"]
epochs_dir = config["paths"]["epochs_dir"]
output_dir = os.path.join(source_dir, "time_varying")
os.makedirs(output_dir, exist_ok=True)

# Parameters
sfreq = config.get("params", {}).get("sfreq", 1000)
window_size = config.get("params", {}).get("window_size", 0.2)
step_size = config.get("params", {}).get("step_size", 0.05)
threshold = config.get("params", {}).get("plv_threshold", 0.9)  # strong edge definition

window_samples = int(window_size * sfreq)
step_samples = int(step_size * sfreq)

# ---------------- PLV & Graph Functions ----------------
def compute_plv(x1, x2):
    phase_diff = np.angle(hilbert(x1)) - np.angle(hilbert(x2))
    return np.abs(np.sum(np.exp(1j * phase_diff)) / len(x1))

def graph_features(plv_matrix, threshold):
    G = nx.from_numpy_array(plv_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"      ⚠️ Graph not fully connected — using largest component with {len(G.nodes)} nodes")

    try:
        degree_centrality = np.mean(list(nx.degree_centrality(G).values()))
        betweenness = np.mean(list(nx.betweenness_centrality(G, weight='weight').values()))
        clustering = np.mean(list(nx.clustering(G, weight='weight').values()))
        efficiency = nx.global_efficiency(G)
        return [degree_centrality, betweenness, clustering, efficiency]
    except Exception as e:
        print(f"    ❌ Graph feature extraction failed: {e}")
        return None

def filter_top_k_edges(plv_matrix, top_k=0.05):
    n = plv_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    plv_values = plv_matrix[triu_indices]
    k_cutoff = int(len(plv_values) * top_k)
    if k_cutoff < 1:
        return np.zeros_like(plv_matrix)
    # Get indices of top K values
    top_k_idx = np.argpartition(plv_values, -k_cutoff)[-k_cutoff:]
    adj_matrix = np.zeros_like(plv_matrix, dtype=int)
    adj_matrix[triu_indices[0][top_k_idx], triu_indices[1][top_k_idx]] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    np.fill_diagonal(adj_matrix, 0)
    print(f"Edges kept: {adj_matrix.sum()//2} / {len(plv_values)}")
    return adj_matrix

def build_graph_from_top_k(plv_matrix, top_k=0.05):
    n = plv_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    plv_values = plv_matrix[triu_indices]
    k_cutoff = int(len(plv_values) * top_k)
    if k_cutoff < 1:
        return None
    # Get indices of top K values
    top_k_idx = np.argpartition(plv_values, -k_cutoff)[-k_cutoff:]
    # Build weighted edge list
    edges = [(triu_indices[0][i], triu_indices[1][i], plv_values[top_k_idx[j]]) for j, i in enumerate(top_k_idx)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(edges)
    # Ensure connectivity
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G

# ---------------- Main Loop ----------------
file_list = sorted([f for f in os.listdir(source_dir) if f.startswith("source_") and f.endswith(".npy")])

for source_file in tqdm(file_list, desc="Processing source files"):
    file_id = source_file.replace("source_", "").replace(".npy", "")
    source_path = os.path.join(source_dir, source_file)
    epoch_path = os.path.join(epochs_dir, f"{file_id}_epo.fif")

    if not os.path.exists(source_path) or not os.path.exists(epoch_path):
        print(f"❌ Missing files for {file_id}")
        continue

    print(f"\n✅ Processing {file_id}")
    source_ts = np.load(source_path)
    epochs = mne.read_epochs(epoch_path, preload=True)
    n_trials = len(epochs)
    epoch_length = epochs.get_data().shape[2]

    if source_ts.shape[1] != n_trials * epoch_length:
        print(f"❌ Shape mismatch for {file_id}")
        continue

    source_reshaped = source_ts.reshape((n_trials, source_ts.shape[0], epoch_length))
    file_rows = []

    for trial_idx, trial_data in enumerate(source_reshaped):
        print(f"\n🔍 Trial {trial_idx} — shape: {trial_data.shape}")
        for start in range(0, trial_data.shape[1] - window_samples + 1, step_samples):
            segment = trial_data[:, start:start + window_samples]
            n_parcels = segment.shape[0]
            plv_matrix = np.zeros((n_parcels, n_parcels))

            for i in range(n_parcels):
                for j in range(i + 1, n_parcels):
                    plv = compute_plv(segment[i], segment[j])
                    plv_matrix[i, j] = plv_matrix[j, i] = plv

            non_zero_plvs = plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
            percentile = 95  # or 90 for a less strict threshold
            threshold = np.percentile(non_zero_plvs, percentile)

            # Apply threshold to create adjacency matrix:
            adj_matrix = (plv_matrix >= threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)

            strong_edges = non_zero_plvs > threshold
            density = np.sum(strong_edges) / len(non_zero_plvs)

            print(f"      ➤ Window PLV stats: min={non_zero_plvs.min():.4f}, max={non_zero_plvs.max():.4f}, "
                  f"mean={non_zero_plvs.mean():.4f}")
            print(f"      ➤ Strong edges (> {threshold}): {np.sum(strong_edges)} / {len(non_zero_plvs)}")
            print(f"      ➤ Edge density: {density:.4f}")

            features = graph_features(plv_matrix, threshold)
            if features:
                row = {
                    "file_id": file_id,
                    "trial": trial_idx,
                    "window_start": start,
                    "degree_centrality": features[0],
                    "betweenness": features[1],
                    "clustering": features[2],
                    "efficiency": features[3],
                    "edge_density": density,
                    "plv_mean": non_zero_plvs.mean(),
                }
                file_rows.append(row)

            G = build_graph_from_top_k(plv_matrix, top_k=0.05)
            if G is not None and len(G) > 1:
                features = graph_features(nx.to_numpy_array(G), threshold)
                if features:
                    row = {
                        "file_id": file_id,
                        "trial": trial_idx,
                        "window_start": start,
                        "degree_centrality": features[0],
                        "betweenness": features[1],
                        "clustering": features[2],
                        "efficiency": features[3],
                        "edge_density": density,
                        "plv_mean": non_zero_plvs.mean(),
                    }
                    file_rows.append(row)

    if file_rows:
        df = pd.DataFrame(file_rows)
        df.to_csv(os.path.join(output_dir, f"graph_{file_id}.csv"), index=False)
        print(f"\n✅ Saved: graph_{file_id}.csv")
    else:
        print(f"\n⚠️ No valid graph features for {file_id}")
