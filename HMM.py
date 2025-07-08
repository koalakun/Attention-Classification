
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# ------------------------- Step 1: Load Config and Data -------------------------
def load_config_and_data():
    with open("e:/intern/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    event_path = config["paths"]["event_csv"]
    graph_dir = config["paths"]["graph_features_csv_dir"]
    output_dir = "e:/intern/outputs"
    os.makedirs(output_dir, exist_ok=True)

    event_df = pd.read_csv(event_path)

    def normalize_file_key(file_id):
        match = re.search(r'(NDAR[A-Z0-9]+).*BLOCK(\d+)', file_id.upper())
        return f"{match.group(1)}_Block{match.group(2)}" if match else file_id

    event_df["file_key"] = event_df["file_id"].apply(normalize_file_key)
    event_df["relative_timestamp"] = event_df.groupby("file_key")["timestamp"].transform(lambda x: x - x.min())

    event_df_with_types = event_df.groupby("file_key").apply(
        lambda x: x[['relative_timestamp', 'event_type']].sort_values('relative_timestamp'),
        include_groups=False
    ).reset_index(drop=True)

    graph_dict = {}
    for fname in os.listdir(graph_dir):
        if fname.endswith("_graph.csv") and "all_files" not in fname.lower():
            df = pd.read_csv(os.path.join(graph_dir, fname))
            graph_dict[fname.replace("_graph.csv", "")] = df

    overlap_keys = sorted(set(event_df["file_key"].unique()) & set(graph_dict.keys()))
    return config, event_df, graph_dict, overlap_keys, output_dir

# ------------------------- Step 2: Extract Sequences -------------------------
def extract_sequences(event_df, graph_dict, overlap_keys):
    all_sequences, all_lengths, all_subject_ids = [], [], []

    for key in tqdm(overlap_keys):
        df = graph_dict[key]
        if df.empty or "t_start" not in df.columns:
            continue

        subject_events = event_df[event_df["file_key"] == key].sort_values("relative_timestamp")
        if len(subject_events) < 3:
            continue

        t_start = df["t_start"].values
        t_end = df["t_end"].values
        subject_sequence = []

        for _, event_row in subject_events.iterrows():
            ts = event_row['relative_timestamp']
            event_type = event_row['event_type']
            mask = (t_start <= ts) & (ts < t_end)
            if not np.any(mask):
                continue
            win_ids = df.loc[mask, "window_idx"].unique()
            if len(win_ids) == 0:
                continue
            wid = win_ids[0]
            win_df = df[df["window_idx"] == wid]
            if "centrality" not in win_df.columns or win_df.empty:
                continue
            values = win_df["centrality"].values
            if len(values) >= 5:
                features = np.array([
                    np.mean(values), np.std(values), np.max(values), np.min(values),
                    np.median(values), len(values),
                    1.0 if event_type == 'TargOnT' else 0.0,
                    1.0 if event_type == 'RespT' else 0.0,
                    ts, np.log(wid + 1)
                ])
                subject_sequence.append(features)

        if len(subject_sequence) >= 3:
            all_sequences.extend(subject_sequence)
            all_lengths.append(len(subject_sequence))
            all_subject_ids.append(key)

    return np.array(all_sequences), all_lengths, all_subject_ids

# ------------------------- Step 3: Train HMM Model -------------------------
def train_hmm_model(X, lengths, max_states=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = hmm.GaussianHMM(n_components=max_states, covariance_type="diag", n_iter=300, random_state=42)
    model.fit(X_scaled, lengths)
    hidden_states = model.predict(X_scaled, lengths)
    return model, hidden_states, scaler, max_states, X_scaled

# ------------------------- Step 4: Analyze and Save -------------------------
def analyze_subject_states(hidden_states, all_lengths, all_subject_ids, n_states, output_dir):
    start_idx = 0
    subject_summary = []

    for subject, seq_len in zip(all_subject_ids, all_lengths):
        end_idx = start_idx + seq_len
        subject_states = hidden_states[start_idx:end_idx]
        state_counts = np.bincount(subject_states, minlength=n_states)
        summary = {'file_key': subject}
        for i in range(n_states):
            summary[f'state_{i}'] = state_counts[i]
        subject_summary.append(summary)
        start_idx = end_idx

    summary_df = pd.DataFrame(subject_summary)
    summary_path = os.path.join(output_dir, "hmm_subject_summary_fixed.csv")
    summary_df.to_csv(summary_path, index=False)
    return summary_df

# ------------------------- Step 5: Merge with Behavioral RT -------------------------
def calculate_reaction_times(event_df):
    def extract_file_key(file_id):
        match = re.search(r"(NDAR[A-Z0-9]+).*BLOCK(\d+)", file_id.upper())
        return f"{match.group(1)}_Block{match.group(2)}" if match else file_id

    rt_data = []
    for file_id in event_df["file_id"].unique():
        fe = event_df[event_df["file_id"] == file_id].sort_values("timestamp")
        targs = fe[fe["event_type"] == "TargOnT"]
        resps = fe[fe["event_type"] == "RespT"]
        for _, targ in targs.iterrows():
            rt_candidates = resps[resps["timestamp"] > targ["timestamp"]]
            if not rt_candidates.empty:
                rt = rt_candidates.iloc[0]["timestamp"] - targ["timestamp"]
                if 0.05 <= rt <= 2.0:
                    rt_data.append({
                        "file_id": file_id,
                        "file_key": extract_file_key(file_id),
                        "reaction_time": rt,
                        "rt_label": "fast" if rt < 0.5 else "slow"
                    })
    return pd.DataFrame(rt_data)

def merge_with_behavioral(summary_df, event_df, output_dir):
    rt_df = calculate_reaction_times(event_df)
    rt_summary = rt_df.groupby("file_key").agg({
        "reaction_time": ["mean", "std", "count"],
        "rt_label": lambda x: (x == "fast").mean()
    }).reset_index()
    rt_summary.columns = ["file_key", "mean_rt", "std_rt", "trial_count", "fast_proportion"]
    merged = summary_df.merge(rt_summary, on="file_key", how="inner")
    merged.to_csv(os.path.join(output_dir, "hmm_rt_analysis_fixed.csv"), index=False)
    return merged

# ------------------------- Main -------------------------
if __name__ == "__main__":
    config, event_df, graph_dict, overlap_keys, output_dir = load_config_and_data()
    all_sequences, all_lengths, all_subject_ids = extract_sequences(event_df, graph_dict, overlap_keys)

    if len(all_sequences) == 0:
        print("❌ No sequences found.")
        exit()

    X = np.array(all_sequences)
    model, hidden_states, scaler, n_states, X_scaled = train_hmm_model(X, all_lengths)
    summary_df = analyze_subject_states(hidden_states, all_lengths, all_subject_ids, n_states, output_dir)
    np.save(os.path.join(output_dir, "hmm_input_features_fixed.npy"), X)
    np.save(os.path.join(output_dir, "hmm_state_sequences_fixed.npy"), hidden_states)
    np.save(os.path.join(output_dir, "hmm_subject_ids_fixed.npy"), np.array(all_subject_ids))
    merge_with_behavioral(summary_df, event_df, output_dir)
    print("✅ Full HMM + RT pipeline completed.")
