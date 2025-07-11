import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings

# Suppress HMM warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
                # Simplified feature vector - less prone to numerical issues
                features = np.array([
                    np.mean(values),
                    np.std(values) + 1e-6,  # Add small constant to prevent zero std
                    np.max(values),
                    np.min(values),
                    1.0 if event_type == 'TargOnT' else 0.0,
                    1.0 if event_type == 'RespT' else 0.0
                ])
                subject_sequence.append(features)

        if len(subject_sequence) >= 5:  # Increased minimum sequence length
            all_sequences.extend(subject_sequence)
            all_lengths.append(len(subject_sequence))
            all_subject_ids.append(key)

    return np.array(all_sequences), all_lengths, all_subject_ids

# ------------------------- Step 3: Train HMM Model -------------------------
def train_hmm_model(X, lengths, n_components=2):
    """
    Train HMM with better numerical stability
    """
    print(f"🔄 Training HMM with {n_components} states...")
    print(f"   Data shape: {X.shape}")
    print(f"   Sequences: {len(lengths)}")
    print(f"   Total observations: {sum(lengths)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add small noise to prevent numerical issues
    X_scaled += np.random.normal(0, 0.001, X_scaled.shape)
    
    # Try different initialization strategies
    for attempt in range(3):
        try:
            # Create model with more conservative settings
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=100,  # Reduced iterations
                tol=1e-3,    # Looser tolerance
                random_state=42 + attempt,
                init_params="stmc",  # Initialize all parameters
                params="stmc"        # Train all parameters
            )
            
            print(f"   Attempt {attempt + 1}: Fitting model...")
            model.fit(X_scaled, lengths)
            
            # Check if model is valid
            if np.isnan(model.startprob_).any() or np.isnan(model.transmat_).any():
                print(f"   ❌ Attempt {attempt + 1}: NaN values detected")
                continue
                
            if np.sum(model.startprob_) == 0 or np.any(np.sum(model.transmat_, axis=1) == 0):
                print(f"   ❌ Attempt {attempt + 1}: Zero probabilities detected")
                continue
            
            print(f"   ✅ Attempt {attempt + 1}: Model trained successfully")
            
            # Predict states
            hidden_states = model.predict(X_scaled, lengths)
            
            return model, hidden_states, scaler, n_components, X_scaled
            
        except Exception as e:
            print(f"   ❌ Attempt {attempt + 1}: {str(e)}")
            continue
    
    # If all attempts fail, try with fewer states
    if n_components > 2:
        print(f"   🔄 Retrying with {n_components-1} states...")
        return train_hmm_model(X, lengths, n_components-1)
    else:
        raise ValueError("❌ Could not train stable HMM model")

# ------------------------- Step 4: Analyze and Save -------------------------
def analyze_subject_states(hidden_states, all_lengths, all_subject_ids, n_states, output_dir):
    print(f"🔍 Analyzing {len(all_subject_ids)} subjects with {n_states} states...")
    
    start_idx = 0
    subject_summary = []

    for subject, seq_len in zip(all_subject_ids, all_lengths):
        end_idx = start_idx + seq_len
        subject_states = hidden_states[start_idx:end_idx]
        state_counts = np.bincount(subject_states, minlength=n_states)
        state_proportions = state_counts / seq_len
        
        summary = {'file_key': subject, 'sequence_length': seq_len}
        for i in range(n_states):
            summary[f'state_{i}_count'] = state_counts[i]
            summary[f'state_{i}_prop'] = state_proportions[i]
        
        # Add state transitions
        transitions = []
        for i in range(len(subject_states) - 1):
            transitions.append(f"{subject_states[i]}->{subject_states[i+1]}")
        summary['n_transitions'] = len(set(transitions))
        summary['most_common_state'] = np.argmax(state_counts)
        
        subject_summary.append(summary)
        start_idx = end_idx

    summary_df = pd.DataFrame(subject_summary)
    summary_path = os.path.join(output_dir, "hmm_subject_summary_fixed.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"   ✅ Saved summary: {summary_path}")
    print(f"   📊 State distribution across subjects:")
    for i in range(n_states):
        prop_col = f'state_{i}_prop'
        mean_prop = summary_df[prop_col].mean()
        print(f"      State {i}: {mean_prop:.3f} (±{summary_df[prop_col].std():.3f})")
    
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
    print("🔄 Calculating reaction times...")
    rt_df = calculate_reaction_times(event_df)
    
    if rt_df.empty:
        print("   ⚠️ No valid reaction times found")
        return summary_df
    
    rt_summary = rt_df.groupby("file_key").agg({
        "reaction_time": ["mean", "std", "count"],
        "rt_label": lambda x: (x == "fast").mean()
    }).reset_index()
    rt_summary.columns = ["file_key", "mean_rt", "std_rt", "trial_count", "fast_proportion"]
    
    merged = summary_df.merge(rt_summary, on="file_key", how="left")
    merged_path = os.path.join(output_dir, "hmm_rt_analysis_fixed.csv")
    merged.to_csv(merged_path, index=False)
    
    print(f"   ✅ Saved merged analysis: {merged_path}")
    print(f"   📊 Subjects with RT data: {len(rt_summary)}/{len(summary_df)}")
    
    return merged

# ------------------------- Main -------------------------
if __name__ == "__main__":
    print("🚀 Starting HMM Analysis Pipeline...")
    
    config, event_df, graph_dict, overlap_keys, output_dir = load_config_and_data()
    print(f"   📁 Found {len(overlap_keys)} files with both events and graph features")
    
    all_sequences, all_lengths, all_subject_ids = extract_sequences(event_df, graph_dict, overlap_keys)

    if len(all_sequences) == 0:
        print("❌ No sequences found.")
        exit()

    print(f"   ✅ Extracted sequences for {len(all_subject_ids)} subjects")
    
    X = np.array(all_sequences)
    
    # Try with 2 states first (more stable)
    try:
        model, hidden_states, scaler, n_states, X_scaled = train_hmm_model(X, all_lengths, n_components=2)
        
        # Save model artifacts
        summary_df = analyze_subject_states(hidden_states, all_lengths, all_subject_ids, n_states, output_dir)
        
        # Save raw data
        np.save(os.path.join(output_dir, "hmm_input_features_fixed.npy"), X)
        np.save(os.path.join(output_dir, "hmm_state_sequences_fixed.npy"), hidden_states)
        np.save(os.path.join(output_dir, "hmm_subject_ids_fixed.npy"), np.array(all_subject_ids))
        
        # Merge with behavioral data
        final_df = merge_with_behavioral(summary_df, event_df, output_dir)
        
        print("\n🎯 HMM Analysis Complete!")
        print(f"   📊 States identified: {n_states}")
        print(f"   👥 Subjects analyzed: {len(all_subject_ids)}")
        print(f"   📈 Total observations: {len(X)}")
        print(f"   💾 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ HMM training failed: {e}")
        print("   Try reducing the number of states or check data quality")
