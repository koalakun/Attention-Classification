import os
import numpy as np
import mne
from datetime import datetime
from ICA import load_mat_to_mne  # Adjust to your actual import
import glob

# ---------------- CONFIG ----------------
mat_path = r"C:\Users\user\Downloads\SAIIT\SAIIT\NDARCD401HGZ_Block1.mat"
sfp_path = r"E:\intern\GSN_HydroCel_129.sfp"
qa_log_dir = r"e:\intern\qa_logs"
ica_plot_dir = r"e:\intern\ica_plots"
features_dir = r"e:\intern\features"

# ---------------- STEP 1: Verify Channel Coordinates ----------------
def load_expected_coords(sfp_path):
    expected_coords = {}
    ordered_labels = []
    with open(sfp_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                ch_name = parts[0]
                coords = list(map(float, parts[1:]))
                expected_coords[ch_name] = coords
                ordered_labels.append(ch_name)
    return expected_coords, ordered_labels

raw, _, _, ica = load_mat_to_mne(mat_path, sfp_path=sfp_path)
actual_coords = {
    ch['ch_name']: ch['loc'][:3]
    for ch in raw.info['chs']
}
expected_coords, ordered_labels = load_expected_coords(sfp_path)

print(f"\n{'Channel':<8}  {'Expected':<40}  {'Actual':<40}  Match")
print("-" * 105)
for ch in ordered_labels:
    if ch not in actual_coords:
        print(f"{ch:<8}  {'Missing in EEG file':<40}")
        continue
    exp = np.array(expected_coords[ch])
    act = np.array(actual_coords[ch])
    match = np.allclose(exp, act, atol=1e-4)
    print(f"{ch:<8}  {str(np.round(exp, 4)):<40}  {str(np.round(act, 4)):<40}  {'✅' if match else '❌'}")

# ---------------- STEP 2: Check run_id timestamp in QA log ----------------
base_name = os.path.splitext(os.path.basename(mat_path))[0]
qa_logs = sorted(glob.glob(os.path.join(qa_log_dir, f"{base_name}_*.txt")))
if qa_logs:
    latest_log = qa_logs[-1]
    print(f"\n📝 Found QA Log: {os.path.basename(latest_log)}")
    with open(latest_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        run_line = [l for l in lines if "Run ID" in l]
        if run_line:
            print(f"   ✅ Run ID: {run_line[0].strip()}")
        else:
            print("   ❌ Run ID not found in QA log!")
else:
    print("❌ No QA log found.")

# ---------------- STEP 3: Check ICA plot saved with run_id ----------------
ica_plots = sorted(glob.glob(os.path.join(ica_plot_dir, f"{base_name}_*.png")))
if ica_plots:
    latest_ica = ica_plots[-1]
    print(f"\n📷 ICA plot found: {os.path.basename(latest_ica)}")
else:
    print("❌ ICA plot with timestamp not found!")

# ---------------- STEP 4: 
#  ----------------
if hasattr(ica, "exclude") and ica.exclude:
    print(f"\n🧠 ICA Exclusions: {ica.exclude} ✅")
else:
    print("❌ No ICA exclusions found!")

# ---------------- STEP 5 (Optional): Feature file existence ----------------
feat_channel = os.path.join(features_dir, f"channel_{base_name}.npy")
feat_source = os.path.join(features_dir, f"source_{base_name}.npy")

print("\n📦 Feature Files:")
print(f"  {'Exists' if os.path.exists(feat_channel) else 'Missing'} - channel_{base_name}.npy")
print(f"  {'Exists' if os.path.exists(feat_source) else 'Missing'} - source_{base_name}.npy")
# ---------------- STEP 6: Check for empty feature files ----------------
if os.path.exists(feat_source):
    source_data = np.load(feat_source)
    if source_data.size == 0:
        print("❌ Source feature file is empty!")
    else:
        print(f"✅ Source feature file loaded with shape: {source_data.shape}")

# ---------------- STEP 7: Load and concatenate all source feature files ----------------
feature_files = glob.glob(r"E:/intern/features/source_*.npy")

all_feats = []
for fpath in feature_files:
    data = np.load(fpath)
    all_feats.append(data)
    print(f"Loaded {fpath}: {data.shape}")

# Concatenate along the first axis (epochs)
X = np.vstack(all_feats)
print(f"✅ Concatenated feature matrix: {X.shape}")