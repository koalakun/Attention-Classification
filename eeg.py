import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from scipy.signal import welch, hilbert, coherence
from antropy import perm_entropy, higuchi_fd
import mne
import networkx as nx
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse
from tqdm import tqdm
import warnings
from mne import read_bem_solution
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns


# Suppress expected warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- Config ----------------------
folders = [
    r'C:\Users\user\Downloads\SAIIT\SAIIT',
    r'C:\Users\user\Downloads\SAIIT\__MACOSX\SAIIT'
]
tmin, tmax = -0.3, 0.0  # Pre-stimulus window
freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # delta to gamma

# Set your fsaverage label directory manually (disable auto-downloads)
subjects_dir = r"C:\Users\user\mne_data\MNE-fsaverage-data"
fs_dir = os.path.join(subjects_dir, 'fsaverage')
label_dir = os.path.join(fs_dir, 'label')
os.makedirs(label_dir, exist_ok=True)



# ---------------------- Event Extraction ----------------------
def extract_mne_events(result):
    events = []
    for i, evt in enumerate(result.event):
        sample = getattr(evt, 'sample', None)
        value = getattr(evt, 'value', None)
        evt_type = getattr(evt, 'type', None)
        try:
            latency = int(sample) if sample is not None else None
        except Exception:
            latency = None
        event_code = None
        for candidate in [value, evt_type]:
            try:
                if isinstance(candidate, (np.ndarray, list)):
                    candidate = candidate[0]
                event_code = int(candidate)
                break
            except Exception:
                continue
        if latency is not None and event_code is not None:
            events.append([latency, 0, event_code])
    print(f"Extracted {len(events)} usable events.")
    return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)

# ---------------------- Load .mat to MNE Raw ----------------------
def load_mat_to_mne(file_path):
    if os.path.basename(file_path).startswith("._"):
        print(f"⏭️  Skipping invalid file: {file_path}")
        return None, None, None
    print(f"\n>>> Loading {os.path.basename(file_path)}")
    try:
        mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None, None, None
    result = mat.get('result', None)
    if result is None:
        return None, None, None
    data = np.asarray(result.data)
    if data.shape[0] > data.shape[1]:
        data = data.T
    sfreq = int(result.srate)
    ch_names = [str(c.labels[0]) if isinstance(c.labels, (list, np.ndarray)) else str(c.labels) for c in result.chanlocs]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * len(ch_names))
    raw = mne.io.RawArray(data, info)

    # Load custom .sfp coordinates manually
    def load_sfp_coordinates(sfp_path):
        ch_names = []
        ch_pos = []
        with open(sfp_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                ch_name, x, y, z = parts
                ch_names.append(ch_name)
                ch_pos.append([float(x), float(y), float(z)])
        return ch_names, np.array(ch_pos)

    sfp_path = r"C:\Users\user\Downloads\GSN_HydroCel_129.sfp"
    sfp_names, sfp_locs = load_sfp_coordinates(sfp_path)
    montage_dict = dict(zip(sfp_names, sfp_locs))
    custom_montage = mne.channels.make_dig_montage(ch_pos=montage_dict, coord_frame='head')
    raw.set_montage(custom_montage, on_missing='ignore')

    # Set average reference
    raw.set_eeg_reference('average', projection=True)

    print(f"EEG shape: {data.shape}")
    print(f"Sampling rate: {sfreq}")
    print(f"Number of channels: {len(ch_names)}")
    print(f"First 5 channel labels: {ch_names[:5]}")
    return raw, result, sfreq


# ---------------------- Feature Extraction Functions ----------------------
def extract_time_features(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    var_val = np.var(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Calculate skewness and kurtosis
    skew_val = skew(data)
    kurt_val = kurtosis(data)
    
    # Calculate signal energy
    energy = np.sum(data ** 2)
    
    # Calculate zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2
    return [
        mean_val, std_val, var_val, min_val, max_val, 
        skew_val, kurt_val, energy, zero_crossings
    ]

def bandpower(data, sfreq, freq_bands):
    band_powers = []
    for ch in data:
        freqs, psd = welch(ch, sfreq, nperseg=256, noverlap=128)
        for fmin, fmax in freq_bands:
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
            band_powers.append(band_power)
    return band_powers

def relative_bandpower(data, sfreq, freq_bands):
    rel_band_powers = []
    for ch in data:
        freqs, psd = welch(ch, sfreq, nperseg=256, noverlap=128)
        total_power = np.trapezoid(psd, freqs)
        for fmin, fmax in freq_bands:
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
            rel_band_powers.append(band_power / total_power if total_power > 0 else 0)
    return rel_band_powers

def compute_plv(data):
    analytic_signal = hilbert(data)
    phase_info = np.angle(analytic_signal)
    plv = np.abs(np.mean(np.exp(1j * phase_info), axis=0))
    plv_matrix = np.abs(np.corrcoef(phase_info))
    return plv, plv_matrix

def compute_coherence_features(data, sfreq):
    f, Cxy = coherence(data[0], data[1], sfreq, nperseg=256, noverlap=128)
    coherence_features = [
        np.mean(Cxy),
        np.std(Cxy),
        np.max(Cxy),
        np.min(Cxy),
        np.sum(Cxy),
        np.trapezoid(Cxy, f)
    ]
    return coherence_features

def theta_beta_ratio(data, sfreq):
    features = []
    for ch in data:
        f, Pxx = welch(ch, sfreq)
        theta = np.trapezoid(Pxx[(f >= 4) & (f <= 8)], f[(f >= 4) & (f <= 8)])
        beta = np.trapezoid(Pxx[(f >= 13) & (f <= 30)], f[(f >= 13) & (f <= 30)])
        features.append(theta / beta if beta > 0 else 0)
    return features

def graph_metrics_from_plv(plv_matrix):
    G = nx.from_numpy_array(plv_matrix)
    metrics = [
        nx.degree_centrality(G),
        nx.betweenness_centrality(G),
        nx.closeness_centrality(G),
        nx.eigenvector_centrality(G)
    ]
    return [item for sublist in metrics for item in sublist.values()]

def compute_nonlinear_features(data):
    features = []
    for ch in data:
        try:
            pe = perm_entropy(ch, normalize=True)
        except Exception:
            pe = 0.0
        try:
            hfd = higuchi_fd(ch)
        except Exception:
            hfd = 0.0
        features.extend([pe, hfd])
    return features

# ---------------------- Main Processing Function ----------------------
def process_eeg_file(file_path):
    raw, result, sfreq = load_mat_to_mne(file_path)
    if raw is None:
        return None, None

    try:
        mapped = set(raw.get_montage().ch_names) & set(raw.info['ch_names'])
        if len(mapped) == 0:
            raise ValueError("No valid EEG positions matched with standard montage.")
    except Exception as e:
        print(f"Skipping source localization due to montage mismatch: {e}")
        source_features = None

    events = extract_mne_events(result)
    if len(events) == 0:
        print("No usable events.")
        return None, None

    event_id = {str(e): e for e in np.unique(events[:, 2]) if e != 999}
    try:
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=None, preload=True,
            event_repeated='drop'
        )
    except RuntimeError as e:
        if 'Event time samples were not unique' in str(e):
            print(f"Duplicate event timestamps detected in {os.path.basename(file_path)}. Dropping repeated events.")
            return None, None
        else:
            raise

    print(f"Extracting features from {len(epochs)} pre-stimulus epochs...")

    # --------- Channel-level features ---------
    channel_features = []
    for data in epochs.get_data():
        feats = []
        # Time features per channel
        for ch in data:
            feats += list(np.ravel(extract_time_features(ch)))
        feats += list(np.ravel(bandpower(data, sfreq, freq_bands)))
        feats += list(np.ravel(relative_bandpower(data, sfreq, freq_bands)))
        plv_vector, plv_matrix = compute_plv(data)
        feats += list(np.ravel(plv_vector))
        feats += list(np.ravel(compute_coherence_features(data, sfreq)))
        feats += list(np.ravel(compute_nonlinear_features(data)))
        feats += list(np.ravel(theta_beta_ratio(data, sfreq)))
        feats += list(np.ravel(graph_metrics_from_plv(plv_matrix)))
        try:
            feats = np.array([float(x) for x in np.ravel(feats)])
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print("Feature flattening failed. Check this epoch's shape.")
            print("Feature preview (truncated):", feats[:5])
            raise e
        if np.all(np.isnan(feats)) or np.all(np.isinf(feats)):
            print(f"Skipping one epoch due to invalid features (NaN/Inf).")
            continue
        channel_features.append(feats)
    channel_features = np.array(channel_features)

    # --------- Source-level features ---------
    source_features = None
    if len(mapped) > 0:
        try:
            import nibabel
            raw.filter(1., 40., fir_design='firwin')
            noise_cov = mne.compute_covariance(epochs, tmax=0.0)
            fs_dir = os.path.join(subjects_dir, 'fsaverage')
            # DO NOT overwrite subjects_dir here!
            src = mne.setup_source_space(
                subject='fsaverage',
                spacing='oct6',
                subjects_dir=subjects_dir,
                add_dist=False
            )
            bem_path = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
            bem = read_bem_solution(bem_path)
            trans = 'fsaverage'
            fwd = mne.make_forward_solution(
                raw.info, trans=trans, src=src, bem=bem,
                eeg=True, mindist=5.0, n_jobs=1
            )
            inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
            snr = 3.0
            lambda2 = 1.0 / snr ** 2
            method = 'dSPM'
            source_features = []
            for i, epoch in enumerate(epochs.get_data()):
                evoked = mne.EvokedArray(epoch, raw.info, tmin=epochs.times[0])
                stc = apply_inverse(evoked, inverse_operator, lambda2, method=method)
                # Only load labels once per process
                if not hasattr(process_eeg_file, "labels"):
                    print("Using subjects_dir:", subjects_dir)
                    lh_annot = os.path.join(subjects_dir, 'fsaverage', 'label', 'lh.Schaefer2018_100Parcels_7Networks.annot')
                    rh_annot = os.path.join(subjects_dir, 'fsaverage', 'label', 'rh.Schaefer2018_100Parcels_7Networks.annot')

                    if not os.path.isfile(lh_annot) or not os.path.isfile(rh_annot):
                        raise FileNotFoundError("Schaefer annotation files not found in label folder. Please check paths.")

                    labels_lh = mne.read_labels_from_annot('fsaverage', 'Schaefer2018_100Parcels_7Networks', hemi='lh', subjects_dir=subjects_dir)
                    labels_rh = mne.read_labels_from_annot('fsaverage', 'Schaefer2018_100Parcels_7Networks', hemi='rh', subjects_dir=subjects_dir)
                    process_eeg_file.labels = labels_lh + labels_rh
                labels = process_eeg_file.labels
                parcel_ts = stc.extract_label_time_course(labels, src, mode='mean')
                source_feats = np.concatenate([np.mean(parcel_ts, axis=1), np.std(parcel_ts, axis=1)])
                source_feats = np.nan_to_num(source_feats, nan=0.0, posinf=0.0, neginf=0.0)
                source_features.append(source_feats)
            source_features = np.array(source_features)
        except ImportError:
            print("⚠️ nibabel is required for source localization. Run: pip install nibabel")
            source_features = None
        except Exception as e:
            print(f"⚠️ Source localization failed: {e}")
            source_features = None

    return channel_features, source_features

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "features")
    os.makedirs(save_dir, exist_ok=True)

    mat_files = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(folder, file))

    all_channel_feats = []
    all_source_feats = []

    for file_path in tqdm(mat_files, desc="Processing files"):
        ch_feats, src_feats = process_eeg_file(file_path)
        
        if ch_feats is not None and len(ch_feats) > 0:
            all_channel_feats.append(ch_feats)
            print(f"\n✅ Channel-level feature matrix shape: {ch_feats.shape}")
            print("🧠 First channel-level feature vector (truncated):")
            print(ch_feats[0][:10])
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            np.save(os.path.join(save_dir, f"channel_{base_name}.npy"), ch_feats)

        if src_feats is not None and len(src_feats) > 0:
            all_source_feats.append(src_feats)
            print(f"\n✅ Source-level feature matrix shape: {src_feats.shape}")
            print("🧠 First source-level feature vector (truncated):")
            print(src_feats[0][:10])
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            np.save(os.path.join(save_dir, f"source_{base_name}.npy"), src_feats)

    if all_channel_feats:
        all_ch = np.concatenate(all_channel_feats, axis=0)
        all_ch = np.nan_to_num(all_ch)
        ch_save_path = os.path.join(os.path.dirname(__file__), f"pre_stim_features_channel_{len(all_ch)}.npy")
        np.save(ch_save_path, all_ch)
        print(f"\nSaved all channel-level features: {all_ch.shape} to {ch_save_path}")

    if all_source_feats:
        all_src = np.concatenate(all_source_feats, axis=0)
        all_src = np.nan_to_num(all_src)
        src_save_path = os.path.join(os.path.dirname(__file__), f"pre_stim_features_source_{len(all_src)}.npy")
        np.save(src_save_path, all_src)
        print(f"\nSaved all source-level features: {all_src.shape} to {src_save_path}")
