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
import matplotlib.pyplot as plt
from mne.preprocessing import compute_proj_eog, ICA, annotate_amplitude
from mne.utils import use_log_level
import yaml
from datetime import datetime




# Suppress expected warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- Config ----------------------
# Load configuration
try:
    with open("e:/intern/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load config.yaml: {e}")

# Access paths and parameters from config
folders = config["paths"]["raw_mat_dirs"]
fif_dir = config["paths"]["cleaned_data_dir"]
ica_plot_dir = config["paths"]["ica_plot_dir"]
features_dir = config["paths"]["features_dir"]
erp_plot_dir = config["paths"]["erp_plot_dir"]
epochs_dir = config["paths"]["epochs_dir"]
qa_log_dir = config["paths"]["qa_log_dir"]
os.makedirs(qa_log_dir, exist_ok=True)

l_freq = config["preprocessing"]["bandpass_filter"]["l_freq"]
h_freq = config["preprocessing"]["bandpass_filter"]["h_freq"]

n_components = config["preprocessing"]["ica"]["n_components"]
random_state = config["preprocessing"]["ica"]["random_state"]
max_iter = config["preprocessing"]["ica"]["max_iter"]

tmin = config["epoching"]["tmin"]
tmax = config["epoching"]["tmax"]
baseline = config["epoching"]["baseline"]

sfp_path = config["preprocessing"]["sfp_path"]  # <-- Add this line


freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # delta to gamma

# Set your fsaverage label directory manually (disable auto-downloads)
subjects_dir = r"C:\Users\user\mne_data\MNE-fsaverage-data"
fs_dir = os.path.join(subjects_dir, 'fsaverage')
label_dir = os.path.join(fs_dir, 'label')
os.makedirs(label_dir, exist_ok=True)

os.makedirs(ica_plot_dir, exist_ok=True)

# ---------------------- Event Extraction ----------------------
def extract_mne_events(result):
    events = []

    # Check if 'event' exists
    if not hasattr(result, 'event'):
        print("⚠️ No 'event' found in this .mat result")
        return np.empty((0, 3), dtype=int)

    print(f"🔍 Found {len(result.event)} events in result")

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

    print(f"✅ Extracted {len(events)} usable events.\n")
    return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)

# ---------------------- Load .mat to MNE Raw ----------------------
def load_mat_to_mne(file_path, apply_avg_ref=True, apply_proj=True, ica_plot_dir=None, sfp_path=None, return_ica=False):
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

    # Load custom .sfp coordinates from YAML-defined path
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
        return dict(zip(ch_names, ch_pos))

    # Use YAML-defined path
    sfp_coords = load_sfp_coordinates(sfp_path)

    # Filter SFP montage to only include existing raw channels
    filtered_montage = {name: pos for name, pos in sfp_coords.items() if name in raw.ch_names}

    # Create and apply the custom montage
    custom_montage = mne.channels.make_dig_montage(ch_pos=filtered_montage, coord_frame='head')
    raw.set_montage(custom_montage, on_missing='ignore')

    # 1. Apply bandpass filter (1–40 Hz) early for all processing
    raw.filter(l_freq, h_freq, fir_design='firwin')

    # 2. Set average EEG reference only if requested
    if apply_avg_ref:
        raw.set_eeg_reference('average', projection=True)
        if apply_proj:
            try:
                raw.apply_proj()
            except AttributeError:
                print("⚠️ Projection application failed: 'proj_applied' attribute not found.")




    # 3. ICA artifact removal
    ica = None
    try:
        print("🧠 Applying MNE ICA with muscle component detection...")
        
        # Define filename early for use throughout
        ica_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Step 2: Fit MNE ICA (using infomax for better muscle detection)
        ica = ICA(n_components=20, method="infomax", random_state=97, max_iter=800)
        ica.fit(raw)
        print(f"✅ ICA fitted with {ica.n_components_} components")
        
        # Step 3: Plot ICA Components (Fixed visualization)
        if ica_plot_dir is not None:
            try:
                # Create enhanced component plot (removed invalid outlines parameter)
                fig = ica.plot_components(colorbar=True, show=False)
                save_path = os.path.join(ica_plot_dir, f"{ica_filename}_ica_components.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"💾 Saved enhanced ICA component plot: {save_path}")
            except Exception as e:
                print(f"⚠️ Could not save ICA plot: {e}")
        
        # Step 4: Detect muscle components using PSD analysis
        muscle_ics = []
        eog_ics = []
        ecg_ics = []

        #  Muscle detection (20-60 Hz power) 
        try:
            # 🔧 FIX 1: Use clean, fresh ICA sources (advisor's key insight)
            print("🔍 Analyzing ICA components for muscle-band PSD...")
            ica_sources = ica.get_sources(raw.copy().load_data())  # Fresh copy!
            
            for i in range(ica.n_components_):
                # Get component time series from clean sources
                component_ts = ica_sources.get_data()[i]
                
                # Compute PSD for muscle detection (20-60 Hz)
                freqs, psd = welch(component_ts, fs=raw.info['sfreq'], nperseg=256)
                
                # Calculate power in muscle frequency band (20-60 Hz)
                muscle_band = (freqs >= 20) & (freqs <= 60)
                total_band = (freqs >= 1) & (freqs <= 40)
                
                muscle_power = np.mean(psd[muscle_band])
                total_power = np.mean(psd[total_band])
                
                # Muscle detection criterion
                muscle_ratio = muscle_power / total_power if total_power > 0 else 0
                
                #  Add debug printing 
                print(f"Component {i}: muscle_ratio = {muscle_ratio:.3f}")
                
                # Apply threshold
                if muscle_ratio > 0.5:  
                    muscle_ics.append(i)
                    
            print(f"⚡ Muscle-related components detected: {muscle_ics}")
            
        except Exception as e:
            print(f"⚠️ Muscle detection failed: {e}")
        
        # muscle PSD plot
        if muscle_ics and ica_plot_dir is not None:
            try:
                from mne.time_frequency import psd_array_welch  # ✅ safer import
                ica_sources = ica.get_sources(raw).get_data()

                # Compute PSDs for muscle components
                psds = []
                freqs = None
                for idx in muscle_ics:
                    psd, freqs = psd_array_welch(
                        ica_sources[idx],
                        sfreq=raw.info['sfreq'],
                        fmin=1, fmax=60,
                        n_fft=256
                    )
                    psds.append(psd)

                # Plot muscle-related component PSDs
                plt.figure(figsize=(10, 6))
                for i, idx in enumerate(muscle_ics):
                    plt.semilogy(freqs, psds[i], label=f"ICA{idx:02d}", linewidth=2)

                plt.xlabel("Frequency (Hz)", fontsize=12)
                plt.ylabel("Power Spectral Density (μV²/Hz)", fontsize=12)
                plt.title("PSD of Muscle-Related ICA Components", fontsize=14, fontweight='bold')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                muscle_psd_path = os.path.join(ica_plot_dir, f"{ica_filename}_muscle_psd.png")
                plt.savefig(muscle_psd_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"💾 Saved muscle PSD plot: {muscle_psd_path}")

            except Exception as e:
                print(f"⚠️ Could not save muscle PSD plot: {e}")
        
        # EOG detection (robust version with fallback and logging)
        try:
            eog_channels = ['Fp1', 'Fp2', 'AF3', 'AF4', 'AFz', 'Fpz']
            proxy_ch = next((ch for ch in eog_channels if ch in raw.ch_names), None)

            if proxy_ch:
                eog_ics, _ = ica.find_bads_eog(raw, ch_name=proxy_ch, threshold=2.0)
                print(f"👁️ EOG-like components detected using {proxy_ch}: {eog_ics}")
            else:
                eog_ics = []
                print("⚠️ No suitable EOG proxy channel found in EEG. EOG detection skipped.")
                print(f"Available channels: {raw.ch_names[:10]}...")  # Debug info
        except Exception as e:
            print(f"⚠️ EOG detection failed: {e}")
            eog_ics = []
        
        # ECG detection
        try:
            ecg_ics, _ = ica.find_bads_ecg(raw, threshold=2.0)
            print(f"❤️ ECG-like components detected: {ecg_ics}")
        except Exception as e:
            print(f"⚠️ ECG detection failed: {e}")
        
        # Kurtosis-based artifact detection
        try:
            ica_data = ica.get_sources(raw).get_data()
            kurtosis_scores = [kurtosis(component) for component in ica_data]
            kurtosis_ics = [i for i, kurt in enumerate(kurtosis_scores) if abs(kurt) > 4]
            print(f"📊 High-kurtosis components detected: {kurtosis_ics}")
            print(f"📊 Kurtosis scores: {np.round(kurtosis_scores, 2)}")
        except Exception as e:
            print(f"⚠️ Kurtosis detection failed: {e}")
            kurtosis_ics = []
        
        # Combine all artifact components
        artifact_ics = list(set(muscle_ics + eog_ics + ecg_ics + kurtosis_ics))

        # 🧠 NEW: Map exclusion reasons for each ICA component
        exclude_reason = {}
        for ic in muscle_ics:
            exclude_reason[ic] = "muscle"
        for ic in eog_ics:
            exclude_reason[ic] = "EOG"
        for ic in ecg_ics:
            exclude_reason[ic] = "ECG"
        for ic in kurtosis_ics:
            exclude_reason[ic] = "kurtosis"

        # Step 5: Apply ICA removal
        if artifact_ics:
            print(f"🗑️ Excluding {len(artifact_ics)} artifact components: {artifact_ics}")
            ica.exclude = artifact_ics
            
            # Plot excluded components for verification
            if ica_plot_dir is not None and len(artifact_ics) > 0:
                try:
                    fig = ica.plot_components(picks=artifact_ics, colorbar=True, show=False)
                    excluded_path = os.path.join(ica_plot_dir, f"{ica_filename}_ica_excluded.png")
                    fig.savefig(excluded_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"💾 Saved excluded components plot: {excluded_path}")
                except Exception as e:
                    print(f"⚠️ Could not save excluded components plot: {e}")
            
            # Apply ICA cleaning
            raw = ica.apply(raw, exclude=artifact_ics)
            print(f"✅ Applied ICA: removed {len(artifact_ics)} artifact components")
            
            # Generate artifact removal summary
            summary = {
                'muscle': muscle_ics,
                'eog': eog_ics, 
                'ecg': ecg_ics,
                'kurtosis': kurtosis_ics,
                'total_excluded': artifact_ics
            }
            print(f"📋 Artifact removal summary: {summary}")
            
        else:
            print("⚠️ No artifact components detected. Data unchanged.")
        
        # 3. Advanced HTML Report with MNE Report
        if ica_plot_dir is not None:
            try:
                from mne import Report
                
                # Create comprehensive HTML report
                report = Report(title=f"ICA Report: {ica_filename}")
                
                # Create title with exclusion reasons
                if artifact_ics:
                    excluded_with_reason = ", ".join(
                        f"IC{ic:02d} ({exclude_reason.get(ic, 'unknown')})"
                        for ic in sorted(artifact_ics)
                    )
                    ica_title = f"ICA Components (Excluded: {excluded_with_reason})"
                else:
                    ica_title = "ICA Components (No exclusions)"
                
                # 🔍 Add detailed text summary BEFORE the ICA plots
                if artifact_ics:
                    exclusion_text = "<b>🗑️ Excluded ICA Components with Detailed Reasons:</b><br><br>"
                    
                    # Group by artifact type for better organization
                    artifact_groups = {
                        'muscle': [ic for ic in artifact_ics if exclude_reason.get(ic) == 'muscle'],
                        'EOG': [ic for ic in artifact_ics if exclude_reason.get(ic) == 'EOG'],
                        'ECG': [ic for ic in artifact_ics if exclude_reason.get(ic) == 'ECG'],
                        'kurtosis': [ic for ic in artifact_ics if exclude_reason.get(ic) == 'kurtosis']
                    }
                    
                    for artifact_type, components in artifact_groups.items():
                        if components:
                            exclusion_text += f"<b>🔹 {artifact_type.upper()} artifacts:</b> "
                            exclusion_text += ", ".join([f"IC{ic:03d}" for ic in sorted(components)])
                            exclusion_text += "<br>"
                        else:
                            exclusion_text += f"<b>🔹 {artifact_type.upper()} artifacts:</b> None detected<br>"

                    exclusion_text += f"<br><b>Total excluded:</b> {len(artifact_ics)} out of {ica.n_components_} components<br>"
                    exclusion_text += f"<b>Clean components:</b> {ica.n_components_ - len(artifact_ics)} components retained"
                    

                    report.add_html(
                        exclusion_text,
                        title="🔍 Exclusion Summary"
                    )
                else:
                    report.add_html(
                        "<b>No artifacts detected!</b><br><br>"
                        "All ICA components passed quality control checks:<br>"
                        "• No muscle artifacts (high-frequency power < 50%)<br>"
                        "• No eye movement artifacts detected<br>"
                        "• No cardiac artifacts detected<br>"
                        "• All components have normal kurtosis values<br><br>"
                        f"<b>📊 All {ica.n_components_} components retained for analysis.</b>",
                        title="Quality Control Results"
                    )

                # Add ICA components to report with enhanced title
                report.add_ica(
                    ica=ica,
                    title=ica_title,
                    inst=raw,
                    n_jobs=1
                )
                
                # For processing summary
                processing_summary = f"""
                <b>📋 Processing Summary for {ica_filename}:</b><br><br>
                <b>🔧 Preprocessing:</b><br>
                • Bandpass filter: {l_freq}–{h_freq} Hz<br>
                • Reference: Average EEG reference<br>
                • ICA method: {ica.method} with {ica.n_components_} components<br><br>
                <b>🎯 Artifact Detection Results:</b><br>
                • Muscle components: {len(muscle_ics)} detected<br>
                • EOG components: {len(eog_ics)} detected<br>
                • ECG components: {len(ecg_ics)} detected<br>
                • High-kurtosis components: {len(kurtosis_ics)} detected<br><br>
                <b>⚡ Final outcome:</b> {len(artifact_ics)} components excluded, {ica.n_components_ - len(artifact_ics)} retained
                """
                
                report.add_html(
                    processing_summary,
                    title="📋 Processing Details"
                )
                
                # Save HTML report
                html_path = os.path.join(ica_plot_dir, f"{ica_filename}_ica_report.html")
                report.save(html_path, overwrite=True, open_browser=False)
                print(f"💾 Saved comprehensive HTML ICA report: {html_path}")
                
                # Save ICA solution
                ica_save_path = os.path.join(ica_plot_dir, f"{ica_filename}_ica_solution.fif")
                ica.save(ica_save_path, overwrite=True)
                print(f"💾 Saved ICA solution: {ica_save_path}")
                
            except Exception as e:
                print(f"⚠️ Could not save ICA report: {e}")
            
    except Exception as e:
        print(f"⚠️ MNE ICA processing failed: {e}")
        import traceback
        traceback.print_exc()
        ica = None

    # Optionally reset any existing bads
    raw.info['bads'] = []

    # Amplitude-based annotation (replace the old auto-bad detection block)
    annotations, _ = annotate_amplitude(raw, peak=dict(eeg=100e-6))
    raw.set_annotations(annotations)
    print(f"Amplitude-based annotation: {len(annotations)} segments marked as bad.")

    print(f"EEG shape: {data.shape}")
    print(f"Sampling rate: {sfreq}")
    print(f"Number of channels: {len(ch_names)}")
    print(f"First 5 channel labels: {ch_names[:5]}")

    # Save cleaned raw data for reuse (e.g., ERP checking)
    return raw, result, sfreq, ica


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
def save_qa_log(log_lines, qa_log_dir, base_name, run_id=None):
    # Always include Run ID at the top of the log if provided
    if run_id:
        if not any(f"Run ID:" in line for line in log_lines):
            log_lines.insert(0, f"Run ID: {run_id}")
        log_file_path = os.path.join(qa_log_dir, f"{base_name}_qa_{run_id}.txt")
    else:
        log_file_path = os.path.join(qa_log_dir, f"{base_name}_qa.txt")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"📄 QA log saved to {log_file_path}")

def save_cleaned_raw(raw, cleaned_dir, base_name):
    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(cleaned_dir, f"{base_name}_cleaned_raw.fif")
    try:
        raw.save(cleaned_path, overwrite=True)
        print(f"✅ Saved cleaned EEG data to {cleaned_path}")
    except Exception as e:
        print(f"❌ Failed to save cleaned EEG: {e}")

def extract_channel_features(epochs, sfreq, freq_bands):
    channel_features = []
    for data in epochs.get_data():
        feats = []
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
    return np.array(channel_features)

def extract_source_features(epochs_src, raw_src, mapped, subjects_dir):
    source_features = None
    if raw_src is not None and len(mapped) > 0:
        try:
            import nibabel
            noise_cov = mne.compute_covariance(epochs_src, tmax=0.0)
            fs_dir = os.path.join(subjects_dir, 'fsaverage')
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
                raw_src.info, trans=trans, src=src, bem=bem,
                eeg=True, mindist=5.0, n_jobs=1
            )
            inverse_operator = make_inverse_operator(raw_src.info, fwd, noise_cov, loose=0.2, depth=0.8)
            snr = 3.0
            lambda2 = 1.0 / snr ** 2
            method = 'dSPM'
            source_features = []
            for i, epoch in enumerate(epochs_src.get_data()):
                evoked = mne.EvokedArray(epoch, raw_src.info, tmin=epochs_src.times[0])
                with use_log_level('WARNING'):
                    stc = apply_inverse(evoked, inverse_operator, lambda2, method=method)
                    if not hasattr(extract_source_features, "labels"):
                        lh_annot = os.path.join(subjects_dir, 'fsaverage', 'label', 'lh.Schaefer2018_100Parcels_7Networks.annot')
                        rh_annot = os.path.join(subjects_dir, 'fsaverage', 'label', 'rh.Schaefer2018_100Parcels_7Networks.annot')
                        if not os.path.isfile(lh_annot) or not os.path.isfile(rh_annot):
                            raise FileNotFoundError("Schaefer annotation files not found in label folder. Please check paths.")
                        labels_lh = mne.read_labels_from_annot('fsaverage', 'Schaefer2018_100Parcels_7Networks', hemi='lh', subjects_dir=subjects_dir)
                        labels_rh = mne.read_labels_from_annot('fsaverage', 'Schaefer2018_100Parcels_7Networks', hemi='rh', subjects_dir=subjects_dir)
                        extract_source_features.labels = labels_lh + labels_rh
                    labels = extract_source_features.labels
                    parcel_ts = stc.extract_label_time_course(labels, src, mode='mean')
                # Save full time series for this epoch group (all epochs, all parcels, all times)
                ts_save_path = os.path.join(features_dir, f"source_{base_name}.npy")
                if not os.path.exists(ts_save_path):
                    np.save(ts_save_path, parcel_ts)
                    print(f"💾 Saved parcel time series to: {ts_save_path}")

                # Also compute summary features
                source_feats = np.concatenate([np.mean(parcel_ts, axis=1), np.std(parcel_ts, axis=1)])
                source_feats = np.nan_to_num(source_feats, nan=0.0, posinf=0.0, neginf=0.0)
                source_features.append(source_feats)

                # Accumulate parcel time series across epochs
                if not hasattr(extract_source_features, "ts_buffer"):
                    extract_source_features.ts_buffer = []

                extract_source_features.ts_buffer.append(parcel_ts)

                # After all epochs, save combined time series
                if i == len(epochs_src.get_data()) - 1:  # last epoch
                    full_ts = np.concatenate(extract_source_features.ts_buffer, axis=1)  # (n_parcels, total_time)
                    ts_save_path = os.path.join(features_dir, f"source_{base_name}.npy")
                    np.save(ts_save_path, full_ts)
                    print(f"💾 Saved full parcel time series to: {ts_save_path}")
                    del extract_source_features.ts_buffer
            source_features = np.array(source_features)
        except ImportError:
            print("⚠️ nibabel is required for source localization. Run: pip install nibabel")
            source_features = None
        except Exception as e:
            print(f"⚠️ Source localization failed: {e}")
            source_features = None
    return source_features

def process_eeg_file(file_path, run_id):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    log_lines = [f"\n📄 QA Log for {base_name}"]

    # Channel-level: average reference
    raw_chan, result, sfreq, ica = load_mat_to_mne(
        file_path, apply_avg_ref=True, apply_proj=True, ica_plot_dir=ica_plot_dir, sfp_path=sfp_path
    )
    save_cleaned_raw(raw_chan, fif_dir, base_name)
    if raw_chan is None:
        log_lines.append(f"❌ Failed to load file: {file_path}")
        save_qa_log(log_lines, qa_log_dir, base_name, run_id)
        return None, None

    log_lines.append(f"✅ Loaded file: {file_path}")
    log_lines.append(f"➡️ Filtered {l_freq}–{h_freq} Hz, {raw_chan.info['nchan']} channels")

    try:
        mapped = set(raw_chan.get_montage().ch_names) & set(raw_chan.info['ch_names'])
        if len(mapped) == 0:
            raise ValueError("No valid EEG positions matched with standard montage.")
    except Exception as e:
        log_lines.append(f"❌ Montage mismatch: {e}")
        print(f"Skipping source localization due to montage mismatch: {e}")
        mapped = set() 
        source_features = None

    events = extract_mne_events(result)
    if len(events) == 0:
        log_lines.append("❌ No usable events")
        save_qa_log(log_lines, qa_log_dir, base_name, run_id)
        return None, None

    event_id = {str(e): e for e in np.unique(events[:, 2]) if e != 999}
    try:
        epochs = mne.Epochs(
            raw_chan, events, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=baseline, preload=True,
            event_repeated='drop',
            reject_by_annotation=True
        )
    except RuntimeError as e:
        if 'Event time samples were not unique' in str(e):
            log_lines.append("❌ Duplicate event timestamps detected. Dropping repeated events.")
            print(f"Duplicate event timestamps detected in {os.path.basename(file_path)}. Dropping repeated events.")
            save_qa_log(log_lines, qa_log_dir, base_name, run_id)
            return None, None
        else:
            log_lines.append(f"⚠️ Epoching failed: {str(e)}")
            raise

    os.makedirs(epochs_dir, exist_ok=True)
    epochs_path = os.path.join(epochs_dir, f"{base_name}_epo.fif")
    try:
        epochs.save(epochs_path, overwrite=True)
        log_lines.append(f"✅ Saved epochs to: {epochs_path}")
        log_lines.append(f"🧠 Extracted {len(epochs)} valid epochs")
    except Exception as e:
        log_lines.append(f"❌ Failed to save epochs: {e}")

    print(f"Extracting features from {len(epochs)} pre-stimulus epochs...")

    # --------- Channel-level features ---------
    channel_features = extract_channel_features(epochs, sfreq, freq_bands)

    # Check for consistent feature shapes
    feature_shapes = [feats.shape for feats in channel_features]
    if len(set(feature_shapes)) > 1:
        print(f"⚠️ Feature vector shapes vary in {base_name}: {set(feature_shapes)}")

    # --------- Source-level features ---------
    source_features = extract_source_features(
        epochs, raw_chan, mapped, subjects_dir
    )

    save_qa_log(log_lines, qa_log_dir, base_name, run_id)
    return channel_features, source_features

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    save_dir = features_dir
    os.makedirs(save_dir, exist_ok=True)

    # Add run_id for this execution
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ch_feats, src_feats = process_eeg_file(file_path, run_id)
        
        if ch_feats is not None and len(ch_feats) > 0:
            all_channel_feats.append(ch_feats)
            print(f"\n✅ Channel-level feature matrix shape: {ch_feats.shape}")
            print("🧠 First channel-level feature vector (truncated):")
            print(ch_feats[0][:10])
            # Consistent naming for per-file channel features
            np.save(os.path.join(save_dir, f"pre_stim_features_channel_{base_name}.npy"), ch_feats)

        if src_feats is not None and len(src_feats) > 0:
            all_source_feats.append(src_feats)
            print(f"\n✅ Source-level feature matrix shape: {src_feats.shape}")
            print("🧠 First source-level feature vector (truncated):")
            print(src_feats[0][:10])
            # Consistent naming for per-file source features
            np.save(os.path.join(save_dir, f"pre_stim_features_source_{base_name}.npy"), src_feats)

    if all_channel_feats:
        all_ch = np.concatenate(all_channel_feats, axis=0)
        all_ch = np.nan_to_num(all_ch)
        ch_save_path = os.path.join(save_dir, f"pre_stim_features_channel_{len(all_ch)}.npy")
        np.save(ch_save_path, all_ch)
        print(f"\nSaved all channel-level features: {all_ch.shape} to {ch_save_path}")

    if all_source_feats:
        all_src = np.concatenate(all_source_feats, axis=0)
        all_src = np.nan_to_num(all_src)
        src_save_path = os.path.join(save_dir, f"pre_stim_features_source_{len(all_src)}.npy")
        np.save(src_save_path, all_src)
        print(f"\nSaved all source-level features: {all_src.shape} to {src_save_path}")

    print(f"\n🎉 Processing complete! Processed {len(mat_files)} files.")

# ✅ Script ends here - no more code needed!

