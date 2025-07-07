import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
import yaml
from mne.time_frequency import tfr_multitaper

# ---------------- Config ----------------
with open("e:/intern/config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

epochs_dir = config["paths"]["epochs_dir"]
erp_plot_dir = config["paths"]["erp_plot_dir"]
erp_viz_dir = config["paths"]["erp_visualizations"]
os.makedirs(erp_plot_dir, exist_ok=True)
os.makedirs(erp_viz_dir, exist_ok=True)

# ---------------- Helper Functions ----------------
def compute_plv_matrix(data):
    n_channels = data.shape[0]
    analytic_signal = hilbert(data)
    phases = np.angle(analytic_signal)
    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = phases[i] - phases[j]
            plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv_matrix

# ---------------- Script ----------------
for file in os.listdir(epochs_dir):
    if not file.endswith("_epo.fif"):
        continue

    epo_path = os.path.join(epochs_dir, file)
    block_id = os.path.splitext(file)[0].replace('_epo', '')
    print(f"\n>>> Processing {block_id}")

    try:
        epochs = mne.read_epochs(epo_path, preload=True)
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")
        continue

    if len(epochs.event_id) < 1:
        print(f"⚠️ No events in {file}, skipping.")
        continue

    # ----------- ERP Overlay (Butterfly Plot from Averaged ERP) -----------
    try:
        evoked = epochs.average()
        fig = evoked.plot(spatial_colors=True, gfp=True, show=False, titles=f"ERP Overlay – {block_id}")
        save_path = os.path.join(erp_plot_dir, f"{block_id}_erp_overlay.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved ERP overlay (butterfly) plot: {save_path}")
    except Exception as e:
        print(f"❌ ERP overlay failed: {e}")

    # ----------- ERP by Event Type (Condition Overlays) -----------
    event_labels = config.get("event_labels", {})
    evokeds = {}
    for cond in epochs.event_id:
        cond_epochs = epochs[cond]
        if len(cond_epochs) == 0:
            continue
        # Use meaningful label instead of generic "Event X"
        label = event_labels.get(int(cond), f"Event {cond}")
        evokeds[label] = cond_epochs.average()

    if len(evokeds) >= 2:
        try:
            picks = 'Pz' if 'Pz' in epochs.ch_names else 'eeg'
            fig = mne.viz.plot_compare_evokeds(evokeds, picks=picks, ci=True, show=False)
            fig[0].suptitle(f"ERP Overlay – {block_id}")
            save_path = os.path.join(erp_plot_dir, f"{block_id}_event_overlay.png")
            fig[0].savefig(save_path, dpi=150)
            plt.close(fig[0])
            print(f"✅ Saved ERP event overlay: {save_path}")
        except Exception as e:
            print(f"❌ Event ERP overlay failed: {e}")

    # ----------- Simulated Source-Level + Network Viz -----------
    times = epochs.times
    n_parcels = 5
    parcel_ts_high = np.random.randn(n_parcels, len(times)) * 0.1 + 0.5 * np.sin(2 * np.pi * 5 * times)
    parcel_ts_low = np.random.randn(n_parcels, len(times)) * 0.1 + 0.5 * np.sin(2 * np.pi * 5 * times + 0.5)

    # 1. Parcel Time Series
    plt.figure(figsize=(10, 4))
    plt.plot(times, parcel_ts_high.mean(axis=0), label="High Attention")
    plt.plot(times, parcel_ts_low.mean(axis=0), label="Low Attention")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f"Parcel Activation - {block_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(erp_viz_dir, f"{block_id}_parcel_timeseries.png"), dpi=150)
    plt.close()

    # 2. PLV Matrix
    plv_matrix = compute_plv_matrix(parcel_ts_high)
    plt.figure(figsize=(6, 5))
    sns.heatmap(plv_matrix, cmap="coolwarm", square=True, annot=True)
    plt.title(f"PLV Matrix – {block_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(erp_viz_dir, f"{block_id}_plv_matrix_static.png"), dpi=150)
    plt.close()

    # 3. Betweenness Centrality Dynamics (simulated)
    betweenness = np.random.rand(len(times)) * 0.05 + 0.1
    plt.figure(figsize=(10, 3))
    plt.plot(times, betweenness, label="Betweenness Centrality")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f"Graph Betweenness Over Time – {block_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Centrality")
    plt.tight_layout()
    plt.savefig(os.path.join(erp_viz_dir, f"{block_id}_betweenness_centrality.png"), dpi=150)
    plt.close()

# ----------- Time-Frequency Analysis (Multitaper TFR) -----------
freqs = np.arange(12, 40, 2)      
n_cycles = 2                       

# Create save directory for TFR data
tfr_save_dir = config["paths"]["tfr_data_dir"]
os.makedirs(tfr_save_dir, exist_ok=True)

for file in os.listdir(epochs_dir):
    if not file.endswith("_epo.fif"):
        continue

    epo_path = os.path.join(epochs_dir, file)
    block_id = os.path.splitext(file)[0].replace('_epo', '')
    print(f"\n>>> Processing TFR for {block_id}")

    try:
        epochs = mne.read_epochs(epo_path, preload=True)
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")
        continue

    for cond in epochs.event_id.keys():
        condition_epochs = epochs[cond]
        if len(condition_epochs) < 5:
            print(f"⚠️ Not enough epochs for {cond}, skipping TFR.")
            continue

        # Add meaningful label
        event_labels = config.get("event_labels", {})
        label = event_labels.get(int(cond), f"Event {cond}")
        
        try:
            power = condition_epochs.compute_tfr(
                freqs=freqs,
                n_cycles=n_cycles,
                method="multitaper",
                time_bandwidth=2.0,
                return_itc=False,
                average=True,
                picks="eeg",
                decim=3,
                n_jobs=1
            )

            # Save TFR plot with meaningful title
            pick = 'Pz' if 'Pz' in epochs.ch_names else 0
            figs = power.plot(
                picks=pick,
                baseline=(-0.3, 0), mode='logratio', show=False,
                title=f"Time-Frequency – {block_id} – {label}" 
            )
            save_img_path = os.path.join(erp_viz_dir, f"{block_id}_tfr_{cond}_{label.replace(' ', '_').replace('–', '-')}.png")
            if isinstance(figs, list):
                figs[0].savefig(save_img_path, dpi=150)
                plt.close(figs[0])
            else:
                figs.savefig(save_img_path, dpi=150)
                plt.close(figs)

            print(f"✅ Saved time-frequency plot for {cond}: {save_img_path}")

            # Save TFR data to the config-specified directory
            tfr_path = os.path.join(tfr_save_dir, f"{block_id}_tfr_{cond}.h5")
            power.save(tfr_path, overwrite=True)
            print(f"💾 Saved TFR data: {tfr_path}")

        except Exception as e:
            print(f"❌ Failed TFR for {cond} in {block_id}: {e}")