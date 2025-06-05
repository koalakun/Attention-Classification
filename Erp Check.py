import os
import mne
import matplotlib.pyplot as plt
import yaml

# ---------------- Config ----------------
with open("e:/intern/config.yaml", "r") as f:
    config = yaml.safe_load(f)

epochs_dir = os.path.join(os.path.dirname(__file__), "epochs")
erp_plot_dir = config["paths"]["erp_plot_dir"]
os.makedirs(erp_plot_dir, exist_ok=True)

# ---------------- Plotting Function ----------------
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

    if len(epochs.event_id) == 0:
        print(f"⚠️ No event_id found in {file}, skipping.")
        continue

    evokeds = []
    for cond in epochs.event_id.keys():
        n_epochs = len(epochs[cond])
        if n_epochs == 0:
            print(f"⚠️  No remaining epochs for condition {cond}, skipping.")
            continue
        print(f"🧪 Condition {cond}: {n_epochs} epochs remaining.")
        evokeds.append(epochs[cond].average())

    if not evokeds:
        print(f"❌ No ERPs to plot for {block_id} (all conditions empty after rejection).")
        continue

    fig, ax = plt.subplots(figsize=(10, 5))
    for evk in evokeds:
        evk.plot(picks="eeg", spatial_colors=True, show=False, axes=ax)

    ax.set_title(f"ERP Overlay – {block_id}")
    save_path = os.path.join(erp_plot_dir, f"{block_id}_erp_overlay.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"✅ Saved ERP overlay plot to: {save_path}")
