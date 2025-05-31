import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set paths
source_dir = Path("E:/intern/features")
output_dir = source_dir / "heatmaps_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# Collect all source-level feature files
source_files = list(source_dir.glob("source_*.npy"))

# Load and process
all_data = []
file_names = []
parcel_means = []

for file in source_files:
    data = np.load(file)
    if data.ndim != 2 or data.shape[0] == 0:
        continue
    all_data.append(data)
    file_names.append(file.stem)
    parcel_means.append(np.mean(data, axis=0))  # Mean per parcel

# Combine
all_data_concat = np.concatenate(all_data, axis=0)
parcel_means = np.array(parcel_means)

# 1. Band Power Approximation Heatmap
avg_band_power = np.mean(all_data_concat, axis=0).reshape(1, -1)
plt.figure(figsize=(16, 2))
sns.heatmap(avg_band_power, cmap="viridis", cbar=True)
plt.title("Band Power Approximation Heatmap (Mean Activity per Parcel)")
plt.xlabel("Parcel Index")
plt.yticks([])
plt.tight_layout()
plt.savefig(output_dir / "band_power_heatmap.png")
plt.close()

# 2. Averaged Parcel Activity Per File Heatmap
df = pd.DataFrame(parcel_means, index=file_names)
plt.figure(figsize=(16, len(file_names) * 0.3 + 3))
sns.heatmap(df, cmap="plasma", cbar=True)
plt.title("Averaged Parcel Activity Per File")
plt.xlabel("Parcel Index")
plt.ylabel("File")
plt.tight_layout()
plt.savefig(output_dir / "avg_parcel_activity_per_file.png")
plt.close()

# 3. Correlation Heatmap Between Parcels
correlation_matrix = np.corrcoef(all_data_concat.T)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap Between Parcels")
plt.xlabel("Parcel Index")
plt.ylabel("Parcel Index")
plt.tight_layout()
plt.savefig(output_dir / "parcel_correlation_heatmap.png")
plt.close()
