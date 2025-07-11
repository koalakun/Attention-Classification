import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Paths
hmm_summary_path = "e:/intern/outputs/hmm_rt_analysis_fixed.csv"
source_feature_dir = "e:/intern/features"
npy_prefix = "pre_stim_features_source_"

# Load HMM + RT summary
hmm_df = pd.read_csv(hmm_summary_path)
hmm_df["file_key"] = hmm_df["file_key"].str.upper()

print(f"📊 Loaded HMM data: {len(hmm_df)} rows")
print(f"📋 Available columns: {list(hmm_df.columns)}")

# Check if rt_label exists, if not create it from available data
if 'rt_label' not in hmm_df.columns:
    print("⚠️ rt_label not found, creating from mean_rt...")
    median_rt = hmm_df['mean_rt'].median()
    hmm_df['rt_label'] = hmm_df['mean_rt'].apply(lambda x: 'fast' if x < median_rt else 'slow')
    print(f"✅ Created rt_label from mean_rt (median split: {median_rt:.3f})")

print(f"RT label distribution: {hmm_df['rt_label'].value_counts()}")

# Load and merge pre-stim source features
source_features = []
file_keys = []

for fname in os.listdir(source_feature_dir):
    if fname.startswith(npy_prefix) and fname.endswith(".npy"):
        file_key = fname.replace(npy_prefix, "").replace(".npy", "").upper()
        npy_path = os.path.join(source_feature_dir, fname)
        
        try:
            data = np.load(npy_path)
            if data.ndim > 1:
                data = data.flatten()
            
            # Check for valid data
            if len(data) == 0 or np.all(np.isnan(data)) or np.all(data == 0):
                print(f"⚠️ Skipping {file_key}: invalid data")
                continue
                
            source_features.append(data)
            file_keys.append(file_key)
            
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")
            continue

print(f"📁 Loaded {len(source_features)} feature files")

# Convert to DataFrame with proper handling of variable lengths
if source_features:
    # Find maximum feature length for padding
    max_length = max(len(features) for features in source_features)
    print(f"🔧 Feature vector lengths: min={min(len(f) for f in source_features)}, max={max_length}")
    
    # Pad shorter feature vectors with zeros
    padded_features = []
    for features in source_features:
        if len(features) < max_length:
            padded = np.zeros(max_length)
            padded[:len(features)] = features
            padded_features.append(padded)
        else:
            padded_features.append(features)
    
    source_df = pd.DataFrame(padded_features)
    source_df["file_key"] = file_keys
else:
    print("❌ No valid feature files found!")
    exit()

# Merge with HMM data
print(f"🔗 Merging datasets...")
merged_df = pd.merge(hmm_df, source_df, on="file_key", how="inner")
print(f"📊 Merged dataset: {len(merged_df)} samples")

# Drop rows without labels
initial_count = len(merged_df)
merged_df = merged_df.dropna(subset=["rt_label"])
print(f"🧹 Dropped {initial_count - len(merged_df)} rows with missing RT labels")

if len(merged_df) == 0:
    print("❌ No samples with valid RT labels!")
    exit()

# Check class balance
print(f"📈 Class distribution:")
print(merged_df["rt_label"].value_counts(normalize=True))

# Encode target with explicit mapping
label_mapping = {"fast": 0, "slow": 1}
y = merged_df["rt_label"].map(label_mapping).values

# Verify encoding worked
if np.any(pd.isna(y)):
    print("❌ Error in label encoding!")
    print(f"Unique labels: {merged_df['rt_label'].unique()}")
    exit()

# Drop non-feature columns more carefully
non_feature_cols = {
    "file_key", "rt_label", "sequence_length", "n_transitions", 
    "most_common_state", "mean_rt", "std_rt", "trial_count", "fast_proportion"
}

# Find actual non-feature columns that exist
existing_non_feature_cols = [col for col in merged_df.columns if col in non_feature_cols]
X = merged_df.drop(columns=existing_non_feature_cols)

print(f"🎯 Feature matrix shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

# Check for problematic features
print(f"🔍 Data quality check:")
print(f"   NaN values: {X.isnull().sum().sum()}")
print(f"   Infinite values: {np.isinf(X.values).sum()}")
print(f"   Zero variance features: {(X.var() == 0).sum()}")

# Remove zero variance features
zero_var_cols = X.columns[X.var() == 0].tolist()
if zero_var_cols:
    print(f"🗑️ Removing {len(zero_var_cols)} zero-variance features")
    X = X.drop(columns=zero_var_cols)

# Handle remaining NaN/inf values
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# Fix column names to be all strings 
X.columns = X.columns.astype(str)

# Feature scaling (important for some algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

print(f"📚 Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train model with better parameters
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

# Cross-validation before final training
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"🎯 Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fast', 'Slow']))

print("\n📉 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\n🎯 Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Feature importance analysis
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

# Enhanced visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Feature importance plot
sns.barplot(x=importances.values, y=importances.index, ax=ax1)
ax1.set_title("Top 15 Feature Importances")
ax1.set_xlabel("Importance")

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Fast', 'Slow'], yticklabels=['Fast', 'Slow'])
ax2.set_title("Confusion Matrix")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Save results and model
results = {
    'test_accuracy': accuracy_score(y_test, y_pred),
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'feature_count': X.shape[1],
    'sample_count': len(y),
    'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
}

print(f"\n📋 Final Results Summary:")
for key, value in results.items():
    print(f"   {key}: {value}")

# Save the trained model and results
output_dir = "e:/intern/outputs"
os.makedirs(output_dir, exist_ok=True)

# Save model
model_path = os.path.join(output_dir, "rt_classifier_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump({'model': clf, 'scaler': scaler, 'feature_names': X.columns.tolist()}, f)
print(f"💾 Saved trained model to: {model_path}")

# Save results
results_path = os.path.join(output_dir, "classification_results.csv")
pd.DataFrame([results]).to_csv(results_path, index=False)
print(f"💾 Saved results to: {results_path}")
