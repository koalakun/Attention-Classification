import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import pickle
import scipy.stats
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Disable plot popups - we don't want random matplotlib windows popping up
import matplotlib.pyplot as plt
plt.ioff()

# File paths - adjust these if your data is located elsewhere
hmm_summary_path = "e:/intern/outputs/hmm_rt_analysis_fixed.csv"
source_feature_dir = "e:/intern/features"
npy_prefix = "pre_stim_features_source_"

print("Brain Signal RT Classifier - Optimized Version")
print("=" * 50)

# Load the HMM analysis results that contain reaction time data
hmm_df = pd.read_csv(hmm_summary_path)
hmm_df["file_key"] = hmm_df["file_key"].str.upper()  # Standardize case for matching

print(f"Loaded HMM data: {len(hmm_df)} rows")

# Create binary labels based on reaction time - faster than median = "fast", slower = "slow"
# This is a pretty standard approach for binary classification of continuous variables
if 'rt_label' not in hmm_df.columns:
    median_rt = hmm_df['mean_rt'].median()
    hmm_df['rt_label'] = hmm_df['mean_rt'].apply(lambda x: 'fast' if x < median_rt else 'slow')
    print(f"Created rt_label from mean_rt (median split: {median_rt:.3f})")

print(f"RT label distribution:\n{hmm_df['rt_label'].value_counts()}")

# Load the brain connectivity features from .npy files
# These contain the actual brain signal data we want to classify
source_features = []
file_keys = []

for fname in os.listdir(source_feature_dir):
    if fname.startswith(npy_prefix) and fname.endswith(".npy"):
        # Extract the file identifier from the filename
        file_key = fname.replace(npy_prefix, "").replace(".npy", "").upper()
        npy_path = os.path.join(source_feature_dir, fname)
        
        try:
            data = np.load(npy_path)
            # Some files might be matrices, others vectors - flatten everything to be consistent
            if data.ndim > 1:
                data = data.flatten()
            
            # Skip empty or all-zero files - they won't help with classification
            if len(data) == 0 or np.all(np.isnan(data)) or np.all(data == 0):
                continue
                
            source_features.append(data)
            file_keys.append(file_key)
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

print(f"Loaded {len(source_features)} feature files")

def extract_enhanced_brain_features(data):
    """
    Extract meaningful statistical features from raw brain connectivity data.
    This is where we turn thousands of numbers into interpretable metrics.
    """
    features = {}
    
    # Basic statistical measures - these capture the overall distribution
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['skewness'] = scipy.stats.skew(data)  # Is the distribution lopsided?
    features['kurtosis'] = scipy.stats.kurtosis(data)  # How heavy are the tails?
    features['median'] = np.median(data)
    features['q75_q25'] = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile range
    
    # Brain connectivity specific measures - these matter more for neural data
    features['total_strength'] = np.sum(np.abs(data))  # Overall connectivity strength
    features['max_connection'] = np.max(np.abs(data))  # Strongest single connection
    features['mean_abs'] = np.mean(np.abs(data))  # Average connection strength
    features['std_abs'] = np.std(np.abs(data))  # Variability in connection strengths
    
    # Additional distribution characteristics
    features['q90'] = np.percentile(data, 90)  # High-end values
    features['q10'] = np.percentile(data, 10)  # Low-end values
    features['range'] = np.max(data) - np.min(data)  # Full spread
    features['cv'] = features['std'] / (abs(features['mean']) + 1e-8)  # Coefficient of variation
    
    # Network sparsity measures - how many strong vs weak connections?
    threshold_90 = np.percentile(np.abs(data), 90)
    threshold_95 = np.percentile(np.abs(data), 95)
    
    strong_90 = np.abs(data) > threshold_90
    strong_95 = np.abs(data) > threshold_95
    
    features['sparsity_90'] = np.sum(strong_90) / len(data)  # Fraction of strong connections
    features['sparsity_95'] = np.sum(strong_95) / len(data)  # Fraction of very strong connections
    features['strong_conn_mean_90'] = np.mean(data[strong_90]) if np.any(strong_90) else 0
    features['strong_conn_mean_95'] = np.mean(data[strong_95]) if np.any(strong_95) else 0
    
    # Robust statistics - less sensitive to outliers
    features['mad'] = np.median(np.abs(data - np.median(data)))  # Median absolute deviation
    if len(data) > 10:  # Only compute if we have enough data points
        features['trimmed_mean'] = scipy.stats.trim_mean(data, 0.1)  # Mean after removing 10% outliers
    else:
        features['trimmed_mean'] = features['mean']
    
    return list(features.values()), list(features.keys())

# Apply feature engineering to all our brain data files
print(f"Processing {len(source_features)} feature files...")

enhanced_features = []
feature_names = None

for features in source_features:
    brain_feats, feat_names = extract_enhanced_brain_features(features)
    enhanced_features.append(brain_feats)
    if feature_names is None:  # Store feature names from first file
        feature_names = feat_names

# Create a DataFrame with our engineered features
engineered_df = pd.DataFrame(enhanced_features, columns=feature_names)
engineered_df["file_key"] = file_keys

print(f"Extracted {len(feature_names)} enhanced features")

# Merge brain features with reaction time labels
# This is where we connect the brain data to the behavioral outcomes
merged_df = pd.merge(hmm_df, engineered_df, on="file_key", how="inner")

print(f"Combined dataset: {len(merged_df)} samples, {merged_df.shape[1]} total columns")

# Set up the classification problem
label_mapping = {"fast": 0, "slow": 1}  # Convert text labels to numbers
y = merged_df["rt_label"].map(label_mapping).values

# Sanity check - make sure label encoding worked properly
if np.any(pd.isna(y)):
    print("Error in label encoding!")
    exit()

# Separate features from metadata columns
# We only want to use actual brain/behavioral features for prediction
non_feature_cols = ["file_key", "rt_label", "sequence_length", "n_transitions", 
                   "most_common_state", "mean_rt", "std_rt", "trial_count", "fast_proportion"]
existing_non_feature_cols = [col for col in merged_df.columns if col in non_feature_cols]
X = merged_df.drop(columns=existing_non_feature_cols)

print(f"Initial feature matrix: {X.shape}")

# Data cleaning - machine learning algorithms hate missing/infinite values
X = X.fillna(0).replace([np.inf, -np.inf], 0)
X.columns = X.columns.astype(str)  # Ensure column names are strings

# Remove features that don't vary at all - they won't help with prediction
zero_var_cols = X.columns[X.var() == 0].tolist()
if zero_var_cols:
    print(f"Removing {len(zero_var_cols)} zero-variance features")
    X = X.drop(columns=zero_var_cols)

print(f"After cleaning: {X.shape}")

# ========================= FEATURE SELECTION =========================
print("\nFeature Selection...")

# Use only the top 10 most informative features to avoid overfitting
# With only 69 samples, we need to be conservative about feature count
k_optimal = min(10, X.shape[1])
print(f"Using k={k_optimal} features...")

selector = SelectKBest(score_func=f_classif, k=k_optimal)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

print(f"Selected features: {X.shape}")

# Standardize features - most ML algorithms work better when features are on similar scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================= HANDLE CLASS IMBALANCE =========================
print("\nApplying SMOTE...")

# SMOTE creates synthetic samples to balance the classes
# This helps when we have unequal numbers of fast vs slow trials
sm = SMOTE(random_state=42, k_neighbors=3)
try:
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
    print(f"After SMOTE: {len(y_resampled)} samples")
    
    # Train on SMOTE-balanced data, but test on original data for fair evaluation
    X_train, _, y_train, _ = train_test_split(
        X_resampled, y_resampled, stratify=y_resampled, test_size=0.25, random_state=42
    )
    
    # Keep original test set to avoid inflating performance with synthetic data
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.25, random_state=42
    )
    
except ValueError as e:
    print(f"SMOTE failed: {e}, using original data")
    # Fallback to regular train/test split if SMOTE doesn't work
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.25, random_state=42
    )

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# ========================= MODEL TRAINING =========================
print("\nModel Training...")

# Test several different algorithms to see which works best for our data
# Each has different strengths for small datasets like ours
models = {
    'LogisticRegression': LogisticRegression(C=0.1, random_state=42, class_weight='balanced', max_iter=1000),
    'SVM': SVC(probability=True, C=1.0, random_state=42, class_weight='balanced'),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_iter=100),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
}

model_scores = {}
trained_models = {}

print("Model Evaluation:")
for name, model in models.items():
    # Use cross-validation to get a more reliable estimate of performance
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model_scores[name] = cv_scores.mean()
    
    # Train the full model for final evaluation
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Test on held-out data to see how well it generalizes
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"   {name}: CV={cv_scores.mean():.3f}, Test={test_acc:.3f}")

# Pick the model with the best cross-validation performance
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")

# ========================= FINAL EVALUATION =========================
print("\nFinal Evaluation...")

# Get predictions and probabilities for detailed analysis
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_accuracy:.3f}")
print(f"CV Accuracy: {model_scores[best_model_name]:.3f}")

# Detailed performance breakdown by class
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fast', 'Slow']))

# Confusion matrix shows exactly which predictions were right/wrong
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# If possible, show which features the model thinks are most important
if hasattr(best_model, 'coef_'):
    print(f"\nTop Features:")
    feature_importance = np.abs(best_model.coef_[0])
    feature_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

# ========================= SAVE EVERYTHING =========================
print("\nSaving Results...")

output_dir = "e:/intern/outputs"
os.makedirs(output_dir, exist_ok=True)

# Save the trained model and all preprocessing steps so we can use it later
model_path = os.path.join(output_dir, "optimized_rt_classifier.pkl")
save_dict = {
    'model': best_model,
    'scaler': scaler,  # Need this to scale new data the same way
    'selector': selector,  # Need this to select the same features
    'feature_names': X.columns.tolist(),
    'label_mapping': label_mapping,
    'test_accuracy': test_accuracy,
    'cv_accuracy': model_scores[best_model_name]
}

with open(model_path, 'wb') as f:
    pickle.dump(save_dict, f)

print(f"Saved model to: {model_path}")

# ========================= FINAL SUMMARY =========================
print("\n" + "="*50)
print("OPTIMIZED RESULTS SUMMARY")
print("="*50)
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"CV Accuracy: {model_scores[best_model_name]:.3f}")
print(f"Features Used: {X.shape[1]}")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")
print("="*50)

print("\nClassification Complete!")
print(f"Key Improvements: Enhanced Features + SMOTE + k={k_optimal}")

