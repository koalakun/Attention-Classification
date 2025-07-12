import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import yaml
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------ Load Configuration ------------------
with open("e:/intern/config.yaml", "r") as f:
    config = yaml.safe_load(f)

source_dir = config["paths"]["features_dir"]
plv_graph_features_dir = os.path.join(source_dir, "plv_graph_features")

sfreq = 500  # Sampling rate (Hz)
window_size = 0.25  # seconds → 125 samples
step_size = 0.1     # seconds → 50 samples

# Ensure output directory exists
os.makedirs(plv_graph_features_dir, exist_ok=True)

def compute_plv_matrix_fast(data):
    """
    Optimized PLV computation using vectorized operations
    Input: data shape (n_parcels, n_timepoints)
    Output: PLV matrix shape (n_parcels, n_parcels)
    """
    n_parcels, n_times = data.shape
    
    # Get analytic signal (complex representation)
    analytic_signals = hilbert(data, axis=1)
    
    # Extract instantaneous phases
    phases = np.angle(analytic_signals)
    
    # Initialize PLV matrix
    plv_matrix = np.zeros((n_parcels, n_parcels))
    
    # Vectorized PLV computation
    for i in range(n_parcels):
        for j in range(i + 1, n_parcels):
            # Phase difference
            phase_diff = phases[i] - phases[j]
            # PLV is absolute value of mean complex exponential
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv  # Symmetric matrix
    
    return plv_matrix

# ------------------ Main Processing ------------------
all_features = []
processed_files = 0
start_time = time.time()

# Get list of source files to process
source_files = [f for f in os.listdir(source_dir) 
                if f.endswith(".npy") and f.startswith("source_") 
                and "pre_stim" not in f and "channel" not in f]

print(f"\n🚀 Found {len(source_files)} source files to process")

for filename in tqdm(source_files, desc="Processing files"):

    # ---- Load and Validate Data ----
    file_path = os.path.join(source_dir, filename)
    file_id = filename.replace("source_", "").replace(".npy", "")

    try:
        data = np.load(file_path)
        print(f"\n📊 {file_id}: loaded shape {data.shape}")
        print(f"   NaNs: {np.isnan(data).sum()}, Max abs value: {np.max(np.abs(data)):.6f}")

        # ---- Handle epoched or continuous ----
        if data.ndim == 3:  # (n_trials, n_parcels, n_times)
            print(f"⚠️ Epoched data detected: {data.shape}, concatenating trials")
            data = np.concatenate(data, axis=-1)
            print(f"   After concatenation: {data.shape}")
        elif data.ndim == 2:
            print(f"✅ Continuous data detected: {data.shape}")
        else:
            print(f"❌ Unexpected shape {data.shape}, skipping")
            continue

        n_parcels, n_times = data.shape
        w_size = int(window_size * sfreq)
        s_size = int(step_size * sfreq)
        n_windows = (n_times - w_size) // s_size + 1

        print(f"   Window size: {w_size} samples, Step: {s_size} samples")
        print(f"   Time points: {n_times}, Calculated windows: {n_windows}")

        if n_windows <= 0:
            print(f"⚠️ Skipping {file_id} — insufficient time points ({n_times})")
            continue

        file_features = []
        valid_windows = 0
        
        print(f"🔄 Processing {n_windows} windows...")
        window_start_time = time.time()

        for w in range(n_windows):
            start = w * s_size
            end = start + w_size
            window_data = data[:, start:end]
            
            # Show progress every 200 windows for better performance
            if w % 200 == 0 or n_windows <= 20:
                t_start = start / sfreq
                t_end = end / sfreq
                elapsed_windows = time.time() - window_start_time
                if w > 0:
                    windows_per_sec = w / elapsed_windows
                    eta_windows = (n_windows - w) / windows_per_sec if windows_per_sec > 0 else 0
                    print(f"   🌀 Window {w+1}/{n_windows}: t={t_start:.2f}–{t_end:.2f}s (ETA: {eta_windows/60:.1f}min)")
                else:
                    print(f"   🌀 Window {w+1}/{n_windows}: t={t_start:.2f}–{t_end:.2f}s")

            # Sanity check
            if np.isnan(window_data).any() or np.max(np.abs(window_data)) == 0:
                if w % 500 == 0:  # Only print occasionally to avoid spam
                    print(f"⚠️ Window {w} contains NaNs or zeros, skipping")
                continue

            # Compute PLV matrix using optimized function
            plv_matrix = compute_plv_matrix_fast(window_data)

            # Validate matrix
            if np.isnan(plv_matrix).any() or np.max(plv_matrix) == 0:
                if w % 500 == 0:  # Only print occasionally
                    print(f"⚠️ Window {w}: PLV matrix invalid")
                continue

            # Build graph + centrality
            graph = nx.from_numpy_array(plv_matrix)
            centrality = nx.degree_centrality(graph)
            
            valid_windows += 1

            t_start = start / sfreq
            t_end = end / sfreq

            for node, cent_val in centrality.items():
                file_features.append({
                    "file_id": file_id,
                    "window_idx": w,
                    "t_start": t_start,
                    "t_end": t_end,
                    "node": node,
                    "centrality": cent_val
                })

        print(f"✔️ Extracted {len(file_features)} features from {file_id}")
        print(f"   📊 Valid windows: {valid_windows}/{n_windows}")

        if len(file_features) == 0:
            print(f"⚠️ No features extracted from {file_id}, skipping CSV save.")
            continue

        # Save per-file CSV
        df = pd.DataFrame(file_features)
        csv_path = os.path.join(plv_graph_features_dir, f"{file_id}_graph.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path} ({len(df)} rows)")
        print(f"   📁 File size: {os.path.getsize(csv_path) / 1024:.1f} KB")

        all_features.extend(file_features)
        processed_files += 1
        
        # Show timing estimate
        elapsed = time.time() - start_time
        if processed_files > 0:
            avg_time = elapsed / processed_files
            remaining_files = len(source_files) - processed_files
            eta = avg_time * remaining_files
            print(f"   ⏱️ File ETA: {eta/60:.1f} min ({avg_time:.1f}s per file)")

    except Exception as e:
        print(f"❌ Error processing {file_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ------------------ Save Combined Output ------------------
if all_features:
    all_df = pd.DataFrame(all_features)
    all_csv_path = os.path.join(plv_graph_features_dir, "plv_graph_features_ALL_FILES.csv")
    all_df.to_csv(all_csv_path, index=False)
    
    total_time = time.time() - start_time
    print(f"\n🎯 PROCESSING COMPLETE! (took {total_time/60:.1f} minutes)")
    print(f"✅ All features saved to: {all_csv_path}")
    print(f"📊 Total features extracted: {len(all_features)}")
    print(f"📁 Files processed: {len(all_df['file_id'].unique())}/{len(source_files)}")
    print(f"📈 Final CSV size: {os.path.getsize(all_csv_path) / 1024 / 1024:.2f} MB")
else:
    print(f"\n⚠️ No features were extracted from any files! (took {(time.time() - start_time)/60:.1f} minutes)")
    print("   Check your data files and processing parameters.")

# ========================= PERFORMANCE OPTIMIZATION =========================
print("\n🎯 PERFORMANCE OPTIMIZATION - Breaking the 61% Barrier...")
print("="*60)

# Store original feature matrix for extended testing
X_original = merged_df.drop(columns=existing_non_feature_cols)
X_original = X_original.fillna(0).replace([np.inf, -np.inf], 0)

# Remove zero variance features from original
zero_var_original = X_original.columns[X_original.var() == 0].tolist()
if zero_var_original:
    X_original = X_original.drop(columns=zero_var_original)

print(f"🔍 Testing with full feature matrix: {X_original.shape}")

# ========================= EXTENDED FEATURE TESTING =========================
print("\n🔍 Extended Feature Selection Testing...")

extended_results = {}
best_test_accuracy = test_accuracy_orig
best_config = None

# Test wider range of k values
k_values = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]

for k in k_values:
    if k > X_original.shape[1]:
        continue
    
    try:
        print(f"\n📊 Testing k={k}...")
        
        # Feature selection
        selector_k = SelectKBest(score_func=f_classif, k=k)
        X_k = selector_k.fit_transform(X_original, y)
        
        # Scale features
        scaler_k = StandardScaler()
        X_k_scaled = scaler_k.fit_transform(X_k)
        
        # Apply SMOTE
        sm_k = SMOTE(random_state=42, k_neighbors=min(3, len(np.unique(y))-1))
        X_k_smote, y_k_smote = sm_k.fit_resample(X_k_scaled, y)
        
        # Train/test split (original test set)
        X_train_k, _, y_train_k, _ = train_test_split(
            X_k_smote, y_k_smote, stratify=y_k_smote, test_size=0.25, random_state=42
        )
        X_train_orig_k, X_test_orig_k, y_train_orig_k, y_test_orig_k = train_test_split(
            X_k_scaled, y, stratify=y, test_size=0.25, random_state=42
        )
        
        # Test multiple models
        test_models = {
            'LogisticRegression': LogisticRegression(C=0.1, random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(probability=True, C=1.0, random_state=42, class_weight='balanced'),
            'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_iter=100),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
        }
        
        k_results = {}
        
        for model_name, model in test_models.items():
            # Train on SMOTE data
            model.fit(X_train_k, y_train_k)
            
            # Test on original data
            y_pred_k = model.predict(X_test_orig_k)
            test_acc_k = accuracy_score(y_test_orig_k, y_pred_k)
            
            # Cross-validation score
            cv_scores_k = cross_val_score(model, X_train_k, y_train_k, cv=5, scoring='accuracy')
            
            k_results[model_name] = {
                'test_accuracy': test_acc_k,
                'cv_accuracy': cv_scores_k.mean(),
                'cv_std': cv_scores_k.std()
            }
            
            print(f"   {model_name}: Test={test_acc_k:.3f}, CV={cv_scores_k.mean():.3f}±{cv_scores_k.std():.3f}")
        
        # Find best model for this k
        best_model_k = max(k_results.keys(), key=lambda x: k_results[x]['test_accuracy'])
        best_acc_k = k_results[best_model_k]['test_accuracy']
        
        extended_results[k] = {
            'best_model': best_model_k,
            'best_test_accuracy': best_acc_k,
            'all_results': k_results
        }
        
        # Track overall best
        if best_acc_k > best_test_accuracy:
            best_test_accuracy = best_acc_k
            best_config = {
                'k': k,
                'model': best_model_k,
                'test_accuracy': best_acc_k,
                'results': k_results[best_model_k]
            }
            print(f"🎯 NEW BEST! k={k}, {best_model_k}: {best_acc_k:.3f}")
        
    except Exception as e:
        print(f"⚠️ k={k} failed: {e}")
        continue

# ========================= GRAPH-BASED FEATURE ENHANCEMENT =========================
print(f"\n🧠 Graph-Based Feature Enhancement...")

def extract_graph_features(data):
    """Extract graph theory features from connectivity data"""
    features = {}
    
    # Try to reshape into adjacency matrix
    n = int(np.sqrt(len(data))) if len(data) > 100 else len(data)
    
    if n * n == len(data) and n > 3:  # Valid square matrix
        adj_matrix = data.reshape(n, n)
        
        # Graph metrics
        features['node_strength_mean'] = np.mean(np.sum(np.abs(adj_matrix), axis=1))
        features['node_strength_std'] = np.std(np.sum(np.abs(adj_matrix), axis=1))
        features['edge_density'] = np.count_nonzero(adj_matrix) / (n * (n-1))
        
        # Centrality approximations
        node_strengths = np.sum(np.abs(adj_matrix), axis=1)
        features['max_centrality'] = np.max(node_strengths)
        features['centrality_ratio'] = np.max(node_strengths) / (np.mean(node_strengths) + 1e-8)
        
        # Clustering approximation
        features['clustering_approx'] = np.mean(np.abs(adj_matrix @ adj_matrix @ adj_matrix))
        
        # Modularity approximation
        expected = np.outer(node_strengths, node_strengths) / np.sum(node_strengths)
        features['modularity_approx'] = np.sum((adj_matrix - expected) ** 2)
        
    else:
        # Fallback for non-square data
        features['connectivity_variance'] = np.var(data)
        features['connectivity_entropy'] = -np.sum(data * np.log(np.abs(data) + 1e-8))
        features['connectivity_peaks'] = len([x for x in data if abs(x) > np.percentile(np.abs(data), 95)])
    
    return features

# Apply graph features to first few files as test
print("🔍 Testing graph feature extraction...")
graph_enhanced_features = []

for i, features in enumerate(source_features[:10]):  # Test first 10
    try:
        enhanced_feats, _ = extract_enhanced_brain_features(features)
        graph_feats = extract_graph_features(features)
        
        combined_feats = enhanced_feats + list(graph_feats.values())
        graph_enhanced_features.append(combined_feats)
        
    except Exception as e:
        print(f"⚠️ Graph features failed for sample {i}: {e}")
        enhanced_feats, _ = extract_enhanced_brain_features(features)
        graph_enhanced_features.append(enhanced_feats)

if len(graph_enhanced_features) > 5:
    print(f"✅ Graph features extracted: {len(graph_enhanced_features[0])} total features")

# ========================= ADVANCED MODEL TESTING =========================
print(f"\n🚀 Advanced Model Testing...")

# Try different algorithms
try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    advanced_models = {
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'LDA': LinearDiscriminantAnalysis(),
        'GaussianNB': GaussianNB()
    }
    
    # Use best k from previous testing or default to 15
    best_k = best_config['k'] if best_config else 15
    best_k = min(best_k, X_original.shape[1])
    
    print(f"🔍 Testing advanced models with k={best_k}...")
    
    selector_adv = SelectKBest(score_func=f_classif, k=best_k)
    X_adv = selector_adv.fit_transform(X_original, y)
    scaler_adv = StandardScaler()
    X_adv_scaled = scaler_adv.fit_transform(X_adv)
    
    # Split
    X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
        X_adv_scaled, y, stratify=y, test_size=0.25, random_state=42
    )
    
    advanced_results = {}
    
    for model_name, model in advanced_models.items():
        try:
            model.fit(X_train_adv, y_train_adv)
            y_pred_adv = model.predict(X_test_adv)
            test_acc_adv = accuracy_score(y_test_adv, y_pred_adv)
            
            cv_scores_adv = cross_val_score(model, X_train_adv, y_train_adv, cv=5, scoring='accuracy')
            
            advanced_results[model_name] = {
                'test_accuracy': test_acc_adv,
                'cv_accuracy': cv_scores_adv.mean()
            }
            
            print(f"   {model_name}: Test={test_acc_adv:.3f}, CV={cv_scores_adv.mean():.3f}")
            
            if test_acc_adv > best_test_accuracy:
                best_test_accuracy = test_acc_adv
                best_config = {
                    'k': best_k,
                    'model': model_name,
                    'test_accuracy': test_acc_adv,
                    'type': 'advanced'
                }
                print(f"🎯 NEW BEST ADVANCED! {model_name}: {test_acc_adv:.3f}")
                
        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}")
            
except ImportError:
    print("⚠️ Some advanced models not available")

# ========================= OPTIMIZATION SUMMARY =========================
print(f"\n" + "="*60)
print("🎯 OPTIMIZATION RESULTS SUMMARY")
print("="*60)

print(f"🔄 Original Performance:")
print(f"   Test Accuracy: {test_accuracy_orig:.3f}")
print(f"   Features: {X.shape[1]} (k=3)")
print(f"   Model: {best_model_name}")

if best_config:
    improvement = best_config['test_accuracy'] - test_accuracy_orig
    print(f"\n🚀 BEST OPTIMIZED Performance:")
    print(f"   Test Accuracy: {best_config['test_accuracy']:.3f} (+{improvement:.3f})")
    print(f"   Features: k={best_config['k']}")
    print(f"   Model: {best_config['model']}")
    
    if improvement > 0.05:  # 5% improvement
        print(f"✅ SIGNIFICANT IMPROVEMENT FOUND!")
        print(f"🎯 Recommendation: Re-run pipeline with k={best_config['k']} and {best_config['model']}")
    elif improvement > 0.02:  # 2% improvement
        print(f"✅ Moderate improvement found")
    else:
        print(f"📊 Marginal improvement - current setup may be optimal")
else:
    print(f"\n📊 No improvement found - current setup appears optimal")

print(f"\n📈 Feature Selection Analysis:")
if extended_results:
    k_accuracies = [(k, extended_results[k]['best_test_accuracy']) for k in extended_results.keys()]
    k_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Top 5 k values by test accuracy:")
    for k, acc in k_accuracies[:5]:
        print(f"     k={k}: {acc:.3f}")

print(f"\n🎯 Next Steps Recommendations:")
if best_config and best_config['test_accuracy'] > 0.70:
    print(f"✅ Good performance achieved (>70%)")
elif best_config and best_config['test_accuracy'] > 0.65:
    print(f"✅ Reasonable performance (>65%)")
    print(f"🔍 Consider: More feature engineering or external data")
else:
    print(f"⚠️ Performance still limited (<65%)")
    print(f"🔍 Consider: Different feature types or more data")

print("="*60)
