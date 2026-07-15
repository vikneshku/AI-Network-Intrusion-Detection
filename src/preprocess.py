import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(directory_path):
    print("--- Step 3 & 4: Loading and Preprocessing Data ---")
    
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory_path}")
        
    print(f"Found {len(csv_files)} network traffic data files. Combining them...")
    
    list_of_dfs = []
    for file in csv_files:
        print(f"Reading: {os.path.basename(file)}")
        
        # 1. Read a larger initial chunk to make sure we hit hidden attack blocks
        df_temp = pd.read_csv(file, nrows=100000, encoding='latin-1')
        
        # 2. Shuffle rows completely so target attacks and BENIGN rows mix together
        df_temp = df_temp.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 3. Restrict to a 20,000 row snapshot to keep it lightweight on the cloud container
        df_temp = df_temp.head(20000)
        
        list_of_dfs.append(df_temp)
        
    df = pd.concat(list_of_dfs, ignore_index=True)
    print(f"Combined dataset contains {df.shape[0]} total rows.")
    
    df.columns = df.columns.str.strip()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    target_col = 'Label'
    if target_col not in df.columns:
        target_col = [col for col in df.columns if 'label' in col.lower()][0]
    
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).str.strip() 
    
    X = X.select_dtypes(include=[np.number])
    
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes) > 0:
        print(f"\n[INFO] Filtering out rare attack classes with less than 2 samples: {list(rare_classes)}")
        keep_mask = ~y.isin(rare_classes)
        X = X[keep_mask].reset_index(drop=True)
        y = y[keep_mask].reset_index(drop=True)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Dataset cleaned. Remaining rows: {X.shape[0]}, Features: {X.shape[1]}")
    return X, y_encoded, label_encoder

def scale_and_split(X, y):
    # 1. Safely extract feature names if it's a DataFrame, otherwise handle array fallback
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X_arr = np.asarray(X)
        
    y_arr = np.asarray(y)
    
    # 2. Complete a pure NumPy Stratified Downsampling to prevent majority class guessing
    unique_classes, counts = np.unique(y_arr, return_counts=True)
    min_size = counts.min()
    max_samples_per_class = max(int(min_size * 3), 500)
    
    sampled_indices = []
    rng = np.random.default_rng(42)
    
    for cls in unique_classes:
        cls_indices = np.where(y_arr == cls)[0]
        if len(cls_indices) > max_samples_per_class:
            chosen = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
            sampled_indices.extend(chosen)
        else:
            sampled_indices.extend(cls_indices)
            
    # Re-sample arrays based on balanced indexes
    X_balanced = X_arr[sampled_indices]
    y_balanced = y_arr[sampled_indices]
    
    # 3. Split dataset into training (80%) and testing (20%) tracks
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # 4. Standardize scaling normalization properties
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names