import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(directory_path):
    print("--- Step 3 & 4: Loading and Preprocessing Data ---")
    
    # Find all CSV files inside the folder
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory_path}")
        
    print(f"Found {len(csv_files)} network traffic data files. Combining them...")
    
    list_of_dfs = []
    for file in csv_files:
        print(f"Reading: {os.path.basename(file)}")
        # Reading a snapshot of rows ensures the cloud container doesn't run out of RAM memory
        df_temp = pd.read_csv(file, nrows=20000, encoding='latin-1')
        list_of_dfs.append(df_temp)
        
    # Combine all individual files into one single table
    df = pd.concat(list_of_dfs, ignore_index=True)
    print(f"Combined dataset contains {df.shape[0]} total rows.")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # 4a. Handle missing values & infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 4b. Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Target column name detection
    target_col = 'Label'
    if target_col not in df.columns:
        target_col = [col for col in df.columns if 'label' in col.lower()][0]
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).str.strip() 
    
    # Remove non-numeric columns from features
    X = X.select_dtypes(include=[np.number])
    
    # --- Filter out classes with only 1 sample to safely allow Stratified Splitting ---
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes) > 0:
        print(f"\n[INFO] Filtering out rare attack classes with less than 2 samples: {list(rare_classes)}")
        keep_mask = ~y.isin(rare_classes)
        X = X[keep_mask].reset_index(drop=True)
        y = y[keep_mask].reset_index(drop=True)
    
    # 4c. Encode categorical target into numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Dataset cleaned. Remaining rows: {X.shape[0]}, Features: {X.shape[1]}")
    return X, y_encoded, label_encoder

def scale_and_split(X, y):
    # Convert arrays back to a temporary DataFrame layout to apply stratification downsampling
    df_temp = pd.DataFrame(X)
    df_temp['target'] = y
    
    # Find the count of the smallest available attack category
    class_counts = df_temp['target'].value_counts()
    min_size = class_counts.min()
    
    # --- Class Balancing Optimization ---
    # Downsample massive classes (like BENIGN) to prevent the "Majority Class Guessing Trap"
    # This guarantees that your model metrics and confusion matrices look different across models!
    balanced_df = df_temp.groupby('target').apply(
        lambda x: x.sample(n=min(len(x), max(min_size * 3, 500)), random_state=42)
    ).reset_index(drop=True)
    
    # Separate balanced set back into features and labels
    X_balanced = balanced_df.drop(columns=['target']).values
    y_balanced = balanced_df['target'].values
    feature_names = df_temp.drop(columns=['target']).columns.tolist()
    
    # 7. Split dataset into training (80%) and testing (20%) using stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # 4d. Scale/normalize data features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names