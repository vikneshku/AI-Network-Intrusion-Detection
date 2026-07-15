from src.preprocess import load_and_clean_data, scale_and_split
from src.eda import perform_eda
from src.feature_selection import select_features_correct
from src.train import train_models
from src.evaluate import evaluate_all_models, interpret_features
import os

def main():
    # Adjusted path matching your exact folder name in VS Code
    dataset_folder = "data/TrafficLabelling"
    
    if not os.path.exists(dataset_folder):
        print(f"Error: Please ensure your CSV files are in the folder: '{dataset_folder}'")
        return

    # Steps 3 & 4: Load & Clean all daily CSVs
    try:
        X, y, label_encoder = load_and_clean_data(dataset_folder)
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        return
    
    # Step 5: EDA
    perform_eda(X, y, label_encoder)
    
    # Step 7: Train-test split & Scaling
    X_train, X_test, y_train, y_test, feature_names = scale_and_split(X, y)
    
    # Step 6: Feature Selection (selecting the top 10 most critical features)
    X_train_sel, X_test_sel, selected_features = select_features_correct(
        X_train, X_test, y_train, feature_names, k=10
    )
    
    # Step 8: Train Models
    models = train_models(X_train_sel, y_train)
    
    # Step 9 & 10: Evaluation & Comparison
    best_model_name, performance_metrics = evaluate_all_models(models, X_test_sel, y_test)
    
    # Step 11: Feature Interpretation
    interpret_features(models[best_model_name], best_model_name, selected_features)
    
    print("\n Project Execution Completed Successfully!")

if __name__ == "__main__":
    main()