from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def evaluate_all_models(models, X_test, y_test):
    print("--- Step 9 & 10: Model Evaluation & Comparison ---")
    results = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, preds)
        # Using macro average to handle multi-class network labels safely
        precision = precision_score(y_test, preds, average='macro', zero_division=0)
        recall = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, preds)
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": cm
        }
    
    # Display comparison table
    df_results = pd.DataFrame(results).T.drop(columns=["Confusion Matrix"])
    print("\n--- Model Comparison Table ---")
    print(df_results.to_string())
    
    # 10. Choose best model based on F1-Score
    best_model_name = max(results, key=lambda k: results[k]["F1-Score"])
    print(f"\n Best Model Selected: {best_model_name}")
    
    return best_model_name, results

def interpret_features(model, model_name, feature_names):
    print(f"\n--- Step 11: Feature Importance Interpretation ({model_name}) ---")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("Top 5 features affecting prediction:")
        print(feat_imp.head(5))
    else:
        print(f"Feature importance calculation is not supported directly for {model_name}.")