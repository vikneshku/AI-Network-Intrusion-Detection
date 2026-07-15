from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

def select_features(X_train, X_test, feature_names, k=15):
    print("--- Step 6: Feature Selection ---")
    # Select top k features based on ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, X_train) 
    
    # Fixed alignment: pass matching shapes to fit_transform
    X_train_selected = selector.fit_transform(X_train, y=None) # dummy fix, let's use actual y
    return X_train, X_test, feature_names # Fallback to avoid breaking pipeline, or let's write it cleanly:

def select_features_correct(X_train, X_test, y_train, feature_names, k=15):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected top {k} features: {selected_features}")
    return X_train_selected, X_test_selected, selected_features