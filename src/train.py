import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    print("Training Machine Learning Models...")
    
    # Identify unique classes present in this data split
    unique_classes = np.unique(y_train)
    
    # Configure models safely
    # We increase max_iter and change solver so Logistic Regression won't crash on small data
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            solver='liblinear' if len(unique_classes) <= 2 else 'lbfgs', 
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"[WARNING] Could not train {name} due to data constraints: {e}")
            # If a model fails due to a data anomaly, we skip it instead of crashing the dashboard
            continue
            
    if not trained_models:
        raise ValueError("All models failed to train. Please check your training data sample size.")
        
    return trained_models