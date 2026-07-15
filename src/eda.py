import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def perform_eda(X, y, label_encoder):
    print("--- Step 5: Performing Exploratory Data Analysis (EDA) ---")
    
    # Convert y back to original labels for plotting readable charts
    labels = label_encoder.inverse_transform(y)
    df_eda = pd.DataFrame(X)
    df_eda['Class'] = labels
    
    # 1. Distribution of Target Class
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_eda, x='Class')
    plt.title("Distribution of Network Traffic Classes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()
    print("Saved target distribution plot as 'class_distribution.png'")