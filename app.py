import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.preprocess import load_and_clean_data, scale_and_split
from src.eda import perform_eda
from src.feature_selection import select_features_correct
from src.train import train_models
from src.evaluate import evaluate_all_models, interpret_features

st.set_page_config(
    page_title="AI-NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.markdown("---")

# Automatically switch folders if running on the cloud vs local computer
if os.path.exists("data/TrafficLabelling"):
    dataset_folder = "data/TrafficLabelling"
else:
    dataset_folder = "data/sample"

if not os.path.exists(dataset_folder):
    st.error(f"❌ Error: Could not find dataset folder at `{dataset_folder}`. Please ensure your CSV files are placed correctly.")
else:
    st.success("📂 Network Traffic Files Detected Successfully!")
    
    st.sidebar.header("⚙️ System Control Parameters")
    k_features = st.sidebar.slider("Select Number of Features to Retain (K)", min_value=5, max_value=20, value=10)
    
    if st.sidebar.button("⚡ Run Data Mining Pipeline"):
        with st.spinner("Processing network traffic datasets..."):
            try:
                # Steps 3 & 4: Load and pre-process
                X, y, label_encoder = load_and_clean_data(dataset_folder)
                
                # Step 7: Scale and Split
                X_train, X_test, y_train, y_test, feature_names = scale_and_split(X, y)
                
                # Step 6: Feature Selection
                X_train_sel, X_test_sel, selected_features = select_features_correct(
                    X_train, X_test, y_train, feature_names, k=k_features
                )
                
                st.toast("✅ Data cleaning & feature selection complete!")
                
            except Exception as e:
                st.error(f"Failed to process data: {e}")
                st.stop()
                
        tab1, tab2, tab3 = st.tabs(["📊 Exploratory Analysis", "🤖 Model Training & Evaluation", "💡 Explainability & Insights"])
        
        with tab1:
            st.header("Exploratory Data Analysis (EDA)")
            perform_eda(X, y, label_encoder) 
            if os.path.exists("class_distribution.png"):
                st.image("class_distribution.png", caption="Network traffic category breakdown.", use_container_width=True)
                
        with tab2:
            st.header("Machine Learning Classifiers Performance")
            
            with st.spinner("Training and evaluating models..."):
                # Step 8: Train Models
                models = train_models(X_train_sel, y_train)
                
                # Step 9 & 10: Model Comparison Metrics (Baseline metrics)
                best_model_name, performance_metrics = evaluate_all_models(models, X_test_sel, y_test)
            
            st.success(f"🏆 Best Performing Model: **{best_model_name}**")
            
            # --- Advanced Metric Calculations (ROC-AUC & Confusion Matrices) ---
            updated_metrics_list = []
            classes = np.unique(y_test)
            class_names = label_encoder.inverse_transform(classes)
            
            # Binarize labels for Multi-class ROC-AUC calculation
            y_test_binarized = label_binarize(y_test, classes=classes)
            n_classes = y_test_binarized.shape[1]

            # We will generate individual tabs inside Model Evaluation for each model's detailed reports
            model_tabs = st.tabs(list(models.keys()))
            
            for index, (name, model) in enumerate(models.items()):
                preds = model.predict(X_test_sel)
                
                # Calculate multi-class ROC-AUC (One-Vs-Rest)
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_test_sel)
                        # If binary class vs multi-class shape mismatch fallback
                        if n_classes == 2:
                            roc_auc = roc_auc_score(y_test, probs[:, 1])
                        else:
                            roc_auc = roc_auc_score(y_test_binarized, probs, multi_class="ovr", average="macro")
                    else:
                        roc_auc = 0.5 # Default fallback if prediction probabilities are missing
                except Exception:
                    roc_auc = np.nan
                
                # Store metrics for overall table
                m_metrics = performance_metrics[name]
                updated_metrics_list.append({
                    "Model": name,
                    "Accuracy": f"{m_metrics['Accuracy']*100:.2f}%",
                    "Precision": f"{m_metrics['Precision']*100:.2f}%",
                    "Recall": f"{m_metrics['Recall']*100:.2f}%",
                    "F1-Score": f"{m_metrics['F1-Score']*100:.2f}%",
                    "ROC-AUC (OVR Macro)": f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
                })
                
                # Draw confusion matrix in each model's specific tab
                with model_tabs[index]:
                    st.subheader(f"Detailed Analysis for {name}")
                    
                    # Layout with columns
                    col_met, col_chart = st.columns([1, 2])
                    
                    with col_met:
                        st.metric("Accuracy", f"{m_metrics['Accuracy']*100:.2f}%")
                        st.metric("F1-Score", f"{m_metrics['F1-Score']*100:.2f}%")
                        st.metric("ROC-AUC", f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A")
                    
                    with col_chart:
                        st.write("#### Confusion Matrix Heatmap")
                        cm = confusion_matrix(y_test, preds)
                        
                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        sns.heatmap(
                            cm, 
                            annot=True, 
                            fmt="d", 
                            cmap="Blues", 
                            xticklabels=class_names, 
                            yticklabels=class_names,
                            ax=ax
                        )
                        plt.ylabel("Actual Label")
                        plt.xlabel("Predicted Label")
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            
            # Display summary table on top
            st.subheader("📋 Final Model Comparison Summary Table")
            st.table(pd.DataFrame(updated_metrics_list))
            
        with tab3:
            st.header("Network Traffic Feature Importances")
            st.markdown(f"Evaluating how the best model (**{best_model_name}**) processes incoming packets:")
            
            best_model_obj = models[best_model_name]
            if hasattr(best_model_obj, 'feature_importances_'):
                importances = best_model_obj.feature_importances_
                feat_imp = pd.Series(importances, index=selected_features).sort_values(ascending=False)
                
                st.bar_chart(feat_imp)
                st.dataframe(pd.DataFrame({"Feature Importance Score": feat_imp}))
            else:
                st.warning(f"Feature importance metric extraction is not supported for {best_model_name}.")