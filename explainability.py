import streamlit as st
import pandas as pd
import numpy as np

def explain_model(model, model_name: str, score: float, X_train: pd.DataFrame, y_train: pd.Series, is_classification: bool):
    """
    Provides a narrative explanation of why the recommended model was chosen,
    plus data insights and top features for interpretability.
    """
    st.header("🤖 Model Explanation Report")
    
    # 1️⃣ Recommendation summary
    metric = "Accuracy" if is_classification else "R²"
    st.markdown(f"**Recommended Model:** `{model_name}`  \n**{metric}:** {score:.3f}")
    
    # 2️⃣ Why this model?
    reasons = {
        "Random Forest": [
            "- Handles both numerical & categorical features out-of-the-box",
            "- Robust to outliers & non-linear relationships",
            "- Low risk of overfitting (ensemble averaging)",
        ],
        "Logistic Regression": [
            "- Fast and interpretable",
            "- Works well on linearly separable data",
            "- Coefficients give direct feature impact",
        ],
        "Decision Tree": [
            "- Simple to visualize & interpret",
            "- Captures non-linear relationships",
            "- Minimal data preprocessing needed",
        ],
        "SVM": [
            "- Effective in high‑dimensional spaces",
            "- Works well with clear margin of separation",
        ],
        "Naive Bayes": [
            "- Fast, works well with high‑dimensional data",
            "- Good baseline for classification",
        ],
        "Linear Regression": [
            "- Simple and interpretable",
            "- Assumes linear relationship",
        ],
            "SVR": [
            "- Captures non-linear relationships via kernels",
        ],
            "Random Forest Regressor": [
            "- Handles mixed feature types",
            "- Robust to outliers and non-linearity",
        ],
            "Decision Tree Regressor": [
            "- Captures non-linear relationships",
            "- Easy to understand & visualize",
        ]
    }
    st.subheader("Why this model?")
    for bullet in reasons.get(model_name, ["- Reliable general-purpose model"]):
        st.markdown(bullet)
    
    # 3️⃣ Data insights
    num_samples, num_features = X_train.shape
    num_numeric = X_train.select_dtypes(include="number").shape[1]
    num_categorical = num_features - num_numeric
    st.subheader("📊 Data Snapshot")
    st.markdown(f"- **Samples:** {num_samples}  \n"
                f"- **Features:** {num_features}  \n"
                f"  - Numeric: {num_numeric}  \n"
                f"  - Categorical (after encoding): {num_categorical}")
    
    # 4️⃣ Top features
    st.subheader("🔝 Top Features")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feats = X_train.columns
        imp_df = (
            pd.DataFrame({"feature": feats, "importance": importances})
              .sort_values("importance", ascending=False)
              .head(5)
              .reset_index(drop=True)
        )
        st.table(imp_df)
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        # for multiclass, take absolute average
        if coefs.ndim > 1:
            coefs = np.mean(np.abs(coefs), axis=0)
        else:
            coefs = np.abs(coefs)
        feats = X_train.columns
        coef_df = (
            pd.DataFrame({"feature": feats, "coef": coefs})
              .sort_values("coef", ascending=False)
              .head(5)
              .reset_index(drop=True)
        )
        st.table(coef_df)
    else:
        st.write("No feature‐importance or coefficients available for this model.")
