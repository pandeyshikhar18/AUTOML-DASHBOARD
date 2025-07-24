import streamlit as st
import pandas as pd
from automl_runner import train_and_select
from explainability import explain_model
from drift_detector import detect_drift

# Page setup
st.set_page_config(page_title="AutoML Recommender", layout="wide")
st.title("🤖 AutoML Model Recommender + Drift Dashboard")

# 1️⃣ Upload dataset
uploaded = st.file_uploader("Upload your dataset (.csv)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # 2️⃣ Choose target
    target = st.selectbox("Select target column", df.columns)

    # 3️⃣ Train & Recommend
    if st.button("🚀 Train & Recommend Model"):
        model, name, score, X_train, y_train, is_cls = train_and_select(df, target)
        st.success(f"✅ Recommended: **{name}** — "
                   f"{'Accuracy' if is_cls else 'R²'} = {score:.3f}")
        # Save everything to session
        st.session_state.model = model
        st.session_state.model_name = name
        st.session_state.model_score = score
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.is_cls = is_cls

    # 4️⃣ Explain Model
    if "model" in st.session_state and st.button("📊 Explain Model"):
        explain_model(
            st.session_state.model,
            st.session_state.model_name,
            st.session_state.model_score,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.is_cls
        )

    st.markdown("---")
    st.subheader("🧪 Data Drift Detection")

    # 5️⃣ Drift Detection
    ref_file = st.file_uploader("Upload Reference CSV", key="ref", type="csv")
    cur_file = st.file_uploader("Upload Current CSV", key="cur", type="csv")
    if ref_file and cur_file and st.button("🔍 Detect Drift"):
        ref_df = pd.read_csv(ref_file)
        cur_df = pd.read_csv(cur_file)
        detect_drift(ref_df, cur_df)
