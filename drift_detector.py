import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from scipy.stats import ks_2samp, chi2_contingency

def detect_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, alpha=0.05):
    st.write("### 🧪 Data Drift Report")
    
    drifted_features = []

    # 1) Numeric features: KS-test + histogram overlay
    num_cols = ref_df.select_dtypes(include="number").columns
    for col in num_cols:
        ref = ref_df[col].dropna()
        cur = cur_df[col].dropna()
        if len(ref) < 2 or len(cur) < 2:
            continue
        
        stat, p = ks_2samp(ref, cur)
        drift = p < alpha
        if drift:
            drifted_features.append((col, 'numeric', p, ref.mean(), cur.mean()))
        
        # Plot overlay
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ref, name="Reference", opacity=0.6))
        fig.add_trace(go.Histogram(x=cur, name="Current", opacity=0.6))
        fig.update_layout(
            title=f"Distribution Change: {col} (p={p:.3f})",
            barmode='overlay',
            xaxis_title=col,
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2) Categorical features: chi‑square + side‑by‑side bar chart
    cat_cols = ref_df.select_dtypes(include="object").columns
    for col in cat_cols:
        ref_counts = ref_df[col].fillna("missing").value_counts()
        cur_counts = cur_df[col].fillna("missing").value_counts()
        # Align indices
        all_vals = sorted(set(ref_counts.index) | set(cur_counts.index))
        ref_vals = [ref_counts.get(v, 0) for v in all_vals]
        cur_vals = [cur_counts.get(v, 0) for v in all_vals]
        
        # Chi-square test
        table = np.array([ref_vals, cur_vals])
        try:
            chi2, p, _, _ = chi2_contingency(table)
        except:
            continue
        drift = p < alpha
        if drift:
            drifted_features.append((col, 'categorical', p, None, None))
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(name='Reference', x=all_vals, y=ref_vals),
            go.Bar(name='Current', x=all_vals, y=cur_vals)
        ])
        fig.update_layout(
            title=f"Category Change: {col} (p={p:.3f})",
            barmode='group',
            xaxis_title=col,
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3) Narrative summary
    if drifted_features:
        st.subheader("⚠️ Drift Summary")
        for feat, ftype, p, ref_mean, cur_mean in drifted_features:
            if ftype == 'numeric':
                st.markdown(
                    f"- **{feat}** (numeric) drifted (KS p = {p:.3f}); "
                    f"mean changed from {ref_mean:.2f} → {cur_mean:.2f}."
                )
            else:
                st.markdown(
                    f"- **{feat}** (categorical) drifted (Chi‑square p = {p:.3f})."
                )
        st.markdown(
            "> Features with significant drift (p < 0.05) may degrade model performance."
        )
    else:
        st.success("✅ No significant data drift detected (all p ≥ 0.05).")
