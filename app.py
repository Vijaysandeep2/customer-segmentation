import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.markdown("An interactive ML app that segments customers into distinct groups based on their behavior and demographics.")

# ── Generate Data ──
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "Income": np.random.randint(20000, 120000, n),
    "Recency": np.random.randint(1, 365, n),
    "MntProducts": np.random.randint(50, 2000, n),
    "NumWebPurchases": np.random.randint(0, 20, n),
    "NumStorePurchases": np.random.randint(0, 20, n),
})

st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)
x_axis = st.sidebar.selectbox("X Axis", df.columns, index=1)
y_axis = st.sidebar.selectbox("Y Axis", df.columns, index=3)

# ── Scale & Cluster ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, df["Cluster"])

# ── Metrics ──
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", n)
col2.metric("Number of Clusters", k)
col3.metric("Silhouette Score", f"{score:.4f}")

st.markdown("---")

# ── Cluster Plot ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Segments")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df[x_axis], df[y_axis],
                         c=df["Cluster"], cmap="Set1", alpha=0.6)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Clusters: {x_axis} vs {y_axis}")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

with col2:
    st.subheader("📈 Cluster Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    df["Cluster"].value_counts().sort_index().plot(
        kind="bar", ax=ax2, color="steelblue", edgecolor="black"
    )
    ax2.set_title("Number of Customers per Cluster")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

st.markdown("---")

# ── Cluster Summary ──
st.subheader("📋 Cluster Summary")
summary = df.groupby("Cluster").mean().round(2)
st.dataframe(summary, use_container_width=True)

# ── Raw Data ──
if st.checkbox("Show Raw Data"):
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("Built by **Bangi Vijay Sandeep** | [GitHub](https://github.com/Vijaysandeep2) | [LinkedIn](https://linkedin.com/in/vijaysandeep-bangi)")
