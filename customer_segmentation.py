import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "CustomerID": range(1, n + 1),
    "Age": np.random.randint(18, 70, n),
    "Income": np.random.randint(20000, 120000, n),
    "Recency": np.random.randint(1, 365, n),
    "MntProducts": np.random.randint(50, 2000, n),
    "NumWebPurchases": np.random.randint(0, 20, n),
    "NumStorePurchases": np.random.randint(0, 20, n),
    "AcceptedCampaign": np.random.randint(0, 2, n),
})

print("Dataset loaded!")
print(df.head())

# 2. EDA
print("\nBasic Statistics:")
print(df.describe())

# 3. Feature Selection & Scaling
features = ["Age", "Income", "Recency", "MntProducts",
            "NumWebPurchases", "NumStorePurchases"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find Optimal K
inertias = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 5. Train Final Model
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nKMeans trained with K={optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, df['Cluster']):.4f}")

# 6. Cluster Summary
cluster_summary = df.groupby("Cluster")[features].mean().round(2)
print("\nCluster Summary:")
print(cluster_summary)

# 7. Save Results
df.to_csv("customer_segments_output.csv", index=False)
cluster_summary.to_csv("cluster_summary.csv")
print("\nResults saved!")
print("Customer Segmentation Complete!")
