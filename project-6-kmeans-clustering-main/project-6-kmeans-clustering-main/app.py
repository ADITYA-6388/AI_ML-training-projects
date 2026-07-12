import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

st.title("🌸 Iris Flower Clustering using K-Means")

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Use only the features from your notebook
X = df[['petal length (cm)', 'petal width (cm)']]

# Train model
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# User input
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.5)

if st.button("Predict"):
    cluster = model.predict([[petal_length, petal_width]])[0]
    st.success(f"Predicted Cluster: {cluster}")

# -------- Scatter Plot ---------
fig, ax = plt.subplots(figsize=(8,6))

colors = ['red', 'blue', 'green']

for i in range(3):
    data = df[df['Cluster'] == i]
    ax.scatter(
        data['petal length (cm)'],
        data['petal width (cm)'],
        color=colors[i],
        label=f'Cluster {i}'
    )

# Plot centroids
centers = model.cluster_centers_
ax.scatter(
    centers[:,0],
    centers[:,1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("K-Means Clustering")
ax.legend()

st.pyplot(fig)
