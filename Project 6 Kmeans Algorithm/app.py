import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Title
st.title("🌸 Iris Flower Clustering using K-Means")

st.write("Enter petal measurements to predict the flower cluster.")

# Load Iris dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Keep only the two features used in your notebook
df = df[['petal length (cm)', 'petal width (cm)']]

# Train K-Means model
model = KMeans(n_clusters=3, random_state=42)
model.fit(df)

# User input
petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=5.0,
    value=1.5,
)

# Predict
if st.button("Predict Cluster"):
    cluster = model.predict([[petal_length, petal_width]])[0]

    st.success(f"Predicted Cluster: {cluster}")

    labels = {
        0: "Cluster 0",
        1: "Cluster 1",
        2: "Cluster 2"
    }

    st.info(f"Flower belongs to **{labels[cluster]}**")