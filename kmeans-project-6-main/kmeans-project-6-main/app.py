import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("Customer Segmentation using K-Means")

st.write("Enter customer details to predict the cluster.")

# Load dataset
df = pd.read_csv("income.csv")

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Income($)']])

# Train model
model = KMeans(n_clusters=3, random_state=42)
model.fit(scaled_data)

# User input
age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
income = st.number_input("Enter Annual Income ($)", min_value=1000, value=50000)

if st.button("Predict Cluster"):
    user_data = scaler.transform([[age, income]])
    cluster = model.predict(user_data)[0]

    st.success(f"Predicted Cluster: {cluster}")

    if cluster == 0:
        st.info("Customer belongs to Cluster 0")
    elif cluster == 1:
        st.info("Customer belongs to Cluster 1")
    else:
        st.info("Customer belongs to Cluster 2")