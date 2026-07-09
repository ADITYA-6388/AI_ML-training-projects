import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

#Page Configuration
st.set_page_config(
    page_title="Insurance Purchase Prediction",
    page_icon="💵",
    layout="centered"
)

st.title("💵Insurance Purchase Prediction")
st.write("Predict Insurance Purchase using Logistic Regression")

# 1. Load Data
# Note: Ensure "insurance_data.csv" is in the same directory as this script!
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "insurance_data.csv"

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(f"Dataset not found!\nExpected location:\n{DATA_FILE}")
    st.stop()
# 3. Train Test Split (Locked with random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    df[['age']], 
    df.bought_insurance, 
    train_size=0.8, 
    random_state=42
)

# 4. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
st.write("### Model Performance")
st.write(f"**Model Accuracy Score:** {model.score(X_test, y_test):.2f}")

# Extracting dynamic coefficients so manual math always matches the model
m = model.coef_[0][0]
b = model.intercept_[0]

st.write("### Model Parameters")
st.write(f"**Coefficient (m):** {m:.4f}")
st.write(f"**Intercept (b):** {b:.4f}")

# 6. Manual Math & Sigmoid Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(age):
    # y = mx + b
    z = m * age + b 
    y = sigmoid(z)
    return y

# 7. Interactive Prediction UI
st.write("### Try Your Own Prediction")
user_age = st.number_input("Enter Age:", min_value=1, max_value=100, value=35)
probability = prediction_function(user_age)

st.write(f"Calculated Probability: **{probability:.3f}**")
if probability >= 0.5:
    st.success(f"With a probability of {probability:.3f}, a person aged {user_age} **will** buy insurance.")
else:
    st.warning(f"With a probability of {probability:.3f}, a person aged {user_age} **will not** buy insurance.")

# ==============================
# Sidebar Branding
# ==============================

st.markdown("---")

st.markdown("""
<style>
.profile-card{
    background: linear-gradient(135deg,#0f172a,#1e293b);
    padding:30px;
    border-radius:18px;
    color:white;
    box-shadow:0 10px 30px rgba(0,0,0,0.25);
    text-align:center;
    margin-top:20px;
}

.profile-name{
    font-size:30px;
    font-weight:700;
    color:#ffffff;
}

.profile-role{
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:20px;
}

.profile-desc{
    font-size:15px;
    color:#e2e8f0;
    line-height:1.7;
    margin-bottom:25px;
}

.social-btn{
    display:inline-block;
    text-decoration:none;
    padding:12px 24px;
    margin:8px;
    border-radius:10px;
    font-weight:600;
    font-size:15px;
    transition:0.3s;
}

.github{
    background:#24292F;
    color:white !important;
}

.linkedin{
    background:#0A66C2;
    color:white !important;
}

.social-btn:hover{
    transform:translateY(-3px);
    opacity:0.9;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="profile-card">

<div class="profile-name">
👨‍💻 Abhay Kumar Gupta
</div>

<div class="profile-role">
Machine Learning Engineer | Deep Learning Enthusiast | Python Developer
</div>

<a class="social-btn github"
href="https://github.com/Abhay-cody"
target="_blank">
🐙 View GitHub Profile
</a>

<a class="social-btn linkedin"
href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397"
target="_blank">
💼 Connect on LinkedIn
</a>

</div>
""", unsafe_allow_html=True)
