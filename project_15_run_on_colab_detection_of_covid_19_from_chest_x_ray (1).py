import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🩺 COVID-19 Chest X-Ray Detection")
st.write(
    "Upload a Chest X-ray image to predict whether it is **COVID** or **NORMAL**."
)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    img = image.resize((299, 299))

    img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "COVID"
        st.error(f"Prediction : {label}")
    else:
        label = "NORMAL"
        st.success(f"Prediction : {label}")

    st.write(f"Model Output : {confidence:.4f}")
