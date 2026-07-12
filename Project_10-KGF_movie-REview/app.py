import streamlit as st
from transformers import pipeline
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="KGF 2 Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:1rem;
padding-bottom:1rem;
padding-left:2rem;
padding-right:2rem;
}

textarea{
font-size:18px !important;
}

.stButton>button{
background:#c62828;
color:white;
font-size:18px;
font-weight:bold;
height:50px;
width:100%;
border-radius:10px;
border:none;
}

.stButton>button:hover{
background:#b71c1c;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load HuggingFace Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🎬 KGF 2 Movie Review Analyzer")
st.caption("NLP Project using Hugging Face Transformers & Large Language Models")

st.divider()

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

left, right = st.columns([1,4])

# =========================================================
# LEFT PANEL
# =========================================================

with left:

    st.subheader("🔍 NLP Tasks")

    st.success("✔ Sentiment Analysis")

    st.success("✔ Opinion Mining")

    st.success("✔ Review Classification")

    st.success("✔ Large Language Models")

    st.success("✔ Hugging Face Transformers")

    st.divider()

    st.subheader("🛠 Technologies")

    st.markdown("""
- 🐍 Python

- 🤗 Transformers

- 🔥 Streamlit

- 🧠 DistilBERT

- 📊 NLP

- 🎬 Movie Reviews
""")

# =========================================================
# RIGHT PANEL
# =========================================================

with right:

    image_path = Path(__file__).parent / "KGF_2.png"

    if image_path.exists():

        image = Image.open(image_path)

        st.image(image, use_container_width=True)

    else:

        st.error("KGF_2.png not found!")

    st.markdown("## ✍ Write Your Review")

    review = st.text_area(
        "",
        height=180,
        placeholder="Example:\nKGF 2 is one of the best action movies ever made with outstanding performances."
    )

    if st.button("🎯 Analyze Review"):

        if review.strip() == "":

            st.warning("Please enter a review.")

        else:

            with st.spinner("Analyzing..."):

                result = classifier(review)[0]

            sentiment = result["label"]

            confidence = result["score"]

            st.divider()

            if sentiment == "POSITIVE":

                st.success("😊 Positive Review")

            else:

                st.error("😔 Negative Review")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(confidence)

            st.markdown("### Prediction Details")

            st.json(result)

st.divider()

st.markdown(
"""
<center>

### ❤️ Made with Streamlit | Hugging Face | Transformers

</center>
""",
unsafe_allow_html=True
)
"""
)
