import streamlit as st
from transformers import pipeline
from PIL import Image
from pathlib import Path

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="KGF 2 Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
padding-top:0rem;
padding-bottom:2rem;
padding-left:0rem;
padding-right:0rem;
}

textarea{
font-size:18px !important;
}

.stButton>button{
width:100%;
background:#d32f2f;
color:white;
height:55px;
font-size:20px;
border-radius:10px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Sentiment Model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# -------------------------------------------------
# Load Front Poster
# -------------------------------------------------

image_path = Path(__file__).parent / "KGF_2.png"

if image_path.exists():
    image = Image.open(image_path)
    st.image(image, use_container_width=True)
else:
    st.error(f"Poster not found!\nExpected path:\n{image_path}")

# -------------------------------------------------
# Title
# -------------------------------------------------

st.markdown(
"""
<h1 style='text-align:center'>
🎬 Analyze Your Own KGF 2 Review
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<p style='text-align:center;color:gray;font-size:20px'>
Large Language Models • Hugging Face Transformers • NLP
</p>
""",
unsafe_allow_html=True
)

# -------------------------------------------------
# User Review
# -------------------------------------------------

review = st.text_area(
    "Write your movie review",
    height=180,
    placeholder="Example: KGF 2 is one of the greatest action movies ever made..."
)

# -------------------------------------------------
# Predict
# -------------------------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        result = classifier(review)[0]

        sentiment = result["label"]
        confidence = result["score"] * 100

        st.divider()

        if sentiment == "POSITIVE":
            st.success("😊 Positive Review")
        else:
            st.error("😔 Negative Review")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(float(result["score"]))

        st.json(result)

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.markdown(
"""
### 🔍 NLP Tasks

✅ Sentiment Analysis

✅ Opinion Mining

✅ Hugging Face Transformers

✅ Large Language Models

✅ Review Understanding

---
Made with ❤️ using Streamlit
"""
)
