import streamlit as st
from transformers import pipeline

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="KGF 2 NLP Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# --------------------------
# Hide Streamlit Style
# --------------------------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{
    padding-top:0rem;
    padding-bottom:0rem;
    padding-left:0rem;
    padding-right:0rem;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:10px;
    background:#c62828;
    color:white;
    font-weight:bold;
}

textarea{
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# --------------------------
# Front Poster
# --------------------------
st.image("KGF_2.png", use_container_width=True)

st.markdown("<h1 style='text-align:center;'>🎬 Analyze Your Own Movie Review</h1>",
unsafe_allow_html=True)

st.markdown(
"<h4 style='text-align:center;color:gray;'>Powered by Hugging Face Transformers & LLMs</h4>",
unsafe_allow_html=True)

# --------------------------
# User Input
# --------------------------
review = st.text_area(
    "Write your review",
    placeholder="Example: KGF 2 is an outstanding movie with breathtaking action sequences..."
)

# --------------------------
# Prediction
# --------------------------
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        result = classifier(review)[0]

        sentiment = result["label"]
        confidence = result["score"] * 100

        st.markdown("---")

        if sentiment == "POSITIVE":
            st.success("😊 Positive Review")
        else:
            st.error("😔 Negative Review")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.json(result)

st.markdown("---")

st.markdown(
"""
<center>

### NLP Tasks Performed

✔ Sentiment Analysis

✔ Large Language Models (LLMs)

✔ Hugging Face Transformers

✔ Review Understanding

✔ Movie Opinion Mining

</center>
""",
unsafe_allow_html=True
)
