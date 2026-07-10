import os
import streamlit as st
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain.chains import ConversationalRetrievalChain

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="🏥 Star Health Insurance Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Star Health Insurance RAG Chatbot")

st.markdown(
"""
Ask questions about **Star Health Insurance Plans**.

Supported Models:

✅ OpenAI GPT

✅ Google Gemini
"""
)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("⚙ Settings")

model_provider = st.sidebar.radio(
    "Select LLM",
    [
        "OpenAI",
        "Gemini"
    ]
)

api_key = st.sidebar.text_input(
    "Enter API Key",
    type="password"
)

st.sidebar.markdown("---")

html_file = st.sidebar.file_uploader(
    "Upload StarHealth HTML",
    type=["html"]
)

st.sidebar.markdown("---")

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.2
)

# -------------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------------------------------------------
# SAVE HTML
# -------------------------------------------------------

if html_file:

    with open("insurance.html", "wb") as f:
        f.write(html_file.getbuffer())

    st.sidebar.success("HTML Saved")

# -------------------------------------------------------
# VECTOR DATABASE
# -------------------------------------------------------

def build_vector_db():

    loader = BSHTMLLoader("insurance.html")

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    if model_provider == "OpenAI":

        embeddings = OpenAIEmbeddings(
            api_key=api_key
        )

    else:

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local("vectorstore")

    return vector_db

# -------------------------------------------------------
# LOAD VECTOR DB
# -------------------------------------------------------

def load_vector_db():

    if model_provider == "OpenAI":

        embeddings = OpenAIEmbeddings(
            api_key=api_key
        )

    else:

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

# -------------------------------------------------------
# CREATE QA CHAIN
# -------------------------------------------------------

def create_chain():

    if not os.path.exists("vectorstore"):

        with st.spinner("Creating Vector Database..."):

            db = build_vector_db()

    else:

        db = load_vector_db()

    retriever = db.as_retriever(
        search_kwargs={"k":4}
    )

    if model_provider == "OpenAI":

        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=temperature
        )

    else:

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature
        )

    qa = ConversationalRetrievalChain.from_llm(

        llm=llm,

        retriever=retriever,

        return_source_documents=True,
    )

    return qa

# -------------------------------------------------------
# LOAD CHAIN
# -------------------------------------------------------

if api_key:

    if st.session_state.qa_chain is None:

        st.session_state.qa_chain = create_chain()
      # -------------------------------------------------------
# CLEAR CHAT
# -------------------------------------------------------

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# -------------------------------------------------------
# SAMPLE QUESTIONS
# -------------------------------------------------------

st.markdown("### 💡 Sample Questions")

sample_questions = [
    "What are different Maternity Health Insurance Plans?",
    "What benefits do Health Insurance policies offer?",
    "Why should you get Health Insurance when you're young?",
    "How many types of Health Insurance policies are there?",
    "What are the different types of health insurance schemes in India?"
]

cols = st.columns(len(sample_questions))

for i, question in enumerate(sample_questions):
    if cols[i].button(f"Q{i+1}"):
        st.session_state["selected_question"] = question

# -------------------------------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------------------------------

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# -------------------------------------------------------
# CHAT INPUT
# -------------------------------------------------------

query = st.chat_input("Ask anything about Star Health Insurance...")

if "selected_question" in st.session_state:
    query = st.session_state.pop("selected_question")

# -------------------------------------------------------
# PROCESS QUERY
# -------------------------------------------------------

if query:

    if not api_key:
        st.error("⚠ Please enter your API Key in the sidebar.")
        st.stop()

    if not os.path.exists("insurance.html"):
        st.error("⚠ Please upload the Star Health HTML file first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append(("user", query))

    with st.chat_message("user"):
        st.markdown(query)

    # Convert history for LangChain
    history = []

    for role, text in st.session_state.chat_history[:-1]:
        if role == "user":
            history.append((text, ""))
        else:
            if history:
                last_user = history[-1][0]
                history[-1] = (last_user, text)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                result = st.session_state.qa_chain.invoke({
                    "question": query,
                    "chat_history": history
                })

                answer = result["answer"]

                st.markdown(answer)

                st.session_state.chat_history.append(
                    ("assistant", answer)
                )

                sources = result.get("source_documents", [])

                if sources:

                    with st.expander("📚 Source Documents"):

                        for i, doc in enumerate(sources, start=1):

                            st.markdown(f"### Source {i}")

                            content = doc.page_content.strip()

                            if len(content) > 800:
                                content = content[:800] + "..."

                            st.write(content)

                            st.markdown("---")

            except Exception as e:
                st.error(f"Error: {e}")

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>🏥 Star Health Insurance RAG Chatbot</h4>
    <p>Powered by LangChain • FAISS • OpenAI • Gemini • Streamlit</p>
    </center>
    """,
    unsafe_allow_html=True,
)
