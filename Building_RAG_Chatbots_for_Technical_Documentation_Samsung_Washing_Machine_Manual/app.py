import os
import streamlit as st

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="Samsung Washing Machine RAG Chatbot",
    page_icon="🧺",
    layout="wide"
)

st.title("🧺 Samsung Washing Machine Manual Chatbot")
st.write("Ask any question about your Samsung Washing Machine Manual.")

# -----------------------
# OpenAI API Key
# -----------------------

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = st.text_input(
        "Enter your OpenAI API Key",
        type="password"
    )

if not OPENAI_API_KEY:
    st.warning("Please provide your OpenAI API Key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------
# Load LLM
# -----------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# -----------------------
# Cache RAG
# -----------------------

@st.cache_resource
def load_rag():

    loader = UnstructuredHTMLLoader("samsung.html")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert Samsung Washing Machine assistant.

Answer ONLY from the provided context.

If the answer is not present, reply:
"I couldn't find this information in the manual."

Question:
{question}

Context:
{context}

Answer:
"""
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain


rag_chain = load_rag()

# -----------------------
# Chat History
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching manual..."):

        answer = rag_chain.invoke(question).content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )