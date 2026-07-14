"""
Context-Aware RAG Chatbot - Streamlit App
(Wraps the notebook pipeline: Wikipedia loading -> chunking -> ChromaDB -> Groq -> memory)

Run with:
    streamlit run streamlit_rag_app.py

Requires environment variables (or Streamlit secrets) for API keys:
    HF, GEM, GROQ
"""

import os
import requests
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEADERS = {"User-Agent": "ZainRAGChatbot/1.0 (student project; contact: your_email@example.com)"}
TOPICS = ["Machine learning", "Artificial intelligence", "Neural network"]

st.set_page_config(page_title="Context-Aware RAG Chatbot", page_icon="📚")
st.title("📚 Context-Aware RAG Chatbot")
st.write("Ask me anything about Machine Learning, Artificial Intelligence, or Neural Networks.")

hf_key = st.secrets.get("HF", os.environ.get("HF", ""))
gem_key = st.secrets.get("GEM", os.environ.get("GEM", ""))
groq_key = st.secrets.get("GROQ", os.environ.get("GROQ", ""))

MODEL_OPTIONS = {
    "Groq (llama-3.3-70b-versatile)": "groq",
    "Gemini (gemini-2.0-flash)": "gemini",
    "Hugging Face (Llama-3.1-8B-Instruct)": "huggingface",
}

with st.sidebar:
    st.header("Model settings")
    selected_label = st.selectbox("Choose an LLM", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[selected_label]


# ---------------------------------------------------------------------------
# Wikipedia loading (same function as the notebook)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Build the retriever once -- expensive part (Wikipedia fetch, chunking,
# embedding), shared across whichever LLM is selected.
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"

@st.cache_resource(show_spinner="Loading knowledge base...")
def build_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If chroma_db is already populated (as you've uploaded it), just load it —
    # don't re-fetch Wikipedia and re-embed everything on every cold start.
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        documents = []
        for topic in TOPICS:
            content = fetch_full_wikipedia_page(topic)
            if content:
                documents.append(Document(page_content=content, metadata={"title": topic}))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR
        )

    return vectorstore.as_retriever(search_kwargs={"k": 3})
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------------------------------------------------------------------------
# Build the LLM based on the sidebar selection -- cheap, cached per model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_llm(model_choice: str):
    if model_choice == "groq":
        if not groq_key:
            st.error("No Groq API key found. Set GROQ as an environment variable or Streamlit secret.")
            st.stop()
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=groq_key)

    if model_choice == "gemini":
        if not gem_key:
            st.error("No Gemini API key found. Set GEM as an environment variable or Streamlit secret.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, api_key=gem_key)

    if model_choice == "huggingface":
        if not hf_key:
            st.error("No Hugging Face API key found. Set HF as an environment variable or Streamlit secret.")
            st.stop()
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=hf_key,
        )
        return ChatHuggingFace(llm=llm_endpoint)

    raise ValueError(f"Unknown model choice: {model_choice}")


# ---------------------------------------------------------------------------
# Build the chain: retriever (shared) + selected LLM + fresh memory
# ---------------------------------------------------------------------------
def build_chain(model_choice: str):

    retriever = build_retriever()
    llm = build_llm(model_choice)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_chain(selected_model)
    st.session_state.active_model = selected_model

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.active_model != selected_model:

    st.session_state.qa_chain = build_chain(selected_model)

    st.session_state.active_model = selected_model

    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"question": user_input})
            answer = result["answer"]
            sources = result.get("source_documents", [])

        st.write(answer)

        if sources:
            with st.expander("Sources"):
                seen_titles = set()
                for doc in sources:
                    title = doc.metadata.get("title", "Unknown")
                    if title not in seen_titles:
                        st.write(f"- {title}")
                        seen_titles.add(title)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("🗑 Clear Conversation"):

    st.session_state.messages = []

    st.session_state.qa_chain = build_chain(selected_model)

    st.rerun()
