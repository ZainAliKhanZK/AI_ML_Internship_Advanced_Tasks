# 📚 Context-Aware RAG Chatbot

A conversational chatbot that retrieves answers from a custom knowledge base using Retrieval-Augmented Generation (RAG), while maintaining memory of the conversation across turns. Built with LangChain, ChromaDB, and Streamlit, with support for switching between multiple LLM providers.

## 🔗 Live Demo

Try the deployed app here: **[zk chat app](https://zkchat.streamlit.app/)**

## 🎯 Problem Statement

Standard chatbots either rely solely on an LLM's pretrained knowledge (which can be outdated or lack domain-specific detail) or have no memory of earlier turns in a conversation, making multi-turn interactions feel disconnected and repetitive.

## 🚀 Goal

Build a context-aware chatbot that retrieves relevant information from a custom knowledge base using Retrieval-Augmented Generation, while maintaining conversational memory across turns, and deploy it as an interactive Streamlit app with support for switching between multiple LLM providers.

## 📊 Results

- Built a full RAG pipeline: Wikipedia pages fetched via a direct API call, split into 500-character chunks with overlap, embedded using `all-MiniLM-L6-v2`, and stored in a persistent ChromaDB vector store.
- Verified retrieval quality — queries such as *"What is a neural network?"* correctly pulled relevant chunks from the indexed articles.
- Confirmed conversational memory works correctly — a follow-up question using a pronoun (*"What is his wife's name?"* after asking about Bill Gates) was correctly resolved using chat history, without restating the subject.
- Deployed as a Streamlit app with live switching between **three LLM providers** (Groq, Gemini, and Hugging Face), each maintaining its own independent conversation memory.
- Added source attribution in the UI, showing which Wikipedia articles each answer was grounded in.

## 🧠 Approach

1. **Knowledge base**: Fetched full Wikipedia articles (Machine Learning, Artificial Intelligence, Neural Network) via Wikipedia's REST API.
2. **Chunking**: Split articles into 500-character chunks with 50-character overlap using `RecursiveCharacterTextSplitter`, preserving context across chunk boundaries.
3. **Embeddings & vector store**: Generated embeddings with `sentence-transformers/all-MiniLM-L6-v2` and stored them in a persistent **ChromaDB** vector store.
4. **Retrieval**: Configured a retriever to pull the top-3 most relevant chunks for any user query.
5. **LLM integration**: Connected three interchangeable LLM providers — **Groq** (llama-3.3-70b-versatile), **Google Gemini** (gemini-2.0-flash), and **Hugging Face** (Llama-3.1-8B-Instruct) — selectable live from the app.
6. **Conversational memory**: Used `ConversationBufferMemory` so the chatbot resolves references to earlier turns (e.g. pronouns) correctly.
7. **Chain assembly**: Combined retrieval, memory, and the selected LLM using `ConversationalRetrievalChain`.
8. **Deployment**: Built an interactive Streamlit app with a chat interface, source citations, and a sidebar model switcher.

## 🛠️ Tech Stack

- Python
- LangChain (`langchain-classic`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq`, `langchain-google-genai`)
- ChromaDB
- Streamlit
- Groq API / Google Gemini API / Hugging Face Inference API

## 📁 Project Structure

```
├── streamlit_rag_app.py    # Streamlit deployment app
├── chroma_db/               # Persisted vector store (generated on first run)
├── requirements.txt
└── README.md
```

## 💻 Usage

**Setup:**
```bash
git clone https://github.com/ZainAliKhanZK/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

**Set API keys as environment variables** (only the providers you want to use need valid keys):
```bash
# Windows (PowerShell)
$env:GROQ="your_groq_api_key"
$env:GEM="your_gemini_api_key"
$env:HF="your_huggingface_api_key"
```

**Run the app:**
```bash
streamlit run streamlit_rag_app.py
```

The first run builds the knowledge base (fetches Wikipedia pages, chunks, and embeds them) — subsequent runs reuse the cached vector store for faster startup.

## ⚙️ Switching Models

Use the sidebar dropdown to switch between:
- **Groq** (llama-3.3-70b-versatile)
- **Gemini** (gemini-2.0-flash)
- **Hugging Face** (Llama-3.1-8B-Instruct)

Each model maintains its own independent conversation memory — switching models starts a fresh conversation.

## ⚠️ Notes & Limitations

- Knowledge base is currently limited to three Wikipedia topics (Machine Learning, Artificial Intelligence, Neural Network) — easily extendable by adding more topics to the `TOPICS` list.
- Requires a valid API key for whichever LLM provider is selected; free-tier quotas apply and may be rate-limited or exhausted depending on usage.
- Wikipedia API requests require a `User-Agent` header — omitting this causes request failures.

## 👤 Author

**Zain Ali Khan**
[GitHub](https://github.com/ZainAliKhanZK) · [Hugging Face](https://huggingface.co/ZainAliKhanZK)
