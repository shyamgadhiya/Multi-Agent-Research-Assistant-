import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langsmith import Client
import streamlit as st

# ── Environment variables ──────────────────────────────────────────────────────
# Set these in your .env or shell before running:
#
#   GOOGLE_API_KEY       = your Gemini API key
#   TAVILY_API_KEY       = your Tavily search API key
#   HUGGINGFACEHUB_API_TOKEN = your Hugging Face token
#   LANGCHAIN_API_KEY    = your LangSmith API key
#   LANGCHAIN_TRACING_V2 = "true"
#   LANGCHAIN_PROJECT     = "multi-agent-research-assistant"
#
# LangSmith tracing is activated automatically when LANGCHAIN_TRACING_V2=true
# ──────────────────────────────────────────────────────────────────────────────

# # LLM — Gemini 3.1 flash for planner/critic/writer (reasoning-heavy)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.1-flash-lite-preview",
#     temperature=0,
#     google_api_key=os.environ["GOOGLE_API_KEY"],
# )

# # LLM — Gemini 3.1 Flash for researcher nodes (faster, parallel calls)
# llm_fast = ChatGoogleGenerativeAI(
#     model="gemini-3.1-flash-lite-preview",
#     temperature=0,
#     google_api_key=os.environ["GOOGLE_API_KEY"],
# )

# # Embeddings — for RAG vector store
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={
#         "device": "cuda",  # change to "cuda" if you have a GPU
#         "token": os.environ["HUGGINGFACEHUB_API_TOKEN"],
#     },
#     encode_kwargs={"normalize_embeddings": True},
# )


# LLM — Gemini 3.1 flash for planner/critic/writer
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0,
    google_api_key=st.secrets.get("GOOGLE_API_KEY", ""),
)

# LLM — Gemini 3.1 flash for researcher nodes
llm_fast = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0,
    google_api_key=st.secrets.get("GOOGLE_API_KEY", ""),
)

# Embeddings — for RAG vector store
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device": "cpu",  # Streamlit Cloud usually does not provide CUDA
        "token": st.secrets.get("HUGGINGFACEHUB_API_TOKEN", ""),
    },
    encode_kwargs={"normalize_embeddings": True},
)

# Web search tool (Tavily)
web_search = TavilySearch(max_results=4)

# LangSmith client (optional — for manual run logging / eval)
langsmith_client = Client()


def load_vectorstore(path: str) -> FAISS | None:
    """
    Load FAISS index from an explicit path every time — no caching.
    Each call opens a fresh index from disk so switching collections
    always loads exactly the right documents.
 
    path must be an absolute or relative path to the collection folder
    containing faiss.index and index.pkl written by FAISS.save_local().
    """
    if not path:
        return None
 
    # Resolve to absolute path to avoid any working-directory ambiguity
    abs_path = os.path.abspath(path)
 
    if not os.path.isdir(abs_path):
        return None
 
    # Both files must exist — if either is missing the index is corrupt
    if not os.path.exists(os.path.join(abs_path, "index.faiss")) and \
       not os.path.exists(os.path.join(abs_path, "faiss.index")):
        return None
 
    # Load fresh from disk — FAISS.load_local creates a new in-memory index object
    # each time it is called, so there is no cross-collection contamination
    return FAISS.load_local(
        abs_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
