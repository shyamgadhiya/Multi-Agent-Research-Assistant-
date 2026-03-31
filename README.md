# Multi-Agent-Research-Assistant-
Multi-Agent Research Assistant, an autonomous AI system designed using LangGraph  to overcome the limitations of single-turn language models

Developed an autonomous multi-agent research system that decomposes complex user queries into parallel workflows and generates high-quality, citation-backed reports. The system leverages a hybrid Retrieval-Augmented Generation (RAG) pipeline combining live web search (Tavily API) and local vector search (FAISS) to ensure accurate and context-rich responses.

A Corrective RAG (CRAG) framework is implemented to evaluate retrieved document relevance and filter low-quality context before passing it to the language model, significantly reducing hallucinations. The system also includes a self-reflection loop with dynamic quality scoring, enabling agents to detect incomplete answers and automatically trigger refined re-search.

Multiple specialized AI agents powered by Google Gemini collaborate for reasoning, validation, and synthesis. The application features an interactive Streamlit interface with real-time execution tracing and SQLite-based session persistence for maintaining conversational context.

# Key Features:

Multi-agent orchestration using LangGraph
Hybrid RAG pipeline (FAISS + Tavily Search API)
Corrective RAG (CRAG) for hallucination reduction
Self-improving agent loop with quality scoring
Google Gemini-powered agent personas
Streamlit UI with live workflow visualization
SQLite-based memory for session persistence
