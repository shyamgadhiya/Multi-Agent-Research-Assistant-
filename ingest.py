"""
Run this once before main.py to build the FAISS vector store.

Usage:
    python ingest.py --docs_dir ./docs

Supports: .txt, .pdf, .md files in the given directory.
"""

import argparse
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from setup import embeddings

VECTORSTORE_PATH = "vectorstore"

def ingest(docs_dir: str):
    docs_path = Path(docs_dir)
    all_docs = []

    for file in docs_path.rglob("*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix in (".txt", ".md"):
            loader = TextLoader(str(file))
        else:
            continue

        print(f"Loading: {file.name}")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file.name
        all_docs.extend(docs)

    if not all_docs:
        print("No documents found. Add .txt, .pdf or .md files to the docs directory.")
        return

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks from {len(all_docs)} documents.")

    # Build and save FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to ./{VECTORSTORE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="./docs", help="Path to documents folder")
    args = parser.parse_args()
    ingest(args.docs_dir)
