import os
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
from openai import OpenAI
from typing import List

# --- Persistent storage directory ---
PERSIST_DIR = Path("data/chroma")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Initialize or get collection ---
def _get_chroma_collection(collection_name: str):
    # ✅ Updated for Chroma v0.5+
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# --- Embed texts using SentenceTransformer ---
def _embed_texts(texts: List[str]):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode(texts)

# --- Retrieve relevant documents based on query ---
def retrieve_relevant_docs(collection_name: str, query: str, top_k: int = 3):
    collection = _get_chroma_collection(collection_name)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    q_emb = model.encode([query]).tolist()[0]

    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = []
    for docs_list in results.get("documents", []):
        docs.extend(docs_list)
    return docs

# --- Generate answer using OpenAI (modern API) ---
def generate_answer_with_openai(context: str, question: str, max_tokens=256):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set. Set it as environment variable to use OpenAI."

    client = OpenAI(api_key=api_key)
    prompt = f"""Context:\n{context}\n\nQuestion:\n{question}\n\nProvide a concise answer citing parts of the context. If the answer is not in the context, say you don't know."""

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    return resp.choices[0].message.content

# --- Main function to answer user query ---
def answer_query(collection_name: str, question: str, top_k: int = 3):
    docs = retrieve_relevant_docs(collection_name, question, top_k=top_k)
    if not docs:
        return "No relevant documents found in the database."

    context = "\n---\n".join(docs)

    if os.getenv("OPENAI_API_KEY"):
        return generate_answer_with_openai(context, question)
    else:
        return f"Retrieved {len(docs)} docs (no OpenAI key set):\n\n{context}"
