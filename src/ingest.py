import os
import argparse
import uuid
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# --- Constants ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))

# --- Extract text from PDF ---
def extract_text_from_pdf(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

# --- Split text into chunks ---
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# --- Ingest PDF into ChromaDB ---
def ingest_pdf(pdf_path: str, collection_name: str = "docs", embedding_model_name: str = EMBEDDING_MODEL_NAME):
    print(f"Ingesting {pdf_path} into collection '{collection_name}'...")

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    if not chunks:
        print("No text extracted from PDF. Skipping ingestion.")
        return 0

    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)

    # ✅ Use the new Chroma v0.5+ API
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(name=collection_name)

    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings.tolist())

    print(f"✅ Added {len(chunks)} chunks. Persisted to {PERSIST_DIR}")
    return len(chunks)

# --- CLI entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file")
    parser.add_argument("--collection_name", default="docs", help="Name of the Chroma collection")
    args = parser.parse_args()

    ingest_pdf(args.pdf_path, args.collection_name)
