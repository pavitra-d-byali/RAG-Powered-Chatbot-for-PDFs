# 🧠 RAG-Powered Chatbot for PDFs

A minimal **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDFs and ask context-aware questions using **Streamlit**.  
Built with **ChromaDB**, **SentenceTransformers**, and **OpenAI GPT / HuggingFace models**.

---

##  Features
- Upload any PDF and chat with it.
- PDF text extraction → chunking → embeddings → retrieval → LLM answer.
- Local **Chroma vector database** (persistent).
- Works with **OpenAI** or **HuggingFace** APIs.

---

##  Quick Start
```bash
python -m venv .venv
.\.venv\Scripts\activate      # (Windows)
pip install -r requirements.txt

# Set API key
setx OPENAI_API_KEY "sk-..."

# Run app
streamlit run app/streamlit_app.py


Open http://localhost:8501
 in your browser.

📁 Project Structure
app/streamlit_app.py    → Streamlit UI
src/ingest.py           → PDF ingestion & embedding
src/rag.py              → Retrieval & response generation
data/pdfs/              → Your uploaded PDFs
data/chroma/            → Local vector DB

 Example
python -m src.ingest --pdf_path data/pdfs/sample.pdf --collection_name my_docs


Then chat via the Streamlit UI — e.g.

“Summarize this document and list its main points.”

 Stack

Streamlit · ChromaDB · SentenceTransformers · OpenAI GPT / HuggingFace · LangChain Text Splitter# RAG-Powered-Chatbot-for-PDFs
