# ================================================================
# app/streamlit_app.py — RAG-Powered Chatbot for PDFs
# ================================================================

import os
import sys
from pathlib import Path

# --- Ensure project root (so src imports work correctly) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ../
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Load environment variables automatically (.env file) ---
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass  # dotenv missing is fine; user can set env manually

# --- Now import Streamlit and project modules ---
import streamlit as st
import pathlib
from src.ingest import ingest_pdf
from src.rag import answer_query

# --- Streamlit page setup ---
st.set_page_config(page_title='RAG PDF Chatbot', layout='wide')
st.title('🧠 RAG-Powered Chatbot for PDFs')

# --- Sidebar settings ---
with st.sidebar:
    st.header('Settings')
    collection_name = st.text_input('Collection name', value='docs')
    chunk_size = st.slider('Chunk size', min_value=200, max_value=1200, value=500, step=50)
    chunk_overlap = st.slider('Chunk overlap', min_value=0, max_value=300, value=100, step=10)
    st.markdown('---')
    st.markdown('**LLM:** This demo prefers OpenAI (set `OPENAI_API_KEY`).')
    st.markdown('If not set, it will only show retrieved chunks.')

# --- File upload and ingestion ---
uploaded = st.file_uploader('Upload a PDF', type=['pdf'])
if uploaded is not None:
    temp_dir = pathlib.Path('data/pdfs')
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded.name

    with open(temp_path, 'wb') as f:
        f.write(uploaded.getbuffer())

    st.success(f'Saved {uploaded.name} to data/pdfs/')

    if st.button('Ingest uploaded PDF'):
        with st.spinner('Ingesting...'):
            n = ingest_pdf(str(temp_path), collection_name=collection_name)
            st.success(f'Ingested {n} chunks into collection "{collection_name}"')

# --- Query interface ---
query = st.text_input('Ask a question about your documents')
if st.button('Ask'):
    if not query.strip():
        st.warning('Enter a question first')
    else:
        with st.spinner('Retrieving and generating answer...'):
            resp = answer_query(collection_name, query, top_k=3)
            st.markdown('**Answer:**')
            st.write(resp)
            st.markdown('---')
            st.markdown('**Notes:**')
            st.write('For improved results, set your `OPENAI_API_KEY` in the .env file.')
