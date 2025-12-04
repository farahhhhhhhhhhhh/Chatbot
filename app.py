import streamlit as st
from pypdf import PdfReader
import numpy as np
import faiss
from huggingface_hub import InferenceClient

# ----------------------------------------------------------
# Streamlit Title
# ----------------------------------------------------------
st.title("Chatbot")
st.write("Ask any question about the MS1 Checklist PDF")


# ----------------------------------------------------------
# 1. LOAD PDF
# ----------------------------------------------------------
def load_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

pdf_path = "ms1.pdf"
full_text = load_text(pdf_path)


# ----------------------------------------------------------
# 2. SPLIT TEXT INTO CHUNKS
# ----------------------------------------------------------
def chunk_text(text, chunk_size=400, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = chunk_text(full_text)


# ----------------------------------------------------------
# 3. EMBEDDINGS (REMOTE VIA HUGGINGFACE API)
# ----------------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

embed_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=HF_TOKEN
)

def embed(texts):
    """Get embeddings from HuggingFace Inference API."""
    result = embed_client.feature_extraction(texts)
    return np.array(result).astype("float32")

chunk_embeddings = embed(chunks)


# ----------------------------------------------------------
# 4. FAISS VECTOR STORE
# ----------------------------------------------------------
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(chunk_embeddings)


# ----------------------------------------------------------
# 5. LLM CLIENT (GEMMA 2B IT)
# ----------------------------------------------------------
llm_client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN
)

def llm(prompt):
    """Generate answer using Gemma 2B IT."""
    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.2
    )
    return response.choices[0].message["content"]


# ----------------------------------------------------------
# 6. RAG RETRIEVAL + GENERATION
# ----------------------------------------------------------
def retrieve(query, k=3):
    query_emb = embed([query])
    distances, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

def answer_rag(query):
    retrieved = retrieve(query)
    context = "\n\n".join(retrieved)

    prompt = f"""
You are a helpful assistant. Answer the question ONLY using the context provided.

Context:
{context}

Question: {query}
"""

    answer = llm(prompt)
    return answer, retrieved


# ----------------------------------------------------------
# 7. STREAMLIT UI
# ----------------------------------------------------------
user_query = st.text_input("Enter your question about Milestone 1:")

if st.button("Get Answer"):
    if user_query.strip():
        answer, retrieved_chunks = answer_rag(user_query)

        st.write("### ✅ Answer:")
        st.write(answer)

        st.write("---")
        with st.expander("📄 Retrieved Source Chunks"):
            for i, ch in enumerate(retrieved_chunks, 1):
                st.write(f"**Chunk {i}:**")
                st.write(ch)
                st.write("---")

