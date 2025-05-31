import streamlit as st
import numpy as np
import faiss
import json
import subprocess
from sentence_transformers import SentenceTransformer

# Load index and chunks
index = faiss.read_index("builder_index.faiss")
with open("chunk_lookup.json", "r") as f:
    chunk_lookup = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def query_local_llm(context, question):
    prompt = f"""Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    
    # Run using Ollama
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

# UI
st.title("🏗️ Local Builder Report Chatbot")
query = st.text_input("Ask a question about builder reports:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    valid_chunks = [chunk_lookup[str(i)] for i in I[0] if str(i) in chunk_lookup]
    context = "\n".join(valid_chunks)

    with st.spinner("Thinking..."):
        answer = query_local_llm(context, query)

    st.success(answer)