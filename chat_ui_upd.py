import streamlit as st
import numpy as np
import faiss
import json
import io
from sentence_transformers import SentenceTransformer
import ollama
import os
st.write("🗂️ Saving feedback to directory:", os.getcwd())

# --- Function to query local LLM via Ollama API ---
def query_local_llm(context, question):
    prompt = f"""Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}],
        options={"temperature": 0}  # deterministic output
    )
    return response['message']['content']

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load FAISS index and chunk mapping ---
index = faiss.read_index("builder_index.faiss")
with open("chunk_lookup.json", "r") as f:
    chunk_lookup = json.load(f)

# --- Load Sentence Transformer model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI ---
st.title("🏗️ Local Builder Report Chatbot")

# Input and ask button
query = st.text_input("Ask a question about builder reports:")
submit = st.button("Ask")

# Process question only when 'Ask' button is clicked
if submit and query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    valid_chunks = [chunk_lookup[str(i)] for i in I[0] if str(i) in chunk_lookup]
    context = "\n".join(valid_chunks)

    with st.spinner("Thinking..."):
        answer = query_local_llm(context, query)
        st.session_state.chat_history.append({"question": query, "answer": answer})
        st.session_state.last_answer = answer  # ✅ Save answer to session

    st.success(answer)

    # --- Feedback Section ---
feedback = st.radio("Was this answer helpful?", ["👍", "👎"], key=f"feedback_{len(st.session_state.chat_history)}")

if st.button("Submit Feedback"):
    try:
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{query}\t{st.session_state.last_answer.strip()}\t{feedback}\n")
        st.success("✅ Feedback saved!")
    except Exception as e:
        st.error(f"❌ Error saving feedback: {e}")


# --- Display chat history ---
if st.session_state.chat_history:
    st.markdown("### 🔄 Chat History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")

    # --- Download chat history as TXT ---
    history_text = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in st.session_state.chat_history])
    st.download_button("💾 Download Chat History as TXT", data=history_text, file_name="chat_history.txt")

    # --- Clear chat history ---
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
