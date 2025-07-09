import gradio as gr
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import hashlib
import os
import pickle

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global in-memory session cache
session_cache = {}

# Extract text from the first N pages using OCR
def extract_text_from_pdf(pdf_path, max_pages=10):
    images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
    all_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        all_text += text + "\n"
    return all_text

# Chunk long text into smaller passages
def chunk_text(text, max_tokens=500):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) <= max_tokens:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Build or load FAISS index from cached files
def create_or_load_index(chunks, cache_key):
    index_path = f"/tmp/{cache_key}_faiss.index"
    chunk_path = f"/tmp/{cache_key}_chunks.pkl"

    if os.path.exists(index_path) and os.path.exists(chunk_path):
        index = faiss.read_index(index_path)
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)
    else:
        embeddings = embed_model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        faiss.write_index(index, index_path)
        with open(chunk_path, "wb") as f:
            pickle.dump(chunks, f)
    return index, chunks

# Find relevant chunks and return text
def answer_question(query, index, chunks, top_k=3):
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec), top_k)
    context = "\n\n".join([chunks[i] for i in I[0]])
    return f"📘 Context:\n{context}\n\n❓ Q: {query}\n\n💬 A: (This is the most relevant content from your PDF. You can now use this or plug it into an LLM.)"

# Gradio processing function
def process_pdf_and_question(pdf_file, question, pages_to_process):
    if not pdf_file:
        return "Please upload a PDF file."

    pdf_bytes = pdf_file.read()
    cache_key = hashlib.md5(pdf_bytes).hexdigest()
    pdf_path = f"/tmp/{cache_key}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    if cache_key not in session_cache:
        text = extract_text_from_pdf(pdf_path, max_pages=pages_to_process)
        chunks = chunk_text(text)
        index, chunks = create_or_load_index(chunks, cache_key)
        session_cache[cache_key] = (index, chunks)
    else:
        index, chunks = session_cache[cache_key]

    return answer_question(question, index, chunks)

# Gradio Interface
gr.Interface(
    fn=process_pdf_and_question,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(lines=2, placeholder="Ask your question...", label="Question"),
        gr.Slider(minimum=1, maximum=30, value=10, label="Pages to process")
    ],
    outputs="text",
    title="📄 Offline PDF Q&A with OCR + Embeddings",
    description="Upload a scanned or text-based PDF, ask questions, and get context from its content."
).launch(server_name="127.0.0.1", share=True)
