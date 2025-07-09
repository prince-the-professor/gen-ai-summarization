import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_or_load_index(chunks, index_path="faiss.index", chunk_path="chunks.pkl"):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists(index_path) and os.path.exists(chunk_path):
        print(f"📂 Loading cached index from: {index_path}")
        index = faiss.read_index(index_path)
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("⚙️ Creating new FAISS index and embeddings...")
        embeddings = model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        # Save index and chunks
        faiss.write_index(index, index_path)
        with open(chunk_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"✅ Saved index: {index_path}")
        print(f"✅ Saved chunks: {chunk_path}")

    return index, chunks, model
