from transformers import pipeline, AutoTokenizer

def answer_query(query, index, chunks, embed_model, top_k=2):
    query_embedding = embed_model.encode([query])
    _, I = index.search(query_embedding, top_k)

    context = "\n\n".join([chunks[i] for i in I[0]])
    
    prompt = f"""Context:
{context}

Question: {query}
Answer:"""

    # 🔍 Token length visualization
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    num_tokens = len(tokenizer(prompt)["input_ids"])
    print(f"🔢 Prompt token length: {num_tokens} tokens")

    # ⚠ Optional: Truncate if too long for flan-t5-base (512 tokens max)
    if num_tokens > 512:
        print("⚠ Prompt too long. Trimming context to fit within 512 tokens.")
        context = "\n\n".join([chunks[i] for i in I[0][:1]])  # Reduce top_k to 1
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    answer = qa_pipeline(prompt, max_new_tokens=300)[0]['generated_text']
    return answer
