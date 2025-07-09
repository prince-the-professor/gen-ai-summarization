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
