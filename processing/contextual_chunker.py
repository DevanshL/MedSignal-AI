from agents.llm import generate
from processing.chunker import chunk_text

def add_context_to_chunk(document_text: str, chunk: str) -> str:
    prompt = f"""Given this document excerpt:
<document>
{document_text[:2000]}
</document>

Give a 1-2 sentence context for this chunk that situates it within the document. Be specific about drug names, study type, population, timeframe if present.

<chunk>
{chunk}
</chunk>

Context (1-2 sentences only):"""
    
    context = generate(prompt)
    return f"{context}\n\n{chunk}"

def contextual_chunk_documents(docs: list[dict], text_field: str = "text") -> list[dict]:
    chunks = []
    for doc in docs:
        text = doc.get(text_field, "")
        if not text:
            continue
        raw_chunks = chunk_text(text)
        for i, chunk in enumerate(raw_chunks):
            enriched = add_context_to_chunk(text, chunk)
            chunks.append({**doc, "chunk_id": i, "chunk_text": enriched})
    return chunks