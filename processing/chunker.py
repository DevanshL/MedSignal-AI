from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)

def chunk_documents(docs: list[dict], text_field: str = "text") -> list[dict]:
    chunks = []
    for doc in docs:
        text = doc.get(text_field, "")
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({**doc, "chunk_id": i, "chunk_text": chunk})
    return chunks

if __name__ == "__main__":
    sample = {"id": "123", "text": "Semaglutide is a GLP-1 receptor agonist. " * 50}
    chunks = chunk_documents([sample])
    print(f"Generated {len(chunks)} chunks")
    print(chunks[0]["chunk_text"][:100])