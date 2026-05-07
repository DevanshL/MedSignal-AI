from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()

if __name__ == "__main__":
    texts = ["semaglutide causes nausea", "GLP-1 receptor agonist side effects"]
    embeddings = embed_texts(texts)
    print(f"Embedding dim: {len(embeddings[0])}")
    print(f"Count: {len(embeddings)}")