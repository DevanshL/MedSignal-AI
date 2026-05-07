import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(name: str = "medsignal"):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def upsert_chunks(chunks: list[dict], embeddings: list[list[float]], collection_name: str = "medsignal"):
    col = get_or_create_collection(collection_name)
    col.upsert(
        ids=[f"{c.get('report_id', c.get('nct_id', c.get('id', 'doc')))}_{c['chunk_id']}" for c in chunks],
        embeddings=embeddings,
        documents=[c["chunk_text"] for c in chunks],
        metadatas=[{k: v for k, v in c.items() if k not in ("chunk_text",) and isinstance(v, (str, int, float, bool))} for c in chunks]
    )
    print(f"Upserted {len(chunks)} chunks")

def query_collection(embedding: list[float], n_results: int = 5, collection_name: str = "medsignal"):
    col = get_or_create_collection(collection_name)
    return col.query(query_embeddings=[embedding], n_results=n_results)

if __name__ == "__main__":
    col = get_or_create_collection()
    print(f"Collection ready. Count: {col.count()}")