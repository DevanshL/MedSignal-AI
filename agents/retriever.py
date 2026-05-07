from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from processing.reranker import rerank

load_dotenv()

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def retriever_node(state: dict) -> dict:
    query = state.get("plan", state["query"])
    collection = chroma_client.get_or_create_collection("medsignal")
    
    # Dense retrieval
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20
    )
    
    docs = []
    for i, doc_text in enumerate(dense_results["documents"][0]):
        docs.append({
            "text": doc_text,
            "metadata": dense_results["metadatas"][0][i],
            "distance": dense_results["distances"][0][i],
            "id": dense_results["ids"][0][i]
        })
    
    # BM25 sparse retrieval on top dense results
    corpus = [d["text"].split() for d in docs]
    if corpus:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.split())
        for i, doc in enumerate(docs):
            doc["bm25_score"] = float(scores[i])
            doc["hybrid_score"] = (1 - doc["distance"]) * 0.7 + doc["bm25_score"] * 0.3
        docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    reranked = rerank(query, docs[:15], top_k=10)
    return {**state, "retrieved_docs": reranked}

if __name__ == "__main__":
    result = retriever_node({"query": "semaglutide renal adverse events", "plan": "semaglutide renal impairment adverse effects"})
    print(f"Retrieved {len(result['retrieved_docs'])} docs")
    print(result['retrieved_docs'][0]['text'][:200])