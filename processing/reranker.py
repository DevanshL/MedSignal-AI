from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    for i, doc in enumerate(docs):
        doc["rerank_score"] = float(scores[i])
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]