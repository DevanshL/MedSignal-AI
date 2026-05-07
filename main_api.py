from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from agents.graph import build_graph

app = FastAPI(title="MedSignal AI", version="1.0")
graph = build_graph()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    critic_passed: bool
    citations: list[str]
    critique: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    result = graph.invoke({
        "query": request.query,
        "plan": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "critic_passed": False,
        "critique": ""
    })
    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        critic_passed=result["critic_passed"],
        citations=result["citations"],
        critique=result.get("critique", "")
    )

@app.get("/health")
async def health():
    return {"status": "ok"}