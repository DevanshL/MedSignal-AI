from typing import TypedDict, Optional
from langchain_core.documents import Document

class MedSignalState(TypedDict):
    query: str
    plan: str
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    answer: str
    citations: list[str]
    confidence: float
    critic_passed: bool
    critique: Optional[str]