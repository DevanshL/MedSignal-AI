from dotenv import load_dotenv
load_dotenv()

import time
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from agents.graph import build_graph

graph = build_graph()

def run_query(question: str) -> dict:
    return graph.invoke({
        "query": question,
        "plan": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "critic_passed": False,
        "critique": ""
    })

def test_answer_cites_sources():
    result = run_query("What are the adverse events of semaglutide?")
    assert "[Source" in result["answer"], "Answer must contain citations"
    print("✅ Citation test passed")

def test_critic_passes():
    result = run_query("What are the gastrointestinal adverse events of semaglutide?")
    assert result["critic_passed"] == True, f"Critic failed: {result['critique']}"
    print("✅ Critic test passed")

def test_confidence_threshold():
    result = run_query("What is the pancreatitis risk with semaglutide?")
    assert result["confidence"] >= 0.6, f"Low confidence: {result['confidence']}"
    print(f"✅ Confidence test passed: {result['confidence']}")

def test_insufficient_evidence_response():
    result = run_query("What are semaglutide adverse events on mars colonists?")
    # Should say insufficient evidence, not hallucinate
    answer_lower = result["answer"].lower()
    assert any(phrase in answer_lower for phrase in [
        "insufficient", "not found", "no evidence", "cannot", "not available"
    ]), "Should flag insufficient evidence for impossible query"
    print("✅ Insufficient evidence test passed")

if __name__ == "__main__":
    print("=== DeepEval Custom Tests ===\n")
    test_answer_cites_sources()
    print("Waiting 20s for rate limit...")
    time.sleep(20)
    test_critic_passes()
    print("Waiting 20s for rate limit...")
    time.sleep(20)
    test_confidence_threshold()
    print("Waiting 20s for rate limit...")
    time.sleep(20)
    test_insufficient_evidence_response()
    print("\n=== All tests passed ===")