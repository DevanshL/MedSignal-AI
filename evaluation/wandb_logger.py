from dotenv import load_dotenv
load_dotenv()

import wandb
import time
from agents.graph import build_graph

def log_query_to_wandb(query: str, result: dict, latency: float):
    wandb.log({
        "query": query,
        "answer_length": len(result["answer"]),
        "confidence": result["confidence"],
        "critic_passed": int(result["critic_passed"]),
        "num_citations": len(result["citations"]),
        "latency_seconds": latency
    })

def run_logged_eval(questions: list[str], project: str = "medsignal-eval"):
    wandb.init(project=project, name="eval-run-1")
    graph = build_graph()
    
    for q in questions:
        start = time.time()
        result = graph.invoke({
            "query": q,
            "plan": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "critic_passed": False,
            "critique": ""
        })
        latency = time.time() - start
        log_query_to_wandb(q, result, latency)
        print(f"Logged: {q[:50]}... | conf={result['confidence']} | {latency:.2f}s")
    
    wandb.finish()

if __name__ == "__main__":
    test_queries = [
        "What are the adverse events of semaglutide?",
        "Is semaglutide safe for renal patients?",
        "What is the pancreatitis risk with semaglutide?",
        "Does semaglutide cause hypoglycemia?",
        "What are the cardiovascular effects of semaglutide?"
    ]
    run_logged_eval(test_queries)