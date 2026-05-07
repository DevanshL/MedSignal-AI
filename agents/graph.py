from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from agents.state import MedSignalState
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.synthesizer import synthesizer_node
from agents.critic import critic_node

def should_retry(state: MedSignalState) -> str:
    if state["critic_passed"]:
        return "end"
    return "end"  # for now always end; add retry loop in W5

def build_graph():
    graph = StateGraph(MedSignalState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", should_retry, {"end": END})
    
    return graph.compile()

if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "query": "What are the adverse events of semaglutide in patients with renal impairment?",
        "plan": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "critic_passed": False,
        "critique": ""
    })
    
    print("\n=== ANSWER ===")
    print(result["answer"])
    print(f"\n=== CONFIDENCE: {result['confidence']} ===")
    print(f"Critic passed: {result['critic_passed']}")
    print(f"Critique: {result['critique']}")
    print("\n=== CITATIONS ===")
    for c in result["citations"]:
        print(c)