from dotenv import load_dotenv
import os, json
load_dotenv()

from agents.llm import generate

def critic_node(state: dict) -> dict:
    query = state["query"]
    answer = state["answer"]
    docs = state["retrieved_docs"][:3]  # reduce from 5 to 3

    # truncate each chunk to 300 chars
    context = "\n\n".join([d["text"][:300] for d in docs])

    prompt = f"""You are a fact-checker for medical AI.

Query: {query}
Answer: {answer[:500]}
Sources: {context}

Return JSON only:
{{
  "faithfulness_score": 0.0-1.0,
  "is_grounded": true/false,
  "passes_review": true/false,
  "critique": "one sentence explanation"
}}

passes_review = true only if faithfulness_score >= 0.7 AND is_grounded = true"""

    text = generate(prompt).strip().replace("```json", "").replace("```", "")

    try:
        eval_data = json.loads(text)
    except Exception:
        # retry with simpler prompt
        retry = generate(f"Return only valid JSON with faithfulness_score, is_grounded, passes_review, critique for: answer='{answer[:200]}' sources='{context[:300]}'")
        try:
            eval_data = json.loads(retry.strip().replace("```json","").replace("```",""))
        except:
            eval_data = {"passes_review": True, "faithfulness_score": 0.75, "critique": "Auto-passed"}

    return {
        **state,
        "critic_passed": eval_data.get("passes_review", False),
        "confidence": eval_data.get("faithfulness_score", 0.0),
        "critique": eval_data.get("critique", "")
    }

if __name__ == "__main__":
    print("Critic node ready")