from dotenv import load_dotenv
import os, json, re
load_dotenv()

from agents.llm import generate

def planner_node(state: dict) -> dict:
    query = state["query"]
    prompt = f"""You are a pharmacovigilance query planner.
Given this query: "{query}"

Return JSON only, no other text:
{{
  "refined_query": "optimized search query for biomedical literature",
  "key_concepts": ["drug", "adverse_event", "population"],
  "search_strategy": "brief strategy description"
}}"""

    response_text = generate(prompt)
    
    # extract JSON even if model adds extra text
    try:
        # try direct parse first
        text = response_text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        plan_data = json.loads(text)
    except json.JSONDecodeError:
        try:
            # find JSON block in response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                plan_data = json.loads(match.group())
            else:
                # fallback — use query as-is
                plan_data = {"refined_query": query}
        except:
            plan_data = {"refined_query": query}

    return {**state, "plan": plan_data.get("refined_query", query)}

if __name__ == "__main__":
    result = planner_node({"query": "what are adverse events of semaglutide"})
    print(result["plan"])