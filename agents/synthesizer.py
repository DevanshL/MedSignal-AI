from dotenv import load_dotenv
import os
load_dotenv()

from agents.llm import generate

def synthesizer_node(state: dict) -> dict:
    query = state["query"]
    docs = state["retrieved_docs"][:5]

    context = "\n\n".join([
        f"[Source {i+1} | {d['metadata'].get('source','unknown')}]\n{d['text']}"
        for i, d in enumerate(docs)
    ])

    prompt = f"""You are a pharmacovigilance expert. Answer clearly using ONLY the provided sources.

Query: {query}

Sources:
{context}

Rules:
- Write a clean, direct answer in plain English
- Do NOT mention source numbers like "Source 1", "Source 2", "Source 3" anywhere
- Do NOT say "Source X does not mention" or "Source X reports"
- Do NOT use bullet points
- Do NOT add hedging like "more research needed" or "may vary"
- State facts directly from the evidence
- If evidence is limited, say "Limited evidence available" once, then state what IS known
- Keep answer under 150 words

Answer:"""

    answer_text = generate(prompt)
    citations = list(set([
        f"Source {i+1}: {d['metadata'].get('source', 'unknown')} - {d['id']}"
        for i, d in enumerate(docs)
    ]))

    return {**state, "answer": answer_text, "citations": citations}

if __name__ == "__main__":
    from agents.retriever import retriever_node
    state = retriever_node({"query": "semaglutide nausea adverse events",
                            "plan": "semaglutide nausea adverse events"})
    result = synthesizer_node(state)
    print(result["answer"])