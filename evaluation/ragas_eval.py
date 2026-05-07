from dotenv import load_dotenv
load_dotenv()

import os, time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from agents.graph import build_graph
from evaluation.golden_dataset import GOLDEN_QA


class GroqSingleN(ChatGroq):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def generate(self, messages, stop=None, **kwargs):
        kwargs.pop("n", None)
        return super().generate(messages, stop=stop, **kwargs)

    async def agenerate(self, messages, stop=None, **kwargs):
        kwargs.pop("n", None)
        return await super().agenerate(messages, stop=stop, **kwargs)

    def batch(self, messages, stop=None, **kwargs):
        kwargs.pop("n", None)
        return super().batch(messages, stop=stop, **kwargs)

    async def abatch(self, messages, stop=None, **kwargs):
        kwargs.pop("n", None)
        return await super().abatch(messages, stop=stop, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        kwargs.pop("n", None)
        return super().invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        kwargs.pop("n", None)
        return await super().ainvoke(input, config=config, **kwargs)


ragas_llm = LangchainLLMWrapper(GroqSingleN(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
))

faithfulness_metric = Faithfulness(llm=ragas_llm)


def run_ragas_evaluation(sample_size: int = 5):
    graph = build_graph()
    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running eval on {sample_size} questions...")
    for i, qa in enumerate(GOLDEN_QA[:sample_size]):
        print(f"  [{i+1}/{sample_size}] {qa['question'][:60]}...")
        try:
            result = graph.invoke({
                "query": qa["question"],
                "plan": "",
                "retrieved_docs": [],
                "reranked_docs": [],
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "critic_passed": False,
                "critique": ""
            })
            if not result.get("answer") or len(result["answer"]) < 10:
                print(f"  Skip Q{i+1}: empty answer")
                continue
            if not result.get("retrieved_docs"):
                print(f"  Skip Q{i+1}: no docs retrieved")
                continue
            questions.append(qa["question"])
            answers.append(result["answer"])
            contexts.append([d["text"] for d in result["retrieved_docs"][:3]])
            ground_truths.append(qa["ground_truth"])
            time.sleep(10)
        except Exception as e:
            print(f"  Error Q{i+1}: {e}")
            continue

    if not questions:
        print("No results collected.")
        return None

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    scores = evaluate(dataset, metrics=[faithfulness_metric])
    return scores


if __name__ == "__main__":
    scores = run_ragas_evaluation(sample_size=5)
    if scores:
        print("\n=== RAGAS SCORES ===")
        print(scores)
        try:
            print(f"Faithfulness: {scores['faithfulness']:.4f}")
        except:
            print(f"Faithfulness: {scores['faithfulness']}")