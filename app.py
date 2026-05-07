from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import time
from agents.graph import build_graph
from processing.ner import extract_entities

st.set_page_config(page_title="MedSignal AI", page_icon="💊", layout="wide")

@st.cache_resource
def load_graph():
    return build_graph()

graph = load_graph()

def is_medical_query(query: str) -> bool:
    medical_keywords = [
        "drug", "adverse", "event", "side effect", "safety", "trial",
        "patient", "dose", "symptom", "disease", "treatment", "medication",
        "clinical", "risk", "therapy", "semaglutide", "glp", "diabetes",
        "obesity", "renal", "cardiac", "liver", "nausea", "pain", "cancer",
        "inhibitor", "receptor", "agonist", "pharma", "fda", "study",
        "effect", "reaction", "toxicity", "interaction", "blood", "heart",
        "kidney", "eye", "ocular", "psychiatric", "weight", "insulin",
        "glucose", "injection", "weekly", "dose", "approved", "trial"
    ]
    return any(kw in query.lower() for kw in medical_keywords)

# header
st.title("💊 MedSignal AI")
st.caption("Pharmacovigilance intelligence — ask about drug safety signals")

with st.sidebar:
    st.markdown("**Data Sources**")
    st.markdown("PubMed · FDA · ClinicalTrials · PMC")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("Query → Plan → Retrieve → Synthesize → Fact-check")
    st.divider()
    st.markdown("**Eval**")
    st.markdown("Faithfulness: 0.64 · DeepEval: 4/4")
    st.divider()
    st.caption("LangGraph · ChromaDB · Groq · HuggingFace")

# query input
query = st.text_input(
    "",
    placeholder="e.g. What adverse events does semaglutide cause in renal patients?",
    label_visibility="collapsed"
)

run = st.button("🔍 Analyze", type="primary")

if run and not query.strip():
    st.warning("Please enter a question.")

elif run and query.strip():
    if not is_medical_query(query):
        st.warning("⚠️ MedSignal AI is designed for pharmacovigilance questions. Please ask about drug safety, adverse events, or clinical trials.")
    else:
        with st.spinner("Analyzing..."):
            start = time.time()
            result = graph.invoke({
                "query": query,
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

        # metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{result['confidence']:.0%}")
        m2.metric("Fact Check", "✅ Passed" if result["critic_passed"] else "⚠️ Review")
        m3.metric("Sources", len(result["citations"]))
        m4.metric("Time", f"{latency:.1f}s")

        st.divider()

        # answer
        st.markdown("### Answer")
        if result["critic_passed"]:
            st.success(result["answer"])
        else:
            st.warning(result["answer"])
            if result.get("critique"):
                with st.expander("⚠️ Fact-check note"):
                    st.write(result["critique"])

        # NER entities — deduplicated, filtered
        entities = extract_entities(result["answer"])
        if entities:
            st.markdown("### 🔬 Detected Entities")
            icons = {
                        "Medication": "🔵",
                        "Disease_disorder": "🔴",
                        "Sign_symptom": "🟡"
                    }
            cols = st.columns(4)
            for i, e in enumerate(entities):
                icon = icons.get(e["entity"], "⚪")
                cols[i % 4].markdown(f"{icon} **{e['word']}** `{e['entity']}`")

        st.divider()

        # sources
        with st.expander("📚 Sources"):
            sources = list(set([
                c.split(":")[1].strip().split("-")[0].strip() 
                if ":" in c else c 
                for c in result["citations"]
            ]))
            source_labels = {
                "pubmed": "📄 PubMed Research Papers",
                "openfda": "🏛️ FDA Adverse Event Reports", 
                "clinicaltrials": "🔬 ClinicalTrials.gov",
                "pmc_fulltext": "📑 PMC Full-text Articles"
            }
            for s in sources:
                label = source_labels.get(s.lower().strip(), f"📄 {s}")
                st.markdown(f"- {label}")

                # evidence
                with st.expander("🔍 Retrieved Evidence"):
                    for i, doc in enumerate(result["retrieved_docs"][:5]):
                        source = doc["metadata"].get("source", "unknown")
                        score = doc.get("rerank_score", 0)
                        st.markdown(f"**[{i+1}]** `{source}` · relevance `{score:.3f}`")
                        st.text(doc["text"][:400])
                        if i < 4:
                            st.divider()