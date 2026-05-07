# 💊 MedSignal AI

A pharmacovigilance intelligence platform that monitors PubMed, FDA adverse event reports, ClinicalTrials.gov, and PMC full-text papers to surface drug safety signals — answering natural language queries like *"what are adverse events for semaglutide in elderly patients?"* with grounded, cited, hallucination-checked answers.

---

## 🏗️ Architecture & System Design

MedSignal AI implements a multi-phase RAG architecture that bridges raw clinical data collection with agentic, fact-checked synthesis.

```mermaid
graph TD
    subgraph Ingestion [1. Multi-Source Ingestion]
        PubMed[PubMed Abstracts] --> Chunk[Chunking]
        PMC[PMC Full-Text] --> Chunk
        ClinTrials[ClinicalTrials.gov] --> Chunk
        OpenFDA[FDA Adverse Events] --> Chunk
        Chunk --> Embed[BGE Embeddings]
        Embed --> Chroma[(ChromaDB Vector Store)]
    end

    subgraph AgentLoop [2. Agentic Reasoning Loop]
        Query[User Query] --> Planner[1. Planner Node]
        Planner --> Retriever[2. Retriever Node]
        Chroma -.-> Retriever
        Retriever --> BM25[BM25 Hybrid Search]
        BM25 --> Rerank[Cross-Encoder Reranker]
        Rerank --> Synth[3. Synthesizer Node]
        Synth --> Critic[4. Critic Fact-Checker Node]
        Critic -- Approve --> UI[Streamlit UI]
    end
```

### Key Engineering Decisions

1. **Multi-Source Clinical Ingestion:**
   Fetches and parses data from four sources:
   * **PubMed Abstracts:** Individual papers via NCBI E-utilities API — avoids diversity collapse from concatenated batches.
   * **PMC Full-Text:** Structured full-text bodies with title, abstract, and body extraction.
   * **ClinicalTrials.gov:** Study protocols, criteria, and summaries.
   * **OpenFDA:** Live post-market adverse event reporting via FAERS.

2. **Hybrid Search + Cross-Encoder Reranking:**
   * Dense retrieval with `BAAI/bge-small-en-v1.5` captures semantic similarity.
   * Sparse BM25 retrieval captures exact drug name and terminology matches.
   * Top-20 candidates reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` for high-accuracy query-document scoring.
   * This approach reduces retrieval failure rate by ~49-67% over dense-only retrieval.

3. **4-Agent LangGraph Loop:**
   * **Planner:** Refines raw user query into optimized biomedical search terms.
   * **Retriever:** Executes hybrid search and reranking against 1314 chunks.
   * **Synthesizer:** Generates a direct plaintext answer from retrieved evidence only.
   * **Critic:** Runs structured JSON faithfulness evaluation — flags hallucinations before answer reaches the user.

4. **Groq Client with Automatic Fallback:**
   Primary model `llama-3.3-70b-versatile` automatically falls back to `llama-3.1-8b-instant` on rate limit errors — keeps pipeline running without interruption.

5. **Biomedical NER:**
   `d4data/biomedical-ner-all` via HuggingFace Transformers pipeline extracts `Medication`, `Disease_disorder`, and `Sign_symptom` entities from answers — displayed live in the UI.

6. **Python 3.13 Compatibility:**
   Uses `d4data/biomedical-ner-all` instead of `scispacy` which fails to build on Python 3.13 due to C++ binary compilation errors.

---

## 📂 Repository Structure

```
MedSignal/
│
├── agents/
│   ├── graph.py                 # LangGraph assembly and routing
│   ├── state.py                 # State schema (TypedDict)
│   ├── planner.py               # Query refinement node
│   ├── retriever.py             # Hybrid search + reranking node
│   ├── synthesizer.py           # Answer synthesis node
│   ├── critic.py                # Hallucination checking node
│   └── llm.py                   # Groq client with 8b fallback
│
├── ingestion/
│   ├── pubmed.py                # PubMed XML parser, individual papers
│   ├── pmc.py                   # PMC full-text XML cleaner
│   ├── clinicaltrials.py        # ClinicalTrials.gov API fetcher
│   └── openfda.py               # FDA FAERS adverse event retriever
│
├── processing/
│   ├── chunker.py               # Recursive character splitter
│   ├── embedder.py              # BGE-small embeddings
│   ├── reranker.py              # Cross-encoder reranker
│   └── ner.py                   # Biomedical NER pipeline
│
├── storage/
│   └── chroma_store.py          # ChromaDB client and batch upserts
│
├── evaluation/
│   ├── golden_dataset.py        # 20 ground truth clinical QA pairs
│   ├── ragas_eval.py            # RAGAS faithfulness evaluation
│   └── deepeval_tests.py        # Custom assertion tests
│
├── app.py                       # Streamlit UI
├── main.py                      # Full ingestion pipeline
├── requirements.txt
└── .env                         # API keys (not committed)
```

---

## 🚀 Setup

```bash
git clone https://github.com/your-username/MedSignal.git
cd MedSignal
python3 -m venv medsignal-env
source medsignal-env/bin/activate
pip install -r requirements.txt
```

Create `.env` in project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
NCBI_API_KEY=your_ncbi_key
```

---

## ⚡ Usage

**Run full ingestion pipeline:**
```bash
python main.py
```

**Start Streamlit UI:**
```bash
streamlit run app.py
```

**Run RAGAS evaluation:**
```bash
python -m evaluation.ragas_eval
```

**Run DeepEval tests:**
```bash
python -m evaluation.deepeval_tests
```

---

## 📊 Evaluation Results

| Metric | Score |
|---|---|
| RAGAS Faithfulness | 0.64 |
| DeepEval Assertions | 4/4 passed |
| Critic Confidence | 0.9 – 1.0 |
| Response Latency p95 | ~2.7s |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (Llama 3.1 8B / 3.3 70B) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | ChromaDB (local, persistent) |
| NER | d4data/biomedical-ner-all |
| Eval | RAGAS + DeepEval + Weights & Biases |
| UI | Streamlit |
| Data Sources | PubMed · FDA FAERS · ClinicalTrials.gov · PMC |