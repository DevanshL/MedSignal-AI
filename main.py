from ingestion.pubmed import fetch_pubmed
from ingestion.openfda import fetch_adverse_events
from ingestion.clinicaltrials import fetch_trials
from ingestion.pmc import fetch_pmc_fulltext
from processing.chunker import chunk_documents as contextual_chunk_documents
from processing.embedder import embed_texts
from storage.chroma_store import get_or_create_collection
import chromadb

DRUG = "semaglutide"

def upsert_batched(chunks, embeddings, batch_size=500):
    col = get_or_create_collection()
    for i in range(0, len(chunks), batch_size):
        bc = chunks[i:i+batch_size]
        be = embeddings[i:i+batch_size]
        col.upsert(
            ids=[f"{c.get('report_id', c.get('nct_id', c.get('id','doc')))}_{c['chunk_id']}" for c in bc],
            embeddings=be,
            documents=[c['chunk_text'] for c in bc],
            metadatas=[{k:v for k,v in c.items() if k not in ('chunk_text',) and isinstance(v,(str,int,float,bool))} for c in bc]
        )

def run_ingestion():
    # clear old collection
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("medsignal")
        print("Cleared old collection")
    except:
        pass

    print("=== PubMed (individual papers) ===")
    pubmed_docs = fetch_pubmed(f"{DRUG} adverse events", max_results=100)

    print("=== OpenFDA ===")
    events = fetch_adverse_events(DRUG, limit=100)
    fda_docs = [{"id": e["report_id"], "text": " ".join(e["reactions"]),
                 "source": "openfda", "serious": str(e["serious"])} for e in events if e["report_id"]]

    print("=== ClinicalTrials ===")
    trials = fetch_trials(f"{DRUG} adverse events", max_results=100)
    trial_docs = [{"id": t["nct_id"], "text": t["summary"], "source": "clinicaltrials"}
                  for t in trials if t["summary"]]

    print("=== PMC Full Text (individual papers) ===")
    pmc_docs = fetch_pmc_fulltext(f"{DRUG} adverse events", max_results=50)

    all_docs = pubmed_docs + fda_docs + trial_docs + pmc_docs
    print(f"\nTotal individual docs: {len(all_docs)}")

    print("\n=== Contextual Chunking ===")
    chunks = contextual_chunk_documents(all_docs)
    print(f"Total contextual chunks: {len(chunks)}")

    print("\n=== Embedding ===")
    embeddings = embed_texts([c["chunk_text"] for c in chunks])

    print("\n=== Storing in Chroma ===")
    upsert_batched(chunks, embeddings)

    col = get_or_create_collection()
    print(f"\nFinal chunk count: {col.count()}")
    print("Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()