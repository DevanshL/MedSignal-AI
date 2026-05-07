import requests
import time

def fetch_pubmed(query: str, max_results: int = 500) -> list[dict]:
    # Step 1: get IDs
    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    ).json()
    ids = search["esearchresult"]["idlist"]

    # Step 2: fetch each abstract individually
    docs = []
    batch_size = 20
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        fetch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "rettype": "abstract",
                "retmode": "xml"
            }
        )
        # split by article
        import re
        articles = re.split(r'<PubmedArticle>', fetch.text)[1:]
        for j, article in enumerate(articles):
            pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
            title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article, re.DOTALL)
            abstract_match = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
            
            pmid = pmid_match.group(1) if pmid_match else batch_ids[j] if j < len(batch_ids) else f"unknown_{i}_{j}"
            title = title_match.group(1) if title_match else ""
            abstract = abstract_match.group(1) if abstract_match else ""
            
            if abstract:
                docs.append({
                    "id": f"pubmed_{pmid}",
                    "text": f"Title: {title}\n\nAbstract: {abstract}",
                    "source": "pubmed",
                    "pmid": pmid
                })
        time.sleep(0.5)  # NCBI rate limit

    print(f"Fetched {len(docs)} individual PubMed papers")
    return docs

if __name__ == "__main__":
    docs = fetch_pubmed("semaglutide adverse events", max_results=50)
    print(f"Total: {len(docs)} papers")
    print(docs[0]["text"][:300])