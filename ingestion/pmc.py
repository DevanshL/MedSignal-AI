import requests
import re
import time

def fetch_pmc_fulltext(query: str, max_results: int = 100) -> list[dict]:
    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pmc", "term": query, "retmax": max_results, "retmode": "json"}
    ).json()
    ids = search["esearchresult"]["idlist"]

    docs = []
    batch_size = 10
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        fetch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pmc", "id": ",".join(batch_ids), "rettype": "xml", "retmode": "text"}
        )
        articles = re.split(r'<article ', fetch.text)[1:]
        for j, article in enumerate(articles):
            title_match = re.search(r'<article-title>(.*?)</article-title>', article, re.DOTALL)
            abstract_match = re.search(r'<abstract>(.*?)</abstract>', article, re.DOTALL)
            body_match = re.search(r'<body>(.*?)</body>', article, re.DOTALL)

            title = re.sub(r'<[^>]+>', '', title_match.group(1)) if title_match else ""
            abstract = re.sub(r'<[^>]+>', '', abstract_match.group(1)) if abstract_match else ""
            body = re.sub(r'<[^>]+>', '', body_match.group(1))[:3000] if body_match else ""

            text = f"Title: {title}\n\nAbstract: {abstract}\n\nBody: {body}".strip()
            if len(text) > 100:
                docs.append({
                    "id": f"pmc_{batch_ids[j] if j < len(batch_ids) else i+j}",
                    "text": text,
                    "source": "pmc_fulltext"
                })
        time.sleep(0.5)

    print(f"Fetched {len(docs)} individual PMC papers")
    return docs

if __name__ == "__main__":
    docs = fetch_pmc_fulltext("semaglutide adverse events", max_results=20)
    print(f"Total: {len(docs)} papers")
    print(docs[0]["text"][:300])