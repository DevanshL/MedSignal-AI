import requests

def fetch_trials(query: str, max_results: int = 100) -> list[dict]:
    r = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={
            "query.term": query,
            "pageSize": max_results,
            "format": "json"
        }
    )
    studies = r.json().get("studies", [])
    trials = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        desc_mod = proto.get("descriptionModule", {})
        status_mod = proto.get("statusModule", {})
        trials.append({
            "nct_id": id_mod.get("nctId"),
            "title": id_mod.get("briefTitle"),
            "status": status_mod.get("overallStatus"),
            "summary": desc_mod.get("briefSummary", "")[:500]
        })
    return trials

if __name__ == "__main__":
    trials = fetch_trials("semaglutide adverse events")
    print(f"Fetched {len(trials)} trials")
    print(trials[0])