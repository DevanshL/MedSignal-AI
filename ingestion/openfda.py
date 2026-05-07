import requests

def fetch_adverse_events(drug: str, limit: int = 100) -> list[dict]:
    r = requests.get(
        "https://api.fda.gov/drug/event.json",
        params={
            "search": f"patient.drug.medicinalproduct:{drug}",
            "limit": limit
        }
    )
    data = r.json()
    results = data.get("results", [])
    
    # extract useful fields only
    events = []
    for e in results:
        events.append({
            "report_id": e.get("safetyreportid"),
            "serious": e.get("serious"),
            "reactions": [
                r["reactionmeddrapt"] 
                for r in e.get("patient", {}).get("reaction", [])
            ],
            "drugs": [
                d.get("medicinalproduct", "") 
                for d in e.get("patient", {}).get("drug", [])
            ]
        })
    return events

if __name__ == "__main__":
    events = fetch_adverse_events("semaglutide")
    print(f"Fetched {len(events)} adverse event reports")
    print(events[0])