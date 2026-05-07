from transformers import pipeline

ner_pipeline = pipeline("token-classification",
                        model="d4data/biomedical-ner-all",
                        aggregation_strategy="first")

KEEP_LABELS = {"Medication", "Disease_disorder", "Sign_symptom"}

def extract_entities(text: str) -> list[dict]:
    try:
        entities = ner_pipeline(text[:512])
        seen = set()
        results = []
        for e in entities:
            word = e["word"].strip()
            word_lower = word.lower()
            if (e["score"] > 0.65
                    and len(word) > 3
                    and not word.isdigit()
                    and word_lower not in seen
                    and e["entity_group"] in KEEP_LABELS):
                seen.add(word_lower)
                results.append({
                    "entity": e["entity_group"],
                    "word": word,
                    "score": round(float(e["score"]), 3)
                })
        return results
    except Exception as e:
        print(f"NER error: {e}")
        return []

if __name__ == "__main__":
    text = "Semaglutide caused nausea and pancreatitis in diabetic patients"
    for e in extract_entities(text):
        print(f"{e['entity']}: {e['word']} ({e['score']})")