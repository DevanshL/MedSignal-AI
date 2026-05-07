from dotenv import load_dotenv
import os
import requests

load_dotenv("/Users/srinaimishamudari/Desktop/MedSignal/.env")

results = {}

# 1. HuggingFace
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.getenv('HF_TOKEN'))
    user = api.whoami()
    results["HF_TOKEN"] = f"✅ Valid (user: {user['name']})"
except Exception as e:
    results["HF_TOKEN"] = f"❌ Error: {e}"

# 2. Google Gemini
try:
    r = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={os.getenv('GOOGLE_API_KEY')}",
        timeout=10
    )
    results["GOOGLE_API_KEY"] = "✅ Valid" if r.status_code == 200 else f"❌ Invalid (HTTP {r.status_code})"
except Exception as e:
    results["GOOGLE_API_KEY"] = f"❌ Error: {e}"

# 3. Groq
try:
    r = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        timeout=10
    )
    results["GROQ_API_KEY"] = "✅ Valid" if r.status_code == 200 else f"❌ Invalid (HTTP {r.status_code})"
except Exception as e:
    results["GROQ_API_KEY"] = f"❌ Error: {e}"

# 4. Cohere
try:
    r = requests.get(
        "https://api.cohere.com/v2/models",
        headers={"Authorization": f"Bearer {os.getenv('COHERE_API_KEY')}"},
        timeout=10
    )
    results["COHERE_API_KEY"] = "✅ Valid" if r.status_code == 200 else f"❌ Invalid (HTTP {r.status_code})"
except Exception as e:
    results["COHERE_API_KEY"] = f"❌ Error: {e}"

# 5. NCBI
try:
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
        params={"api_key": os.getenv("NCBI_API_KEY"), "retmode": "json"},
        timeout=10
    )
    results["NCBI_API_KEY"] = "✅ Valid" if r.status_code == 200 else f"❌ Invalid (HTTP {r.status_code})"
except Exception as e:
    results["NCBI_API_KEY"] = f"❌ Error: {e}"

print("\n===== API KEY VALIDATION =====")
for key, status in results.items():
    print(f"{key:20s} → {status}")
print("==============================\n")
