import requests
from sentence_transformers import SentenceTransformer

# 1. SETUP: Load the embedding model
print("Loading the search brain...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. DETECT: Find your index name automatically
BASE_URL = "http://localhost:8080/api/v1"
index_name = "default" # Fallback name

try:
    # Ask the server: "What boxes (indexes) do you have?"
    list_response = requests.get(f"{BASE_URL}/index")
    if list_response.status_code == 200:
        indexes = list_response.json().get("indexes", [])
        if indexes:
            index_name = indexes[0]
            print(f"Detected Index: '{index_name}'")
except Exception as e:
    print(f"Note: Could not list indexes, using default. Error: {e}")

# 3. ASK: Get the user's question
query_text = input("\nWhat would you like to know from your PDF? ")

# 4. TRANSLATE: Turn the question into a vector
query_vector = model.encode(query_text).tolist()

# 5. SEARCH: Use the exact path for searching within an index
print(f"Searching Endee in '{index_name}'...")
ENDEE_SEARCH_URL = f"{BASE_URL}/index/{index_name}/search"

payload = {
    "vector": query_vector,
    "k": 3 
}

try:
    response = requests.post(ENDEE_SEARCH_URL, json=payload)

    if response.status_code == 200:
        results = response.json().get("results", [])
        print(f"\n--- Found {len(results)} relevant matches ---")
        
        for i, res in enumerate(results):
            # Check 'metadata' or 'content' depending on how you saved it in app.py
            metadata = res.get("metadata", {})
            text = metadata.get("text") or res.get("text") or "No text content found in this match."
            score = res.get("score", 0)
            
            print(f"\n[Match {i+1}] (Confidence: {score:.4f}):")
            print(f"Content: {text[:500]}...") # Showing first 500 chars
    else:
        print(f"Search failed (Status {response.status_code}): {response.text}")
        print(f"Tried URL: {ENDEE_SEARCH_URL}")

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to Endee. Is your Docker container running?")