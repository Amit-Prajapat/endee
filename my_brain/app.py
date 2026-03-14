import pypdf
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1️⃣ READ PDF
print("Reading your PDF...")
reader = pypdf.PdfReader("notes.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

# 2️⃣ CHUNK TEXT
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(full_text)
print(f"Created {len(chunks)} chunks!")

# 3️⃣ CREATE EMBEDDINGS
print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# 4️⃣ INSERT INTO ENDEE
print("Uploading vectors to Endee...")

vectors = []

for i, emb in enumerate(embeddings):
    vectors.append({
        "id": str(i),
        "vector": emb.tolist(),
        "metadata": {
            "text": chunks[i]
        }
    })

payload = {
    "vectors": vectors
}

response = requests.post(
    "http://localhost:8080/api/v1/index/default/vectors",
    json=payload
)

print("Status Code:", response.status_code)
print("Server response:", response.text)

# print("Server response:", response.text)
print("Done! Vectors uploaded to Endee.")
payload = {
    "vectors": [
        {
            "id": str(i),
            "values": embeddings[i].tolist(),
            "metadata": {"text": chunks[i]}
        }
        for i in range(len(embeddings))
    ]
}