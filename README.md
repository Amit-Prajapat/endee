# Personal PDF Brain (RAG Pipeline) 🧠

An end-to-end Retrieval-Augmented Generation (RAG) system built to perform semantic search over in-depth Data Structures and Algorithms (DSA) notes. 

## 🚀 Technical Architecture
* **Vector Database:** Endee (Containerized via Docker, built from source)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Language/Logic:** Python
* **Document Processing:** PyPDF & LangChain Text Splitters

## 🛠️ What I Built
1. **Infrastructure:** Resolved cross-platform build configurations (CRLF/LF, CPU architecture flags) to successfully compile and run the Endee OSS C++ vector database inside a Docker container on a Windows host.
2. **Ingestion Engine (`app.py`):** Developed a script to parse unstructured PDF notes, chunk the text with optimal overlap, generate vector embeddings, and upsert them into the Endee database via REST API.
3. **Retrieval Engine (`search.py`):** Built an interactive CLI tool that dynamically discovers the active database index, vectorizes user queries, and retrieves the top-K most semantically relevant paragraphs with confidence scores.

## 💡 Key Learnings
Gained hands-on experience with vector similarity search, container orchestration, resolving port conflicts, and API endpoint debugging.
