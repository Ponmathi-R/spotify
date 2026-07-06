# PDF Hybrid RAG GenAI App

A GenAI PDF scraping and question-answering application using **FastAPI**, **Streamlit**, **LangChain**, **OpenAI**, **LangSmith**, **BM25**, **Vector Store**, **Reranker**, and **Neo4j Knowledge Graph**.

## Features

* Upload or load PDF files
* Extract text from PDFs
* Split PDF text into chunks
* Store embeddings in vector database
* Perform keyword search using BM25
* Use hybrid retrieval: vector search + BM25
* Use reranking for better answers
* Store document relationships in Neo4j knowledge graph
* Ask questions from PDF content
* FastAPI backend
* Streamlit frontend
* LangSmith tracing and debugging
* OpenAI API key support

## Project Structure

```text
pdf_hybrid_rag_app/
│
├── backend.py
├── frontend.py
├── ingest.py
├── rag_pipeline.py
├── graph_store.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   └── sample.pdf
│
├── vector_store/
│
└── bm25_store/
```

## Tech Stack

* Python
* FastAPI
* Streamlit
* LangChain
* OpenAI
* LangSmith
* ChromaDB
* BM25
* Sentence Transformers
* Neo4j

## Setup Instructions

### 1. Create project folder

```powershell
mkdir pdf_hybrid_rag_app
cd pdf_hybrid_rag_app
```

### 2. Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create `.env` file

```env
OPENAI_API_KEY=your_openai_api_key_here

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=pdf-hybrid-rag-app

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

### 5. Add PDF file

Place your PDF inside the `data` folder.

Example:

```text
data/sample.pdf
```

### 6. Run ingestion

```powershell
python ingest.py
```

### 7. Run FastAPI backend

```powershell
uvicorn backend:app --reload
```

FastAPI will run at:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 8. Run Streamlit frontend

Open another terminal and run:

```powershell
streamlit run frontend.py
```

Streamlit will run at:

```text
http://localhost:8501
```

## How the App Works

1. PDF is loaded.
2. Text is extracted from the PDF.
3. Text is split into smaller chunks.
4. Chunks are stored in vector database.
5. BM25 keyword index is created.
6. Neo4j knowledge graph is created.
7. User asks a question.
8. App searches using vector search and BM25.
9. Results are reranked.
10. Final context is sent to OpenAI.
11. Answer is displayed in Streamlit.

## Example Questions

```text
What is this PDF about?
Summarize the main points.
Explain this document in simple words.
What are the key topics?
Give me important points from this PDF.
```

## Notes

Do not share your API keys publicly.

Do not upload `.env` file to GitHub.

Add `.env` to `.gitignore`.

## Author

Created by Ponmathi Radhakrishnan.
