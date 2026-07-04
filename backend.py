import os
import pickle
import subprocess
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

from graph_utils import graph_search

load_dotenv()

app = FastAPI(title="Advanced PDF Hybrid RAG App")

VECTOR_DB_PATH = "faiss_index"
BM25_PATH = "bm25.pkl"
UPLOAD_FOLDER = "uploaded_pdfs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class QuestionRequest(BaseModel):
    question: str
    chat_history: List[dict] = []


def vector_search(question, top_k=8):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(question, k=top_k)

    for d in docs:
        d.metadata["retriever"] = "faiss_vector"

    return docs


def bm25_search(question, top_k=8):
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_question = question.lower().split()
    scores = bm25.get_scores(tokenized_question)

    top_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for i in top_indexes:
        doc = chunks[i]
        doc.metadata["retriever"] = "bm25"
        doc.metadata["score"] = float(scores[i])
        results.append(doc)

    return results


def deduplicate_docs(documents):
    unique_docs = []
    seen = set()

    for doc in documents:
        text = doc.page_content.strip()

        if text not in seen:
            unique_docs.append(doc)
            seen.add(text)

    return unique_docs


def cross_encoder_rerank(question, documents, top_k=6):
    documents = deduplicate_docs(documents)

    if not documents:
        return []

    pairs = [(question, doc.page_content) for doc in documents]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    final_docs = []

    for score, doc in scored_docs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        final_docs.append(doc)

    return final_docs


def rewrite_question(question, chat_history):
    if not chat_history:
        return question

    history_text = "\n".join(
        [
            f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}"
            for item in chat_history[-3:]
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Rewrite the user's latest question into a standalone question.

Chat history:
{history_text}

Latest question:
{question}

Standalone question:
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def multi_query_generation(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Create 3 different search queries for retrieving PDF content.
Return only the queries, one per line.

Question:
{question}
"""

    response = llm.invoke(prompt)

    queries = [
        q.strip("- ").strip()
        for q in response.content.splitlines()
        if q.strip()
    ]

    return queries[:3]


def hybrid_retrieve(question):
    queries = [question] + multi_query_generation(question)

    all_docs = []

    for q in queries:
        all_docs.extend(vector_search(q))
        all_docs.extend(bm25_search(q))
        all_docs.extend(graph_search(q))

    final_docs = cross_encoder_rerank(question, all_docs)

    return final_docs


@app.get("/")
def home():
    return {
        "message": "Advanced PDF Hybrid RAG API is running"
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = subprocess.run(
        ["python", "ingest.py"],
        capture_output=True,
        text=True
    )

    return {
        "message": "PDF uploaded and indexed successfully",
        "file": file.filename,
        "ingestion_output": result.stdout,
        "ingestion_error": result.stderr
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    original_question = request.question

    standalone_question = rewrite_question(
        original_question,
        request.chat_history
    )

    final_docs = hybrid_retrieve(standalone_question)

    context = "\n\n".join(
        [
            f"""
Source: {doc.metadata.get("source", "unknown")}
Page: {doc.metadata.get("page", "unknown")}
Retriever: {doc.metadata.get("retriever", "unknown")}
Content:
{doc.page_content}
"""
            for doc in final_docs
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a PDF question-answering assistant.

Rules:
1. Answer only using the PDF context.
2. If the answer is not present, say:
"I could not find this information in the PDF."
3. Include page numbers when possible.
4. Give a clear and beginner-friendly answer.

PDF Context:
{context}

Question:
{standalone_question}

Answer:
"""

    response = llm.invoke(prompt)

    confidence = "High" if len(final_docs) >= 4 else "Medium" if len(final_docs) >= 2 else "Low"

    return {
        "question": original_question,
        "standalone_question": standalone_question,
        "answer": response.content,
        "confidence": confidence,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "retriever": doc.metadata.get("retriever", "unknown"),
                "rerank_score": round(doc.metadata.get("rerank_score", 0), 3),
                "content": doc.page_content[:600]
            }
            for doc in final_docs
        ]
    }