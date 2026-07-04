import os
import pickle
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from graph_utils import create_graph

load_dotenv()

PDF_FOLDER = "uploaded_pdfs"
DEFAULT_PDF = "spotify_web_app_architecture.pdf"

VECTOR_DB_PATH = "faiss_index"
BM25_PATH = "bm25.pkl"


def load_all_pdfs():
    documents = []

    pdf_files = []

    if os.path.exists(DEFAULT_PDF):
        pdf_files.append(DEFAULT_PDF)

    if os.path.exists(PDF_FOLDER):
        for file in os.listdir(PDF_FOLDER):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(PDF_FOLDER, file))

    if not pdf_files:
        raise FileNotFoundError("No PDF found. Add a PDF file first.")

    for pdf in pdf_files:
        print(f"Loading PDF: {pdf}")
        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(pdf)

        documents.extend(docs)

    return documents


def main():
    print("Loading PDFs...")
    documents = load_all_pdfs()

    print("Splitting PDFs into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("Creating FAISS vector database...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    print("Creating BM25 index...")
    tokenized_chunks = [
        doc.page_content.lower().split()
        for doc in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks
            },
            f
        )

    print("Creating NetworkX knowledge graph...")
    create_graph(chunks)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()