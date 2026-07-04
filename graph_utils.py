import pickle
import re
import networkx as nx
from langchain_core.documents import Document

GRAPH_PATH = "graph.pkl"


STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
    "have", "has", "had", "you", "your", "into", "about", "between", "using",
    "used", "also", "can", "will", "shall", "may", "not", "but", "all", "any",
    "pdf", "page", "document", "there", "their", "them", "then", "than"
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords(text: str, max_keywords: int = 40):
    text = text.lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text)

    keywords = []
    for word in words:
        if word not in STOPWORDS and len(word) > 3:
            keywords.append(word)

    return list(dict.fromkeys(keywords))[:max_keywords]


def create_graph(chunks):
    graph = nx.Graph()

    for i, doc in enumerate(chunks):
        chunk_id = f"chunk_{i}"

        graph.add_node(
            chunk_id,
            type="chunk",
            content=doc.page_content,
            page=doc.metadata.get("page", 0),
            source=doc.metadata.get("source", "unknown")
        )

        keywords = extract_keywords(doc.page_content)

        for keyword in keywords:
            keyword_id = f"keyword_{keyword}"

            graph.add_node(
                keyword_id,
                type="keyword",
                name=keyword
            )

            graph.add_edge(chunk_id, keyword_id, relation="HAS_KEYWORD")

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)

    return graph


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def graph_search(question: str, top_k: int = 5):
    graph = load_graph()
    question_keywords = extract_keywords(question)

    chunk_scores = {}

    for keyword in question_keywords:
        keyword_node = f"keyword_{keyword}"

        if keyword_node in graph:
            for node in graph.neighbors(keyword_node):
                if graph.nodes[node].get("type") == "chunk":
                    chunk_scores[node] = chunk_scores.get(node, 0) + 1

    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []

    for chunk_id, score in sorted_chunks:
        data = graph.nodes[chunk_id]

        results.append(
            Document(
                page_content=data["content"],
                metadata={
                    "page": data.get("page", 0),
                    "source": data.get("source", "unknown"),
                    "retriever": "networkx_graph",
                    "score": score
                }
            )
        )

    return results