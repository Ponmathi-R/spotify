import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Advanced PDF Hybrid RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Advanced PDF Hybrid RAG App")

st.write(
    "Upload PDFs and ask questions. This app uses FAISS, BM25, NetworkX Knowledge Graph, CrossEncoder reranking, FastAPI, Streamlit, LangChain, OpenAI, and LangSmith."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2, tab3 = st.tabs(
    [
        "Ask PDF",
        "Upload PDF",
        "Chat History"
    ]
)

with tab1:
    st.subheader("Ask a Question")

    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching PDF and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "chat_history": st.session_state.chat_history
                        },
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()

                        st.success(data["answer"])

                        st.info(f"Confidence: {data['confidence']}")

                        st.session_state.chat_history.append(
                            {
                                "user": question,
                                "assistant": data["answer"]
                            }
                        )

                        with st.expander("Standalone Question"):
                            st.write(data["standalone_question"])

                        with st.expander("Sources"):
                            for source in data["sources"]:
                                st.markdown(
                                    f"""
                                    **File:** {source['source']}  
                                    **Page:** {source['page']}  
                                    **Retriever:** {source['retriever']}  
                                    **Rerank Score:** {source['rerank_score']}
                                    """
                                )
                                st.write(source["content"])
                                st.divider()

                    else:
                        st.error("Backend error.")
                        st.write(response.text)

                except Exception as e:
                    st.error("FastAPI backend is not running.")
                    st.write(e)

with tab2:
    st.subheader("Upload New PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:
        if st.button("Upload and Re-index"):
            with st.spinner("Uploading and indexing PDF..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                try:
                    response = requests.post(
                        f"{API_URL}/upload-pdf",
                        files=files,
                        timeout=300
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["message"])

                        with st.expander("Ingestion Output"):
                            st.text(data.get("ingestion_output", ""))

                        if data.get("ingestion_error"):
                            with st.expander("Ingestion Warnings / Errors"):
                                st.text(data.get("ingestion_error", ""))

                    else:
                        st.error("Upload failed.")
                        st.write(response.text)

                except Exception as e:
                    st.error("FastAPI backend is not running.")
                    st.write(e)

with tab3:
    st.subheader("Chat History")

    if st.session_state.chat_history:
        df = pd.DataFrame(st.session_state.chat_history)
        st.dataframe(df, use_container_width=True)

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No chat history yet.")