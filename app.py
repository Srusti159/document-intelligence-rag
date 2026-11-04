"""
Document Intelligence RAG System - Streamlit Dashboard
Production-grade RAG interface with multi-stage cosine retrieval, confidence thresholding,
audit-ready source citation tracking, and sub-second Groq LLM inference.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

from core.ingestion import DocumentIngestionEngine
from core.vectorstore import ChromaVectorStoreManager
from core.retriever import MultiStageRetriever
from core.llm import GroqLLMEngine
from core.memory import ConversationMemoryManager

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom Styling -----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .badge-blue { background-color: #1E3A8A; color: #93C5FD; border: 1px solid #3B82F6; }
    .badge-green { background-color: #064E3B; color: #6EE7B7; border: 1px solid #10B981; }
    .badge-purple { background-color: #4C1D95; color: #C4B5FD; border: 1px solid #8B5CF6; }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 0.8rem;
        text-align: center;
    }
    .citation-box {
        background-color: #0F172A;
        border-left: 3px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State Initialization -----------------
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = ConversationMemoryManager(max_history_turns=10)

if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = ChromaVectorStoreManager(
        persist_directory="./chroma_db",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free API key at console.groq.com/keys"
    )

    selected_model = st.selectbox(
        "Groq Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎯 Multi-Stage Retrieval")

    top_k = st.slider("Top-K Candidates", min_value=1, max_value=8, value=4, step=1)
    confidence_threshold = st.slider(
        "Cosine Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Chunks with similarity score below this threshold are discarded to prevent hallucinations."
    )

    st.markdown("---")
    st.markdown("### 📥 Document Ingestion")

    chunk_size = st.number_input("Chunk Size (Chars)", value=800, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (Chars)", value=150, step=25)

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Ingest PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Parsing PDFs and generating MiniLM embeddings..."):
                ingestion_engine = DocumentIngestionEngine(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                total_chunks_added = 0

                for uploaded_file in uploaded_files:
                    chunks, stats = ingestion_engine.extract_from_pdf(
                        uploaded_file,
                        filename=uploaded_file.name
                    )
                    count = st.session_state.vector_manager.add_documents(chunks)
                    total_chunks_added += count
                    st.toast(f"✅ Ingested {uploaded_file.name}: {len(chunks)} chunks")

                st.success(f"Indexed {total_chunks_added} chunks into ChromaDB!")

    st.markdown("---")
    if st.button("🗑️ Reset Vector Database", use_container_width=True):
        st.session_state.vector_manager.reset_collection()
        st.session_state.messages.clear()
        st.session_state.memory_manager.clear()
        st.success("Vector store & chat history cleared!")
        st.rerun()

# ----------------- Main View -----------------
st.markdown('<div class="main-header">Document Intelligence RAG System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-stage retrieval over complex PDFs with cosine confidence thresholding, verifiable source citations, and Groq LLM inference.</div>',
    unsafe_allow_html=True
)

# Badges
st.markdown("""
<div>
    <span class="badge badge-blue">Embedding: all-MiniLM-L6-v2</span>
    <span class="badge badge-green">Vector Database: ChromaDB (Persistent)</span>
    <span class="badge badge-purple">Inference: Groq (Llama 3.3 70B / 8B)</span>
    <span class="badge badge-blue">Guardrail: Cosine Threshold Filtering</span>
</div>
""", unsafe_allow_html=True)

# Document Stats Dashboard
total_vectors = st.session_state.vector_manager.get_document_count()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Active Vectors in ChromaDB", value=total_vectors)
with col2:
    st.metric(label="Retrieval Top-K", value=top_k)
with col3:
    st.metric(label="Confidence Threshold", value=f"{int(confidence_threshold * 100)}%")
with col4:
    st.metric(label="Active LLM", value=selected_model.split("-")[0].capitalize())

st.markdown("---")

# ----------------- Chat Interface -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander(f"🔍 Source Citations ({len(msg['citations'])} verified references)"):
                for idx, c in enumerate(msg["citations"], 1):
                    st.markdown(
                        f"**[Citation {idx}]** `{c['source']}` — **Page {c['page']}** | "
                        f"**Cosine Similarity:** `{c['score'] * 100:.1f}%`"
                    )
                    st.info(c["content"])

user_query = st.chat_input("Ask a grounded question about your ingested documents...")

if user_query:
    if total_vectors == 0:
        st.error("⚠️ Please upload and process a PDF document in the sidebar first.")
    else:
        # Display user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Multi-stage retrieval
        retriever = MultiStageRetriever(
            vector_store=st.session_state.vector_manager.vector_store,
            default_top_k=top_k,
            min_confidence_threshold=confidence_threshold
        )

        with st.spinner("Performing multi-stage semantic retrieval & confidence scoring..."):
            retrieval_start = time.perf_counter()
            retrieved_chunks, ret_stats = retriever.retrieve(
                query=user_query,
                top_k=top_k,
                confidence_threshold=confidence_threshold
            )
            retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 1)

        # Format context and citations
        context_str = retriever.format_context_with_citations(retrieved_chunks)
        citations_data = [
            {
                "source": c.source_document,
                "page": c.page_number,
                "score": c.similarity_score,
                "content": c.content,
                "chunk_id": c.chunk_id
            }
            for c in retrieved_chunks
        ]

        # LLM Generation
        llm_engine = GroqLLMEngine(
            api_key=groq_api_key,
            model_name=selected_model,
            temperature=0.1
        )

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            if not ret_stats["is_grounded"]:
                full_response = (
                    f"⚠️ **Low Confidence Guardrail Activated**: No passages in the ingested documents "
                    f"met your confidence threshold of **{int(confidence_threshold * 100)}%** "
                    f"(Highest detected match: `{ret_stats['max_confidence'] * 100:.1f}%`).\n\n"
                    f"To prevent hallucination, an answer was not generated. You may lower the threshold "
                    f"in the sidebar or verify document ingestion."
                )
                response_container.markdown(full_response)
            elif not groq_api_key:
                full_response = "⚠️ **Missing Groq API Key**: Please enter your Groq API key in the sidebar to generate grounded responses."
                response_container.markdown(full_response)
            else:
                chat_history = st.session_state.memory_manager.get_chat_history_for_llm()
                gen_start = time.perf_counter()

                for chunk in llm_engine.generate_grounded_response_stream(
                    query=user_query,
                    context_str=context_str,
                    chat_history=chat_history
                ):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")

                response_container.markdown(full_response)
                gen_ms = round((time.perf_counter() - gen_start) * 1000, 1)

                st.caption(
                    f"⚡ **Inference:** `{gen_ms} ms` | 🎯 **Retrieval:** `{retrieval_ms} ms` | "
                    f"📚 **Citations:** `{len(retrieved_chunks)} chunks` | "
                    f"🔝 **Peak Cosine Similarity:** `{ret_stats['max_confidence'] * 100:.1f}%`"
                )

            # Display Collapsible Citations Inspector
            if citations_data:
                with st.expander(f"🔍 Source Citations ({len(citations_data)} verified references)"):
                    for idx, c in enumerate(citations_data, 1):
                        st.markdown(
                            f"**[Citation {idx}]** `{c['source']}` — **Page {c['page']}** | "
                            f"**Cosine Similarity:** `{c['score'] * 100:.1f}%`"
                        )
                        st.info(c["content"])

        # Save to session memory
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": citations_data
        })
        st.session_state.memory_manager.add_turn(
            query=user_query,
            response=full_response,
            citations=citations_data,
            stats=ret_stats
        )
