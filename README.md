# 📄 Document Intelligence RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-red.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20LPU-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An enterprise-grade, hallucination-resistant **Retrieval-Augmented Generation (RAG)** pipeline designed for multi-page PDF documents. Features **multi-stage cosine similarity filtering**, **strict confidence thresholding**, **verifiable source citation tracking**, and **sub-second LLM inference via Groq**.

---

## 🌟 Key Highlights

- **Overlap-Aware Ingestion:** Recursive character chunking with dynamic overlap and page-level metadata preservation.
- **Persistent ChromaDB Vector Store:** Dense semantic embeddings powered by `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional space).
- **Multi-Stage Cosine Retrieval:** Dynamic similarity scoring with an active confidence threshold slider to discard low-relevance passages.
- **Hallucination Defense Guardrail:** Prompts and retrieval filters enforce strict evidence grounding—if confidence falls below threshold, the system triggers an explicit refusal instead of fabricating answers.
- **Audit-Ready Citations:** Every claim links directly to its source PDF filename, page number, chunk hash, and cosine similarity score.
- **Sub-Second LLM Inference:** Powered by Groq LPU engine supporting **Llama 3.3 70B Versatile**, **Llama 3.1 8B Instant**, and **Mixtral 8x7B**.
- **Interactive Streamlit Interface:** Modern UI with live vector collection metrics, streaming responses, and an expandable citation inspector.

---

## 🏗️ System Architecture

```
[ PDF Documents ]
       │
       ▼
[ PyPDF Parser ] ──▶ [ Recursive Character Splitter ] (Size: 800, Overlap: 150)
                             │
                             ▼
              [ all-MiniLM-L6-v2 Embeddings ]
                             │
                             ▼
              [ Persistent ChromaDB Store ]
                             │
     ┌───────────────────────┴───────────────────────┐
     │                                               │
[ User Query ]                                       │
     │                                               │
     ▼                                               ▼
[ Embedding ] ──▶ [ Multi-Stage Semantic Search ] ◀──┘
                             │
                             ▼
              [ Cosine Confidence Threshold Filter ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Above Threshold? ]             [ Below Threshold? ]
            │                                 │
           YES                                NO
            │                                 │
            ▼                                 ▼
[ Structured Citations Context ]    [ Hallucination Guardrail Refusal ]
            │
            ▼
[ Groq LLM (Llama 3.3 70B) ] ──▶ [ Grounded Streamed Response + Audit Citations ]
```

---

## 📂 Repository Structure

```
document-intelligence-rag/
├── app.py                      # Interactive Streamlit dashboard & UI
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git exclusions (vector DB, cache, env)
├── README.md                   # System documentation & benchmarks
├── core/
│   ├── __init__.py             # Core package export
│   ├── ingestion.py            # PDF parsing, metadata enrichment & chunking
│   ├── vectorstore.py          # ChromaDB lifecycle & all-MiniLM-L6-v2 embeddings
│   ├── retriever.py            # Multi-stage retrieval & cosine threshold scoring
│   ├── llm.py                  # Groq client, prompt guardrails & streaming
│   └── memory.py               # Conversational query history tracking
└── sample_data/
    ├── create_sample_doc.py    # Sample whitepaper generator
    └── multimodal_ai_whitepaper.pdf # 3-page test whitepaper
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Srusti159/document-intelligence-rag.git
cd document-intelligence-rag
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to `.env` and add your free Groq API key from [Groq Console](https://console.groq.com/keys):
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing with Sample Data

You can immediately test the system using the included sample whitepaper (`sample_data/multimodal_ai_whitepaper.pdf`):
1. In the sidebar, upload `sample_data/multimodal_ai_whitepaper.pdf`.
2. Click **🚀 Process & Ingest PDFs**.
3. Ask sample queries:
   - *"What embedding model is utilized and what dimensionality does it produce?"*
   - *"How does the multi-stage retrieval handle chunks below the confidence threshold?"*
   - *"What are the token generation speeds on the Groq LPU?"*
4. Inspect the **Source Citations** accordion to see verified page numbers and cosine similarity scores!

---

## 📊 Technical Specifications

| Component | Technology | Rationale |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` | 384-d dense vector space, high semantic fidelity, CPU-friendly |
| **Vector DB** | ChromaDB (HNSW index) | Local persistence, zero cloud dependencies, sub-millisecond lookups |
| **Chunking Strategy** | Recursive Character Splitter | 800-character chunks with 150-character overlap for semantic continuity |
| **Retrieval Filter** | Cosine Similarity Threshold | Discards chunks with similarity < 0.35–0.40 to prevent hallucination |
| **LLM Inference** | Groq (`llama-3.3-70b-versatile`) | ~250+ tokens/sec, low-latency grounded reasoning |
| **Citation System** | In-prompt structured tagging | Exact page, chunk ID, and source document attribution |

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

Developed by **Srusti** ([@Srusti159](https://github.com/Srusti159))  
*Aspiring AI Developer passionate about Multimodal AI, RAG, and open-source LLMs.*
