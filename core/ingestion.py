"""
Document Ingestion & Chunking Engine
Handles PDF parsing, overlap-aware splitting, and metadata enrichment.
"""

import os
import hashlib
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentIngestionEngine:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def extract_from_pdf(self, file_path_or_bytes, filename: str = "document.pdf") -> Tuple[List[Document], Dict[str, Any]]:
        """
        Extracts text from PDF page-by-page with exact page number metadata.
        Returns extracted documents and ingestion analytics.
        """
        reader = PdfReader(file_path_or_bytes)
        num_pages = len(reader.pages)
        raw_documents: List[Document] = []
        total_chars = 0

        # Compute document content hash for caching/tracking
        hasher = hashlib.sha256()

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            hasher.update(text.encode("utf-8"))
            total_chars += len(text)

            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_idx + 1,
                    "total_pages": num_pages
                }
            )
            raw_documents.append(doc)

        doc_hash = hasher.hexdigest()[:12]

        # Overlap-aware chunking
        chunks = self.text_splitter.split_documents(raw_documents)

        # Enrich chunk metadata with chunk IDs and source tracking
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{filename}_p{chunk.metadata.get('page', 1)}_c{idx}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["doc_hash"] = doc_hash
            chunk.metadata["char_count"] = len(chunk.page_content)

        stats = {
            "filename": filename,
            "total_pages": num_pages,
            "total_characters": total_chars,
            "total_chunks": len(chunks),
            "doc_hash": doc_hash,
            "avg_chunk_size": round(sum(len(c.page_content) for c in chunks) / max(1, len(chunks)), 1)
        }

        return chunks, stats
