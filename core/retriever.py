"""
Multi-Stage Retrieval Engine
Handles semantic search, cosine similarity score calculation, confidence filtering,
and source citation tracking.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma


@dataclass
class ScoredChunk:
    content: str
    metadata: Dict[str, Any]
    similarity_score: float  # Normalized 0.0 to 1.0 (higher = more similar)
    page_number: int
    source_document: str
    chunk_id: str


class MultiStageRetriever:
    def __init__(
        self,
        vector_store: Chroma,
        default_top_k: int = 4,
        min_confidence_threshold: float = 0.40
    ):
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.min_confidence_threshold = min_confidence_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        confidence_threshold: float = None
    ) -> Tuple[List[ScoredChunk], Dict[str, Any]]:
        """
        Retrieves top_k chunks, computes cosine similarity, applies confidence
        threshold filtering, and packages exact citation metadata.
        """
        k = top_k or self.default_top_k
        threshold = confidence_threshold if confidence_threshold is not None else self.min_confidence_threshold

        # Chroma's similarity_search_with_relevance_scores returns (Document, score) where score is in [0, 1]
        try:
            results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k * 2  # Retrieve broader pool for multi-stage filtering
            )
        except Exception:
            # Fallback to standard similarity_search_with_score (returns distance)
            raw_results = self.vector_store.similarity_search_with_score(query=query, k=k * 2)
            results_with_scores = []
            for doc, distance in raw_results:
                # Convert L2/cosine distance to normalized similarity score [0, 1]
                score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                results_with_scores.append((doc, score))

        scored_chunks: List[ScoredChunk] = []

        for doc, score in results_with_scores:
            meta = doc.metadata or {}
            scored_chunk = ScoredChunk(
                content=doc.page_content,
                metadata=meta,
                similarity_score=round(float(score), 4),
                page_number=int(meta.get("page", 1)),
                source_document=str(meta.get("source", "Unknown Document")),
                chunk_id=str(meta.get("chunk_id", "N/A"))
            )
            scored_chunks.append(scored_chunk)

        # Multi-stage stage 2: Apply confidence threshold filtering
        filtered_chunks = [c for c in scored_chunks if c.similarity_score >= threshold]

        # Stage 3: Rank and slice to target top_k
        final_chunks = filtered_chunks[:k]

        # Metrics & citation summary
        stats = {
            "query": query,
            "total_candidates": len(scored_chunks),
            "passed_threshold": len(filtered_chunks),
            "final_selected": len(final_chunks),
            "threshold_applied": threshold,
            "max_confidence": max([c.similarity_score for c in scored_chunks], default=0.0),
            "is_grounded": len(final_chunks) > 0
        }

        return final_chunks, stats

    def format_context_with_citations(self, chunks: List[ScoredChunk]) -> str:
        """
        Formats retrieved chunks into an audit-ready, structured prompt context
        with explicit citation tags.
        """
        if not chunks:
            return "No relevant context found meeting the confidence threshold."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"[CITATION {idx}] (Document: {chunk.source_document}, "
                f"Page: {chunk.page_number}, Confidence: {chunk.similarity_score * 100:.1f}%)\n"
                f"{chunk.content.strip()}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
