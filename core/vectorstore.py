"""
ChromaDB Vector Store Manager
Initializes all-MiniLM-L6-v2 embeddings and manages persistent ChromaDB collections.
"""

import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class ChromaVectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "document_intelligence"
    ):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.collection_name = collection_name

        # Initialize SentenceTransformers embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store: Optional[Chroma] = None
        self._init_or_load()

    def _init_or_load(self):
        """Loads existing ChromaDB or prepares collection."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]) -> int:
        """Adds documents/chunks to persistent ChromaDB collection."""
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        return len(documents)

    def get_document_count(self) -> int:
        """Returns total vector count in the collection."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def reset_collection(self):
        """Purges and resets the current collection and storage."""
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass

        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except Exception:
                pass

        self._init_or_load()
