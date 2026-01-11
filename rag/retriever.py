"""
RAG Retriever using SQLite vector database with Gemini embeddings.

This module wraps the existing QAMP utilities to provide a unified interface
for retrieval across both the evaluation script and the Streamlit app.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add QAMP to path so we can import from scripts.common
_qamp_path = Path(__file__).parent.parent / "QAMP"
if str(_qamp_path) not in sys.path:
    sys.path.insert(0, str(_qamp_path))

from scripts.common import (
    get_db_connection,
    get_embedding_client,
    embed_texts,
    retrieve_top_k,
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
)


# Default paths relative to project root
DEFAULT_DB_PATH = Path(__file__).parent.parent / "QAMP" / "data" / "qamp.db"


class RAGRetriever:
    """
    Unified RAG retriever using SQLite + Gemini embeddings.
    
    Wraps the existing QAMP utilities (common.py) to provide a clean interface
    for retrieving relevant Qiskit documentation chunks.
    
    Usage:
        retriever = RAGRetriever()
        context = retriever.retrieve_context("How do I use Estimator?", top_k=5)
    
    Attributes:
        db_path: Path to the SQLite database containing embeddings
        embed_model: Name of the embedding model (must match what was used to build the DB)
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            db_path: Path to SQLite database. Defaults to QAMP/data/qamp.db
            embed_model: Embedding model name. Must match the model used to build the DB.
        """
        if db_path is None:
            self.db_path = str(DEFAULT_DB_PATH)
        else:
            self.db_path = db_path
            
        self.embed_model = embed_model
        self._client = None
        self._conn = None
        
        # Validate database exists
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"RAG database not found at {self.db_path}. "
                "Run the QAMP ingestion pipeline first."
            )
    
    @property
    def client(self):
        """Lazy-initialize the embedding client."""
        if self._client is None:
            self._client = get_embedding_client()
        return self._client
    
    @property
    def conn(self):
        """Lazy-initialize the database connection."""
        if self._conn is None:
            self._conn = get_db_connection(self.db_path)
        return self._conn
    
    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def _embed_query(self, query: str):
        """Embed a query string using Gemini."""
        embeddings = embed_texts(
            self.client,
            [query],
            task_type="RETRIEVAL_QUERY",
            model=self.embed_model,
            request_label="rag_query",
        )
        if not embeddings:
            raise ValueError("Failed to embed query")
        return embeddings[0]
    
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            query: The query string to search for
            top_k: Number of chunks to retrieve
            source_filter: Optional source path to filter by
            
        Returns:
            List of dicts, each containing:
                - id: Chunk identifier (e.g., "docs/api/qiskit/circuit.mdx#0")
                - text: The chunk text content
                - url: GitHub URL to the source file
                - source: Source path
                - chunk_index: Index of chunk within the file
                - score: Cosine similarity score (0-1)
        """
        query_embedding = self._embed_query(query)
        
        results = retrieve_top_k(
            self.conn,
            query_embedding,
            k=top_k,
            source_filter=source_filter,
        )
        
        # Add score to each chunk dict
        chunks_with_scores = []
        for chunk, score in results:
            chunk_with_score = dict(chunk)
            chunk_with_score["score"] = score
            chunks_with_scores.append(chunk_with_score)
        
        return chunks_with_scores
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_filter: Optional[str] = None,
    ) -> str:
        """
        Retrieve context string for LLM prompts.
        
        This is the main method for use in code generation workflows.
        Returns concatenated chunk texts suitable for injection into prompts.
        
        Args:
            query: The query string (typically the task prompt)
            top_k: Number of chunks to retrieve
            source_filter: Optional source path to filter by
            
        Returns:
            Concatenated context string (chunks joined by "\\n\\n")
        """
        chunks = self.retrieve(query, top_k=top_k, source_filter=source_filter)
        
        if not chunks:
            return ""
        
        context_parts = [chunk["text"] for chunk in chunks]
        return "\n\n".join(context_parts)


def retrieve_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: Optional[str] = None,
) -> str:
    """
    Convenience function for one-off context retrieval.
    
    Creates a temporary RAGRetriever, retrieves context, and closes the connection.
    For repeated queries, create a RAGRetriever instance instead.
    
    Args:
        query: The query string
        top_k: Number of chunks to retrieve
        db_path: Optional path to SQLite database
        
    Returns:
        Concatenated context string
    """
    with RAGRetriever(db_path=db_path) as retriever:
        return retriever.retrieve_context(query, top_k=top_k)
