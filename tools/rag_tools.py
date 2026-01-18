"""
RAG Tools for AutoGen Agents

This module provides tool functions that wrap RAGRetriever functionality
for use with AutoGen's function calling feature.
"""

from typing import Annotated
from rag import RAGRetriever
from utils.trace_logger import get_rag_trace_callback

# Global retriever instance (initialized once, reused across calls)
_retriever = None

# Session-based RAG query limiting
MAX_RAG_QUERIES_PER_TASK = 3
_rag_query_count = 0

def reset_rag_query_count():
    """Reset the RAG query counter. Call this at the start of each task."""
    global _rag_query_count
    _rag_query_count = 0

def get_rag_query_count() -> int:
    """Get the current RAG query count for this session."""
    return _rag_query_count

def get_retriever() -> RAGRetriever:
    """
    Get or create the global RAGRetriever instance.
    
    Returns:
        RAGRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(db_path="QAMP/data/qamp.db")
    return _retriever



def retrieve_qiskit_docs(
    query: Annotated[str, "Search query for Qiskit documentation (e.g., 'How to use Estimator primitive?')"],
    top_k: Annotated[int, "Number of documentation chunks to retrieve (default: 5)"] = 5,
    min_score: Annotated[float, "Minimum similarity score threshold 0.0-1.0 (default: 0.0)"] = 0.0,
) -> str:
    """
    Retrieve relevant Qiskit documentation chunks using RAG.
    
    This tool searches the Qiskit documentation database and returns
    the most relevant chunks based on semantic similarity to the query.
    Use this when you need specific information about Qiskit APIs,
    quantum gates, circuits, or other Qiskit functionality.
    
    Args:
        query: Search query describing what you need to know about Qiskit
        top_k: Maximum number of documentation chunks to retrieve (1-20)
        min_score: Minimum similarity score (0.0-1.0). Chunks below this threshold are filtered out.
    
    Returns:
        Formatted string containing relevant documentation chunks with metadata.
        Returns "No relevant documentation found." if no chunks meet the criteria.
    
    Example:
        >>> result = retrieve_qiskit_docs("How do I create a Bell state?", top_k=3)
        >>> print(result)
    """
    # Check RAG query limit
    global _rag_query_count
    if _rag_query_count >= MAX_RAG_QUERIES_PER_TASK:
        return (
            f"RAG query limit reached ({MAX_RAG_QUERIES_PER_TASK} queries per task). "
            f"Please proceed with your existing knowledge and the information already retrieved. "
            f"Generate the implementation based on what you know about Qiskit."
        )
    
    # Increment query count
    _rag_query_count += 1
    
    # Validate parameters
    if not query or not query.strip():
        return "Error: Query cannot be empty."
    
    if top_k < 1 or top_k > 20:
        return "Error: top_k must be between 1 and 20."
    
    if min_score < 0.0 or min_score > 1.0:
        return "Error: min_score must be between 0.0 and 1.0."
    
    try:
        # Get retriever instance
        retriever = get_retriever()
        
        # Retrieve chunks with scores
        all_chunks = retriever.retrieve(query, top_k=top_k)
        
        # Filter by minimum score
        chunks = [c for c in all_chunks if c["score"] >= min_score]
        
        if not chunks:
            return (
                f"No relevant documentation found for query: '{query}'\n"
                f"(Retrieved {len(all_chunks)} chunks, but none met min_score={min_score} threshold)"
            )
        
        # Format results for LLM consumption
        result_parts = [
            f"Found {len(chunks)} relevant documentation chunk(s) (filtered from {len(all_chunks)} total):\n"
        ]
        
        for i, chunk in enumerate(chunks, 1):
            result_parts.append(f"\n{'='*60}")
            result_parts.append(f"Chunk {i}/{len(chunks)} (similarity: {chunk['score']:.3f})")
            result_parts.append(f"{'='*60}")
            result_parts.append(f"Source: {chunk.get('source', 'N/A')}")
            result_parts.append(f"URL: {chunk.get('url', 'N/A')}")
            result_parts.append(f"\nContent:\n{chunk['text']}")
        
        # Invoke trace callback if set
        trace_callback = get_rag_trace_callback()
        if trace_callback:
            trace_callback(query, top_k, min_score, chunks)
        
        return "\n".join(result_parts)
    
    except FileNotFoundError as e:
        return f"Error: RAG database not found. {str(e)}"
    except Exception as e:
        return f"Error retrieving documentation: {str(e)}"


def close_retriever():
    """
    Close the global RAGRetriever instance and release resources.
    
    Call this when shutting down the application to properly close
    the database connection.
    """
    global _retriever
    if _retriever is not None:
        _retriever.close()
        _retriever = None


# Export main functions
__all__ = [
    "retrieve_qiskit_docs",
    "get_retriever",
    "close_retriever",
    "reset_rag_query_count",
    "get_rag_query_count",
    "MAX_RAG_QUERIES_PER_TASK",
]

# Made with Bob