"""
RAG module for Qiskit documentation retrieval.

Uses SQLite vector database with Gemini embeddings.
"""

from .retriever import RAGRetriever, retrieve_context

__all__ = ["RAGRetriever", "retrieve_context"]
