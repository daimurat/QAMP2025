"""
Adapter to bridge the existing RAG retriever to the orchestrator interface.

Accepts multiple queries, deduplicates by doc_id, and returns RetrievedDocument
instances with timestamps.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional

from models import RetrievedDocument


class MultiQueryRAGRetriever:
    """
    Wraps a single-query retriever (rag.RAGRetriever) to support multiple queries.
    """

    def __init__(self, base_retriever):
        """
        Args:
            base_retriever: Object with retrieve(query: str, top_k: int) -> List[Dict]
        """
        self.base = base_retriever

    async def retrieve(self, queries: List[str], top_k: int = 5) -> List[RetrievedDocument]:
        docs: List[RetrievedDocument] = []
        seen = set()

        for query in queries:
            chunks = await self._maybe_await(self.base.retrieve(query, top_k=top_k))
            for chunk in chunks or []:
                doc_id = chunk.get("id") or chunk.get("doc_id")
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                docs.append(
                    RetrievedDocument(
                        doc_id=doc_id,
                        text=chunk.get("text", ""),
                        source=chunk.get("source", ""),
                        url=chunk.get("url", ""),
                        relevance_score=float(chunk.get("score", 0.0)),
                        retrieved_at=datetime.now(),
                        query_used=query,
                    )
                )
        return docs

    async def _maybe_await(self, value):
        if asyncio.iscoroutine(value):
            return await value
        return value


__all__ = ["MultiQueryRAGRetriever"]
