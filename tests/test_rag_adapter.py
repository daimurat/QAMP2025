import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.rag_adapter import MultiQueryRAGRetriever  # noqa: E402


class StubBaseRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 5):
        self.calls += 1
        return [
            {"id": f"id-{query}-1", "text": "t1", "source": "s", "url": "u", "score": 0.9},
            {"id": "shared", "text": "t2", "source": "s", "url": "u", "score": 0.8},
        ]


@pytest.mark.asyncio
async def test_multi_query_rag_retriever_dedupes():
    base = StubBaseRetriever()
    retriever = MultiQueryRAGRetriever(base)
    docs = await retriever.retrieve(["q1", "q2"], top_k=2)
    ids = {d.doc_id for d in docs}
    assert len(ids) == 3  # id-q1-1, id-q2-1, shared
    assert base.calls == 2
