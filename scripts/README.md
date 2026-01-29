# QAMP

RAG pipeline for Qiskit documentation.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
GOOGLE_API_KEY=...
GITHUB_TOKEN=...      # optional, avoids rate limits
```

## Pipeline

### 1. Chunk docs

Fetches `.mdx` files from `docs/api`, skips versioned folders, chunks by tokens (~800 tokens, 200 overlap).

```bash
python scripts/qiskit_docs_pipeline.py --path docs/api --out data/chunks.jsonl
```

Incremental by default (compares file SHAs). Use `--fresh` to rebuild.

### 2. Embed

Ingests chunks into SQLite with Gemini embeddings.

```bash
python scripts/ingest_to_sqlite.py --input data/chunks.jsonl --db data/qamp.db
```

Resumes from existing DB entries. Use `--fresh` to start over.

### 3. Query

```bash
python scripts/rag_query.py --query "How do I use the Estimator?" --k 5
```

Filter by source:
```bash
python scripts/rag_query.py --query "Batch execution" \
  --source-filter "Qiskit/documentation/docs/api/qiskit-ibm-runtime/batch.mdx"
```

## Schema

`chunks` table:
- `id` — `path#chunk_index`
- `text`, `source`, `url`
- `embedding` (BLOB), `embedding_norm`, `embedding_dim`
- `embedding_model`, `created_at`
