import argparse
import json
import sys
import os
import time
import sqlite3
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm
from scripts.common import (
    get_db_connection,
    init_database,
    get_embedding_client,
    embed_texts,
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_BATCH_SIZE,
)


def ingest_chunks(
    input_file: str,
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_model: str = DEFAULT_EMBED_MODEL,
    stop_on_error: bool = False,
    fresh: bool = False,
):
    """
    Ingest chunks from JSONL into SQLite with embeddings.
    
    Args:
        input_file: Path to JSONL file
        db_path: Path to SQLite database
        batch_size: Number of texts to embed per batch
        embed_model: Embedding model name
        stop_on_error: If True, stop on first error; if False, continue
        fresh: If True, start fresh by clearing the database
    """
    if fresh:
        print("Starting fresh. Clearing database...")
        if os.path.exists(db_path):
            os.remove(db_path)

    print(f"Initializing database at {db_path}...")
    init_database(db_path)
    
    print("Connecting to Google Gemini API...")
    client = get_embedding_client()
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Load existing chunk IDs for resumability
    existing_ids = set()
    if not fresh:
        print("Checking for existing chunks in database...")
        try:
            cursor.execute("SELECT id FROM chunks")
            rows = cursor.fetchall()
            existing_ids = {row["id"] for row in rows}
            print(f"Found {len(existing_ids)} existing chunks.")
        except sqlite3.OperationalError:
            # Table might not exist yet
            pass

    total_chunks = 0
    processed_chunks = 0
    skipped_chunks = 0
    failed_chunks = 0
    embedding_dim = None
    batch_num = 0
    
    print(f"Reading chunks from {input_file}...")
    
    batch_texts = []
    batch_metadata = []

    def flush_batch():
        nonlocal batch_texts, batch_metadata, batch_num
        nonlocal processed_chunks, failed_chunks, embedding_dim

        if not batch_texts:
            return

        batch_num += 1
        count = len(batch_texts)
        approx_tokens = sum(max(len(text) // 4, 1) for text in batch_texts)
        print(
            f"[Batch {batch_num}] Embedding {count} chunks (est_tokens~{approx_tokens})...",
            flush=True,
        )
        start = time.perf_counter()

        processed, failed, dim = process_batch(
            cursor, conn, client, batch_texts, batch_metadata,
            embed_model, stop_on_error, batch_label=f"batch-{batch_num}"
        )

        elapsed = time.perf_counter() - start
        processed_chunks += processed
        failed_chunks += failed
        if dim and embedding_dim is None:
            embedding_dim = dim

        status = "completed" if processed else "failed"
        print(
            f"[Batch {batch_num}] {status} in {elapsed:.1f}s "
            f"(processed={processed}, failed={failed})",
            flush=True,
        )

        batch_texts = []
        batch_metadata = []
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line.strip())
                    text = obj.get("text", "")
                    chunk_id = obj.get("id", f"chunk_{line_num}")
                    
                    if not text:
                        continue

                    if chunk_id in existing_ids:
                        skipped_chunks += 1
                        continue
                    
                    batch_texts.append(text)
                    batch_metadata.append({
                        "id": chunk_id,
                        "source": obj.get("source"),
                        "url": obj.get("url"),
                        "chunk_index": obj.get("chunk_index"),
                        "text": text,
                    })
                    
                    total_chunks += 1
                    
                    if len(batch_texts) >= batch_size:
                        flush_batch()
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                    if stop_on_error:
                        raise
                    failed_chunks += 1
                    continue
        
        flush_batch()
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        print(f"Error during ingestion: {e}", file=sys.stderr)
        raise
    
    finally:
        conn.close()
    
    print("\n" + "="*60)
    print("Ingestion Summary:")
    print(f"  Total chunks read: {total_chunks}")
    print(f"  Skipped (already in DB): {skipped_chunks}")
    print(f"  Successfully processed: {processed_chunks}")
    print(f"  Failed: {failed_chunks}")
    if embedding_dim:
        print(f"  Embedding dimension: {embedding_dim}")
    print("="*60)


def process_batch(
    cursor,
    conn,
    client,
    texts: list,
    metadata: list,
    embed_model: str,
    stop_on_error: bool,
    batch_label: str,
) -> tuple[int, int, Optional[int]]:
    """
    Process a single batch of texts: embed and insert into database.
    
    Returns:
        (processed_count, failed_count, embedding_dimension)
    """
    try:
        embeddings = embed_texts(
            client,
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            model=embed_model,
            request_label=batch_label,
        )
        
        if len(embeddings) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
        
        embedding_dim = len(embeddings[0]) if embeddings else None
        
        for i, (meta, embedding) in enumerate(zip(metadata, embeddings)):
            try:
                # Convert embedding to float32 and compute norm
                embedding_float32 = embedding.astype(np.float32)
                norm = float(np.linalg.norm(embedding_float32))
                
                embedding_blob = embedding_float32.tobytes()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, source, url, chunk_index, text, embedding, embedding_dim, embedding_norm, embedding_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    meta["id"],
                    meta["source"],
                    meta["url"],
                    meta["chunk_index"],
                    meta["text"],
                    embedding_blob,
                    embedding_dim,
                    norm,
                    embed_model,
                ))
            
            except Exception as e:
                print(f"Error inserting chunk {meta.get('id', 'unknown')}: {e}", file=sys.stderr)
                if stop_on_error:
                    raise
                return 0, len(texts), embedding_dim
        
        conn.commit()
        return len(texts), 0, embedding_dim
    
    except Exception as e:
        print(f"Error processing batch: {e}", file=sys.stderr)
        if stop_on_error:
            raise
        return 0, len(texts), None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest chunks from JSONL into SQLite with embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/chunks.jsonl",
        help="Input JSONL file path (default: data/chunks.jsonl)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embeddings (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing on first error",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, clearing the database and state",
    )
    
    args = parser.parse_args()
    
    ingest_chunks(
        input_file=args.input,
        db_path=args.db,
        batch_size=args.batch_size,
        embed_model=args.embed_model,
        stop_on_error=args.stop_on_error,
        fresh=args.fresh,
    )

