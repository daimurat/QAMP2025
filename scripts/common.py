import os
import sys
import sqlite3
import time
import numpy as np
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

DEFAULT_DB_PATH = "data/qamp.db"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_GEN_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 5

# Rate limiting configuration.
# Free tier is quite restrictive; default to 15 RPM (4s between requests).
# Set EMBED_REQUESTS_PER_MIN=0 to disable rate limiting entirely.
DEFAULT_EMBED_RPM = int(os.getenv("EMBED_REQUESTS_PER_MIN", "15"))
EMBED_MAX_RETRIES = 5
EMBED_BACKOFF_BASE_SECONDS = 10.0
EMBED_BACKOFF_MAX_SECONDS = 120.0


class FixedIntervalRateLimiter:
    """
    Rate limiter that enforces a fixed minimum interval between requests.
    
    Unlike a token bucket, this does NOT allow initial bursting - it paces
    requests evenly from the start to stay within API rate limits.
    
    For example, at 15 RPM this enforces ~4 seconds between each request.
    """
    def __init__(self, requests_per_minute: int):
        self.rpm = max(requests_per_minute, 0)
        if self.rpm > 0:
            self.min_interval = 60.0 / self.rpm
        else:
            self.min_interval = 0
        self.last_request_time: Optional[float] = None

    def acquire(self) -> float:
        """
        Wait if necessary to maintain the rate limit.
        
        Returns:
            The number of seconds waited (0 if no wait was needed).
        """
        if self.rpm <= 0 or self.min_interval <= 0:
            return 0.0

        now = time.monotonic()
        waited = 0.0

        if self.last_request_time is not None:
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
                waited = sleep_time

        self.last_request_time = time.monotonic()
        return waited


# Global rate limiter instance
EMBED_RATE_LIMITER: Optional[FixedIntervalRateLimiter] = None
if DEFAULT_EMBED_RPM > 0:
    EMBED_RATE_LIMITER = FixedIntervalRateLimiter(DEFAULT_EMBED_RPM)
    print(
        f"[rate-limiter] Initialized at {DEFAULT_EMBED_RPM} RPM "
        f"(~{60.0/DEFAULT_EMBED_RPM:.1f}s between requests)",
        file=sys.stderr,
    )


def _is_rate_limit_error(err: Exception) -> bool:
    if isinstance(err, errors.ClientError):
        if getattr(err, "code", None) == 429:
            return True
        status = getattr(err, "status", "")
        if isinstance(status, str) and status.upper() == "RESOURCE_EXHAUSTED":
            return True
    return False


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the SQLite database with the chunks table schema."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT,
            url TEXT,
            chunk_index INTEGER,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding_norm REAL NOT NULL,
            embedding_model TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)
    """)
    
    conn.commit()
    conn.close()


def get_embedding_client(api_key: Optional[str] = None) -> genai.Client:
    """Get Google Gemini embedding client."""
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment (.env)")
    return genai.Client(api_key=api_key)


def embed_texts(
    client: genai.Client,
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = DEFAULT_EMBED_MODEL,
    request_label: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Embed a batch of texts using Google Gemini embeddings.
    
    Args:
        client: Google genai client
        texts: List of text strings to embed
        task_type: "RETRIEVAL_DOCUMENT" for documents, "RETRIEVAL_QUERY" for queries
        model: Embedding model name
        request_label: Optional label for logging (e.g., batch identifier)
        
    Returns:
        List of numpy arrays (float32) representing embeddings
    """
    if not texts:
        return []

    batch_size = len(texts)
    label = request_label or "embed_texts"
    
    # Wait for rate limit before making request
    if EMBED_RATE_LIMITER:
        waited = EMBED_RATE_LIMITER.acquire()
        if waited > 0.5:  # Only log if we actually waited significantly
            print(f"[{label}] waited {waited:.1f}s for rate limit", file=sys.stderr)

    attempt = 0
    backoff = EMBED_BACKOFF_BASE_SECONDS
    last_exception = None

    while attempt < max(1, EMBED_MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768
                ),
            )

            embeddings = getattr(result, "embeddings", None)
            if not embeddings:
                raise ValueError("No embeddings returned from API")

            embedding_arrays = []
            for emb in embeddings:
                if hasattr(emb, "values"):
                    arr = np.array(emb.values, dtype=np.float32)
                    embedding_arrays.append(arr)
                else:
                    raise ValueError(f"Unexpected embedding structure: {emb}")

            return embedding_arrays

        except Exception as e:
            last_exception = e
            should_retry = _is_rate_limit_error(e)
            error_code = getattr(e, "code", None)
            error_status = getattr(e, "status", None)
            detail = getattr(e, "message", str(e))
            print(
                f"[{label}] attempt {attempt + 1}/{EMBED_MAX_RETRIES} failed "
                f"(batch_size={batch_size}, model={model}, code={error_code}, status={error_status}): {detail}",
                file=sys.stderr,
            )
            if should_retry and attempt < max(1, EMBED_MAX_RETRIES) - 1:
                print(f"[{label}] retrying in {backoff:.0f}s...", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, EMBED_BACKOFF_MAX_SECONDS)
                attempt += 1
                continue
            raise RuntimeError(f"Error embedding texts: {e}") from e

    raise RuntimeError(f"Error embedding texts: {last_exception}") from last_exception


def compute_cosine_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: List[np.ndarray],
    doc_norms: List[float]
) -> np.ndarray:
    """
    Compute cosine similarities between query embedding and document embeddings.
    
    Args:
        query_embedding: Query vector (float32 numpy array)
        doc_embeddings: List of document vectors (float32 numpy arrays)
        doc_norms: Precomputed L2 norms for each document
        
    Returns:
        Array of cosine similarity scores
    """
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return np.zeros(len(doc_embeddings))
    
    doc_matrix = np.array(doc_embeddings)
    
    dot_products = np.dot(doc_matrix, query_embedding)
    
    doc_norms_array = np.array(doc_norms)
    similarities = dot_products / (doc_norms_array * query_norm)
    
    return similarities


def retrieve_top_k(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    k: int = DEFAULT_TOP_K,
    source_filter: Optional[str] = None
) -> List[Tuple[dict, float]]:
    """
    Retrieve top-k chunks by cosine similarity.
    
    Args:
        conn: SQLite connection
        query_embedding: Query vector (float32 numpy array)
        k: Number of top results to return
        source_filter: Optional source path to filter by
        
    Returns:
        List of tuples (chunk_dict, score) sorted by score descending
        chunk_dict contains: id, text, url, source, chunk_index
    """
    cursor = conn.cursor()
        
    if source_filter:
        cursor.execute("""
            SELECT id, text, url, source, chunk_index, embedding, embedding_norm
            FROM chunks
            WHERE source = ?
        """, (source_filter,))
    else:
        cursor.execute("""
            SELECT id, text, url, source, chunk_index, embedding, embedding_norm
            FROM chunks
        """)
    
    rows = cursor.fetchall()
    
    if not rows:
        return []
    
    doc_embeddings = []
    doc_norms = []
    chunk_data = []
    
    for row in rows:
        embedding_bytes = row["embedding"]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        doc_embeddings.append(embedding)
        doc_norms.append(row["embedding_norm"])
        chunk_data.append({
            "id": row["id"],
            "text": row["text"],
            "url": row["url"],
            "source": row["source"],
            "chunk_index": row["chunk_index"]
        })
    
    similarities = compute_cosine_similarity(query_embedding, doc_embeddings, doc_norms)
    
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = [
        (chunk_data[idx], float(similarities[idx]))
        for idx in top_indices
    ]
    
    return results
