#!/usr/bin/env python3

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm


@dataclass
class PipelineConfig:
    owner: str
    repo: str
    root_path: str
    ref: str
    token: Optional[str]
    out_path: Path
    include_exts: List[str]
    chunk_size: int
    chunk_overlap: int
    fresh: bool


VERSION_DIR_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")


def build_github_session(token: Optional[str]) -> requests.Session:
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "qiskit-docs-chunker/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def iter_contents(
    session: requests.Session,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> Iterable[Dict]:
    """Yields all file/dir entries from a given repo path."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("type") == "file":
        yield data  # Single file case
    elif isinstance(data, list):
        for entry in data:
            yield entry
    else:
        return


def is_version_dir(name: str) -> bool:
    """Return True if the directory name looks like a version number."""
    return bool(VERSION_DIR_PATTERN.match(name))


def is_skipped_dir(name: str) -> bool:
    """Return True if the directory should be skipped (dev, release-notes, etc.)."""
    return name.lower() in {"dev", "release-notes"}


def list_files_recursive(
    session: requests.Session,
    owner: str,
    repo: str,
    root_path: str,
    ref: str,
    include_exts: List[str],
) -> List[Dict]:
    """
    Recursively lists all files under root_path using the Contents API,
    skipping versioned directories (e.g., 0.42, 1.1.0), dev, and release-notes.
    """
    results: List[Dict] = []
    stack = [root_path]

    pbar = tqdm(desc="Discovering files", unit="dir")
    while stack:
        current = stack.pop()
        pbar.update(1)
        try:
            for entry in iter_contents(session, owner, repo, current, ref):
                entry_type = entry.get("type")
                name: str = entry.get("name", "")
                if entry_type == "dir":
                    if is_version_dir(name) or is_skipped_dir(name):
                        # Skip historical version directories, dev, and release-notes
                        continue
                    stack.append(entry["path"])
                elif entry_type == "file":
                    if any(name.lower().endswith(ext) for ext in include_exts):
                        results.append(entry)
        except requests.HTTPError as e:
            print(f"WARN: Could not access {current}: {e}")
    pbar.close()
    return results


def download_text(session: requests.Session, download_url: str) -> str:
    response = session.get(download_url, timeout=60)
    response.raise_for_status()
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.text


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Prefer token-based chunking (approx. tokens), fallback to character splitter
    with markdown-aware separators.
    """
    # Try token-based splitting first
    try:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )
        return splitter.split_text(text)
    except Exception:
        pass

    # Fallback to character splitter with heading-friendly separators
    fallback_chunk_size = max(chunk_size * 4, 1000)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=chunk_overlap * 4,
        add_start_index=False,
        separators=[
            "\n### ",
            "\n## ",
            "\n# ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )
    return splitter.split_text(text)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_existing_file_shas(out_path: Path) -> Dict[str, Optional[str]]:
    """
    Scan the existing JSONL and build a mapping of path -> file_sha.
    If a record lacks file_sha, the value is None.
    """
    if not out_path.exists():
        return {}
    shas: Dict[str, Optional[str]] = {}
    try:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = doc.get("id", "")
                if "#" in chunk_id:
                    path = chunk_id.rsplit("#", 1)[0]
                else:
                    path = doc.get("path") or ""
                if not path:
                    continue
                shas[path] = doc.get("file_sha")
    except Exception as e:
        print(f"WARN: Could not read existing output {out_path}: {e}")
    return shas


def fetch_commit_sha(session: requests.Session, owner: str, repo: str, ref: str) -> Optional[str]:
    """Resolve the commit SHA for the given ref."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        sha = data.get("sha")
        if isinstance(sha, str):
            return sha
    except requests.HTTPError as e:
        print(f"WARN: Could not resolve commit for {ref}: {e}")
    return None


def iter_existing_chunks(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream existing JSONL entries, ignoring malformed lines."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"WARN: Could not read existing chunks from {path}: {e}")
        return []


class State(TypedDict, total=False):
    files: List[Dict[str, Any]]
    commit_sha: Optional[str]


def make_node_discover_files(config: PipelineConfig):
    """
    Factory to create the 'discover' node.
    This node finds all relevant files using the reliable Contents API.
    """
    def node(state: State) -> Dict[str, Any]:
        session = build_github_session(config.token)
        commit_sha = fetch_commit_sha(session, config.owner, config.repo, config.ref)
        files_from_api = list_files_recursive(
            session=session,
            owner=config.owner,
            repo=config.repo,
            root_path=config.root_path,
            ref=config.ref,
            include_exts=config.include_exts,
        )
        
        files = [
            {
                "path": f.get("path"),
                "download_url": f.get("download_url"),
                "sha": f.get("sha"),
            }
            for f in files_from_api
            if f.get("path") and f.get("download_url") and f.get("sha")
        ]

        print(f"Discovered {len(files)} files under {config.root_path}")
        if commit_sha:
            print(f"Reference {config.ref} resolves to commit {commit_sha}")
        return {"files": files, "commit_sha": commit_sha}
    return node


def make_node_process_files(config: PipelineConfig):
    """
    Factory to create the 'process' node.
    This node downloads, chunks, and writes the file content incrementally.
    """
    def node(state: State) -> Dict[str, Any]:
        session = build_github_session(config.token)
        ensure_parent_dir(config.out_path)

        files: List[Dict[str, Any]] = state.get("files", []) or []
        commit_sha = state.get("commit_sha") or config.ref

        if not files:
            print("No files found to process.")
            return {}

        if config.fresh:
            print("Starting fresh. Clearing previous output.")
            if config.out_path.exists():
                config.out_path.unlink()

        previous_shas = {} if config.fresh else load_existing_file_shas(config.out_path)
        new_shas = {f["path"]: f.get("sha") for f in files if f.get("path")}

        removed_paths = set(previous_shas) - set(new_shas)
        changed_paths = {
            path for path, sha in new_shas.items()
            if previous_shas.get(path) != sha
        }
        unchanged_paths = set(new_shas) - changed_paths

        if not changed_paths and not removed_paths and not config.fresh and config.out_path.exists():
            print("No changes detected; skipping processing.")
            return {}

        temp_out = config.out_path.with_suffix(config.out_path.suffix + ".tmp")
        with temp_out.open("w", encoding="utf-8") as out_f:
            # Keep unchanged chunks from previous run
            if config.out_path.exists() and not config.fresh:
                for doc in iter_existing_chunks(config.out_path):
                    chunk_id = doc.get("id", "")
                    path_prefix = chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else ""
                    if path_prefix in unchanged_paths:
                        out_f.write(json.dumps(doc, ensure_ascii=False))
                        out_f.write("\n")

            # Process added or changed files
            for file_entry in tqdm(files, desc="Processing files", unit="file"):
                path = file_entry.get("path", "")
                if path not in changed_paths:
                    continue

                download_url = file_entry.get("download_url")
                file_sha = file_entry.get("sha")

                if not download_url:
                    print(f"Skipping {path} (no download_url).")
                    continue

                try:
                    text = download_text(session, download_url)
                except requests.HTTPError as http_err:
                    print(f"Failed to download {path}: {http_err}")
                    continue
                except requests.RequestException as req_err:
                    print(f"Network error for {path}: {req_err}")
                    continue

                text_chunks = chunk_text(
                    text=text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )

                for i, chunk in enumerate(text_chunks):
                    doc = {
                        "id": f"{path}#{i}",
                        "path": path,
                        "source": f"{config.owner}/{config.repo}/{path}",
                        "url": f"https://github.com/{config.owner}/{config.repo}/blob/{config.ref}/{path}",
                        "commit_sha": commit_sha,
                        "file_sha": file_sha,
                        "chunk_index": i,
                        "text": chunk,
                    }
                    out_f.write(json.dumps(doc, ensure_ascii=False))
                    out_f.write("\n")

        temp_out.replace(config.out_path)

        print(
            f"Processed {len(changed_paths)} updated/added files. "
            f"Removed {len(removed_paths)} files."
        )
        return {}
    return node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Qiskit docs via GitHub API, chunk with LangGraph/LangChain, and write JSONL."
        )
    )
    parser.add_argument("--owner", default="Qiskit", help="GitHub owner")
    parser.add_argument("--repo", default="documentation", help="GitHub repo")
    parser.add_argument(
        "--path",
        dest="root_path",
        default="docs/api",
    )
    parser.add_argument(
        "--ref", default="main", help="Git reference (branch, tag, or commit SHA)"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="./data/chunks.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800, help="Chunk size (tokens) for splitting"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap for splitting"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring previous state and overwriting output",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("No GITHUB_TOKEN found in environment (.env). Using unauthenticated API calls (may hit rate limits).")

    config = PipelineConfig(
        owner=args.owner,
        repo=args.repo,
        root_path=args.root_path,
        ref=args.ref,
        token=token,
        out_path=Path(args.out_path),
        include_exts=[".mdx"],
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fresh=args.fresh,
    )

    initial_state: State = {}

    graph = StateGraph(State)
    graph.add_node("discover", make_node_discover_files(config))
    graph.add_node("process", make_node_process_files(config))

    graph.add_edge(START, "discover")
    graph.add_edge("discover", "process")
    graph.add_edge("process", END)

    app = graph.compile()
    
    print(f"Starting pipeline for {config.owner}/{config.repo}/{config.root_path}@{config.ref}")
    # Run the graph
    app.invoke(initial_state)
    print("Pipeline finished.")


if __name__ == "__main__":
    main()