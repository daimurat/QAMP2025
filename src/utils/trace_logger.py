"""
Trace Logger for Multi-Agent Observability

This module provides data models and utilities for capturing, 
analyzing, and persisting trace data from agent interactions.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass
class AgentMessage:
    """Represents a single message in the agent conversation."""
    agent_name: str
    role: str  # "assistant", "user", "function", "tool"
    content: str  # Full content for decision tracing
    timestamp: str
    tool_calls: list[dict] | None = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGQuery:
    """Represents a RAG retrieval operation."""
    query_id: str
    query_text: str
    top_k: int
    min_score: float
    retrieved_chunks: list[dict]  # {source, url, score, text_preview (200 chars)}
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """Model configuration used for the task."""
    model: str
    temperature: float
    api_type: str
    base_url: str | None = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskTrace:
    """Complete trace for a single task execution."""
    task_id: str
    entry_point: str
    prompt: str
    model_config: ModelConfig
    messages: list[AgentMessage] = field(default_factory=list)
    rag_queries: list[RAGQuery] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    final_code: str | None = None
    latency_s: float = 0.0
    passed: bool | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "entry_point": self.entry_point,
            "prompt": self.prompt,
            "timestamp": self.timestamp,
            "latency_s": self.latency_s,
            "passed": self.passed,
            "error": self.error,
            "model_config": self.model_config.to_dict(),
            "token_usage": self.token_usage.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "rag_queries": [r.to_dict() for r in self.rag_queries],
            "final_code": self.final_code,
        }


class TraceLogger:
    """
    Manages trace data collection and persistence for a single task.
    
    Usage:
        logger = TraceLogger(task_id="qiskitHumanEval/47", entry_point="create_bell_state", 
                             prompt="...", model_config=ModelConfig(...))
        logger.capture_from_chat_history(result.chat_history)
        logger.add_rag_query(query, top_k, min_score, chunks)
        logger.save(traces_dir)
    """
    
    RAG_PREVIEW_LENGTH = 200
    TOOL_RESPONSE_TRUNCATE_LENGTH = 500
    
    def __init__(
        self,
        task_id: str,
        entry_point: str,
        prompt: str,
        model_config: ModelConfig
    ):
        self.trace = TaskTrace(
            task_id=task_id,
            entry_point=entry_point,
            prompt=prompt,
            model_config=model_config
        )
        self._rag_query_counter = 0
    
    def capture_from_chat_history(self, chat_history: list[dict]) -> None:
        """
        Parse and capture messages from AutoGen chat history.
        
        Args:
            chat_history: List of message dicts from result.chat_history
        """
        for msg in chat_history:
            content = msg.get("content", "") or ""
            
            # Truncate TERMINATE messages
            if content.strip() == "TERMINATE":
                content = "[TERMINATE]"
            
            # Extract tool calls if present
            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = []
                for tc in msg["tool_calls"]:
                    tool_call_info = {
                        "name": tc.get("function", {}).get("name", "unknown"),
                        "arguments": tc.get("function", {}).get("arguments", "")
                    }
                    # Truncate long tool arguments
                    if len(tool_call_info["arguments"]) > self.TOOL_RESPONSE_TRUNCATE_LENGTH:
                        tool_call_info["arguments"] = (
                            tool_call_info["arguments"][:self.TOOL_RESPONSE_TRUNCATE_LENGTH] + "..."
                        )
                    tool_calls.append(tool_call_info)
            
            agent_msg = AgentMessage(
                agent_name=msg.get("name", msg.get("role", "unknown")),
                role=msg.get("role", "unknown"),
                content=content,
                timestamp=datetime.now().isoformat(),
                tool_calls=tool_calls
            )
            self.trace.messages.append(agent_msg)
    
    def add_rag_query(
        self,
        query: str,
        top_k: int,
        min_score: float,
        chunks: list[dict]
    ) -> None:
        """
        Log a RAG retrieval operation.
        
        Args:
            query: The search query text
            top_k: Number of chunks requested
            min_score: Minimum similarity threshold
            chunks: List of retrieved chunks with their metadata
        """
        self._rag_query_counter += 1
        
        # Create preview versions of chunks
        preview_chunks = []
        for chunk in chunks:
            preview = {
                "source": chunk.get("source", "N/A"),
                "url": chunk.get("url", "N/A"),
                "score": chunk.get("score", 0.0),
                "text_preview": (chunk.get("text", "")[:self.RAG_PREVIEW_LENGTH] + "..."
                                if len(chunk.get("text", "")) > self.RAG_PREVIEW_LENGTH
                                else chunk.get("text", ""))
            }
            preview_chunks.append(preview)
        
        rag_query = RAGQuery(
            query_id=f"rag_{self._rag_query_counter:03d}",
            query_text=query,
            top_k=top_k,
            min_score=min_score,
            retrieved_chunks=preview_chunks,
            timestamp=datetime.now().isoformat()
        )
        self.trace.rag_queries.append(rag_query)
    
    def set_result(
        self,
        final_code: str | None,
        latency_s: float,
        passed: bool | None = None,
        error: str | None = None
    ) -> None:
        """Set the final result of the task execution."""
        self.trace.final_code = final_code
        self.trace.latency_s = latency_s
        self.trace.passed = passed
        self.trace.error = error
    
    def set_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0
    ) -> None:
        """Set token usage statistics."""
        self.trace.token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    
    def generate_summary(self) -> str:
        """Generate a human-readable markdown summary of the trace."""
        lines = []
        
        # Header
        status = "✓ PASS" if self.trace.passed else "✗ FAIL" if self.trace.passed is False else "?"
        lines.append(f"# Task: {self.trace.entry_point}")
        lines.append(f"**ID**: {self.trace.task_id}")
        lines.append(f"**Result**: {status} | **Latency**: {self.trace.latency_s:.2f}s")
        lines.append("")
        
        # Model info
        lines.append("## Model Configuration")
        lines.append(f"- **Model**: {self.trace.model_config.model}")
        lines.append(f"- **Temperature**: {self.trace.model_config.temperature}")
        lines.append(f"- **API Type**: {self.trace.model_config.api_type}")
        if self.trace.model_config.base_url:
            lines.append(f"- **Base URL**: {self.trace.model_config.base_url}")
        lines.append("")
        
        # Token usage
        if self.trace.token_usage.total_tokens > 0:
            lines.append("## Token Usage")
            lines.append(f"- Prompt: {self.trace.token_usage.prompt_tokens:,}")
            lines.append(f"- Completion: {self.trace.token_usage.completion_tokens:,}")
            lines.append(f"- Total: {self.trace.token_usage.total_tokens:,}")
            lines.append("")
        
        # Agent flow
        lines.append("## Agent Flow")
        lines.append("")
        
        for i, msg in enumerate(self.trace.messages, 1):
            role_icon = {"assistant": "🤖", "user": "👤", "function": "⚙️", "tool": "🔧"}.get(msg.role, "📝")
            lines.append(f"### {i}. {role_icon} {msg.agent_name}")
            
            # Show tool calls if any
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"**Tool Call**: `{tc['name']}`")
            
            # Truncate very long content for summary
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "\n\n*[Content truncated for summary...]*"
            
            lines.append(f"\n{content}\n")
        
        # RAG queries table
        if self.trace.rag_queries:
            lines.append("## RAG Retrievals")
            lines.append("")
            lines.append("| Query | Top Score | Chunks |")
            lines.append("|-------|-----------|--------|")
            for rq in self.trace.rag_queries:
                top_score = max((c["score"] for c in rq.retrieved_chunks), default=0)
                query_preview = rq.query_text[:50] + "..." if len(rq.query_text) > 50 else rq.query_text
                lines.append(f"| {query_preview} | {top_score:.3f} | {len(rq.retrieved_chunks)} |")
            lines.append("")
        
        # Error details
        if self.trace.error:
            lines.append("## Error")
            lines.append(f"```\n{self.trace.error[:500]}\n```")
            lines.append("")
        
        return "\n".join(lines)
    
    def save(self, traces_dir: Path) -> Path:
        """
        Save trace data to the traces directory.
        
        Args:
            traces_dir: Base traces directory
            
        Returns:
            Path to the task-specific trace folder
        """
        # Create task-specific folder
        task_folder_name = f"{self.trace.entry_point}"
        task_dir = traces_dir / task_folder_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trace.json (machine-readable)
        trace_json_path = task_dir / "trace.json"
        with open(trace_json_path, "w", encoding="utf-8") as f:
            json.dump(self.trace.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save summary.md (human-readable)
        summary_path = task_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.generate_summary())
        
        # Save chat_history.json (full messages)
        chat_path = task_dir / "chat_history.json"
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.trace.messages], f, indent=2, ensure_ascii=False)
        
        # Save RAG queries if any
        if self.trace.rag_queries:
            rag_dir = task_dir / "rag_queries"
            rag_dir.mkdir(exist_ok=True)
            for rq in self.trace.rag_queries:
                rq_path = rag_dir / f"{rq.query_id}.json"
                with open(rq_path, "w", encoding="utf-8") as f:
                    json.dump(rq.to_dict(), f, indent=2, ensure_ascii=False)
        
        return task_dir


# Global RAG trace callback for integration with rag_tools.py
_rag_trace_callback: Callable[[str, int, float, list], None] | None = None


def set_rag_trace_callback(callback: Callable[[str, int, float, list], None]) -> None:
    """
    Set the callback for RAG query tracing.
    
    The callback receives: (query, top_k, min_score, chunks)
    """
    global _rag_trace_callback
    _rag_trace_callback = callback


def get_rag_trace_callback() -> Callable[[str, int, float, list], None] | None:
    """Get the current RAG trace callback."""
    return _rag_trace_callback


def clear_rag_trace_callback() -> None:
    """Clear the RAG trace callback."""
    global _rag_trace_callback
    _rag_trace_callback = None


__all__ = [
    "AgentMessage",
    "RAGQuery", 
    "ModelConfig",
    "TokenUsage",
    "TaskTrace",
    "TraceLogger",
    "set_rag_trace_callback",
    "get_rag_trace_callback",
    "clear_rag_trace_callback",
]
