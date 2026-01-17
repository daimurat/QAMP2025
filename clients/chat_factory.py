"""
Factory for creating chat clients based on provider selection.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from clients.groq_chat import GroqChatClient
from clients.openrouter_chat import OpenRouterChatClient


class ChatProvider(str, Enum):
    """Supported chat providers."""

    GROQ = "groq"
    OPENROUTER = "openrouter"


def create_chat_client(
    provider: ChatProvider | str,
    model: str,
    api_key: Optional[str] = None,
    max_tokens: int = 4000,
    timeout_seconds: int = 60,
    **kwargs: Any,
) -> GroqChatClient | OpenRouterChatClient:
    """
    Create a chat client based on the provider.

    Args:
        provider: The provider to use ("groq" or "openrouter").
        model: The model name to use.
        api_key: Optional API key (falls back to environment variable).
        max_tokens: Maximum tokens for completion.
        timeout_seconds: Request timeout in seconds.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A chat client instance.
    """
    if isinstance(provider, str):
        provider = ChatProvider(provider.lower())

    if provider == ChatProvider.GROQ:
        return GroqChatClient(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    elif provider == ChatProvider.OPENROUTER:
        return OpenRouterChatClient(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            sort_by=kwargs.get("sort_by", "price"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = ["ChatProvider", "create_chat_client"]
