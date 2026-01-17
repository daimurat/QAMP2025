"""
OpenRouter chat client wrapper that matches the agents' expected interface.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class OpenRouterChatClient:
    """
    OpenRouter client using the OpenAI-compatible API.
    
    Supports reasoning models with the reasoning.enabled flag,
    and provider sorting (e.g., by price).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4000,
        timeout_seconds: int = 60,
        sort_by: str = "price",
    ):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        )
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.sort_by = sort_by

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:

        def _call():
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout_seconds,
                "extra_body": {
                    "provider": {
                        "sort": self.sort_by,
                    },
                },
            }
            if response_format:
                kwargs["response_format"] = response_format
            return self.client.chat.completions.create(**kwargs)

        resp = await asyncio.to_thread(_call)
        content = resp.choices[0].message.content
        return type("Resp", (), {"content": content})


__all__ = ["OpenRouterChatClient"]
