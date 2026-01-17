"""
Groq chat client wrapper that matches the agents' expected interface.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from groq import Groq


class GroqRateLimiter:
    """
    Proactive rate limiter that enforces a minimum interval between requests.
    
    Free tier Groq has strict limits:
    - ~30 requests per minute (RPM)
    - ~6000-10000 tokens per minute (TPM)
    
    Since large prompts with RAG context consume many tokens, we need
    conservative pacing to avoid hitting TPM limits.
    """
    
    _instance: Optional["GroqRateLimiter"] = None
    _lock = asyncio.Lock()
    
    def __init__(self, requests_per_minute: int):
        self.rpm = max(requests_per_minute, 0)
        if self.rpm > 0:
            self.min_interval = 60.0 / self.rpm
        else:
            self.min_interval = 0
        self.last_request_time: Optional[float] = None
        self._acquire_lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> "GroqRateLimiter":
        # Default to 10 RPM (6s between requests) for free tier
        # This is conservative to avoid TPM limits with large prompts
        if cls._instance is None:
            rpm = int(os.getenv("GROQ_REQUESTS_PER_MIN", "10"))
            cls._instance = cls(rpm)
            if rpm > 0:
                print(
                    f"[groq-rate-limiter] Initialized at {rpm} RPM "
                    f"(~{60.0/rpm:.1f}s between requests)",
                    file=sys.stderr,
                )
        return cls._instance
    
    async def acquire(self) -> float:
        """
        Wait if necessary to maintain the rate limit.
        
        Returns:
            The number of seconds waited (0 if no wait was needed).
        """
        if self.rpm <= 0 or self.min_interval <= 0:
            return 0.0

        async with self._acquire_lock:
            now = time.monotonic()
            waited = 0.0

            if self.last_request_time is not None:
                elapsed = now - self.last_request_time
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    await asyncio.sleep(sleep_time)
                    waited = sleep_time

            self.last_request_time = time.monotonic()
            return waited


class GroqChatClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4000,
        timeout_seconds: int = 60,
    ):
        self.model = model
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._rate_limiter = GroqRateLimiter.get_instance()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        # Wait for rate limit before making request
        waited = await self._rate_limiter.acquire()
        if waited > 0.5:
            print(f"[groq] waited {waited:.1f}s for rate limit", file=sys.stderr)
        
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=self.max_tokens,
                response_format=response_format,
                timeout=self.timeout_seconds,
            )

        resp = await asyncio.to_thread(_call)
        content = resp.choices[0].message.content
        return type("Resp", (), {"content": content})


__all__ = ["GroqChatClient", "GroqRateLimiter"]
