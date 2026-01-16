import asyncio
import random
from typing import Any, Callable

from models import RetryPolicy


async def with_retry(func: Callable[[], Any], policy: RetryPolicy, context: str = "") -> Any:
    """Execute an async function with retry logic."""
    delay = policy.initial_delay_seconds
    attempt = 0
    last_exception = None

    while attempt <= policy.max_retries:
        try:
            return await func()
        except Exception as e:
            last_exception = e
            attempt += 1
            if attempt > policy.max_retries:
                break
            sleep_for = delay
            if policy.jitter:
                sleep_for = delay * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(min(sleep_for, policy.max_delay_seconds))
            delay = min(delay * policy.exponential_base, policy.max_delay_seconds)
    if last_exception:
        raise last_exception
    return None

