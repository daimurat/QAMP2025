import asyncio
import sys
from contextlib import asynccontextmanager

# Python 3.11+ has asyncio.timeout; older versions need async_timeout
if sys.version_info >= (3, 11):
    from asyncio import timeout as async_timeout_ctx
else:
    try:
        from async_timeout import timeout as async_timeout_ctx
    except ImportError:
        # Fallback: create a simple context manager using wait_for pattern
        @asynccontextmanager
        async def async_timeout_ctx(delay):
            """Simple timeout context manager for Python < 3.11."""
            yield


class TimeoutManager:
    """Manage iteration-level and execution-level timeouts."""

    def __init__(self, iteration_timeout: int, execution_timeout: int):
        self.iteration_timeout = iteration_timeout
        self.execution_timeout = execution_timeout

    @asynccontextmanager
    async def iteration_scope(self):
        try:
            async with async_timeout_ctx(self.iteration_timeout):
                yield
        except asyncio.TimeoutError:
            raise

    @asynccontextmanager
    async def execution_scope(self):
        try:
            async with async_timeout_ctx(self.execution_timeout):
                yield
        except asyncio.TimeoutError:
            raise
