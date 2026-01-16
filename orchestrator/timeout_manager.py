import asyncio
from contextlib import asynccontextmanager


class TimeoutManager:
    """Manage iteration-level and execution-level timeouts."""

    def __init__(self, iteration_timeout: int, execution_timeout: int):
        self.iteration_timeout = iteration_timeout
        self.execution_timeout = execution_timeout

    @asynccontextmanager
    async def iteration_scope(self):
        try:
            async with asyncio.timeout(self.iteration_timeout):
                yield
        except asyncio.TimeoutError:
            raise

    @asynccontextmanager
    async def execution_scope(self):
        try:
            async with asyncio.timeout(self.execution_timeout):
                yield
        except asyncio.TimeoutError:
            raise

