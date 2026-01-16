import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import LocalCodeSandbox  # noqa: E402
from models import ExecutionResult  # noqa: E402


@pytest.mark.asyncio
async def test_sandbox_executes_code(tmp_path: Path):
    sandbox = LocalCodeSandbox()
    code = "print('hi')"
    result = await sandbox.execute(
        code=code,
        timeout_seconds=5,
        allowed_imports=["math"],
        working_directory=str(tmp_path),
        max_memory_mb=128,
    )
    assert result.success is True
    assert "hi" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_blocks_disallowed_import(tmp_path: Path):
    sandbox = LocalCodeSandbox()
    code = "import os\nprint('x')"
    result = await sandbox.execute(
        code=code,
        timeout_seconds=5,
        allowed_imports=["math"],
        working_directory=str(tmp_path),
        max_memory_mb=128,
    )
    assert result.success is False
    assert result.error_type == "ImportError"


@pytest.mark.asyncio
async def test_sandbox_times_out(tmp_path: Path):
    sandbox = LocalCodeSandbox()
    code = "import time\ntime.sleep(2)"
    result = await sandbox.execute(
        code=code,
        timeout_seconds=1,
        allowed_imports=["time"],
        working_directory=str(tmp_path),
        max_memory_mb=128,
    )
    assert result.success is False
    assert result.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_sandbox_captures_runtime_error(tmp_path: Path):
    sandbox = LocalCodeSandbox()
    code = "1/0"
    result = await sandbox.execute(
        code=code,
        timeout_seconds=5,
        allowed_imports=[],
        working_directory=str(tmp_path),
        max_memory_mb=128,
    )
    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert "ZeroDivisionError" in (result.stderr or result.error_traceback or "")
