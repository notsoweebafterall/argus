"""Unit tests for Argus System Tools."""

from __future__ import annotations

import pytest
from tools.system_tools import read_file, run_python, write_file


def test_run_python_success():
    """Verify executing Python code and capturing result & stdout."""
    code = """
print("Hello from sandbox")
x = 10
y = 25
result = {"sum": x + y}
"""
    res = run_python(code=code)
    assert "Hello from sandbox" in res["stdout"]
    assert res["result"] == {"sum": 35}


def test_run_python_context_vars():
    """Verify context variables are injected into sandbox scope."""
    code = """
result = [x * factor for x in items]
"""
    res = run_python(code=code, context_vars={"items": [1, 2, 3], "factor": 4})
    assert res["result"] == [4, 8, 12]


def test_run_python_execution_error():
    """Verify execution errors raise RuntimeError with execution_error."""
    code = """
a = 1 / 0
"""
    with pytest.raises(RuntimeError) as exc_info:
        run_python(code=code)
    assert "execution_error" in str(exc_info.value)
    assert "ZeroDivisionError" in str(exc_info.value)


def test_run_python_timeout():
    """Verify timeout triggers TimeoutError."""
    code = """
import time
while True:
    time.sleep(0.1)
"""
    with pytest.raises(TimeoutError) as exc_info:
        run_python(code=code, timeout=0.5)
    assert "timeout" in str(exc_info.value)


def test_run_python_socket_import_blocked():
    """Verify socket and network module imports are blocked in sandbox."""
    code = """
import socket
s = socket.socket()
"""
    with pytest.raises(ImportError) as exc_info:
        run_python(code=code)
    assert "Network access disabled" in str(exc_info.value)


def test_write_and_read_file_direct():
    """Verify direct scoped write_file and read_file execution."""
    task_id = "test_task_direct"
    rel_path = "subfolder/manifest.json"
    content = '{"status": "ready", "count": 100}'

    write_res = write_file(task_id=task_id, relative_path=rel_path, content=content)
    assert write_res["status"] == "written"
    assert write_res["bytes_written"] == len(content.encode("utf-8"))

    read_res = read_file(task_id=task_id, relative_path=rel_path)
    assert read_res["content"] == content
    assert read_res["size_bytes"] == len(content.encode("utf-8"))


def test_scoped_file_path_traversal_direct():
    """Verify path traversal is rejected for direct write and read calls."""
    task_id = "test_task_traversal_direct"

    with pytest.raises(ValueError) as exc_info:
        write_file(task_id=task_id, relative_path="../forbidden.txt", content="hack")
    assert "Path traversal detected" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        read_file(task_id=task_id, relative_path="../../forbidden.txt")
    assert "Path traversal detected" in str(exc_info.value)
