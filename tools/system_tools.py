"""System tools for Argus ML."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agents.contracts import ToolSpec
from pipeline.config import settings

# Scratch directory for sandboxed subprocess execution
SCRATCH_DIR = settings.workspace_root / "experiments" / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Runs directory for scoped file read/write operations
RUNS_DIR = settings.workspace_root / "experiments" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

RUNNER_TEMPLATE = r"""
import sys
import json
import os

# Best-effort import-level sandbox: disable socket and network library imports
class BlockedImportFinder:
    BLOCKED_MODULES = {"socket", "http", "urllib", "requests", "aiohttp", "httpx"}

    def find_spec(self, fullname, path, target=None):
        root_mod = fullname.split(".")[0]
        if root_mod in self.BLOCKED_MODULES:
            raise ImportError(f"Network access disabled in sandbox: cannot import '{fullname}'")
        return None

sys.meta_path.insert(0, BlockedImportFinder())
for mod in list(sys.modules.keys()):
    if mod.split(".")[0] in BlockedImportFinder.BLOCKED_MODULES:
        del sys.modules[mod]

# Context variables loaded from input payload
context_vars_path = sys.argv[1]
output_result_path = sys.argv[2]
code_to_run_path = sys.argv[3]
scratch_dir = sys.argv[4]

# Switch working directory to ephemeral scratch dir under experiments/scratch
os.chdir(scratch_dir)

with open(context_vars_path, "r", encoding="utf-8") as f:
    context = json.load(f)

with open(code_to_run_path, "r", encoding="utf-8") as f:
    code = f.read()

# Execution scope initialized with context variables
local_env = dict(context)
local_env["__result__"] = None

# Execute code
try:
    exec(code, local_env, local_env)
    res = local_env.get("result", local_env.get("__result__", None))
    # Test JSON serializability
    try:
        json.dumps(res)
        serializable_res = res
    except (TypeError, OverflowError):
        serializable_res = str(res)

    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump({"result": serializable_res}, f)
except Exception as e:
    import traceback
    sys.stderr.write(traceback.format_exc())
    sys.exit(1)
"""


def _check_code_imports(code: str) -> None:
    """Pre-execution AST check to catch socket / network imports early."""
    blocked = {"socket", "http", "urllib", "requests", "aiohttp", "httpx"}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in blocked:
                        raise ImportError(f"Network access disabled in sandbox: forbidden import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in blocked:
                        raise ImportError(f"Network access disabled in sandbox: forbidden import '{node.module}'")
    except SyntaxError:
        pass


def run_python(code: str, context_vars: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    """Execute Python code snippet in an isolated subprocess.

    Provides best-effort sandboxing for this batch:
    - Runs in a standalone subprocess with working directory pinned to an ephemeral scratch folder under experiments/scratch/.
    - Enforces a hard execution timeout (default: 5.0s).
    - Enforces AST check and sys.meta_path hook to reject socket and network library imports.
    - Captures stdout and stderr streams separately from return values.
    - Does not claim hardened kernel-level container isolation.

    Returns {"result": Any, "stdout": str}.
    """
    _check_code_imports(code)

    context = context_vars or {}
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as temp_run_dir:
        temp_path = Path(temp_run_dir)
        context_file = temp_path / "context.json"
        code_file = temp_path / "code.py"
        output_file = temp_path / "output.json"
        runner_file = temp_path / "runner.py"

        with open(context_file, "w", encoding="utf-8") as f:
            json.dump(context, f)

        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        with open(runner_file, "w", encoding="utf-8") as f:
            f.write(RUNNER_TEMPLATE)

        python_exe = sys.executable

        try:
            process = subprocess.run(
                [
                    python_exe,
                    str(runner_file),
                    str(context_file),
                    str(output_file),
                    str(code_file),
                    str(temp_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"timeout: Code execution exceeded {timeout}s limit") from exc

        if process.returncode != 0:
            err = process.stderr.strip() or process.stdout.strip() or "Unknown execution error"
            raise RuntimeError(f"execution_error: {err}")

        result: Any = None
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                result = data.get("result")

        return {
            "result": result,
            "stdout": process.stdout,
        }


def _get_scoped_path(task_id: str, relative_path: str) -> Path:
    """Validate and resolve target file path strictly inside experiments/runs/{task_id}/."""
    if not task_id or not task_id.strip():
        raise ValueError("task_id must not be empty")

    base_dir = (RUNS_DIR / task_id.strip()).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    target_path = (base_dir / relative_path).resolve()

    if not target_path.is_relative_to(base_dir):
        raise ValueError(
            f"Path traversal detected: '{relative_path}' resolves outside allowed run directory '{base_dir}'"
        )

    return target_path


# Maximum file write size (50MB) to guard against runaway disk usage
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024


def write_file(task_id: str, relative_path: str, content: str) -> dict[str, Any]:
    """Write text content to a file strictly scoped within experiments/runs/{task_id}/.

    Guarded by:
      - Path traversal rejection via _get_scoped_path
      - Max file size limit (50MB) to protect against runaway/adversarial disk exhaustion
    """
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"file_size_exceeded: Content size ({len(encoded)} bytes) exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES} bytes (50MB)."
        )

    target_path = _get_scoped_path(task_id, relative_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(encoded)

    return {
        "file_path": str(target_path),
        "bytes_written": len(encoded),
        "status": "written",
    }


def read_file(task_id: str, relative_path: str) -> dict[str, Any]:
    """Read text content from a file strictly scoped within experiments/runs/{task_id}/."""
    target_path = _get_scoped_path(task_id, relative_path)
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(f"File not found: '{relative_path}' in run directory for task '{task_id}'")

    content = target_path.read_text(encoding="utf-8")
    return {
        "file_path": str(target_path),
        "content": content,
        "size_bytes": len(content.encode("utf-8")),
    }


RUN_PYTHON_SPEC = ToolSpec(
    name="run_python",
    description="Executes a Python snippet in a subprocess with working directory pinned to scratch dir, hard timeout, and network imports blocked.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code snippet to execute. Assign output to 'result' to return values.",
            },
            "context_vars": {
                "type": "object",
                "description": "Dictionary of variables to inject into the execution scope.",
            },
        },
        "required": ["code"],
    },
    handler=run_python,
)

WRITE_FILE_SPEC = ToolSpec(
    name="write_file",
    description="Writes text content to a file scoped strictly to the designated run directory (experiments/runs/{task_id}/).",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Unique task identifier owning the run directory.",
            },
            "relative_path": {
                "type": "string",
                "description": "Relative file path within the run directory.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["task_id", "relative_path", "content"],
    },
    handler=write_file,
)

READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description="Reads text content from a file scoped strictly to the designated run directory (experiments/runs/{task_id}/).",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Unique task identifier owning the run directory.",
            },
            "relative_path": {
                "type": "string",
                "description": "Relative file path within the run directory.",
            },
        },
        "required": ["task_id", "relative_path"],
    },
    handler=read_file,
)
