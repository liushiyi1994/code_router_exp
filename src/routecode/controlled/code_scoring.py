from __future__ import annotations

import ast
import contextlib
import faulthandler
import io
import json
import multiprocessing
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


DEFAULT_IMPORTS = [
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import functools",
    "import hashlib",
    "import string",
    "from typing import *",
    "from collections import *",
]


@dataclass(frozen=True)
class CodeScore:
    parsed_answer: str
    quality: float


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args: Any, **kwargs: Any) -> str:
        raise OSError

    def readline(self, *args: Any, **kwargs: Any) -> str:
        raise OSError

    def readlines(self, *args: Any, **kwargs: Any) -> list[str]:
        raise OSError

    def readable(self) -> bool:
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore[type-arg]
    _stream = "stdin"


def score_python_code_output(raw_text: str, gold_json: str, *, timeout_s: float = 6.0) -> CodeScore:
    """Score function-style code tasks using embedded HumanEval/MBPP checks.

    This is intentionally scoped to the controlled broad Stage 0 records where
    the prompt already carries the assert/check code. It is not a full
    LiveCodeBench scorer.
    """

    try:
        spec = json.loads(gold_json)
    except json.JSONDecodeError:
        return CodeScore("invalid_code_gold", 0.0)
    benchmark = str(spec.get("benchmark", "")).lower()
    tests = str(spec.get("tests", "")).strip()
    if benchmark not in {"humaneval", "mbpp"} or not tests:
        return CodeScore("unsupported_code_task", 0.0)
    entry_point = str(spec.get("entry_point", "")).strip()
    code = extract_python_code(raw_text, entry_point=entry_point or None)
    if not code.strip():
        return CodeScore("no_code", 0.0)
    program = build_check_program(benchmark, code, tests, entry_point)
    result = run_python_check(program, timeout_s=timeout_s)
    return CodeScore(result, 1.0 if result == "passed" else 0.0)


def build_check_program(benchmark: str, code: str, tests: str, entry_point: str = "") -> str:
    imports = "\n".join(DEFAULT_IMPORTS)
    if benchmark == "humaneval":
        if not entry_point:
            return f"{imports}\n{code}\nraise AssertionError('missing entry point')"
        return f"{imports}\n{code}\n{tests}\ncheck({entry_point})"
    return f"{imports}\n{code}\n{tests}"


def extract_python_code(text: str, *, entry_point: str | None = None) -> str:
    cleaned = strip_thinking(str(text or ""))
    fenced = extract_fenced_python(cleaned)
    candidate = fenced if fenced.strip() else cleaned
    valid = extract_longest_valid_code(candidate)
    if valid.strip():
        return sanitize_valid_code(valid, entry_point=entry_point)
    return ""


def strip_thinking(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def extract_fenced_python(text: str) -> str:
    matches = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def extract_longest_valid_code(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").splitlines()
    if len(lines) > 140:
        lines = lines[:140]
    best = ""
    best_count = 0
    for start in range(len(lines)):
        for end in range(start, len(lines)):
            snippet = "\n".join(lines[start : end + 1])
            if not snippet.strip():
                continue
            try:
                ast.parse(snippet)
            except (SyntaxError, MemoryError):
                continue
            count = sum(1 for line in snippet.splitlines() if line.strip())
            if count > best_count:
                best = snippet
                best_count = count
    return best


def sanitize_valid_code(code: str, *, entry_point: str | None = None) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not entry_point:
        return code
    definitions: dict[str, ast.AST] = {}
    imports: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
            definitions[node.targets[0].id] = node
    if entry_point not in definitions:
        return code
    reachable = reachable_definitions(entry_point, definitions)
    kept = imports + [node for name, node in definitions.items() if name in reachable]
    return "\n".join(ast.unparse(node) for node in kept)


def reachable_definitions(entry_point: str, definitions: dict[str, ast.AST]) -> set[str]:
    deps = {name: definition_dependencies(node) for name, node in definitions.items()}
    seen: set[str] = set()
    pending = [entry_point]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(sorted((deps.get(name, set()) & set(definitions)) - seen))
    return seen


def definition_dependencies(node: ast.AST) -> set[str]:
    deps: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            deps.add(child.id)
        elif isinstance(child, ast.Attribute):
            deps.add(child.attr)
    return deps


def run_python_check(program: str, *, timeout_s: float) -> str:
    queue: multiprocessing.Queue[str] = multiprocessing.Queue()
    process = multiprocessing.Process(target=_execute_program, args=(program, float(timeout_s), queue))
    process.start()
    process.join(timeout=float(timeout_s) + 1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)
        return "timed out"
    if queue.empty():
        return "timed out"
    return queue.get()


def _execute_program(program: str, timeout_s: float, queue: multiprocessing.Queue[str]) -> None:
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as dirname:
        saved = {
            "os_chdir": os.chdir,
            "os_getcwd": os.getcwd,
            "os_remove": os.remove,
            "os_rmdir": os.rmdir,
            "shutil_rmtree": shutil.rmtree,
        }
        try:
            os.chdir(dirname)
            reliability_guard()
            with swallow_io(), time_limit(timeout_s):
                exec(program, {})
            queue.put("passed")
        except TimeoutException:
            queue.put("timed out")
        except BaseException as exc:
            queue.put(f"failed: {type(exc).__name__}")
        finally:
            os.chdir = saved["os_chdir"]  # type: ignore[method-assign]
            os.getcwd = saved["os_getcwd"]  # type: ignore[method-assign]
            os.remove = saved["os_remove"]  # type: ignore[method-assign]
            os.rmdir = saved["os_rmdir"]  # type: ignore[method-assign]
            shutil.rmtree = saved["shutil_rmtree"]  # type: ignore[method-assign]
            os.chdir(cwd)


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum: int, frame: Any) -> None:
        raise TimeoutException("timed out")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), redirect_stdin(stream):
        yield


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    import builtins
    import subprocess

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None  # type: ignore[assignment]
    os.system = None  # type: ignore[assignment]
    os.putenv = None  # type: ignore[assignment]
    os.remove = None  # type: ignore[assignment]
    os.removedirs = None  # type: ignore[assignment]
    os.rmdir = None  # type: ignore[assignment]
    os.fchdir = None  # type: ignore[assignment]
    os.setuid = None  # type: ignore[assignment]
    os.fork = None  # type: ignore[assignment]
    os.forkpty = None  # type: ignore[assignment]
    os.killpg = None  # type: ignore[assignment]
    os.rename = None  # type: ignore[assignment]
    os.renames = None  # type: ignore[assignment]
    os.truncate = None  # type: ignore[assignment]
    os.replace = None  # type: ignore[assignment]
    os.unlink = None  # type: ignore[assignment]
    os.fchmod = None  # type: ignore[assignment]
    os.fchown = None  # type: ignore[assignment]
    os.chmod = None  # type: ignore[assignment]
    os.chown = None  # type: ignore[assignment]
    os.chroot = None  # type: ignore[assignment]
    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]
    subprocess.Popen = None  # type: ignore[assignment]
    builtins.help = None
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
