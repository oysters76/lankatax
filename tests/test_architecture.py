"""Import-boundary tests for the layered package.

Rules:
- ``backend`` stays transport-free: no Typer or Rich.
- FastAPI is allowed only in ``backend/api``, the HTTP adapter.
- Hatchet is confined to ``backend/jobs``, the execution adapter.
- Services must not import the Hatchet adapter.

The CLI may import Typer and Rich; it is not scanned here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(__file__).parent.parent / "src" / "lankatax_ai"
BACKEND = PACKAGE_ROOT / "backend"
API = BACKEND / "api"


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def _imported_roots(path: Path) -> set[str]:
    return {module.split(".")[0] for module in _imported_modules(path)}


def _offenders(root: Path, banned: set[str], *, allowed: Path | None = None) -> list[str]:
    found: list[str] = []
    for path in _python_files(root):
        if allowed is not None and path.is_relative_to(allowed):
            continue
        hits = _imported_roots(path) & banned
        if hits:
            relative = path.relative_to(PACKAGE_ROOT)
            found.append(f"{relative} imports {', '.join(sorted(hits))}")
    return found


def test_backend_does_not_import_cli_frameworks() -> None:
    offenders = _offenders(BACKEND, {"typer", "rich"})
    assert not offenders, "backend must stay transport-free:\n" + "\n".join(offenders)


def test_fastapi_is_confined_to_the_api_package() -> None:
    offenders = _offenders(BACKEND, {"fastapi"}, allowed=API)
    assert not offenders, "FastAPI belongs in backend/api only:\n" + "\n".join(offenders)


def _module_offenders(root: Path, module: str, *, allowed: Path | None = None) -> list[str]:
    found: list[str] = []
    for path in _python_files(root):
        if allowed is not None and path.is_relative_to(allowed):
            continue
        hits = {
            imported
            for imported in _imported_modules(path)
            if imported == module or imported.startswith(f"{module}.")
        }
        if hits:
            relative = path.relative_to(PACKAGE_ROOT)
            found.append(f"{relative} imports {', '.join(sorted(hits))}")
    return found


def test_hatchet_is_confined_to_the_jobs_adapter() -> None:
    offenders = _module_offenders(
        BACKEND,
        "hatchet_sdk",
        allowed=BACKEND / "jobs",
    )
    assert not offenders, "Hatchet belongs in backend/jobs only:\n" + "\n".join(offenders)


def test_services_do_not_import_the_hatchet_adapter() -> None:
    offenders = _module_offenders(
        BACKEND / "services",
        "lankatax_ai.backend.jobs",
    )
    assert not offenders, "Services must not depend on backend/jobs:\n" + "\n".join(offenders)


def test_langgraph_is_confined_to_the_chat_package() -> None:
    offenders = _offenders(BACKEND, {"langgraph"}, allowed=BACKEND / "chat")
    assert not offenders, "LangGraph belongs in backend/chat only:\n" + "\n".join(offenders)
