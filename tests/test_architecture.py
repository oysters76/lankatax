"""Import-boundary tests for the layered package.

Rules:
- ``backend`` stays transport-free: no Typer or Rich.
- FastAPI is allowed only in ``backend/api``, the HTTP adapter.

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


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


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
