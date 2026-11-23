"""Transport-agnostic use cases shared by the CLI and the API.

Each use case lives in its own package (``services/search``, ``services/cache``,
...). Shared pieces stay at this level: ``errors``, ``protocols`` and ``utils``.

Layering rule: this package may import ``rag``, ``db``, ``llm`` and
``ingestion``, but never ``typer``, ``rich``, ``fastapi`` or any other transport
library. Rendering and HTTP concerns stay with the caller.
"""
