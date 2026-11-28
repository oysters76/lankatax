from __future__ import annotations

import pytest

from lankatax_ai.backend.documents.page_marker import format_page_marker
from lankatax_ai.backend.ingestion.chunker.common import (
    extract_pages_and_clean,
    merge_blocks_to_chunks,
)
from lankatax_ai.backend.ingestion.types import BlockType, ChunkConfig, RawBlock

pytestmark = pytest.mark.unit


def test_chunk_starting_with_overlap_keeps_the_previous_page():
    blocks = [
        RawBlock(
            block_type=BlockType.paragraph,
            content=format_page_marker(1) + " ".join(["alpha"] * 60),
        ),
        RawBlock(
            block_type=BlockType.paragraph,
            content=format_page_marker(2) + " ".join(["beta"] * 60),
        ),
    ]

    chunks = merge_blocks_to_chunks([blocks], ChunkConfig(max_tokens=80, overlap_tokens=10))

    assert len(chunks) == 2
    page_start, page_end, _ = extract_pages_and_clean(chunks[1])
    assert (page_start, page_end) == (1, 2)
