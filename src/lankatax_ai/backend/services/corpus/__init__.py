"""Corpus versioning service."""

from lankatax_ai.backend.services.corpus.service import (
    advance_corpus_epoch,
    advance_corpus_epoch_after_ingest,
)

__all__ = ["advance_corpus_epoch", "advance_corpus_epoch_after_ingest"]
