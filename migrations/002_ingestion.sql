CREATE TYPE chunk_strategy_enum as ENUM('structural', 'semantic');
CREATE TYPE document_kind_enum as ENUM('pdf', 'page');

ALTER TABLE documents ADD COLUMN raw_html TEXT; 
ALTER TABLE documents ADD COLUMN document_kind document_kind_enum NOT NULL DEFAULT 'page'; 
ALTER TABLE documents ADD COLUMN ingested_hash TEXT;

ALTER TABLE chunks ADD COLUMN token_count INTEGER;
ALTER TABLE chunks ADD COLUMN chunk_strategy chunk_strategy_enum;

-- Trigram index: enables fast keyword/fuzzy matching for hybrid search
-- (vector similarity finds meaning, trigrams find exact terms like "SSCL")
-- GIN = Generalized Inverted Index, useful for full-text search, trigram matching
-- Slow to write, but since chunks are written once during nightly ingestion, this is fine.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops); 
