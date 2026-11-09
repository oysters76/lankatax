CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      url TEXT NOT NULL UNIQUE,
      title TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      category TEXT NOT NULL,
      raw_text TEXT NOT NULL,
      scraped_at TIMESTAMPTZ DEFAULT now(),
      updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      content TEXT NOT NULL,
      embedding vector(1536),
      chunk_index INTEGER NOT NULL,
      metadata_json JSONB DEFAULT '{}'
);

CREATE TABLE chat_sessions (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      fingerprint TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now(),
      message_count INTEGER DEFAULT 0
);

CREATE TABLE chat_messages (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
      role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
      content TEXT NOT NULL,
      sources_json JSONB DEFAULT '[]',
      tokens_used INTEGER DEFAULT 0,
      cost NUMERIC(10, 6) DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE scrape_runs (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      started_at TIMESTAMPTZ DEFAULT now(),
      finished_at TIMESTAMPTZ,
      pages_scraped INTEGER DEFAULT 0,
      pages_updated INTEGER DEFAULT 0,
      pages_failed INTEGER DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed'))
);

CREATE TABLE eval_results (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      question TEXT NOT NULL,
      expected_answer TEXT NOT NULL,
      actual_answer TEXT NOT NULL,
      faithfulness NUMERIC(3, 2) CHECK (faithfulness BETWEEN 0 AND 1),
      relevance NUMERIC(3, 2) CHECK (relevance BETWEEN 0 AND 1),
      created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE cache_entries (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      query_embedding vector(1536),
      query_text TEXT NOT NULL,
      response TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now(),
      hit_count INTEGER DEFAULT 0
);

CREATE INDEX idx_chunks_embedding ON chunks
       using hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_cache_embedding ON cache_entries
       using hnsw (query_embedding vector_cosine_ops); 


