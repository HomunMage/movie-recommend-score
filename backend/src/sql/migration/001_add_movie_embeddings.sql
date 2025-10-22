-- src/sql/migration/001_add_movie_embeddings.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create movie_embeddings table
CREATE TABLE IF NOT EXISTS movie_embeddings (
    movie_id INTEGER PRIMARY KEY,
    embedding vector(32),  -- Match your EMBEDDING_DIM
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS movie_embeddings_vector_idx 
ON movie_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Optional: Add metadata table for movies if needed
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(255),
    genres VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
CREATE TRIGGER update_movie_embeddings_modtime
    BEFORE UPDATE ON movie_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();