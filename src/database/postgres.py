import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import register_vector


DB_CONFIG = "postgresql://rag_user:rag_password@localhost:5434/rag_hybrid_db"

def setup_database():
    """Creates the table and necessary indexes for Hybrid Search."""
    with psycopg.connect(DB_CONFIG, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        
        cursor = conn.cursor()
        
        # Create the chunks table
        # We use JSONB for metadata to be flexible (metadata of the chunks)
        print("Creating table chunks...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(1536)
            );
            """
        )
        
        # 1. Create HNSW Index for vector search (Semantic)
        # 'vector_cosine_ops' is standard for text embeddings
        print("Creating Vector Chunks Index..")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS chunk_embedding_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64);
            """
        )
        
        # 2. Create GIN Index for keyword search (Lexical)
        # This allows fast full-text search using 'content' column
        print("Creating Text Chunks Index..")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS chunk_content_idx
            ON document_chunks
            USING GIN (to_tsvector('english', content));
            """
        )
        
        # 3. Create the semantic caching table 
        # Semantic caching table used to speed the answer process if the user asked previous answered and stored question
        print("Creating table semantic caching...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache(
                id BIGSERIAL PRIMARY KEY,
                question_text TEXT,
                question_embedding vector(1536),
                answer_text TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
        
        # 4. Create the HSNW Index for vector search for semantic caching
        print("Creating Vector Chunks Index..")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS cache_idx 
            ON semantic_cache 
            USING hnsw (question_embedding vector_cosine_ops);
            """
        )
        
        # 5. Create table for logs
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_log(
                id BIGSERIAL PRIMARY KEY,
                user_query TEXT,
                bot_response TEXT,
                retrieved_doc_ids JSONB,
                was_cache_hit BOOLEAN,
                execution_time INTEGER,
                timestamp TIMESTAMP DEFAULT NOW()
            );
            """
        )
    print("Database setup complete.")

def store_chunk(content, metadata, embedding_fn):
    """Generates an embedding and stores the chunk in Postgres."""
    # Generate the embedding for the text    
    vector = embedding_fn(content)
    # Insert into the DB
    with psycopg.connect(DB_CONFIG) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO document_chunks (content, metadata, embedding)
            VALUES (%s, %s, %s);
            """, (content, Json(metadata), vector)
        )
        conn.commit()
        
def hybrid_search(query_text, embedding_fn, top_k=10, rrf_k=60):
    """
    Performs Hybrid Search (Vector + Keyword) and combines results using RRF.
    """
    query_vector = embedding_fn(query_text)
    
    # We fetch more candidates than we need (e.g., 50) to give RRF enough data to fuse.
    limit_candidates = 50 

    with psycopg.connect(DB_CONFIG) as conn:
        register_vector(conn)
        cursor = conn.cursor()

        # ---------------------------------------------------------
        # 1. Vector Search (Semantic)
        # ---------------------------------------------------------
        # Uses Cosine Distance (<=>)
        # print(f"Running Vector Search for: '{query_text}'")
        cursor.execute("""
            SELECT id, content, metadata
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vector, limit_candidates))
        vector_results = cursor.fetchall() # List of (id, content, metadata)

        # ---------------------------------------------------------
        # 2. Keyword Search (Lexical)
        # ---------------------------------------------------------
        # We use `plainto_tsquery` which handles user input safely (no syntax errors).
        # print(f"Running Keyword Search for: '{query_text}'")
        cursor.execute("""
            SELECT id, content, metadata
            FROM document_chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC
            LIMIT %s
        """, (query_text, query_text, limit_candidates))
        keyword_results = cursor.fetchall()

        # ---------------------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF) Algorithm
        # ---------------------------------------------------------
        # Structure: { doc_id: {"score": 0.0, "data": row_data} }
        fused_scores = {}

        # Helper to update scores
        def add_to_rrf(results, rank_weight=1.0):
            for rank, row in enumerate(results):
                doc_id = row[0]
                # RRF Formula: 1 / (k + rank)
                # Rank is 0-indexed here, so we use (rank + 1)
                base_score = 1.0 / (rrf_k + rank + 1)
                
                # APPLIED HERE: Multiply by the weight
                weighted_score = base_score * rank_weight
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"score": 0.0, "data": row}
                
                fused_scores[doc_id]["score"] += weighted_score 

        # Apply RRF to both lists
        add_to_rrf(vector_results)
        add_to_rrf(keyword_results)

        # Sort by final RRF score descending
        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )

        return sorted_results[:top_k]

