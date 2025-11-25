##############################################################
###################  STEP THREE ##############################
######### Store the chunked text into the pgvectore ##########
##############################################################
import os
import json
from time import sleep
import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import register_vector
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = "postgresql://rag_user:rag_password@localhost:5434/rag_hybrid_db"

# OpenAI api requesting 
MAX_RETRIES = 5
BASE_SLEEP_TIME = 15

client = OpenAI()

def get_embedding(text: str):
    """Generate embedding using openai text-embedding-3-small"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            break
        except (RateLimitError, APIError, TimeoutError) as e:
            retries += 1
            sleep_time = BASE_SLEEP_TIME * (2 ** (retries - 1))
            print(f"Error: {e}")
            sleep(sleep_time)
    return response.data[0].embedding


def setup_database():
    """Creates the table and necessary indexes for Hybrid Search."""
    with psycopg.connect(DB_CONFIG, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        
        cursor = conn.cursor()
        
        # Create the table
        # We use JSONB for metadata to be flexible (metadata of the chunks)
        print("Creating table...")
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
        print("Creating Vector Index..")
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
        print("Creating Text Index..")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS chunk_content_idx
            ON document_chunks
            USING GIN (to_tsvector('english', content));
            """
        )
    print("Database setup complete.")
    
def store_chunk(content, metadata):
    """Generates an embedding and stores the chunk in Postgres."""
    # Generate the embedding for the text    
    vector = get_embedding(content)
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


if __name__ == "__main__":
    # 1. Initialize the DB
    setup_database()
    
    with open("./data/processed/enriched_chunks.json", "r", encoding="utf-8") as f:
        data_chunks = json.load(f)
        
    for idx, chunk in tqdm(enumerate(data_chunks), total=len(data_chunks)):
        text_chunk = chunk['chunk']
        context = chunk['chunk_metada']['context_expansion']
        text = text_chunk.replace('\n', '') + '\n' + context
        metadata = {}
        metadata['section_header'] = chunk['chunk_metada']['section_header']
        metadata['chunk_id'] = idx
        
        store_chunk(
            content=text,
            metadata=metadata
        )
        
        