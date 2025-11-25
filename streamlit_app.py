import yaml
import psycopg
from pgvector.psycopg import register_vector
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = OpenAI()

# Configuration
DB_CONFIG = "postgresql://rag_user:rag_password@localhost:5434/rag_hybrid_db"

# Load prompts safely
try:
    with open("./prompts/generating_answer.yml") as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Prompt file not found. Please ensure './prompts/generating_answer.yml' exists.")
    st.stop()

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def hybrid_search(query_text, top_k=10, rrf_k=60):
    """
    Performs Hybrid Search (Vector + Keyword) and combines results using RRF.
    """
    query_vector = get_embedding(query_text)
    
    limit_candidates = 50 

    with psycopg.connect(DB_CONFIG) as conn:
        register_vector(conn)
        cursor = conn.cursor()

        # 1. Vector Search
        cursor.execute("""
            SELECT id, content, metadata
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vector, limit_candidates))
        vector_results = cursor.fetchall()

        # 2. Keyword Search
        cursor.execute("""
            SELECT id, content, metadata
            FROM document_chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC
            LIMIT %s
        """, (query_text, query_text, limit_candidates))
        keyword_results = cursor.fetchall()

        # 3. RRF
        fused_scores = {}

        def add_to_rrf(results, rank_weight=1.0):
            for rank, row in enumerate(results):
                doc_id = row[0]
                base_score = 1.0 / (rrf_k + rank + 1)
                weighted_score = base_score * rank_weight
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"score": 0.0, "data": row}
                
                fused_scores[doc_id]["score"] += weighted_score 

        add_to_rrf(vector_results)
        add_to_rrf(keyword_results)

        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )

        return sorted_results[:top_k]

st.set_page_config(page_title="Q&A App", layout="wide")

# --- Session State ---
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

st.title("Q&A Assistant For the Book Agentic Design Patterns")

# --- Display previous Q&A at top ---
if st.session_state.last_question:
    st.subheader("Question:")
    st.info(st.session_state.last_question)

    st.subheader("Bot Answer:")
    st.success(st.session_state.last_answer)

# Spacer to push input to bottom
st.markdown("<div style='height: 45vh;'></div>", unsafe_allow_html=True)

# Load the answer generator system prompt
system_prompt = prompts["generating_answer"]["system"]

# --- Input at bottom ---
with st.container():
    st.markdown("### Ask your question")
    question = st.text_area(" ", placeholder="Type your question here...", key="question_box")
    
    if st.button("Submit"):
        if question.strip():
            with st.spinner("Searching knowledge base..."):
                # MOVED INSIDE: Only run search if question exists
                results = hybrid_search(question)
                
                retrieved_texts = ""
                for i, res in enumerate(results):
                    content = res['data'][1]
                    retrieved_texts += content + "\n\n"
                
                # Load the answer generator user prompt
                user_prompt = prompts["generating_answer"]["user"].format(
                    question=question,
                    retrieved_chunks=retrieved_texts
                )
            
            with st.spinner("Generating answer..."):
                # Using standard ChatCompletions (Adjust if using a custom wrapper)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                answer = response.choices[0].message.content

                # Store in session state
                st.session_state.last_question = question
                st.session_state.last_answer = answer

                # Force reload to show Q&A at top
                st.rerun()
        else:
            st.warning("Please enter a question.")