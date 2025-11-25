##############################################################
###################  STEP FIVE ###############################
############## Evaluating The RAG System #####################
##############################################################

import os
import json
import yaml
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI()

# Configuration
DB_CONFIG = "postgresql://rag_user:rag_password@localhost:5434/rag_hybrid_db"


def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def hybrid_search(query_text, top_k=10, rrf_k=60):
    """
    Performs Hybrid Search (Vector + Keyword) and combines results using RRF.
    """
    query_vector = get_embedding(query_text)
    
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


class AnswerEvaluation(BaseModel):
    correctness: int = Field(description="Provide your evaluation as a single integer score (0-5)")


# --- Example Usage ---
if __name__ == "__main__":
    # Load the qa_pairs dataset
    qa_path = "./data/processed/qa_pairs.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    with open("./prompts/generating_answer.yml") as f:
        prompts = yaml.safe_load(f)
        
    system_prompt = prompts["generating_answer"]["system"]
    
    eval_scores = {
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
        "haullucinated": 0
    }
    
    for pair in tqdm(qa_pairs[:100], total=len(qa_pairs[:100])):
        question = pair["question"]
        truth_answer = pair["answer"]
        results = hybrid_search(question)

        retrieved_texts = ""
        for i, res in enumerate(results):
            content = res['data'][1]
            retrieved_texts += content + "\n\n"
            meta = res['data'][2]
            # score = res['score']
            # print(f"{i+1}. Score: {score:.4f} | Text: {content[:100]}...")
        
        # print(f"Question: {question}")
        # print(f"The truth answer: {truth_answer}")
        # print(10*"+")
        # print(f"Retrieved texts: {retrieved_texts}")
        
        user_prompt = prompts["generating_answer"]["user"].format(
            question=question,
            retrieved_chunks=retrieved_texts
        )
        
        # print(f"System prompt: {system_prompt}")
        # print(10*'-')
        # print(f"User prompt: {user_prompt}")
        
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        # print(response.output_text)
        answer = response.output_text
        
        eval_system_prompt = prompts["validating_generated_answer"]["system"]
        
        eval_user_prompt = prompts["validating_generated_answer"]["user"].format(
            question=question,
            retrieved_chunks=retrieved_texts,
            model_answer=answer,
            gold_answer=truth_answer
        )
        
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system", 
                    "content": eval_system_prompt
                },
                {
                    "role": "user",
                    "content": eval_user_prompt
                },
            ],
            text_format=AnswerEvaluation,
        )

        event = response.output_parsed
        
        # print(event.correctness)
        
        if event.correctness == 5:
            eval_scores["correct"] += 1
        elif event.correctness == 4 or event.correctness == 3:
            eval_scores["partially_correct"] += 1
        elif event.correctness == 2 or event.correctness == 1:
            eval_scores["incorrect"] += 1
        else:
            eval_scores["haullucinated"] += 1 
        
        # break
    
    print(f"Number of correct answers: {eval_scores["correct"]}")
    print(f"Number of partially_correct answers: {eval_scores["partially_correct"]}")
    print(f"Number of incorrect answers: {eval_scores["incorrect"]}")
    print(f"Number of haullucinated answers: {eval_scores["haullucinated"]}")