##############################################################
###################  STEP FOUR ###############################
######### Creating Evaluation Q&A Dataset ####################
##############################################################

import yaml
import json
import os
import random
from time import sleep
from pydantic import BaseModel
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ---------- Models ----------
class QuestionAnswer(BaseModel):
    question: str
    answer: str
    
class QAList(BaseModel):
    q_a_pairs: list[QuestionAnswer]


# ---------- Load Data ----------
with open("./data/processed/Agentic_Design_Patterns.mmd", "r") as f:
    full_text = f.read()

with open("./prompts/rag_validation_openai.yml") as f:
    prompts = yaml.safe_load(f)
    
# ---------- Save output ----------
os.makedirs("./data/processed", exist_ok=True)
output_path = "./data/processed/qa_pairs.json"

def split_into_four_parts(text: str):
    length = len(text)
    part_size = length // 4
    parts = [text[i * part_size : (i + 1) * part_size] for i in range(3)]
    parts.append(text[3 * part_size:])
    return parts

parts = split_into_four_parts(full_text)

roles = [
    "computer science student", 
    "Junior AI/ML Engineer", 
    "Senior AI/ML Engineer or Architect",
    "Researcher in AI / Autonomous Systems",
    "Product Manager (AI-focused)",
    "Software Engineer Exploring AI Adoption"
]

models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]

qa_pairs = []

# ---------- Retry Helper ----------
MAX_RETRIES = 6
BASE_SLEEP = 3  # base seconds

def call_with_retry(fn, *args, **kwargs):
    """
    Handles:
    - RateLimitError
    - APIError
    - APITimeoutError
    With exponential backoff and jitter.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)

        except (RateLimitError, APIError, APITimeoutError) as e:
            wait = BASE_SLEEP * attempt + random.uniform(0.5, 1.5)
            print(f"[Retry {attempt}/{MAX_RETRIES}] Error: {e}. Sleeping {wait:.1f}s...")
            sleep(wait)

        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    print("❌ Max retries exceeded — skipping this request.")
    return None


# ---------- Main Loop ----------
for idx, part in enumerate(parts, start=1):

    system_prompt = prompts["rag_validation_prompt"]["system_template"].format(
        text_book=part
    )

    for role in roles:

        user_prompt = prompts["rag_validation_prompt"]["user_template"].format(
            num_questions=10,
            role=role,
        )

        for model in models:

            print(f"\n→ Calling model: {model} | Part {idx} | Role: {role}")

            response = call_with_retry(
                client.responses.parse,
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text_format=QAList
            )

            if response is None:
                continue  # skip this chunk/model if completely failed

            parsed = response.output_parsed
            for q_a in parsed.q_a_pairs:
                qa_pairs.append({
                    "question": q_a.question,
                    "answer": q_a.answer,
                    "role": role,
                    "model": model,
                    "part_index": idx
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

print(f"\n✅ Saved {len(qa_pairs)} QA pairs to {output_path}")
