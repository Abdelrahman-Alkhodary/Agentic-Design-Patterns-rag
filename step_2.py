##############################################################
###################  STEP TWO ################################
######### Read the markdown text and chunk it ################
##############################################################

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm
from time import sleep
import json
import os

load_dotenv()


with open("./data/processed/Agentic_Design_Patterns.mmd", "r") as f:
    full_text = f.read()

rec_splitter = RecursiveCharacterTextSplitter(chunk_size=1750, chunk_overlap=250)
rec_chunks = rec_splitter.split_text(full_text)

client = OpenAI()

book_name = "Agentic Design Patterns"

# -----------------------------------------------------------------------------
# 1. Pydantic Model for Structured Output
# -----------------------------------------------------------------------------
class ChunkMetadata(BaseModel):
    semantic_title: str = Field(description="A human-readable short title for the chunk")
    context_expansion: str = Field(description="Using the cached full book, generate a short context expansion for this chunk, no more than 2 sentences, that explains the broader context and patterns from the book relevant to this text.")
    section_header: str = Field(description="The section title or chapter the chunk belongs to")
    keywords: list[str] = Field(description="Important keywords describing this chunk")

MAX_RETRIES = 5
BASE_SLEEP = 18  # seconds

data_chunks = []

# Ensure the directory exists
output_dir = "./data/processed/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "enriched_chunks.json")


for chunk in tqdm(rec_chunks, total=len(rec_chunks)):
    
    user_message = {
        "role": "user",
        "content": f"Enrich the following chunk using the full book text:\n\n{chunk}"
    }

    system_prompt={
        "role": "system",
        "content": f"""
        You are a helpful RAG assistant. This is the full book '{book_name}'.

        {full_text}
    """
    }
    
    retries = 0
    
    while retries < MAX_RETRIES:
        try:    
            response = client.responses.parse(
                model="gpt-5-nano",
                input=[
                    system_prompt,  # cached full book
                    user_message           # new chunk
                ],
                text_format=ChunkMetadata,
            )

            parsed = response.output_parsed
            # print(response.usage.input_tokens_details.cached_tokens)
            chunk_metada = {}
            chunk_metada["context_expansion"] = parsed.context_expansion
            chunk_metada["semantic_title"] = parsed.semantic_title
            chunk_metada["section_header"] = parsed.section_header
            chunk_metada["keywords"] = parsed.keywords
            
            data_chunks.append({
                "chunk": chunk,
                "chunk_metada": chunk_metada
            })
            
            sleep(BASE_SLEEP)
            
            if len(data_chunks) % 25 == 0:    
                # Save the data_chunks list as a JSON file
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data_chunks, f, ensure_ascii=False, indent=4)
                    
                print(f"Saved {len(data_chunks)} enriched chunks to {output_path}")
            break
        except (RateLimitError, APIError, TimeoutError) as e:
            retries += 1
            sleep_time = BASE_SLEEP * (2 ** (retries - 1)) 
            print(f"Error: {e}")
            sleep(sleep_time)

# Save the data_chunks list as a JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_chunks, f, ensure_ascii=False, indent=4)

print(f"Saved {len(data_chunks)} enriched chunks to {output_path}")
        