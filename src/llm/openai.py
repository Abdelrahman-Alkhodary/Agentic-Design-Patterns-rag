import os
import yaml
from time import sleep
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI api requesting 
MAX_RETRIES = 5
BASE_SLEEP_TIME = 15

class OpenaiLLM:
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        
        with open("./prompts/generating_answer.yml") as f:
            self.prompts = yaml.safe_load(f)
        
        

    def get_embedding(self, text: str):
        """Generate embedding using openai text-embedding-3-small"""
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                break
            except (RateLimitError, APIError, TimeoutError) as e:
                retries += 1
                sleep_time = BASE_SLEEP_TIME * (2 ** (retries - 1))
                print(f"Error: {e}")
                sleep(sleep_time)
        return response.data[0].embedding
    
    def generate_answer(self, question: str, retrieved_texts: str):
        system_prompt = self.prompts["generating_answer"]["system"]
        
        user_prompt = self.prompts["generating_answer"]["user"].format(
            question=question,
            retrieved_chunks=retrieved_texts
        )
        
        response = self.client.responses.create(
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
        
        answer = response.output_text
        
        return answer
