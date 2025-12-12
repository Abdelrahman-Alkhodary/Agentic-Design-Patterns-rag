##############################################################
###################  STEP FIVE ###############################
############## Evaluating The RAG System #####################
##############################################################

import os
import json
import yaml
from src.database.postgres import hybrid_search
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm
from src.llm.openai import OpenaiLLM

load_dotenv()
client = OpenAI()

class AnswerEvaluation(BaseModel):
    correctness: int = Field(description="Provide your evaluation as a single integer score (0-5)")


if __name__ == "__main__":
    # initiate the llm model
    llm = OpenaiLLM()
    # Load the qa_pairs dataset
    qa_path = "./data/processed/qa_pairs.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    with open("./prompts/generating_answer.yml") as f:
        prompts = yaml.safe_load(f)
        
    eval_scores = {
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
        "haullucinated": 0
    }
    
    for pair in tqdm(qa_pairs[:100], total=len(qa_pairs[:100])):
        question = pair["question"]
        truth_answer = pair["answer"]
        results = hybrid_search(question, embedding_fn=llm.get_embedding)

        retrieved_texts = ""
        for i, res in enumerate(results):
            content = res['data'][1]
            retrieved_texts += content + "\n\n"
            meta = res['data'][2]
            # score = res['score']
            # print(f"{i+1}. Score: {score:.4f} | Text: {content[:100]}...")
        
        # Generating answer for the question and the retrieved texts
        answer = llm.generate_answer(
            question=question,
            retrieved_texts=retrieved_texts
        )
        
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
        
        
        if event.correctness == 5:
            eval_scores["correct"] += 1
        elif event.correctness == 4 or event.correctness == 3:
            eval_scores["partially_correct"] += 1
        elif event.correctness == 2 or event.correctness == 1:
            eval_scores["incorrect"] += 1
        else:
            eval_scores["haullucinated"] += 1 
        
    
    print(f"Number of correct answers: {eval_scores["correct"]}")
    print(f"Number of partially_correct answers: {eval_scores["partially_correct"]}")
    print(f"Number of incorrect answers: {eval_scores["incorrect"]}")
    print(f"Number of haullucinated answers: {eval_scores["haullucinated"]}")