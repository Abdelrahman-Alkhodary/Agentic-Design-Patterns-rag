##############################################################
###################  STEP THREE ##############################
######### Store the chunked text into the pgvectore ##########
##############################################################
import json
from src.database.postgres import setup_database, store_chunk
from src.llm.openai import OpenaiLLM
from tqdm import tqdm




def main():
    # 1. Initialize the DB
    setup_database()
    llm = OpenaiLLM()
    
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
            metadata=metadata,
            embedding_fn=llm.get_embedding
        )
    

if __name__ == "__main__":
    main()    
        