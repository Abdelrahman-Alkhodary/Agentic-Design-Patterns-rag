# from openai import OpenAI
# from src import config

# client = OpenAI(api_key=config.OPENAI_API_KEY)

# def get_embedding(text: str) -> list[float]:
#     """Generates vector embedding for the given text."""
#     try:
#         response = client.embeddings.create(
#             input=text, 
#             model=config.EMBEDDING_MODEL
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         return []