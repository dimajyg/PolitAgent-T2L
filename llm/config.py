import os
from dotenv import load_dotenv

load_dotenv()

key_mistral = "here your mistral token"

model_mistral = "mistral-large-latest"
temperature_mistral = 0.7

model_openai = "gpt-4o-mini"
temperature_openai = 0.7
openai_api_key = os.getenv("OPENAI_API_KEY")

model_vllm = "mosaicml/mpt-7b"
temperature_vllm = 0.7
vllm_api_key = os.getenv("VLLM_API_KEY")
