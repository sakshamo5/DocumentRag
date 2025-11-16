import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AVAILABLE_MODELS = {
    "OpenAI GPT-OSS 20B": "openai/gpt-oss-20b",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
}

DEFAULT_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.2
MAX_TOKENS = 2048

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

DEFAULT_K = 5 
SIMILARITY_THRESHOLD = 0.7

FAISS_INDEX_PATH = "faiss_index"
FINETUNED_MODEL_PATH = "models/finetuned_embeddings"
EVAL_RESULTS_PATH = "evaluation_results"
