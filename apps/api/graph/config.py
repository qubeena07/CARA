import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
MAX_RETRIES = 3
RELEVANCE_THRESHOLD = 0.5
HALLUCINATION_THRESHOLD = 0.7
