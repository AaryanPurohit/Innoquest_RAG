import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize sentence transformer for embeddings
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Gemini API with API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize text generation model
_text_model = genai.GenerativeModel("gemini-2.5-flash")

def get_embedding(text: str):
    """Get embedding using sentence transformers."""
    vec = _embedder.encode([text], normalize_embeddings=False)[0]
    return vec.tolist()

def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Generate answer using Gemini API."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    response = _text_model.generate_content(full_prompt)
    # Handle response text extraction
    if hasattr(response, 'text'):
        return response.text
    elif hasattr(response, 'parts') and len(response.parts) > 0:
        return response.parts[0].text
    else:
        return str(response)