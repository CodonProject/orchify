try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception: pass

import os


def req_key() -> str:
    api_key  = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY') or ''
    return api_key

def req_url(fallback: str = '') -> str:
    base_url = os.getenv('BASE_URL') or ''
    return base_url if base_url else fallback

def req_model(fallback: str = '') -> str:
    model_id = os.getenv('MODEL_ID') or ''
    return model_id if model_id else fallback