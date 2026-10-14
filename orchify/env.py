try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception: pass

import os


def req_key() -> str:
    '''Read the API key from the environment.

    Returns:
        str: ``API_KEY`` (or ``OPENAI_API_KEY``) from the environment or ``.env``,
        or an empty string when neither variable is set.
    '''
    api_key  = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY') or ''
    return api_key

def req_url(fallback: str = '') -> str:
    '''Read the API base URL from the environment.

    Args:
        fallback (str): Value returned when ``BASE_URL`` is not set.

    Returns:
        str: The configured base URL or ``fallback``.
    '''
    base_url = os.getenv('BASE_URL') or ''
    return base_url if base_url else fallback

def req_model(fallback: str = '') -> str:
    '''Read the default model id from the environment.

    Args:
        fallback (str): Value returned when ``MODEL_ID`` is not set.

    Returns:
        str: The configured model id or ``fallback``.
    '''
    model_id = os.getenv('MODEL_ID') or ''
    return model_id if model_id else fallback
