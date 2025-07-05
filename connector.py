import os
from dotenv import load_dotenv, find_dotenv #to make system configurations easy

def load_env():
    _ = load_dotenv(find_dotenv())

def get_gemini_api_key():
    load_env()
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    return gemini_api_key
