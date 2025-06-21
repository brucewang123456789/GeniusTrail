import os
import httpx
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root

API_URL = os.getenv("XAI_API_URL")
API_KEY = os.getenv("XAI_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "grok-3-latest",
    "messages": [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."}
    ],
    "stream": False,
}

response = httpx.post(API_URL, headers=headers, json=payload, timeout=10.0)
print("Status:", response.status_code)
print("Response JSON:", response.text)
