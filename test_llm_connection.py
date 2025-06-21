import requests
import os
from dotenv import load_dotenv

# Make sure environment variables can be loaded
load_dotenv()

url = os.getenv("XAI_API_URL") or "https://chat.x.ai/api/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
    "Content-Type": "application/json"
}
payload = {
    "model": os.getenv("VELTRAX_MODEL", "grok-3-latest"),
    "messages": [{"role": "user", "content": "ping"}],
    "stream": False
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print(f"Status: {response.status_code}")
    print("Response:", response.text[:500])
except Exception as e:
    print("Error:", str(e))
