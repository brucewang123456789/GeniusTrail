from config import settings
import httpx

API_KEY = settings.XAI_API_KEY
ENDPOINT = "https://api.openai.com/v1/chat/completions"

class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: list):
        payload = {
            "model": self.model,
            "messages": messages
        }
        response = httpx.post(ENDPOINT, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
