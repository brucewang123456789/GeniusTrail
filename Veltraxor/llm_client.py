import httpx
from config import settings

ENDPOINT = "https://api.x.ai/v1/chat/completions"

class LLMClient:
    def __init__(self, model: str):
        self.model = model
        api_key = settings.XAI_API_KEY
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set in environment")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: list) -> dict:
        """
        Send a chat completion request and return the full JSON response.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        response = httpx.post(ENDPOINT, headers=self.headers, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()

    def stream(self, messages: list, timeout: float = 60.0):
        """
        Send a streaming chat completion request.
        Yields each JSON chunk as it arrives.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        with httpx.stream("POST", ENDPOINT, headers=self.headers, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_lines():
                if not chunk:
                    continue
                try:
                    yield httpx.models.json.loads(chunk)
                except Exception:
                    continue
