import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any


def cli_chat() -> None:
    # Load .env without overriding existing environment vars
    load_dotenv(override=False)

    # Retrieve token from environment or prompt user
    token: str | None = os.getenv("VELTRAX_API_TOKEN") or os.getenv("XAI_API_KEY")
    if not token:
        token = input("Enter API token: ").strip()
        if not token:
            print("API token required.")
            return
        os.environ["VELTRAX_API_TOKEN"] = token

    # Endpoint and headers
    url: str = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat")
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Chat history and timeout configuration
    history: List[Dict[str, str]] = []
    timeout_sec: int = int(os.getenv("CHAT_TIMEOUT", "60"))  # seconds
    print("Type message or 'exit' to quit.")

    while True:
        try:
            user_input: str = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        payload: Dict[str, Any] = {"prompt": user_input, "history": history}
        try:
            resp: requests.Response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_sec,
            )
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                continue

            data: Dict[str, Any] = resp.json()
            answer: str = data.get("response", "")
            print("Bot:", answer)

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print("Request failed:", e)
            continue


if __name__ == "__main__":
    cli_chat()
