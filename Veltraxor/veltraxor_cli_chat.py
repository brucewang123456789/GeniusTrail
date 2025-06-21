import os
import json
import requests
from dotenv import load_dotenv

def cli_chat():
    load_dotenv()
    token = os.getenv("VELTRAX_API_TOKEN") or os.getenv("XAI_API_KEY")
    if not token:
        token = input("Enter API token: ").strip()
        if not token:
            print("API token required.")
            return
        os.environ["VELTRAX_API_TOKEN"] = token
    url = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    history = []
    print("Type message or 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        payload = {"prompt": user_input, "history": history}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                continue
            data = resp.json()
            answer = data.get("response", "")
            print("Bot:", answer)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print("Request failed:", e)
            continue

if __name__ == "__main__":
    cli_chat()
