import os
import requests
import re
from dotenv import load_dotenv
from typing import List, Dict, Any


def summary_replacement(response: str) -> str:
    """Replace 'Final Answer:' with a random summary phrase followed by a comma."""
    summary_phrases = [
        "In summary, ",
        "To put it simply, ",
        "In short, ",
        "Long story short, ",
    ]
    import random

    phrase = random.choice(summary_phrases)
    response = re.sub(r"(?i)Final\s+Answer\s*:?\s*", phrase, response)
    return response


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
    base_url: str = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000")
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
        method_success = False

        # Try POST first
        try:
            resp: requests.Response = requests.post(
                f"{base_url}/chat",
                headers=headers,
                json=payload,
                timeout=timeout_sec,
            )
            if resp.status_code == 200:
                method_success = True
            else:
                print(f"POST Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"POST Request failed: {e}")

        # If POST fails, try GET with query parameters
        if not method_success:
            try:
                resp: requests.Response = requests.get(
                    f"{base_url}/chat?prompt={requests.utils.quote(user_input)}",
                    headers=headers,
                    timeout=timeout_sec,
                )
                if resp.status_code == 200:
                    method_success = True
                else:
                    print(f"GET Error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"GET Request failed: {e}")

        # Handle response if either method succeeded
        if method_success:
            try:
                data: Dict[str, Any] = resp.json()
                raw_answer: str = data.get(
                    "response", "Backend gave me zilch—pathetic!"
                )
                answer: str = summary_replacement(
                    raw_answer
                )  # Replace 'Final Answer:' with summary phrase
                print("Bot:", answer)

                # Update history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})
            except Exception as e:
                print(f"Response parsing failed: {e}")
        else:
            print("Both POST and GET tanked—disgraceful!")


if __name__ == "__main__":
    cli_chat()
