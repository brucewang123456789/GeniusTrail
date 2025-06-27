import os
import sys
import json
import logging
from typing import Dict, Any

import requests
from dotenv import load_dotenv


def main() -> None:
    # Load environment variables from .env in project root
    load_dotenv()

    # Read API token from environment variables
    token: str | None = os.getenv("VELTRAX_API_TOKEN") or os.getenv("XAI_API_KEY")
    if not token:
        logging.error(
            "API token not found. Please set VELTRAX_API_TOKEN or XAI_API_KEY in .env"
        )
        sys.exit(1)

    # Endpoint URL for chat API, guaranteed to be str
    api_url: str = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat")
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Prepare request payload; adjust prompt/history as needed
    payload: Dict[str, Any] = {"prompt": "Hello, this is a test.", "history": []}

    try:
        response: requests.Response = requests.post(
            api_url, headers=headers, json=payload, timeout=10
        )
    except requests.RequestException as exc:
        logging.error("HTTP request to chat API failed: %s", exc)
        sys.exit(1)

    print("Status code:", response.status_code)

    # Try to parse JSON response
    try:
        data: Dict[str, Any] = response.json()
        print("Response JSON:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except ValueError:
        print("Received non-JSON response:")
        print(response.text)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
