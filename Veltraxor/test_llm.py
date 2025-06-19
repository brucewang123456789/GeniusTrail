#!/usr/bin/env python3
"""
test_llm.py

This script tests the LLMClient.chat method by sending a sample message list
and printing out the raw response and extracted content. All comments and
output are in English.
Usage: python test_llm.py
"""

import os
from dotenv import load_dotenv

# Load .env so that settings in LLMClient pick up XAI_API_URL and XAI_API_KEY
load_dotenv()

from llm_client import LLMClient

def main():
    # Instantiate client. If VELTRAX_MODEL is set in .env, pass None here to pick it up.
    client = LLMClient(model=None)

    # Prepare a sample conversation. "system" prompt then a "user" message.
    messages = [
        {"role": "system", "content": "You are Veltraxor, a concise and rigorous assistant."},
        {"role": "user",   "content": "Hello, testing LLMClient.chat"}
    ]

    try:
        # Call chat
        response = client.chat(messages)
        print("LLMClient.chat returned dict:")
        print(response)

        # Extract text content if structure matches expected format
        if isinstance(response, dict) and response.get("choices"):
            choice = response["choices"][0]
            # Typical OpenAI/Grok style: message.content first, fallback to text
            content = choice.get("message", {}).get("content") or choice.get("text") or ""
            print("Extracted content:", content)
        else:
            print("Response is not a dict with 'choices' or has unexpected format.")
    except Exception as e:
        import traceback
        print("Exception occurred during LLMClient.chat:")
        traceback.print_exc()

if __name__ == "__main__":
    main()