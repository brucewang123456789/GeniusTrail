# veltraxor.py — CLI chatbot with Dynamic CoT

from typing import Dict, List, Optional
import argparse
from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

MODEL_NAME = "grok-3-latest"
SYSTEM_PROMPT = (
    "You are Veltraxor, a concise yet rigorous reasoning assistant."
)


def extract_final_line(text: str) -> str:
    """Return the last line beginning with 'Final answer:' if present."""
    for line in reversed(text.splitlines()):
        if line.lower().startswith("final answer"):
            return line.strip()
    return text.strip()


def chat_once(client: LLMClient, question: str) -> str:
    """Run one shot (with optional extra CoT) and return the final reply."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    first_resp = client.chat(messages)
    first_reply = first_resp["choices"][0]["message"]["content"].strip()

    if decide_cot(question, first_reply):
        print("System: Deep reasoning triggered…")
        final_raw = integrate_cot(
            client,
            system_prompt=SYSTEM_PROMPT,
            user_question=question,
            first_reply=first_reply,
        )
    else:
        final_raw = first_reply

    return extract_final_line(final_raw)


def repl(client: LLMClient) -> None:
    """Interactive loop."""
    print("Veltraxor ready. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Session terminated.")
            break
        answer = chat_once(client, question)
        print(f"\nVeltraxor: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Veltraxor — CLI chatbot with Dynamic CoT control"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One-shot prompt; if omitted, enters interactive mode",
    )
    args = parser.parse_args()

    client = LLMClient(model=MODEL_NAME)

    if args.prompt:
        question = " ".join(args.prompt)
        print(chat_once(client, question))
    else:
        repl(client)


if __name__ == "__main__":
    main()