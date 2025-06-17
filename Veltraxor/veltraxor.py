# veltraxor.py — CLI chatbot with Dynamic CoT v5

from typing import Dict, List
from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

MODEL_NAME = "grok-3-latest"
SYSTEM_PROMPT = "You are Veltraxor, a concise yet rigorous reasoning assistant."

def extract_final_line(text: str) -> str:
    """Return the last line starting with 'Final answer:' if present."""
    for line in reversed(text.splitlines()):
        if line.lower().startswith("final answer"):
            return line.strip()
    return text.strip()

def main() -> None:
    client = LLMClient(model=MODEL_NAME)

    print("Veltraxor ready. Type 'exit' to quit.\n")
    while True:
        question = input("User: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Session terminated.")
            break

        # first pass
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        first_resp = client.chat(messages)
        first_reply = first_resp["choices"][0]["message"]["content"].strip()

        # dynamic CoT decision
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

        print(f"\nVeltraxor: {extract_final_line(final_raw)}\n")

if __name__ == "__main__":
    main()
