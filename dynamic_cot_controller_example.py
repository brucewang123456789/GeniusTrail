"""
dynamic_cot_controller_example.py
Demo stub for Dynamic Chain-of-Thought (CoT) controller.
This file provides API signatures and placeholder logic
without exposing production thresholds or adaptive algorithms.
"""

import os
import re
import logging
from typing import Any, Dict, List

LOGGER = logging.getLogger("veltraxor.cot.demo")

# Load patterns and thresholds from environment for demonstration
MIN_TOKENS_GOOD = int(os.getenv("COT_MIN_TOKENS", "25"))
FAST_REGEX = os.getenv("COT_FAST_REGEX", r"\b(what\s+is|who\s+is)\b")
DEEP_REGEX = os.getenv("COT_DEEP_REGEX", r"\b(knight|logic)\b")
MAX_COT_ROUNDS = int(os.getenv("COT_MAX_ROUNDS", "2"))

FAST_PAT = re.compile(FAST_REGEX, re.I)
DEEP_PAT = re.compile(DEEP_REGEX, re.I)
UNCERTAIN = re.compile(r"\b(maybe|unsure|probably)\b", re.I)


def classify_question(q: str) -> str:
    """
    Classify the question into FAST, LIGHT, or DEEP.
    Simplified demo logic for interface illustration.
    """
    if DEEP_PAT.search(q):
        return "DEEP"
    if FAST_PAT.search(q):
        return "FAST"
    if re.search(r"\d", q):
        return "LIGHT"
    return "LIGHT"


def quality_gate(reply: str) -> bool:
    """
    Check if the first reply is good enough to skip further CoT.
    Demo only checks token count and simple uncertainty marker.
    """
    return len(reply.split()) >= MIN_TOKENS_GOOD and not UNCERTAIN.search(reply)


def decide_cot(question: str, first_reply: str) -> bool:
    """
    Decide whether to trigger CoT based on question type and reply quality.
    FAST: always skip
    DEEP: always trigger
    LIGHT: trigger if quality gate fails
    """
    lvl = classify_question(question)
    if lvl == "FAST":
        return False
    if lvl == "DEEP":
        return True
    return not quality_gate(first_reply)


COT_TEMPLATE = (
    "{question}\n\n"
    "Demonstration: Think step by step and justify briefly.\n"
    "End with exactly one line:\n"
    '"Final answer: <concise answer>"'
)


def integrate_cot(
    client: Any,
    system_prompt: str,
    user_question: str,
    first_reply: str,
    max_rounds: int = MAX_COT_ROUNDS,
) -> str:
    """
    Placeholder iterative CoT.
    Appends demo messages to show where real CoT would run.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": first_reply},
    ]
    answer = first_reply
    for i in range(max_rounds):
        if quality_gate(answer):
            LOGGER.info("Demo CoT: quality gate passed at round %d", i)
            break
        LOGGER.info("Demo CoT: round %d placeholder", i + 1)
        messages.append(
            {"role": "user", "content": COT_TEMPLATE.format(question=user_question)}
        )
        resp = client.chat(messages)
        answer = resp["choices"][0]["message"]["content"].strip()
        messages.append({"role": "assistant", "content": answer})
    return answer
