"""
dynamic_cot_controller.py
Dynamic CoT controller v6

Functions
---------
classify_question   : FAST | LIGHT | DEEP
quality_gate        : decides if first reply is sufficient
decide_cot          : whether to trigger extra reasoning
integrate_cot       : up to N CoT rounds with early stopping
"""

from __future__ import annotations
import re
import logging
from typing import Any, Dict, List

LOGGER = logging.getLogger("veltraxor.cot")

# ─────────────────────────────────────────────
# (1) question difficulty classifier
# ─────────────────────────────────────────────
FAST_PAT = re.compile(
    r"\b(what\s+is|who\s+is|chemical\s+formula|capital\s+of|year\s+did)\b",
    re.I,
)
DEEP_PAT = re.compile(
    r"\b(knight|knave|spy|bulb|switch|prove|logic|integer|determine|min|max)\b",
    re.I,
)

def classify_question(q: str) -> str:
    if DEEP_PAT.search(q):
        return "DEEP"
    if FAST_PAT.search(q):
        return "FAST"
    if re.search(r"\d", q):
        return "LIGHT"
    return "LIGHT"


# ─────────────────────────────────────────────
# (2) first-reply quality gate
# ─────────────────────────────────────────────
UNCERTAIN = re.compile(
    r"\b(maybe|not\s+sure|uncertain|probably|possibly|guess)\b", re.I
)
MIN_TOKENS_GOOD = 25

def quality_gate(first_reply: str) -> bool:
    """Return True when reply length and certainty look sufficient."""
    long_enough = len(first_reply.split()) >= MIN_TOKENS_GOOD
    has_uncertain = bool(UNCERTAIN.search(first_reply))
    return long_enough and not has_uncertain


# ─────────────────────────────────────────────
# (3) CoT trigger decision
# ─────────────────────────────────────────────
def decide_cot(question: str, first_reply: str) -> bool:
    lvl = classify_question(question)

    # FAST questions: never trigger extra reasoning
    if lvl == "FAST":
        return False

    # DEEP questions: always trigger extra reasoning
    if lvl == "DEEP":
        return True

    # LIGHT questions: trigger only if first reply looks weak
    return not quality_gate(first_reply)


# ─────────────────────────────────────────────
# (4) iterative CoT executor
# ─────────────────────────────────────────────
COT_TEMPLATE = (
    "{question}\n\n"
    "Think step by step and justify briefly.\n"
    "End with exactly one line:\n"
    "\"Final answer: <your concise answer>\""
)

def integrate_cot(
    client: Any,
    system_prompt: str,
    user_question: str,
    first_reply: str,
    max_rounds: int = 3,
) -> str:
    """Iteratively add reasoning rounds until quality gate passes."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": first_reply},
    ]
    answer = first_reply
    for step in range(max_rounds):
        if quality_gate(answer):
            break
        LOGGER.info("CoT round %d triggered", step + 1)
        messages.append(
            {"role": "user", "content": COT_TEMPLATE.format(question=user_question)}
        )
        try:
            resp = client.chat(messages)
            answer = resp["choices"][0]["message"]["content"].strip()
            messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            LOGGER.error("CoT call failed: %s", exc)
            break
    return answer