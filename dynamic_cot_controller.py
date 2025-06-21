# dynamic_cot_controller.py — extracted external call into call_grok

import re
import logging
from typing import Any, Dict, List

log = logging.getLogger("cot-controller")

FAST_PAT = re.compile(r"\b(what\s+is|who\s+is|chemical\s+formula|capital\s+of|year\s+did)\b", re.I)
DEEP_PAT = re.compile(r"\b(knight|knave|spy|bulb|switch|prove|logic|integer|determine|min|max)\b", re.I)

UNCERTAIN = re.compile(r"\b(maybe|not\s+sure|uncertain|probably|possibly|guess)\b", re.I)
MIN_TOKENS_GOOD = 60

COT_TEMPLATE = (
    "{question}\n\n"
    "Respond as Veltraxor would: irreverent, sarcastic, razor-sharp. "
    "Reason step by step, showing your chain of thought in brief bullets. "
    "Finish with exactly one line:\n"
    '"Final answer: <concise yet savage answer>"'
)


def classify_question(question: str) -> str:
    """Classify question into FAST, DEEP, or LIGHT."""
    try:
        if DEEP_PAT.search(question):
            return "DEEP"
        if FAST_PAT.search(question):
            return "FAST"
        if re.search(r"\d", question):
            return "LIGHT"
    except Exception as exc:
        log.error("Error in classify_question(): %s", exc, exc_info=True)
    return "LIGHT"


def quality_gate(reply: str) -> bool:
    """Return True if reply is long enough and confident."""
    try:
        word_count = len(reply.split())
        unsure = bool(UNCERTAIN.search(reply))
        return word_count >= MIN_TOKENS_GOOD and not unsure
    except Exception as exc:
        log.error("Error in quality_gate(): %s", exc, exc_info=True)
    return False


def decide_cot(question: str, first_reply: str) -> bool:
    """Decide whether to run chain-of-thought reasoning."""
    level = classify_question(question)
    if level == "FAST":
        return False
    if level == "DEEP":
        return True
    return not quality_gate(first_reply)


def call_grok(client: Any, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call the grok API and return its raw response dict.
    Any exceptions are logged and re-raised.
    """
    try:
        return client.chat(messages)
    except Exception as exc:
        log.error("Error calling grok API: %s", exc, exc_info=True)
        raise


def integrate_cot(
    client: Any,
    system_prompt: str,
    user_question: str,
    first_reply: str,
    max_rounds: int = 3,
) -> List[Dict[str, str]]:
    """
    Iteratively add reasoning rounds until quality gate passes.
    Returns the full messages list.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_question},
        {"role": "assistant", "content": first_reply},
    ]
    answer = first_reply

    for idx in range(max_rounds):
        if quality_gate(answer):
            break

        log.info("CoT iteration %d", idx + 1)
        next_prompt = COT_TEMPLATE.format(question=user_question)
        messages.append({"role": "user", "content": next_prompt})

        try:
            resp = call_grok(client, messages)
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content", "").strip()
                if content:
                    answer = content
                    messages.append({"role": "assistant", "content": answer})
                    continue
            log.warning("CoT response empty; breaking")
            break
        except Exception as exc:
            log.error("CoT integration error: %s", exc, exc_info=True)
            break

    return messages