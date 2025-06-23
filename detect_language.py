"""Detect the language of a given text.

Priority:
1. Unicode-script heuristics for short or code-mixed text.
2. Fallback to `langdetect` for other cases.

Run standalone:
    python detect_language.py "Some text here"
or just:
    python detect_language.py
and follow the prompt.
"""

from __future__ import annotations

import re
import sys
from typing import cast

from langdetect import DetectorFactory, detect

# Make `langdetect` deterministic.
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """Return an ISO-639-1 language code such as 'zh', 'en', or 'ko'."""
    # Chinese: CJK Unified Ideographs
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    # Korean: Hangul Jamo + Hangul Syllables
    if re.search(r"[\u1100-\u11FF\uAC00-\uD7AF]", text):
        return "ko"
    # Japanese: Hiragana + Katakana
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text):
        return "ja"
    # Cyrillic block
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    # Arabic script
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    # Devanagari
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    # Fallback to `langdetect`.
    try:
        return cast(str, detect(text))
    except Exception:
        # Default to English if detection fails.
        return "en"


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) > 1:
        input_text: str = " ".join(sys.argv[1:])
    else:
        input_text = input("Please enter text to detect language: ")

    code: str = detect_language(input_text)
    print(f"Detected language code: {code}")


if __name__ == "__main__":
    main()
