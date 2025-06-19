# detect_language.py
# A standalone script to detect the language of a given text.
# Uses Unicode script ranges for common languages as a first pass,
# then falls back to langdetect for other cases.

import re
import sys
from langdetect import detect, DetectorFactory

# Fix random seed for reproducible results from langdetect
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detect the language code of the input text.
    Priority is given to Unicode script detection for reliability on short text.
    Falls back to langdetect library if no script matches.
    Returns ISO 639-1 language code (e.g. 'zh', 'en', 'ko', etc.).
    """
    # Chinese: CJK Unified Ideographs
    if re.search(r'[\u4E00-\u9FFF]', text):
        return "zh"
    # Korean: Hangul Jamo + Hangul Syllables
    if re.search(r'[\u1100-\u11FF\uAC00-\uD7AF]', text):
        return "ko"
    # Japanese: Hiragana + Katakana
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja"
    # Cyrillic: Russian, Ukrainian, etc.
    if re.search(r'[\u0400-\u04FF]', text):
        return "ru"
    # Arabic script
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    # Devanagari: Hindi, Nepali, etc.
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    # Add more Unicode script checks here as needed...

    # Fallback to langdetect for other languages
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"  # default to English on error

def main():
    """
    Entry point for the script. Reads input from command line arguments
    or from stdin prompt, then prints detected language code.
    """
    if len(sys.argv) > 1:
        # Join all arguments as the input text
        input_text = " ".join(sys.argv[1:])
    else:
        # Prompt user for input if no arguments provided
        input_text = input("Please enter text to detect language: ")

    code = detect_language(input_text)
    print(f"Detected language code: {code}")

if __name__ == "__main__":
    main()