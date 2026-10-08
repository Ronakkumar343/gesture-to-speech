"""Configuration loader for Gesture-to-Speech phrases.

Maps gesture keys ("hello", "yes", "stop", "thank_you", "help") to spoken phrases.
Allows customization of spoken phrases without modifying classification or stabilization logic.
Unknown keys in phrases.json are ignored with a warning.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

DEFAULT_PHRASES: Dict[str, str] = {
    "hello": "Hello",
    "yes": "Yes",
    "stop": "Stop",
    "thank_you": "Thank you",
    "help": "I need help",
}

VALID_KEYS = set(DEFAULT_PHRASES.keys())


def load_phrases(path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Loads and validates gesture key to phrase mappings from a JSON file.

    Validation rules:
    - If path is None, defaults to phrases.json located in the project root.
    - File must exist and contain a valid JSON object (dict).
    - Every value must be a non-empty string.
    - Missing keys will fallback to default values with a printed warning.
    - Unknown keys are ignored with a printed warning.
    - Unparseable or invalid root JSON falls back to full DEFAULT_PHRASES with a warning.

    Args:
        path: Optional file path to phrases.json.

    Returns:
        A dictionary mapping the 5 supported gesture keys to non-empty phrase strings.
    """
    if path is None:
        file_path = Path(__file__).resolve().parent / "phrases.json"
    else:
        file_path = Path(path)

    if not file_path.is_file():
        print(f"[CONFIG WARNING]: Phrases file not found at '{file_path}'. Using defaults.", file=sys.stderr)
        return dict(DEFAULT_PHRASES)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[CONFIG WARNING]: Failed to parse JSON from '{file_path}' ({e}). Using defaults.", file=sys.stderr)
        return dict(DEFAULT_PHRASES)

    if not isinstance(data, dict):
        print(f"[CONFIG WARNING]: Root of '{file_path}' must be a JSON object. Using defaults.", file=sys.stderr)
        return dict(DEFAULT_PHRASES)

    phrases = dict(DEFAULT_PHRASES)

    for k, v in data.items():
        if k not in VALID_KEYS:
            print(f"[CONFIG WARNING]: Unknown gesture key '{k}' ignored.", file=sys.stderr)
            continue
        if not isinstance(v, str) or not v.strip():
            print(
                f"[CONFIG WARNING]: Invalid or empty phrase for key '{k}'. Using default '{DEFAULT_PHRASES[k]}'.",
                file=sys.stderr,
            )
            continue
        phrases[k] = v.strip()

    return phrases
