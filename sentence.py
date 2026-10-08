"""Sentence builder module for composing multi-phrase spoken sentences.

Maintains an in-memory buffer of accumulated phrases, exposes the joined text,
and coordinates speech dispatch with sentence recording. Fully unit-testable
without camera or TTS runtime dependencies.
"""

from typing import Any, Callable, List, Optional


class SentenceBuilder:
    """Buffers spoken phrases into a complete sentence with speaking and clear actions."""

    def __init__(self) -> None:
        self._buffer: List[str] = []
        self._last_spoken: Optional[str] = None

    def append(self, phrase: Optional[str]) -> bool:
        """Appends a phrase to the sentence buffer.

        Ignores None, empty strings, and whitespace-only strings.

        Returns:
            True if phrase was appended, False if ignored.
        """
        if phrase is None:
            return False
        cleaned = phrase.strip()
        if not cleaned:
            return False
        self._buffer.append(cleaned)
        return True

    @property
    def text(self) -> str:
        """Returns the joined sentence string from all buffered phrases."""
        return " ".join(self._buffer)

    @property
    def phrases(self) -> List[str]:
        """Returns a copy of the list of phrases currently in the buffer."""
        return list(self._buffer)

    @property
    def last_spoken(self) -> Optional[str]:
        """Returns the last spoken sentence text, or None if none spoken yet."""
        return self._last_spoken

    def clear(self) -> None:
        """Empties the current sentence buffer."""
        self._buffer.clear()

    def speak_sentence(self, speak_fn: Callable[[str], Any]) -> Optional[str]:
        """Speaks the full sentence text via speak_fn, records it, and returns the text.

        Args:
            speak_fn: Callable taking a string to speak.

        Returns:
            The spoken sentence text string, or None if the buffer is empty.
        """
        sentence = self.text
        if not sentence:
            return None
        self._last_spoken = sentence
        speak_fn(sentence)
        return sentence
