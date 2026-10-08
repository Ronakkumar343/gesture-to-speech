"""Session transcript logging and file export module.

Records all spoken phrases and full sentences with precise timestamps,
using an injectable clock for deterministic headless testing.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union


class SessionTranscript:
    """Logs timestamped spoken phrases and exports formatted session records to text files."""

    def __init__(
        self,
        clock: Optional[Callable[[], Union[float, int, datetime, str]]] = None,
    ) -> None:
        """Initializes a transcript session.

        Args:
            clock: Optional injectable callable returning a unix timestamp (float/int),
                   a datetime instance, or an ISO-style timestamp string. Defaults
                   to datetime.now().
        """
        self._clock = clock
        self._entries: List[Tuple[str, str]] = []

    def _format_timestamp(self) -> str:
        """Computes current timestamp string formatted as 'YYYY-MM-DD HH:MM:SS'."""
        if self._clock is not None:
            val = self._clock()
            if isinstance(val, datetime):
                return val.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(val, str):
                return val.strip()
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, phrase: Optional[str]) -> bool:
        """Appends a spoken phrase entry with current timestamp.

        Ignores None, empty strings, and whitespace.

        Returns:
            True if phrase was logged, False if ignored.
        """
        if phrase is None:
            return False
        cleaned = phrase.strip()
        if not cleaned:
            return False
        ts = self._format_timestamp()
        self._entries.append((ts, cleaned))
        return True

    @property
    def entries(self) -> List[Tuple[str, str]]:
        """Returns a copy of the list of logged (timestamp, phrase) entries."""
        return list(self._entries)

    def export(self, path: Union[str, Path]) -> str:
        """Exports the transcript entries to a text file with a header line.

        Format:
            # Gesture-to-Speech Session Transcript
            [2026-10-07 16:52:10] Hello
            [2026-10-07 16:52:14] Yes

        Args:
            path: Target file path to write to.

        Returns:
            The normalized absolute or relative string path written.
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            f.write("# Gesture-to-Speech Session Transcript\n")
            for ts, phrase in self._entries:
                f.write(f"[{ts}] {phrase}\n")

        return str(target)
