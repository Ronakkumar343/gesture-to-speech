"""Temporal gesture stabilization and smoothing module.

Provides:
- GestureStabilizer: Sliding-window majority voting, confidence gating,
  flicker suppression, and HUD state reporting.
- DoublePalmStateMachine: Pure deterministic state machine mapping rapid
  two-palm triggers into emergency "I need help" phrases.
"""

from collections import deque
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import DEFAULT_PHRASES


# Canonical gesture key lookup
_REVERSE_KEY_LOOKUP: Dict[str, str] = {v.lower(): k for k, v in DEFAULT_PHRASES.items()}
for k in DEFAULT_PHRASES.keys():
    _REVERSE_KEY_LOOKUP[k.lower()] = k


def _normalize_gesture_key(phrase_or_key: Optional[str]) -> Optional[str]:
    """Normalizes either a phrase ('Hello') or key ('hello') into a canonical key."""
    if phrase_or_key is None:
        return None
    cleaned = phrase_or_key.strip().lower()
    return _REVERSE_KEY_LOOKUP.get(cleaned, cleaned)


class DoublePalmStateMachine:
    """Detects rapid double-palm gestures within a time window using an injectable clock.

    Rules:
    - First stabilized palm emission -> "Hello" (or configured hello phrase).
    - Second stabilized palm emission within `window_sec` -> "I need help" (or configured help phrase).
    - Continuous hold of a single palm without release stays "Hello".
    - Two palms separated by > `window_sec` produce two separate "Hello"s.
    """

    def __init__(
        self,
        window_sec: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
        phrase_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.window_sec = window_sec
        self._clock = clock
        self._phrase_map = dict(phrase_map) if phrase_map is not None else dict(DEFAULT_PHRASES)

        self._last_hello_time: Optional[float] = None
        self._active_key: Optional[str] = None
        self._active_emitted_key: Optional[str] = None

    def reset(self) -> None:
        """Resets the internal tracking state."""
        self._last_hello_time = None
        self._active_key = None
        self._active_emitted_key = None

    def update(self, stabilized_phrase_or_key: Optional[str]) -> Optional[str]:
        """Processes the stabilized winner and applies double-palm timing rules.

        Args:
            stabilized_phrase_or_key: Winning gesture from stabilizer or None.

        Returns:
            The phrase to emit, or None if no gesture is active.
        """
        if stabilized_phrase_or_key is None:
            self._active_key = None
            self._active_emitted_key = None
            return None

        key = _normalize_gesture_key(stabilized_phrase_or_key)
        if key is None:
            self._active_key = None
            self._active_emitted_key = None
            return None

        now = self._clock()

        # Check if this is a new gesture onset (edge trigger)
        if key != self._active_key:
            if key == "hello":
                if (
                    self._last_hello_time is not None
                    and (now - self._last_hello_time) <= self.window_sec
                ):
                    emitted_key = "help"
                    self._last_hello_time = None
                else:
                    emitted_key = "hello"
                    self._last_hello_time = now
            else:
                emitted_key = key
                self._last_hello_time = None

            self._active_key = key
            self._active_emitted_key = emitted_key
        else:
            # Continuing to hold the same gesture
            emitted_key = self._active_emitted_key or key

        return self._phrase_map.get(emitted_key, emitted_key)


class GestureStabilizer:
    """Sliding-window majority voting stabilizer with confidence smoothing and flicker rejection."""

    def __init__(
        self,
        window_size: int = 5,
        vote_threshold: Optional[int] = None,
        min_confidence: float = 0.6,
        double_palm_window: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
        phrase_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Constructs a stabilizer instance.

        Args:
            window_size: Number of frames in sliding history (default 5).
            vote_threshold: Minimum frame agreement count (default: ceil(N * 0.6)).
            min_confidence: Minimum average confidence required for winner (default 0.6).
            double_palm_window: Max elapsed seconds for double-palm help gesture (default 2.5).
            clock: Time function returning seconds (default time.monotonic).
            phrase_map: Optional dictionary mapping gesture keys to output phrases.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.window_size = window_size
        self.vote_threshold = (
            vote_threshold if vote_threshold is not None else math.ceil(window_size * 0.6)
        )
        self.min_confidence = min_confidence
        self._phrase_map = dict(phrase_map) if phrase_map is not None else dict(DEFAULT_PHRASES)

        self._window: deque[Tuple[Optional[str], float]] = deque(maxlen=window_size)
        self._double_palm = DoublePalmStateMachine(
            window_sec=double_palm_window,
            clock=clock,
            phrase_map=self._phrase_map,
        )

        # HUD State attributes
        self._last_winner: Optional[str] = None
        self._winner_votes: int = 0
        self._winner_confidence: float = 0.0

    @property
    def last_winner(self) -> Optional[str]:
        """Most recently emitted winning phrase string."""
        return self._last_winner

    @property
    def window_contents(self) -> List[Tuple[Optional[str], float]]:
        """Current frame buffer contents as a list of (phrase_or_none, confidence) tuples."""
        return list(self._window)

    @property
    def winner_votes(self) -> int:
        """Vote count of the current winning candidate in the window."""
        return self._winner_votes

    @property
    def winner_confidence(self) -> float:
        """Mean confidence score of current winning candidate."""
        return self._winner_confidence

    @property
    def state(self) -> Dict[str, Any]:
        """Summary dictionary of current stabilizer state for HUD overlays."""
        return {
            "last_winner": self._last_winner,
            "window_size": self.window_size,
            "current_frames": len(self._window),
            "winner_votes": self._winner_votes,
            "winner_confidence": self._winner_confidence,
        }

    def clear(self) -> None:
        """Clears the history window and resets double palm state."""
        self._window.clear()
        self._double_palm.reset()
        self._last_winner = None
        self._winner_votes = 0
        self._winner_confidence = 0.0

    def update(
        self,
        phrase: Optional[str],
        confidence: float = 0.0,
    ) -> Optional[str]:
        """Appends a new frame observation and returns the stabilized phrase if qualified.

        Args:
            phrase: Classified phrase or gesture key for current frame, or None.
            confidence: Confidence score of classification in [0, 1].

        Returns:
            The stabilized phrase string, or None if:
            - Hand is absent or pose is unrecognized.
            - Vote threshold is not met.
            - Mean confidence is below min_confidence.
            - Alternating flicker is detected.
        """
        key = _normalize_gesture_key(phrase) if phrase is not None else None
        conf = float(confidence) if (key is not None and confidence > 0.0) else 0.0

        # Append observation to sliding window
        self._window.append((key, conf))

        # Extract all non-None keys from current window
        non_none_keys = [k for k, _ in self._window if k is not None]

        if not non_none_keys:
            self._winner_votes = 0
            self._winner_confidence = 0.0
            self._double_palm.update(None)
            return None

        # 1. Flicker rejection: count transitions between alternating distinct keys
        transitions = sum(
            1 for i in range(len(non_none_keys) - 1) if non_none_keys[i] != non_none_keys[i + 1]
        )
        if transitions >= 2:
            # Alternating flicker pattern detected (e.g. A -> B -> A)
            self._winner_votes = 0
            self._winner_confidence = 0.0
            self._double_palm.update(None)
            return None

        # 2. Majority voting over window
        vote_counts: Dict[str, int] = {}
        for k in non_none_keys:
            vote_counts[k] = vote_counts.get(k, 0) + 1

        # Find maximum votes
        max_votes = max(vote_counts.values())
        top_candidates = [k for k, v in vote_counts.items() if v == max_votes]

        # In case of tie between two distinct candidates, no majority
        if len(top_candidates) != 1:
            self._winner_votes = 0
            self._winner_confidence = 0.0
            self._double_palm.update(None)
            return None

        candidate_key = top_candidates[0]

        # Check vote threshold
        if max_votes < self.vote_threshold:
            self._winner_votes = max_votes
            self._winner_confidence = 0.0
            self._double_palm.update(None)
            return None

        # 3. Confidence threshold check
        matching_confidences = [c for k, c in self._window if k == candidate_key]
        mean_conf = sum(matching_confidences) / len(matching_confidences)

        if mean_conf < self.min_confidence:
            self._winner_votes = max_votes
            self._winner_confidence = round(mean_conf, 3)
            self._double_palm.update(None)
            return None

        # Passed smoothing and confidence checks!
        self._winner_votes = max_votes
        self._winner_confidence = round(mean_conf, 3)

        # 4. Pass to double-palm temporal state machine
        emitted_phrase = self._double_palm.update(candidate_key)
        self._last_winner = emitted_phrase
        return emitted_phrase
