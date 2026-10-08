"""Deterministic rule-based gesture classification from MediaPipe hand landmarks.

Provides:
- classify_with_confidence(landmarks) -> Optional[Tuple[str, float]]
  Classifies 21 MediaPipe hand landmarks into a spoken phrase string and confidence in (0, 1].
- classify_key_with_confidence(landmarks) -> Optional[Tuple[str, float]]
  Classifies into an internal gesture key ("hello", "yes", "stop", "thank_you") and confidence.
- classify(landmarks) -> Optional[str]
  Backward-compatible v1 classifier returning the spoken phrase string or None.
- classify_key(landmarks) -> Optional[str]
  Returns the internal gesture key or None.

Zero external runtime model dependencies. Fully deterministic and unit-testable.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Landmark index constants according to MediaPipe Hands specification
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Canonical key to default phrase mapping
KEY_TO_PHRASE: Dict[str, str] = {
    "hello": "Hello",
    "yes": "Yes",
    "stop": "Stop",
    "thank_you": "Thank you",
    "help": "I need help",
}


def _extract_coords(pt: Any) -> Optional[Tuple[float, float]]:
    """Extracts (x, y) coordinates from a landmark object, dict, or tuple."""
    if pt is None:
        return None

    # Landmark object with .x and .y attributes (MediaPipe NormalizedLandmark)
    if hasattr(pt, "x") and hasattr(pt, "y"):
        try:
            return float(pt.x), float(pt.y)
        except (TypeError, ValueError):
            return None

    # Dictionary format {'x': ..., 'y': ...}
    if isinstance(pt, dict) and "x" in pt and "y" in pt:
        try:
            return float(pt["x"]), float(pt["y"])
        except (TypeError, ValueError):
            return None

    # Sequence / tuple / list format (x, y)
    if isinstance(pt, (tuple, list)) and len(pt) >= 2:
        try:
            return float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            return None

    return None


def classify_key_with_confidence(
    landmarks: Optional[Sequence[Any]],
) -> Optional[Tuple[str, float]]:
    """Classifies landmarks into an internal gesture key and confidence score in (0, 1].

    Confidence is computed from geometric margins between finger joints relative
    to overall hand scale (wrist to middle MCP distance). Unambiguous, fully-extended
    or clearly-folded joints yield scores near 1.0; borderline poses with narrow
    margins yield lower scores.

    Returns:
        Tuple of (gesture_key, confidence) or None if unrecognized or invalid.
    """
    if landmarks is None:
        return None

    try:
        if len(landmarks) != 21:
            return None
    except TypeError:
        return None

    parsed_points: List[Tuple[float, float]] = []
    for pt in landmarks:
        coords = _extract_coords(pt)
        if coords is None:
            return None
        parsed_points.append(coords)

    # Hand palm scale: distance between WRIST (0) and MIDDLE_MCP (9)
    wrist_x, wrist_y = parsed_points[WRIST]
    m_mcp_x, m_mcp_y = parsed_points[MIDDLE_MCP]
    hand_scale = math.hypot(m_mcp_x - wrist_x, m_mcp_y - wrist_y)
    if hand_scale < 0.05:
        hand_scale = 0.05

    # Finger tip vs pip coordinates (y=0 is top, y=1 is bottom)
    # Extended: tip.y < pip.y (pip.y - tip.y > 0)
    # Folded:   tip.y > pip.y (tip.y - pip.y > 0)
    index_ext_margin = (parsed_points[INDEX_PIP][1] - parsed_points[INDEX_TIP][1]) / hand_scale
    middle_ext_margin = (parsed_points[MIDDLE_PIP][1] - parsed_points[MIDDLE_TIP][1]) / hand_scale
    ring_ext_margin = (parsed_points[RING_PIP][1] - parsed_points[RING_TIP][1]) / hand_scale
    pinky_ext_margin = (parsed_points[PINKY_PIP][1] - parsed_points[PINKY_TIP][1]) / hand_scale

    index_extended = index_ext_margin > 0.0
    middle_extended = middle_ext_margin > 0.0
    ring_extended = ring_ext_margin > 0.0
    pinky_extended = pinky_ext_margin > 0.0

    # Folded margins (> 0 when folded)
    index_fold_margin = -index_ext_margin
    middle_fold_margin = -middle_ext_margin
    ring_fold_margin = -ring_ext_margin
    pinky_fold_margin = -pinky_ext_margin

    # Thumb posture
    thumb_tip_y = parsed_points[THUMB_TIP][1]
    thumb_ip_y = parsed_points[THUMB_IP][1]
    index_mcp_y = parsed_points[INDEX_MCP][1]

    # Thumb pointing up: tip is above IP joint and above index MCP joint
    thumb_up = (thumb_tip_y < thumb_ip_y) and (thumb_tip_y < index_mcp_y)
    thumb_up_margin = min(thumb_ip_y - thumb_tip_y, index_mcp_y - thumb_tip_y) / hand_scale
    thumb_fold_margin = max(thumb_tip_y - thumb_ip_y, thumb_tip_y - index_mcp_y) / hand_scale

    # Helper function to compute confidence score from required margins
    def _calc_conf(margin_specs: List[Tuple[float, float]]) -> float:
        # Scale each margin by its condition-specific reference margin
        ratios = [min(1.0, max(0.0, m / ref)) for m, ref in margin_specs]
        min_r = min(ratios)
        mean_r = sum(ratios) / len(ratios)
        # Combined score: bounded strictly in (0, 1]
        raw_score = 0.35 + 0.65 * (0.5 * min_r + 0.5 * mean_r)
        return round(min(1.0, max(0.05, raw_score)), 3)

    # 1. Open Palm -> "hello"
    # All four fingers extended upward
    if index_extended and middle_extended and ring_extended and pinky_extended:
        specs = [
            (index_ext_margin, 0.35),
            (middle_ext_margin, 0.35),
            (ring_ext_margin, 0.35),
            (pinky_ext_margin, 0.35),
        ]
        return ("hello", _calc_conf(specs))

    # 2. Thumbs Up -> "yes"
    # Thumb pointing up, all four fingers folded
    if thumb_up and (not index_extended) and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        specs = [
            (thumb_up_margin, 0.30),
            (index_fold_margin, 0.30),
            (middle_fold_margin, 0.30),
            (ring_fold_margin, 0.30),
            (pinky_fold_margin, 0.30),
        ]
        return ("yes", _calc_conf(specs))

    # 3. Fist -> "stop"
    # All four fingers folded, thumb not pointing up
    if (not thumb_up) and (not index_extended) and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        specs = [
            (thumb_fold_margin, 0.08),
            (index_fold_margin, 0.30),
            (middle_fold_margin, 0.30),
            (ring_fold_margin, 0.30),
            (pinky_fold_margin, 0.30),
        ]
        return ("stop", _calc_conf(specs))

    # 4. Peace Sign -> "thank_you"
    # Index and middle fingers extended, ring and pinky fingers folded
    if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
        specs = [
            (index_ext_margin, 0.35),
            (middle_ext_margin, 0.35),
            (ring_fold_margin, 0.30),
            (pinky_fold_margin, 0.30),
        ]
        return ("thank_you", _calc_conf(specs))

    return None


def classify_with_confidence(
    landmarks: Optional[Sequence[Any]],
) -> Optional[Tuple[str, float]]:
    """Pure function classifying 21 MediaPipe hand landmarks into a phrase and confidence.

    Args:
        landmarks: Sequence of 21 hand landmarks (objects with x,y, dicts, or tuples).

    Returns:
        Tuple of (phrase, confidence) where phrase is one of "Hello", "Yes", "Stop",
        "Thank you", or None if unrecognized or invalid. Confidence is in (0, 1].
    """
    result = classify_key_with_confidence(landmarks)
    if result is None:
        return None
    key, conf = result
    phrase = KEY_TO_PHRASE.get(key, key)
    return phrase, conf


def classify(landmarks: Optional[Sequence[Any]]) -> Optional[str]:
    """Pure function classifying 21 MediaPipe hand landmarks into a spoken phrase string.

    Backward-compatible with v1 test suite.

    Returns:
        One of "Hello", "Yes", "Stop", "Thank you", or None if input is invalid
        or the gesture is not recognized.
    """
    result = classify_with_confidence(landmarks)
    return result[0] if result is not None else None


def classify_key(landmarks: Optional[Sequence[Any]]) -> Optional[str]:
    """Pure function returning the canonical gesture key string ("hello", etc.) or None."""
    result = classify_key_with_confidence(landmarks)
    return result[0] if result is not None else None
