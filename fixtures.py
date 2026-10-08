"""Shared synthetic landmark fixtures for Gesture-to-Speech.

Used by both test_gestures.py and app.py --demo to guarantee consistent
deterministic behavior across unit tests and live demo executions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0

    def __getitem__(self, item):
        if item in (0, "x"):
            return self.x
        if item in (1, "y"):
            return self.y
        if item in (2, "z"):
            return self.z
        raise IndexError(f"Invalid landmark index or key: {item!r}")


def create_hand(
    thumb_tip_y: float = 0.55,
    thumb_tip_x: float = 0.45,
    thumb_ip_y: float = 0.58,
    thumb_ip_x: float = 0.38,
    thumb_mcp_y: float = 0.65,
    thumb_mcp_x: float = 0.40,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
) -> List[Landmark]:
    """Builds a complete 21-point MediaPipe-compatible hand landmark set."""
    # 0: Wrist
    wrist = Landmark(0.50, 0.85)

    # 1-4: Thumb
    thumb_cmc = Landmark(0.46, 0.75)
    thumb_mcp = Landmark(thumb_mcp_x, thumb_mcp_y)
    thumb_ip = Landmark(thumb_ip_x, thumb_ip_y)
    thumb_tip = Landmark(thumb_tip_x, thumb_tip_y)

    # MCP bases for fingers
    index_mcp = Landmark(0.42, 0.52)
    middle_mcp = Landmark(0.50, 0.50)
    ring_mcp = Landmark(0.58, 0.52)
    pinky_mcp = Landmark(0.66, 0.56)

    # Index finger (5..8)
    if index_extended:
        index_pip = Landmark(0.42, 0.38)
        index_dip = Landmark(0.42, 0.28)
        index_tip = Landmark(0.42, 0.18)
    else:
        index_pip = Landmark(0.42, 0.42)
        index_dip = Landmark(0.42, 0.48)
        index_tip = Landmark(0.42, 0.54)

    # Middle finger (9..12)
    if middle_extended:
        middle_pip = Landmark(0.50, 0.36)
        middle_dip = Landmark(0.50, 0.25)
        middle_tip = Landmark(0.50, 0.15)
    else:
        middle_pip = Landmark(0.50, 0.40)
        middle_dip = Landmark(0.50, 0.46)
        middle_tip = Landmark(0.50, 0.52)

    # Ring finger (13..16)
    if ring_extended:
        ring_pip = Landmark(0.58, 0.38)
        ring_dip = Landmark(0.58, 0.28)
        ring_tip = Landmark(0.58, 0.18)
    else:
        ring_pip = Landmark(0.58, 0.42)
        ring_dip = Landmark(0.58, 0.48)
        ring_tip = Landmark(0.58, 0.54)

    # Pinky finger (17..20)
    if pinky_extended:
        pinky_pip = Landmark(0.66, 0.42)
        pinky_dip = Landmark(0.66, 0.32)
        pinky_tip = Landmark(0.66, 0.22)
    else:
        pinky_pip = Landmark(0.66, 0.46)
        pinky_dip = Landmark(0.66, 0.52)
        pinky_tip = Landmark(0.66, 0.58)

    return [
        wrist,        # 0
        thumb_cmc,    # 1
        thumb_mcp,    # 2
        thumb_ip,     # 3
        thumb_tip,    # 4
        index_mcp,    # 5
        index_pip,    # 6
        index_dip,    # 7
        index_tip,    # 8
        middle_mcp,   # 9
        middle_pip,   # 10
        middle_dip,   # 11
        middle_tip,   # 12
        ring_mcp,     # 13
        ring_pip,     # 14
        ring_dip,     # 15
        ring_tip,     # 16
        pinky_mcp,    # 17
        pinky_pip,    # 18
        pinky_dip,    # 19
        pinky_tip,    # 20
    ]


def get_open_palm_landmarks() -> List[Landmark]:
    """All fingers extended upward, thumb extended outward."""
    return create_hand(
        thumb_tip_y=0.45,
        thumb_tip_x=0.25,
        thumb_ip_y=0.52,
        thumb_ip_x=0.32,
        thumb_mcp_y=0.62,
        thumb_mcp_x=0.40,
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
    )


def get_thumbs_up_landmarks() -> List[Landmark]:
    """Thumb pointing straight up, four fingers folded into fist."""
    return create_hand(
        thumb_tip_y=0.30,
        thumb_tip_x=0.36,
        thumb_ip_y=0.45,
        thumb_ip_x=0.36,
        thumb_mcp_y=0.60,
        thumb_mcp_x=0.38,
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
    )


def get_fist_landmarks() -> List[Landmark]:
    """All 4 fingers folded, thumb tucked across fingers."""
    return create_hand(
        thumb_tip_y=0.55,
        thumb_tip_x=0.45,
        thumb_ip_y=0.58,
        thumb_ip_x=0.38,
        thumb_mcp_y=0.65,
        thumb_mcp_x=0.40,
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
    )


def get_peace_landmarks() -> List[Landmark]:
    """Index and middle fingers extended (V-shape), ring and pinky folded, thumb tucked."""
    return create_hand(
        thumb_tip_y=0.55,
        thumb_tip_x=0.45,
        thumb_ip_y=0.58,
        thumb_ip_x=0.38,
        thumb_mcp_y=0.65,
        thumb_mcp_x=0.40,
        index_extended=True,
        middle_extended=True,
        ring_extended=False,
        pinky_extended=False,
    )


def get_unrecognized_hand_landmarks() -> List[Landmark]:
    """Only pinky extended (not one of the supported phrases)."""
    return create_hand(
        thumb_tip_y=0.55,
        thumb_tip_x=0.45,
        thumb_ip_y=0.58,
        thumb_ip_x=0.38,
        thumb_mcp_y=0.65,
        thumb_mcp_x=0.40,
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=True,
    )


GESTURE_FIXTURES: Dict[str, Dict[str, Any]] = {
    "open_palm": {
        "description": "Open palm (all fingers extended)",
        "landmarks": get_open_palm_landmarks(),
        "expected_phrase": "Hello",
    },
    "thumbs_up": {
        "description": "Thumbs up (thumb pointing up, fingers folded)",
        "landmarks": get_thumbs_up_landmarks(),
        "expected_phrase": "Yes",
    },
    "fist": {
        "description": "Fist (all fingers folded)",
        "landmarks": get_fist_landmarks(),
        "expected_phrase": "Stop",
    },
    "peace_sign": {
        "description": "Peace sign (index and middle fingers extended)",
        "landmarks": get_peace_landmarks(),
        "expected_phrase": "Thank you",
    },
}
