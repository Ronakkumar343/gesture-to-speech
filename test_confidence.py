"""Unit tests for gesture confidence scoring and margins."""

import pytest
from fixtures import (
    create_hand,
    get_fist_landmarks,
    get_open_palm_landmarks,
    get_peace_landmarks,
    get_thumbs_up_landmarks,
    get_unrecognized_hand_landmarks,
)
from gestures import (
    classify,
    classify_key,
    classify_key_with_confidence,
    classify_with_confidence,
)


class TestConfidenceScoring:
    """Test suite for classify_with_confidence across clear and borderline poses."""

    def test_open_palm_confidence_near_one(self):
        landmarks = get_open_palm_landmarks()
        result = classify_with_confidence(landmarks)
        assert result is not None
        phrase, conf = result
        assert phrase == "Hello"
        assert 0.85 <= conf <= 1.0

    def test_thumbs_up_confidence_near_one(self):
        landmarks = get_thumbs_up_landmarks()
        result = classify_with_confidence(landmarks)
        assert result is not None
        phrase, conf = result
        assert phrase == "Yes"
        assert 0.85 <= conf <= 1.0

    def test_fist_confidence_near_one(self):
        landmarks = get_fist_landmarks()
        result = classify_with_confidence(landmarks)
        assert result is not None
        phrase, conf = result
        assert phrase == "Stop"
        assert 0.85 <= conf <= 1.0

    def test_peace_sign_confidence_near_one(self):
        landmarks = get_peace_landmarks()
        result = classify_with_confidence(landmarks)
        assert result is not None
        phrase, conf = result
        assert phrase == "Thank you"
        assert 0.85 <= conf <= 1.0

    def test_borderline_pose_scores_lower_confidence(self):
        """A borderline palm where finger tips barely clear PIPs should have lower confidence."""
        # Standard open palm has tips around 0.15-0.22, PIPs around 0.36-0.42 (margin > 0.15)
        # In a borderline palm, let's create fingertips just barely above pip joints
        borderline_hand = create_hand(
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
        # Override fingertips to be barely above pip
        from fixtures import Landmark

        # Index PIP is at y=0.38; set tip at 0.375 (barely 0.005 higher)
        borderline_hand[8] = Landmark(0.42, 0.375)
        # Middle PIP is at y=0.36; set tip at 0.355
        borderline_hand[12] = Landmark(0.50, 0.355)
        # Ring PIP is at y=0.38; set tip at 0.375
        borderline_hand[16] = Landmark(0.58, 0.375)
        # Pinky PIP is at y=0.42; set tip at 0.415
        borderline_hand[20] = Landmark(0.66, 0.415)

        result = classify_with_confidence(borderline_hand)
        assert result is not None
        phrase, conf = result
        assert phrase == "Hello"
        # Borderline confidence should be significantly lower than unambiguous (~0.95+)
        assert conf < 0.65
        assert conf > 0.0

    def test_classify_key_with_confidence(self):
        landmarks = get_thumbs_up_landmarks()
        res = classify_key_with_confidence(landmarks)
        assert res is not None
        key, conf = res
        assert key == "yes"
        assert conf > 0.8

    def test_classify_key(self):
        assert classify_key(get_fist_landmarks()) == "stop"
        assert classify_key(get_unrecognized_hand_landmarks()) is None

    def test_invalid_and_none_return_none(self):
        assert classify_with_confidence(None) is None
        assert classify_with_confidence([]) is None
        assert classify_with_confidence([None] * 21) is None
