"""Unit tests for Gesture-to-Speech classification and speech engine fallback."""

import pytest
from fixtures import (
    GESTURE_FIXTURES,
    get_open_palm_landmarks,
    get_thumbs_up_landmarks,
    get_fist_landmarks,
    get_peace_landmarks,
    get_unrecognized_hand_landmarks,
)
from gestures import classify
import speech


class TestGestureClassification:
    """Test suite for classify() pure function across all supported gestures."""

    @pytest.mark.parametrize(
        "fixture_key,expected_phrase",
        [
            ("open_palm", "Hello"),
            ("thumbs_up", "Yes"),
            ("fist", "Stop"),
            ("peace_sign", "Thank you"),
        ],
    )
    def test_all_supported_fixtures(self, fixture_key: str, expected_phrase: str):
        fixture = GESTURE_FIXTURES[fixture_key]
        landmarks = fixture["landmarks"]
        assert len(landmarks) == 21
        assert classify(landmarks) == expected_phrase

    def test_open_palm_hello(self):
        landmarks = get_open_palm_landmarks()
        assert classify(landmarks) == "Hello"

    def test_thumbs_up_yes(self):
        landmarks = get_thumbs_up_landmarks()
        assert classify(landmarks) == "Yes"

    def test_fist_stop(self):
        landmarks = get_fist_landmarks()
        assert classify(landmarks) == "Stop"

    def test_peace_sign_thank_you(self):
        landmarks = get_peace_landmarks()
        assert classify(landmarks) == "Thank you"

    def test_unrecognized_posture_returns_none(self):
        landmarks = get_unrecognized_hand_landmarks()
        assert classify(landmarks) is None

    def test_dict_landmark_format_support(self):
        """Ensure classify() handles dictionaries with 'x' and 'y' keys."""
        raw_landmarks = get_open_palm_landmarks()
        dict_landmarks = [{"x": pt.x, "y": pt.y} for pt in raw_landmarks]
        assert classify(dict_landmarks) == "Hello"

    def test_tuple_landmark_format_support(self):
        """Ensure classify() handles (x, y) tuples."""
        raw_landmarks = get_fist_landmarks()
        tuple_landmarks = [(pt.x, pt.y) for pt in raw_landmarks]
        assert classify(tuple_landmarks) == "Stop"


class TestEdgeCasesAndValidation:
    """Test suite for robust edge cases, None, and malformed inputs."""

    def test_none_input_returns_none(self):
        assert classify(None) is None

    def test_empty_list_returns_none(self):
        assert classify([]) is None

    def test_too_few_landmarks_returns_none(self):
        landmarks = get_open_palm_landmarks()[:20]  # 20 instead of 21
        assert classify(landmarks) is None

    def test_too_many_landmarks_returns_none(self):
        landmarks = get_open_palm_landmarks() + [get_open_palm_landmarks()[0]]  # 22 landmarks
        assert classify(landmarks) is None

    def test_invalid_element_types_returns_none(self):
        invalid_list = ["not-a-point"] * 21
        assert classify(invalid_list) is None

    def test_landmarks_with_none_elements_returns_none(self):
        landmarks = get_open_palm_landmarks()
        landmarks[5] = None
        assert classify(landmarks) is None

    def test_non_sequence_input_returns_none(self):
        assert classify(12345) is None
        assert classify("invalid_string") is None


class TestSpeechModule:
    """Test suite for safe, non-crashing speech module fallback."""

    def test_speak_executes_safely(self, capsys):
        speech.speak("Hello")
        captured = capsys.readouterr()
        assert "Hello" in captured.out

    def test_speak_handles_empty_or_none(self, capsys):
        speech.speak("")
        speech.speak(None)
        captured = capsys.readouterr()
        assert captured.out == ""
