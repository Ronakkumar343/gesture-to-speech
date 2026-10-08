"""Unit tests for SentenceBuilder module."""

import pytest
from sentence import SentenceBuilder


class TestSentenceBuilder:
    """Test suite for phrase accumulation, text joining, and speech dispatch."""

    def test_empty_builder_properties(self):
        builder = SentenceBuilder()
        assert builder.text == ""
        assert builder.phrases == []
        assert builder.last_spoken is None

    def test_append_valid_phrases(self):
        builder = SentenceBuilder()
        assert builder.append("Hello") is True
        assert builder.append("Yes") is True
        assert builder.text == "Hello Yes"
        assert builder.phrases == ["Hello", "Yes"]

    def test_append_ignores_none_and_empty(self):
        builder = SentenceBuilder()
        assert builder.append(None) is False
        assert builder.append("") is False
        assert builder.append("   ") is False
        assert builder.text == ""
        assert len(builder.phrases) == 0

    def test_clear_empties_buffer(self):
        builder = SentenceBuilder()
        builder.append("Hello")
        builder.append("Stop")
        assert builder.text == "Hello Stop"
        builder.clear()
        assert builder.text == ""
        assert builder.phrases == []

    def test_speak_sentence_dispatches_and_records(self):
        builder = SentenceBuilder()
        builder.append("Hello")
        builder.append("Thank you")

        spoken_calls = []

        def mock_speak(text: str):
            spoken_calls.append(text)

        result = builder.speak_sentence(mock_speak)
        assert result == "Hello Thank you"
        assert spoken_calls == ["Hello Thank you"]
        assert builder.last_spoken == "Hello Thank you"
        # Buffer remains until explicitly cleared
        assert builder.text == "Hello Thank you"

    def test_speak_sentence_empty_returns_none(self):
        builder = SentenceBuilder()
        spoken_calls = []

        result = builder.speak_sentence(spoken_calls.append)
        assert result is None
        assert spoken_calls == []
        assert builder.last_spoken is None
