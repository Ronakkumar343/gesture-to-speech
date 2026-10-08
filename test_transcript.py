"""Unit tests for SessionTranscript logging and file export."""

from datetime import datetime
from pathlib import Path
import pytest
from transcript import SessionTranscript


class TestSessionTranscript:
    """Test suite for timestamped session logging and text export."""

    def test_log_with_injectable_clock(self):
        fixed_dt = datetime(2026, 10, 7, 16, 52, 10)
        transcript = SessionTranscript(clock=lambda: fixed_dt)

        assert transcript.log("Hello") is True
        assert transcript.log("Yes") is True
        assert len(transcript.entries) == 2
        assert transcript.entries[0] == ("2026-10-07 16:52:10", "Hello")
        assert transcript.entries[1] == ("2026-10-07 16:52:10", "Yes")

    def test_log_ignores_empty_and_none(self):
        transcript = SessionTranscript()
        assert transcript.log(None) is False
        assert transcript.log("") is False
        assert transcript.log("   ") is False
        assert len(transcript.entries) == 0

    def test_export_file_content(self, tmp_path: Path):
        fixed_dt = datetime(2026, 10, 7, 16, 52, 10)
        transcript = SessionTranscript(clock=lambda: fixed_dt)

        transcript.log("Hello")
        transcript.log("I need help")

        export_file = tmp_path / "subdir" / "transcript.txt"
        exported_path = transcript.export(export_file)

        assert Path(exported_path).is_file()
        content = Path(exported_path).read_text(encoding="utf-8")

        lines = content.strip().splitlines()
        assert len(lines) == 3
        assert lines[0] == "# Gesture-to-Speech Session Transcript"
        assert lines[1] == "[2026-10-07 16:52:10] Hello"
        assert lines[2] == "[2026-10-07 16:52:10] I need help"
