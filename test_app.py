"""Integration unit tests for app.py demo mode and CLI parser."""

import pytest
from app import run_demo, main


class TestAppIntegration:
    """Test suite ensuring headless demo mode and CLI entry points function properly."""

    def test_run_demo_returns_zero(self, capsys):
        exit_code = run_demo()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "ALL DEMO CHECKS PASSED SUCCESSFULLY" in captured.out
        assert "[PASS] (a) Stabilizer steady sequence" in captured.out
        assert "[PASS] (b) Flicker sequence" in captured.out
        assert "[PASS] (c) Double-palm temporal logic" in captured.out
        assert "[PASS] (d) SentenceBuilder" in captured.out
        assert "[PASS] (e) Config loader" in captured.out
        assert "[PASS] (f) SessionTranscript export" in captured.out
