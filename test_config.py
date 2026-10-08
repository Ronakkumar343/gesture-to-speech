"""Unit tests for config loader and phrases.json validation."""

import json
from pathlib import Path
import pytest
from config import DEFAULT_PHRASES, load_phrases


class TestConfigLoader:
    """Test suite for robust JSON phrase configuration loading."""

    def test_default_file_loads_shipped_mapping(self):
        phrases = load_phrases()
        assert phrases == DEFAULT_PHRASES
        assert phrases["hello"] == "Hello"
        assert phrases["help"] == "I need help"

    def test_missing_file_returns_defaults_with_warning(self, tmp_path: Path, capsys):
        missing_path = tmp_path / "non_existent.json"
        phrases = load_phrases(missing_path)
        assert phrases == DEFAULT_PHRASES
        captured = capsys.readouterr()
        assert "[CONFIG WARNING]" in captured.err

    def test_invalid_json_syntax_returns_defaults(self, tmp_path: Path, capsys):
        bad_json = tmp_path / "syntax_error.json"
        bad_json.write_text("{ hello: broken_json", encoding="utf-8")

        phrases = load_phrases(bad_json)
        assert phrases == DEFAULT_PHRASES
        captured = capsys.readouterr()
        assert "[CONFIG WARNING]" in captured.err

    def test_non_dict_root_returns_defaults(self, tmp_path: Path, capsys):
        list_json = tmp_path / "array_root.json"
        list_json.write_text('["hello", "world"]', encoding="utf-8")

        phrases = load_phrases(list_json)
        assert phrases == DEFAULT_PHRASES
        captured = capsys.readouterr()
        assert "[CONFIG WARNING]" in captured.err

    def test_custom_valid_phrases(self, tmp_path: Path):
        custom_file = tmp_path / "custom_phrases.json"
        custom_data = {
            "hello": "Greetings",
            "yes": "Affirmative",
            "stop": "Halt",
            "thank_you": "Many thanks",
            "help": "Emergency assistance",
        }
        custom_file.write_text(json.dumps(custom_data), encoding="utf-8")

        phrases = load_phrases(custom_file)
        assert phrases == custom_data

    def test_partial_keys_fill_with_defaults(self, tmp_path: Path, capsys):
        partial_file = tmp_path / "partial.json"
        partial_data = {"hello": "Hi there"}
        partial_file.write_text(json.dumps(partial_data), encoding="utf-8")

        phrases = load_phrases(partial_file)
        assert phrases["hello"] == "Hi there"
        # Others remain defaults
        assert phrases["yes"] == DEFAULT_PHRASES["yes"]
        assert phrases["help"] == DEFAULT_PHRASES["help"]

    def test_invalid_values_revert_to_default(self, tmp_path: Path, capsys):
        invalid_val_file = tmp_path / "invalid_vals.json"
        invalid_val_file.write_text(
            json.dumps({"hello": "", "yes": 1234, "stop": "   "}), encoding="utf-8"
        )

        phrases = load_phrases(invalid_val_file)
        assert phrases["hello"] == DEFAULT_PHRASES["hello"]
        assert phrases["yes"] == DEFAULT_PHRASES["yes"]
        assert phrases["stop"] == DEFAULT_PHRASES["stop"]
        captured = capsys.readouterr()
        assert "[CONFIG WARNING]" in captured.err

    def test_unknown_keys_ignored_with_warning(self, tmp_path: Path, capsys):
        extra_keys_file = tmp_path / "extra_keys.json"
        extra_data = {
            "hello": "Hello",
            "unknown_gesture": "Should be ignored",
        }
        extra_keys_file.write_text(json.dumps(extra_data), encoding="utf-8")

        phrases = load_phrases(extra_keys_file)
        assert "unknown_gesture" not in phrases
        captured = capsys.readouterr()
        assert "[CONFIG WARNING]" in captured.err
        assert "unknown_gesture" in captured.err
