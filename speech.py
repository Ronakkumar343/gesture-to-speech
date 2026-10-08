"""Non-blocking text-to-speech module with graceful fallback.

Uses pyttsx3 if importable and operational. If pyttsx3 is absent, fails to
initialize, or lacks an underlying sound synthesizer (e.g. eSpeak on Linux),
the module falls back silently to console output without interrupting execution.
"""

import queue
import threading
from typing import Optional

_ENGINE_AVAILABLE = False
_speech_queue: "queue.Queue[Optional[str]]" = queue.Queue()
_worker_thread: Optional[threading.Thread] = None

try:
    import pyttsx3

    try:
        # Test engine initialization
        _test_engine = pyttsx3.init()
        # Set a clear speaking rate if possible
        _test_engine.setProperty("rate", 160)
        _ENGINE_AVAILABLE = True
    except Exception as _e:
        _ENGINE_AVAILABLE = False
except ImportError:
    _ENGINE_AVAILABLE = False


def _speech_worker():
    """Background worker processing TTS requests without blocking video loop."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
    except Exception:
        return

    while True:
        text = _speech_queue.get()
        if text is None:
            _speech_queue.task_done()
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[SPEECH ERROR]: Audio output failed ({e})")
        finally:
            _speech_queue.task_done()


if _ENGINE_AVAILABLE:
    _worker_thread = threading.Thread(target=_speech_worker, daemon=True)
    _worker_thread.start()


def is_speech_available() -> bool:
    """Returns True if pyttsx3 initialized and is ready for audio output."""
    return _ENGINE_AVAILABLE


def speak(text: str) -> None:
    """Speaks the text aloud if TTS is available, and always logs it to stdout.

    Args:
        text: The phrase string to speak and display.
    """
    if not text:
        return

    # Always print to stdout for accessibility and headless debugging
    if _ENGINE_AVAILABLE:
        print(f"[SPEECH]: {text}")
        try:
            _speech_queue.put_nowait(text)
        except Exception:
            pass
    else:
        print(f"[SPEECH (Silent)]: {text}")
