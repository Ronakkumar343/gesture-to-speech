# Gesture-to-Speech (v2)

A production-grade Python application that translates static hand gestures into spoken phrases, multi-word sentences, and text overlays for individuals who cannot speak.

The classification engine relies entirely on deterministic, rule-based joint geometry calculated from 21 MediaPipe hand landmarks. There are no black-box machine learning models or probabilistic classifiers used during runtime classification, ensuring deterministic predictability and 100% headless unit testability.

---

## What's New in v2

1. **Geometric Confidence Scoring ([gestures.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/gestures.py))**:
   `classify_with_confidence(landmarks)` measures joint displacement margins relative to hand palm scale (wrist to middle MCP distance). Fully-extended and clearly-folded poses score near `1.0`, while borderline poses score lower.
2. **Sliding-Window Majority Voting & Flicker Suppression ([stabilizer.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/stabilizer.py))**:
   `GestureStabilizer` majority-votes over a sliding buffer of $N$ frames (default 5, agreement threshold $\ge 3$) and gates outputs against minimum confidence (default 0.6). Alternating frames (flicker) are actively suppressed and never emit false triggers.
3. **Emergency "I Need Help" via Double-Palm ([stabilizer.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/stabilizer.py))**:
   Distinguishing spread fingers from standard open palm is fragile under 2D camera perspective variation. Instead, v2 implements an intentional temporal rule: presenting an open palm twice within a 2.5-second window converts the second emission into `"I need help"`. A single continuous palm always remains `"Hello"`. Two palms spaced further apart remain two separate `"Hello"`s.
4. **Sentence Builder ([sentence.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/sentence.py))**:
   `SentenceBuilder` accumulates recognized phrases into cohesive multi-word sentences (e.g. `"Hello Yes Thank you"`), with dedicated actions for clearing and vocal dispatch.
5. **Config-Driven Vocabulary ([phrases.json](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/phrases.json), [config.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/config.py))**:
   Spoken text is decoupled from geometric keys (`"hello"`, `"yes"`, `"stop"`, `"thank_you"`, `"help"`). Remapping phrases in `phrases.json` never touches core logic.
6. **Session Transcript Logging & Export ([transcript.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/transcript.py))**:
   `SessionTranscript` timestamps every spoken phrase and sentence, with CLI export via `--export-transcript <path>`.

---

## Supported Gestures

| Gesture | Geometric Rules | Temporal Rule | Spoken Phrase |
|---|---|---|---|
| **Open Palm** | All 4 fingers extended upward (`tip.y < pip.y`) | Single activation | `"Hello"` |
| **Double Palm** | Open palm activated twice | $\le 2.5\text{s}$ between emissions | `"I need help"` |
| **Thumbs Up** | Thumb pointing upward (`tip.y < ip.y` and `tip.y < index_mcp.y`), all 4 fingers folded | Single activation | `"Yes"` |
| **Fist** | All 4 fingers folded (`tip.y > pip.y`), thumb folded/tucked | Single activation | `"Stop"` |
| **Peace Sign (V)** | Index and middle fingers extended upward, ring and pinky fingers folded | Single activation | `"Thank you"` |

---

## Phrase Configuration (`phrases.json`)

Phrases are mapped in `phrases.json`:
```json
{
  "hello": "Hello",
  "yes": "Yes",
  "stop": "Stop",
  "thank_you": "Thank you",
  "help": "I need help"
}
```

Validation behavior in [config.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/config.py):
- **Missing file or syntax error**: Falls back to built-in defaults with a warning to `stderr`.
- **Missing keys or empty values**: Reverts individual invalid entries to defaults with a warning.
- **Unknown keys**: Documented policy: ignored with a warning to `stderr`.

---

## Project Architecture

- [gestures.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/gestures.py): Pure functions `classify()`, `classify_with_confidence()`, and key classifiers based on 21 MediaPipe coordinates.
- [stabilizer.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/stabilizer.py): `GestureStabilizer` (sliding-window voting, confidence filtering, flicker suppression, HUD metrics) and `DoublePalmStateMachine`.
- [sentence.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/sentence.py): `SentenceBuilder` for accumulating, joining, clearing, and speaking full sentences.
- [config.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/config.py): Validates and loads phrase dictionaries from `phrases.json`.
- [transcript.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/transcript.py): `SessionTranscript` for logging timestamped events and exporting to `.txt`.
- [speech.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/speech.py): Non-blocking text-to-speech engine wrapper with silent console fallback when TTS is absent.
- [fixtures.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/fixtures.py): Synthetic landmark vectors used deterministically across tests and `--demo`.
- [app.py](file:///home/hatch/workspace/agy_workspace/gesture_to_speech/app.py): Live OpenCV + MediaPipe webcam loop with visual HUD, keyboard controls, and end-to-end `--demo` mode.

---

## Installation & Requirements

Requirements:
- Python 3.10+
- `opencv-python`
- `mediapipe`
- `pyttsx3` *(Optional)*: Required for offline audio playback. If absent or if OS sound drivers are unavailable, the application falls back silently to console output and text overlays.

Install dependencies:
```bash
pip install -r requirements.txt
```

*(Optional Linux audio drivers):*
```bash
sudo apt-get install espeak-ng
```

---

## How to Run

### 1. Headless Demo Mode (No Camera or Display Needed)
Runs the complete benchmark suite, verifying all v1 classification fixtures and v2 pipeline features:
```bash
.venv/bin/python app.py --demo
```

### 2. Live Webcam Mode
Runs the real-time webcam feed with landmark tracking, visual HUD, and sentence builder:
```bash
.venv/bin/python app.py
```

#### Keyboard Controls (in webcam window):
- `s`: Speak the full accumulated sentence and record it to the transcript.
- `c`: Clear the sentence buffer.
- `q`: Quit the application.

#### CLI Flags:
- `--demo`: Run headless fixture and pipeline checks.
- `--camera <id>`: Video device index (default: `0`).
- `--debounce <sec>`: Hold duration before gesture trigger (default: `0.5`).
- `--cooldown <sec>`: Cooldown period after gesture trigger (default: `2.0`).
- `--export-transcript <path>`: Write session transcript to file upon exit.
- `--phrases <path>`: Path to a custom phrases JSON configuration.

Example:
```bash
.venv/bin/python app.py --camera 0 --debounce 0.5 --cooldown 2.0 --export-transcript session_log.txt
```

### 3. Running the Test Suite
Run the full headless unit and integration test suite:
```bash
.venv/bin/python -m pytest
```

---

## Honest Verification Status

In accordance with workspace build protocols, here is the exact record of verified functionality:

- [x] **Headless Test Suite (pytest)**: Fully verified (`57 passed in 0.28s`).
  - 20/20 v1 tests passed without regression (`test_gestures.py`).
  - Confidence scoring and joint margin calculations verified (`test_confidence.py`).
  - Sliding-window smoothing, vote thresholds, and flicker suppression verified (`test_stabilizer.py`).
  - Double-palm emergency timing, single-palm hold, and spacing boundaries verified (`test_stabilizer.py`).
  - Sentence accumulation, clear, and speak verified (`test_sentence.py`).
  - Config loading, missing file fallback, malformed JSON recovery, and key isolation verified (`test_config.py`).
  - Session transcript logging, injectable clock, and export file formatting verified (`test_transcript.py`).
  - Demo CLI integration test verified (`test_app.py`).
- [x] **Headless End-to-End Demo (`app.py --demo`)**: Fully verified (exit code `0`).
  - Passed all 4 synthetic gesture fixtures + null + malformed inputs.
  - Passed all 6 v2 checks: steady sequence stabilization, flicker suppression, double-palm temporal logic, sentence builder workflow, config loader, and transcript export.
- [x] **Graceful Silent Speech Fallback**: Fully verified. In this headless container environment lacking `espeak-ng`, `speech.py` detects unavailable audio, suppresses crashes, logs to stdout, and reports `is_speech_available() == False`.
- [ ] **Physical Webcam Video Loop & Audio Speaker Output**: NOT physically verified on this machine due to lack of a physical hardware camera (`/dev/video*`) and physical audio speakers in the container environment. All interactive logic has been isolated and validated via deterministic headless state machines and mocks.
