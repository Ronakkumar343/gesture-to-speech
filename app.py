"""Gesture-to-Speech Main Application (v2).

Provides:
1. Live webcam feed with MediaPipe hand landmark detection, rule-based classification,
   confidence scoring, majority-voting stabilization, emergency double-palm detection,
   sentence builder, session transcript logging, visual HUD, debounce (~0.5s), and cooldown (~2s).
2. Keyboard controls:
   - 's': speak the full accumulated sentence
   - 'c': clear the sentence buffer
   - 'q': quit the application
3. Headless demo mode (`python app.py --demo`) running end-to-end checks on synthetic
   fixtures without needing a camera.
4. CLI flag `--export-transcript PATH`: on exit, exports the session transcript to a file.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from config import DEFAULT_PHRASES, load_phrases
from fixtures import (
    GESTURE_FIXTURES,
    get_open_palm_landmarks,
    get_thumbs_up_landmarks,
)
from gestures import classify, classify_with_confidence
from sentence import SentenceBuilder
from speech import is_speech_available, speak
from stabilizer import GestureStabilizer
from transcript import SessionTranscript


def run_demo() -> int:
    """Headless demo mode evaluating v1 fixture benchmarks and v2 end-to-end features."""
    print("=" * 72)
    print("           GESTURE-TO-SPEECH v2: END-TO-END SYSTEM DEMO")
    print("=" * 72)
    print(f"TTS Audio Engine Active: {is_speech_available()}")
    print("-" * 72)

    all_passed = True

    # ---------------------------------------------------------
    # PART 1: V1 Synthetic Landmark Classification Checks
    # ---------------------------------------------------------
    print("--- [PART 1: CLASSIFICATION FIXTURES] ---")
    for key, fixture in GESTURE_FIXTURES.items():
        landmarks = fixture["landmarks"]
        expected = fixture["expected_phrase"]
        desc = fixture["description"]

        detected = classify(landmarks)
        passed = detected == expected
        if not passed:
            all_passed = False

        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"{status_tag} {desc}")
        print(f"       Expected: {expected!r} | Detected: {detected!r}")
        if detected:
            speak(detected)
        print("-" * 72)

    # Edge cases demo
    none_result = classify(None)
    passed_none = none_result is None
    if not passed_none:
        all_passed = False
    print(f"[{'PASS' if passed_none else 'FAIL'}] No Hand (landmarks=None)")
    print(f"       Expected: None | Detected: {none_result!r}")
    print("-" * 72)

    invalid_result = classify([0.0] * 5)
    passed_invalid = invalid_result is None
    if not passed_invalid:
        all_passed = False
    print(f"[{'PASS' if passed_invalid else 'FAIL'}] Malformed Landmarks (len=5 != 21)")
    print(f"       Expected: None | Detected: {invalid_result!r}")
    print("-" * 72)

    # ---------------------------------------------------------
    # PART 2: V2 Pro-Level Feature Checks
    # ---------------------------------------------------------
    print("--- [PART 2: V2 END-TO-END PIPELINE CHECKS] ---")

    # (a) Stabilizer on steady synthetic sequence emits the right phrase
    stab_a = GestureStabilizer(window_size=5, min_confidence=0.6)
    palm_lms = get_open_palm_landmarks()
    res_a = None
    for _ in range(5):
        c_res = classify_with_confidence(palm_lms)
        p, c = c_res if c_res is not None else (None, 0.0)
        res_a = stab_a.update(p, c)
    passed_a = res_a == "Hello"
    if not passed_a:
        all_passed = False
    print(f"[{'PASS' if passed_a else 'FAIL'}] (a) Stabilizer steady sequence emits 'Hello'")
    print(f"       Expected: 'Hello' | Detected: {res_a!r}")
    print("-" * 72)

    # (b) Flicker sequence emits nothing
    stab_b = GestureStabilizer(window_size=5, min_confidence=0.6)
    palm_res = classify_with_confidence(get_open_palm_landmarks())
    thumbs_res = classify_with_confidence(get_thumbs_up_landmarks())
    flicker_emits = []
    for i in range(10):
        p, c = palm_res if i % 2 == 0 else thumbs_res  # type: ignore[assignment]
        out = stab_b.update(p, c)
        if out is not None:
            flicker_emits.append(out)
    passed_b = len(flicker_emits) == 0
    if not passed_b:
        all_passed = False
    print(f"[{'PASS' if passed_b else 'FAIL'}] (b) Flicker sequence emits nothing")
    print(f"       Emissions during flicker: {flicker_emits}")
    print("-" * 72)

    # (c) Double-palm produces 'I need help', single palm stays 'Hello', far apart stays 'Hello'
    sim_time = 100.0

    def sim_clock() -> float:
        return sim_time

    stab_c = GestureStabilizer(window_size=5, double_palm_window=2.5, clock=sim_clock)
    # Sub-check 1: Single palm held across frames
    single_palm_results = []
    for _ in range(5):
        sim_time += 0.03
        r = stab_c.update("Hello", 0.95)
        if r:
            single_palm_results.append(r)
    single_ok = (len(single_palm_results) > 0) and all(r == "Hello" for r in single_palm_results)

    # Sub-check 2: Release and double-palm within 2.5s
    sim_time += 0.5  # release hand
    for _ in range(3):
        stab_c.update(None, 0.0)
    sim_time += 0.3  # t = 100.0 + 0.15 + 0.5 + 0.3 = 100.95 (elapsed ~0.8s <= 2.5s)
    double_res = None
    for _ in range(3):
        double_res = stab_c.update("Hello", 0.95)
    double_ok = double_res == "I need help"

    # Sub-check 3: Release and second palm > 2.5s later
    sim_time += 0.5
    for _ in range(3):
        stab_c.update(None, 0.0)
    sim_time += 4.0  # elapsed 4s > 2.5s
    far_res = None
    for _ in range(3):
        far_res = stab_c.update("Hello", 0.95)
    far_ok = far_res == "Hello"

    passed_c = single_ok and double_ok and far_ok
    if not passed_c:
        all_passed = False
    print(f"[{'PASS' if passed_c else 'FAIL'}] (c) Double-palm temporal logic")
    print(
        f"       Single hold: {'PASS' if single_ok else 'FAIL'} | "
        f"Quick second: {double_res!r} ({'PASS' if double_ok else 'FAIL'}) | "
        f"Far apart: {far_res!r} ({'PASS' if far_ok else 'FAIL'})"
    )
    print("-" * 72)

    # (d) Sentence builder builds 'Hello Yes', speak_sentence returns it, clear empties
    sb = SentenceBuilder()
    sb.append("Hello")
    sb.append("Yes")
    built_text = sb.text
    spoken_list = []
    spoken_text = sb.speak_sentence(spoken_list.append)
    sb.clear()
    cleared_text = sb.text
    passed_d = (
        built_text == "Hello Yes"
        and spoken_text == "Hello Yes"
        and spoken_list == ["Hello Yes"]
        and cleared_text == ""
    )
    if not passed_d:
        all_passed = False
    print(f"[{'PASS' if passed_d else 'FAIL'}] (d) SentenceBuilder multi-word workflow")
    print(f"       Built: {built_text!r} | Spoken: {spoken_text!r} | Cleared: {cleared_text!r}")
    print("-" * 72)

    # (e) Config loader on shipped phrases.json returns default mapping
    cfg = load_phrases()
    passed_e = cfg == DEFAULT_PHRASES and cfg.get("help") == "I need help"
    if not passed_e:
        all_passed = False
    print(f"[{'PASS' if passed_e else 'FAIL'}] (e) Config loader on shipped phrases.json")
    print(f"       Loaded: {cfg}")
    print("-" * 72)

    # (f) Transcript export to a temp file contains logged phrases
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        temp_export_path = Path(tf.name)

    try:
        tr = SessionTranscript(clock=lambda: "2026-10-07 16:52:10")
        tr.log("Hello")
        tr.log("I need help")
        tr.export(temp_export_path)
        exported_content = temp_export_path.read_text(encoding="utf-8")
        passed_f = (
            "# Gesture-to-Speech Session Transcript" in exported_content
            and "[2026-10-07 16:52:10] Hello" in exported_content
            and "[2026-10-07 16:52:10] I need help" in exported_content
        )
    finally:
        if temp_export_path.exists():
            temp_export_path.unlink()

    if not passed_f:
        all_passed = False
    print(f"[{'PASS' if passed_f else 'FAIL'}] (f) SessionTranscript export file format")
    print("=" * 72)

    if all_passed:
        print("ALL DEMO CHECKS PASSED SUCCESSFULLY (v1 FIXTURES + v2 FEATURES).")
        return 0
    else:
        print("DEMO DETECTED FAILURES IN ONE OR MORE CHECKS.")
        return 1


def run_webcam(
    camera_index: int = 0,
    debounce_sec: float = 0.5,
    cooldown_sec: float = 2.0,
    export_transcript_path: Optional[str] = None,
    phrases_path: Optional[str] = None,
) -> int:
    """Live camera loop with MediaPipe Hands, stabilizer, sentence builder, and HUD."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[ERROR]: Could not open video device at index {camera_index}.")
        print("If you are running in a headless environment, container, or server without a webcam,")
        print("please run in headless demo mode instead:")
        print("    python app.py --demo\n")
        return 1

    # Load phrase mapping
    phrase_map = load_phrases(phrases_path)

    # Initialize v2 pipeline modules
    stabilizer = GestureStabilizer(phrase_map=phrase_map)
    sentence_builder = SentenceBuilder()
    transcript = SessionTranscript()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    current_candidate: Optional[str] = None
    candidate_start_time: float = 0.0
    last_trigger_time: float = 0.0
    active_display_phrase: Optional[str] = None
    active_display_until: float = 0.0

    print("=" * 60)
    print("Gesture-to-Speech live camera started (v2).")
    print(
        f"Debounce: {debounce_sec}s | Cooldown: {cooldown_sec}s | TTS Available: {is_speech_available()}"
    )
    print("Controls: 's': Speak Sentence | 'c': Clear Sentence | 'q': Quit")
    print("=" * 60)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[WARNING]: Ignoring empty camera frame.")
                continue

            now = time.time()
            # Flip horizontally for natural mirror display
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            raw_phrase: Optional[str] = None
            raw_conf: float = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    c_res = classify_with_confidence(hand_landmarks.landmark)
                    if c_res is not None:
                        raw_phrase, raw_conf = c_res
                    # Process only the first hand detected
                    break

            # Update temporal stabilizer
            stabilized_phrase = stabilizer.update(raw_phrase, raw_conf)

            # Debounce and Cooldown State Machine
            in_cooldown = (now - last_trigger_time) < cooldown_sec

            if stabilized_phrase is not None:
                if stabilized_phrase == current_candidate:
                    hold_duration = now - candidate_start_time
                    if hold_duration >= debounce_sec and not in_cooldown:
                        # Trigger phrase
                        speak(stabilized_phrase)
                        transcript.log(stabilized_phrase)
                        sentence_builder.append(stabilized_phrase)

                        last_trigger_time = now
                        active_display_phrase = stabilized_phrase
                        active_display_until = now + cooldown_sec

                        current_candidate = None
                        candidate_start_time = 0.0
                else:
                    current_candidate = stabilized_phrase
                    candidate_start_time = now
            else:
                current_candidate = None
                candidate_start_time = 0.0

            # Render HUD overlays
            # Header bar (height 90px)
            cv2.rectangle(frame, (0, 0), (w, 95), (30, 30, 30), -1)

            tts_status = "Active" if is_speech_available() else "Silent (Console)"
            cv2.putText(
                frame,
                f"Gesture-to-Speech v2 | TTS: {tts_status} | 's': speak | 'c': clear | 'q': quit",
                (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Stabilizer HUD metrics
            winner_str = stabilizer.last_winner or "None"
            votes_str = f"{stabilizer.winner_votes}/{stabilizer.window_size}"
            conf_str = f"{stabilizer.winner_confidence * 100:.0f}%"
            cv2.putText(
                frame,
                f"Stabilizer: [{winner_str}] | Conf: {conf_str} | Agree: {votes_str}",
                (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Sentence buffer HUD
            sentence_str = sentence_builder.text or "<empty>"
            cv2.putText(
                frame,
                f"Sentence: \"{sentence_str}\"",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 100),
                1,
                cv2.LINE_AA,
            )

            # Hold / Cooldown status line below header
            if stabilized_phrase is not None:
                hold_pct = (
                    min(1.0, (now - candidate_start_time) / debounce_sec)
                    if candidate_start_time > 0
                    else 0.0
                )
                status_str = f"Holding: {stabilized_phrase} ({hold_pct * 100:.0f}%)"
                if in_cooldown:
                    status_str = f"Cooldown: ({cooldown_sec - (now - last_trigger_time):.1f}s remaining)"
                cv2.putText(
                    frame,
                    status_str,
                    (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Spoken trigger banner at bottom
            if now < active_display_until and active_display_phrase:
                banner_text = f"SPOKEN: \"{active_display_phrase}\""
                cv2.rectangle(frame, (0, h - 70), (w, h), (0, 150, 0), -1)
                cv2.putText(
                    frame,
                    banner_text,
                    (25, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Gesture-to-Speech v2", frame)
            key = cv2.waitKey(5) & 0xFF

            if key == ord("q"):
                print("Exiting application...")
                break
            elif key == ord("c"):
                sentence_builder.clear()
                print("[APP]: Sentence buffer cleared.")
            elif key == ord("s"):
                spoken = sentence_builder.speak_sentence(speak)
                if spoken:
                    transcript.log(spoken)
                    active_display_phrase = spoken
                    active_display_until = now + cooldown_sec
                    print(f"[APP]: Spoke sentence: \"{spoken}\"")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        if export_transcript_path:
            out_file = transcript.export(export_transcript_path)
            print(f"[APP]: Session transcript exported to '{out_file}'")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gesture-to-Speech Application (v2)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run headless demo mode on synthetic landmarks and v2 pipeline (no camera needed)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.5,
        help="Hold duration in seconds before triggering (default: 0.5)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown duration in seconds after triggering (default: 2.0)",
    )
    parser.add_argument(
        "--export-transcript",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to export session transcript text file on exit",
    )
    parser.add_argument(
        "--phrases",
        type=str,
        default=None,
        metavar="PATH",
        help="Custom phrases JSON file path (default: phrases.json)",
    )

    args = parser.parse_args()

    if args.demo:
        return run_demo()
    return run_webcam(
        camera_index=args.camera,
        debounce_sec=args.debounce,
        cooldown_sec=args.cooldown,
        export_transcript_path=args.export_transcript,
        phrases_path=args.phrases,
    )


if __name__ == "__main__":
    sys.exit(main())
